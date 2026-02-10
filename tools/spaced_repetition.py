#!/usr/bin/env python3
"""
间隔重复模块 — 基于 SM-2 算法（SuperMemo-2）
实现艾宾浩斯遗忘曲线的自动化复习调度
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable


class SpacedRepetitionManager:
    """SM-2 间隔重复算法管理器"""

    REVIEW_FILE = Path(__file__).parent.parent / 'progress' / 'review_cards.json'

    # SM-2 默认参数
    DEFAULT_EF = 2.5
    MIN_EF = 1.3
    MATURE_THRESHOLD = 21  # interval >= 21天视为"成熟"

    # 概念提取配置
    MIN_CONCEPT_LENGTH = 2
    MAX_CONCEPT_LENGTH = 100  # 从 50 提升到 100

    # 每日复习量限制
    MAX_DAILY_REVIEWS = 50

    def __init__(self):
        self.data = self._load_data()

    # ===== 数据持久化 =====

    def _load_data(self) -> dict:
        """加载 review_cards.json，支持备份恢复"""
        if self.REVIEW_FILE.exists():
            try:
                with open(self.REVIEW_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 验证数据结构
                    if 'cards' not in data or 'stats' not in data:
                        raise ValueError('数据结构不完整')
                    return data
            except (json.JSONDecodeError, ValueError) as e:
                # 尝试从备份恢复
                backup_file = self.REVIEW_FILE.with_suffix('.json.bak')
                if backup_file.exists():
                    try:
                        print(f'⚠️  主文件损坏，尝试从备份恢复: {e}')
                        with open(backup_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            # 恢复成功，覆盖主文件
                            self.data = data
                            self._save_data()
                            print('✅ 已从备份恢复')
                            return data
                    except Exception as backup_error:
                        print(f'❌ 备份文件也损坏: {backup_error}')

                # 无法恢复，重置为空
                print(f'⚠️  无法恢复，重置为空数据（原数据已备份到 .corrupted）')
                corrupted = self.REVIEW_FILE.with_suffix('.json.corrupted')
                self.REVIEW_FILE.rename(corrupted)
                return self._default_data()

        return self._default_data()

    def _save_data(self):
        """保存到 review_cards.json，自动创建备份"""
        self.REVIEW_FILE.parent.mkdir(parents=True, exist_ok=True)

        # 如果主文件存在，先备份
        if self.REVIEW_FILE.exists():
            backup_file = self.REVIEW_FILE.with_suffix('.json.bak')
            import shutil
            shutil.copy2(self.REVIEW_FILE, backup_file)

        # 保存主文件
        with open(self.REVIEW_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _default_data() -> dict:
        return {
            'version': 1,
            'cards': {},
            'stats': {
                'total_reviews': 0,
                'average_quality': 0
            }
        }

    # ===== SM-2 核心算法 =====

    @staticmethod
    def sm2(quality: int, repetitions: int, interval: int, ef: float) -> Tuple[int, int, float]:
        """
        SM-2 算法核心计算

        Args:
            quality: 评分 0-5 (0=完全忘记, 5=完美回忆)
            repetitions: 当前连续正确次数
            interval: 当前间隔（天）
            ef: 当前 easiness factor

        Returns:
            (new_repetitions, new_interval, new_ef)
        """
        # 更新 EF
        new_ef = ef + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
        new_ef = max(1.3, round(new_ef, 2))

        if quality >= 3:
            # 记住了
            if repetitions == 0:
                new_interval = 1
            elif repetitions == 1:
                new_interval = 6
            else:
                new_interval = round(interval * new_ef)
            new_repetitions = repetitions + 1
        else:
            # 忘了，重置
            new_repetitions = 0
            new_interval = 1

        return new_repetitions, new_interval, new_ef

    # ===== 概念提取 =====

    @classmethod
    def extract_concepts(cls, morning_theory: str) -> Tuple[List[str], str]:
        """
        从课表 morning_theory 文本中提取概念和上下文

        格式示例:
            "3Blue1Brown第5-8集\n• 行列式 • 逆矩阵/列空间/零空间\n• 非方阵的变换 • 点积的几何意义"

        Returns:
            (concepts, context)
        """
        if not morning_theory or not morning_theory.strip():
            return [], ''

        lines = morning_theory.strip().split('\n')
        context_parts = []
        concept_parts = []
        filtered_concepts = []  # 记录被过滤的概念

        for line in lines:
            line = line.strip()
            if not line:
                continue
            if '•' in line:
                parts = line.split('•')
                for part in parts:
                    part = part.strip()
                    if not part:
                        continue

                    # 检查长度
                    if len(part) < cls.MIN_CONCEPT_LENGTH:
                        filtered_concepts.append((part, '太短'))
                        continue
                    if len(part) > cls.MAX_CONCEPT_LENGTH:
                        filtered_concepts.append((part, '太长'))
                        continue

                    concept_parts.append(part)
            else:
                context_parts.append(line)

        # 记录被过滤的概念
        if filtered_concepts:
            for concept, reason in filtered_concepts:
                print(f'⚠️  概念被过滤（{reason}）: {concept[:50]}...' if len(concept) > 50 else f'⚠️  概念被过滤（{reason}）: {concept}')

        context = '; '.join(context_parts) if context_parts else ''

        # 去重保序
        seen = set()
        unique = []
        for c in concept_parts:
            if c not in seen:
                seen.add(c)
                unique.append(c)

        return unique, context

    # ===== 卡片管理 =====

    def create_card(self, concept: str, week: int, day: int,
                    context: str = '') -> bool:
        """
        创建一张复习卡片

        Args:
            concept: 概念名称（保留原始格式，包括特殊字符如斜杠）
            week: 来源周
            day: 来源天
            context: 学习上下文

        Returns:
            True=新建, False=已存在

        Note:
            概念名称作为字典键使用原始格式（如 "逆矩阵/列空间/零空间"），
            不做任何清理。文件名清理由 obsidian_integration.py 负责。
        """
        if concept in self.data['cards']:
            return False

        today = datetime.now().strftime('%Y-%m-%d')
        tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')

        self.data['cards'][concept] = {
            'concept': concept,
            'source_week': week,
            'source_day': day,
            'source_context': context,
            'created_at': today,
            'easiness_factor': self.DEFAULT_EF,
            'interval': 1,
            'repetitions': 0,
            'next_review': tomorrow,
            'last_review': None,
            'last_quality': None,
            'review_count': 0,
            'history': []
        }
        return True

    def create_cards_from_day(self, week: int, day: int,
                              morning_theory: str) -> List[str]:
        """
        从某天的 morning_theory 批量创建卡片

        Returns:
            新创建的概念列表
        """
        concepts, context = self.extract_concepts(morning_theory)
        new_cards = []

        for concept in concepts:
            if self.create_card(concept, week, day, context):
                new_cards.append(concept)

        if new_cards:
            self._save_data()

        return new_cards

    # ===== 复习流程 =====

    def get_due_cards(self, date_str: str = None, limit: int = None) -> List[dict]:
        """
        获取到期的卡片列表（按紧急度排序：过期越久越靠前）

        Args:
            date_str: 日期字符串 YYYY-MM-DD，默认今天
            limit: 最大返回数量，默认使用 MAX_DAILY_REVIEWS

        Returns:
            按优先级排序的卡片列表
        """
        if date_str is None:
            date_str = datetime.now().strftime('%Y-%m-%d')
        if limit is None:
            limit = self.MAX_DAILY_REVIEWS

        due = []
        for key, card in self.data['cards'].items():
            next_review = card.get('next_review', '')
            if next_review and next_review <= date_str:
                overdue_days = (datetime.strptime(date_str, '%Y-%m-%d')
                                - datetime.strptime(next_review, '%Y-%m-%d')).days
                due.append({
                    'key': key,
                    'concept': card['concept'],
                    'source_week': card['source_week'],
                    'source_day': card['source_day'],
                    'source_context': card.get('source_context', ''),
                    'overdue_days': overdue_days,
                    'review_count': card['review_count'],
                    'easiness_factor': card['easiness_factor'],
                    'interval': card['interval']
                })

        # 按优先级排序：过期天数（降序）+ EF（升序，难的优先）
        due.sort(key=lambda x: (-x['overdue_days'], x['easiness_factor']))

        # 限制返回数量
        if limit > 0 and len(due) > limit:
            return due[:limit]

        return due

    def review_card(self, concept_key: str, quality: int) -> dict:
        """
        对一张卡片评分

        Args:
            concept_key: 概念名称
            quality: 0-5

        Returns:
            更新后的卡片摘要
        """
        quality = max(0, min(5, quality))

        if concept_key not in self.data['cards']:
            return {'error': f'卡片不存在: {concept_key}'}

        card = self.data['cards'][concept_key]
        old_ef = card['easiness_factor']
        old_interval = card['interval']

        # SM-2 计算
        new_reps, new_interval, new_ef = self.sm2(
            quality,
            card['repetitions'],
            card['interval'],
            card['easiness_factor']
        )

        # 更新卡片
        today = datetime.now().strftime('%Y-%m-%d')
        next_review = (datetime.now() + timedelta(days=new_interval)).strftime('%Y-%m-%d')

        card['repetitions'] = new_reps
        card['interval'] = new_interval
        card['easiness_factor'] = new_ef
        card['next_review'] = next_review
        card['last_review'] = today
        card['last_quality'] = quality
        card['review_count'] += 1
        card['history'].append({
            'date': today,
            'quality': quality,
            'interval': new_interval,
            'ef': new_ef
        })

        # 更新全局统计
        self.data['stats']['total_reviews'] += 1
        total = self.data['stats']['total_reviews']
        avg = self.data['stats']['average_quality']
        self.data['stats']['average_quality'] = round(
            avg + (quality - avg) / total, 2
        )

        self._save_data()

        return {
            'concept': concept_key,
            'quality': quality,
            'old_interval': old_interval,
            'new_interval': new_interval,
            'old_ef': old_ef,
            'new_ef': new_ef,
            'next_review': next_review,
            'repetitions': new_reps,
            'review_count': card['review_count'],
            'status': '记住了' if quality >= 3 else '需要重学'
        }

    # ===== 统计 =====

    def get_review_stats(self) -> dict:
        """获取复习统计"""
        cards = self.data['cards']
        today = datetime.now().strftime('%Y-%m-%d')

        total = len(cards)
        due_today = sum(1 for c in cards.values()
                        if c.get('next_review', '') <= today)
        overdue = sum(1 for c in cards.values()
                      if c.get('next_review', '') < today)
        mature = sum(1 for c in cards.values()
                     if c['interval'] >= self.MATURE_THRESHOLD)
        young = total - mature
        reviewed = sum(1 for c in cards.values() if c['review_count'] > 0)
        never_reviewed = total - reviewed

        avg_ef = 0
        if total > 0:
            avg_ef = round(sum(c['easiness_factor'] for c in cards.values()) / total, 2)

        return {
            'total_cards': total,
            'due_today': due_today,
            'overdue': overdue,
            'mature': mature,
            'young': young,
            'reviewed_at_least_once': reviewed,
            'never_reviewed': never_reviewed,
            'average_ef': avg_ef,
            'total_reviews': self.data['stats']['total_reviews'],
            'average_quality': self.data['stats']['average_quality']
        }

    def get_week_review_summary(self, week: int) -> dict:
        """获取指定周创建的卡片复习摘要"""
        week_cards = [c for c in self.data['cards'].values()
                      if c['source_week'] == week]

        total = len(week_cards)
        if total == 0:
            return {'total_concepts': 0, 'reviewed': 0, 'mastered': 0, 'average_quality': 0}

        reviewed = sum(1 for c in week_cards if c['review_count'] > 0)
        mastered = sum(1 for c in week_cards if c['interval'] >= self.MATURE_THRESHOLD)

        qualities = [c['last_quality'] for c in week_cards if c['last_quality'] is not None]
        avg_q = round(sum(qualities) / len(qualities), 1) if qualities else 0

        return {
            'total_concepts': total,
            'reviewed': reviewed,
            'mastered': mastered,
            'average_quality': avg_q
        }

    def get_card_detail(self, concept_key: str) -> Optional[dict]:
        """获取单张卡片详情"""
        return self.data['cards'].get(concept_key)

    def get_learning_analytics(self) -> dict:
        """
        生成学习分析报告

        Returns:
            包含掌握度分布、复习量预测、趋势分析的字典
        """
        cards = list(self.data['cards'].values())
        if not cards:
            return {
                'total_concepts': 0,
                'mastery_distribution': {'struggling': 0, 'learning': 0, 'mastered': 0},
                'review_forecast': [],
                'average_interval': 0,
                'retention_estimate': 0
            }

        # 1. 掌握度分布
        struggling = sum(1 for c in cards if c['easiness_factor'] < 2.0)
        learning = sum(1 for c in cards if 2.0 <= c['easiness_factor'] < 2.5)
        mastered = sum(1 for c in cards if c['easiness_factor'] >= 2.5)

        # 2. 未来7天复习量预测
        today = datetime.now()
        forecast = []
        for day_offset in range(7):
            target_date = (today + timedelta(days=day_offset)).strftime('%Y-%m-%d')
            due_count = sum(1 for c in cards if c.get('next_review', '') == target_date)
            forecast.append({
                'date': target_date,
                'count': due_count
            })

        # 3. 平均复习间隔
        intervals = [c['interval'] for c in cards if c['review_count'] > 0]
        avg_interval = round(sum(intervals) / len(intervals), 1) if intervals else 0

        # 4. 记忆保持率估算（基于平均 EF）
        avg_ef = sum(c['easiness_factor'] for c in cards) / len(cards)
        # EF 越高，保持率越好（简化估算：EF 2.5 = 85%，每 0.1 差异 ±2%）
        retention_estimate = min(100, max(0, 85 + (avg_ef - 2.5) * 20))

        return {
            'total_concepts': len(cards),
            'mastery_distribution': {
                'struggling': struggling,
                'learning': learning,
                'mastered': mastered
            },
            'review_forecast': forecast,
            'average_interval': avg_interval,
            'retention_estimate': round(retention_estimate, 1),
            'average_ef': round(avg_ef, 2)
        }

    # ===== 补建卡片 =====

    def backfill_from_tracker(self, tracker: dict, load_schedule_fn: Callable) -> List[str]:
        """
        为已完成但未建卡的天数补建卡片

        Args:
            tracker: tracker.json 数据
            load_schedule_fn: 返回课表列表的函数

        Returns:
            新建的概念列表
        """
        all_new = []
        done_days = {k: v for k, v in tracker.get('days', {}).items()
                     if v.get('status') == 'done'}

        if not done_days:
            return all_new

        # 检查哪些天还没建卡
        existing_sources = set()
        for card in self.data['cards'].values():
            existing_sources.add(f"W{card['source_week']}D{card['source_day']}")

        days_to_backfill = [k for k in done_days if k not in existing_sources]
        if not days_to_backfill:
            return all_new

        # 加载课表
        schedule = load_schedule_fn()

        for day_key in days_to_backfill:
            # 解析 W1D1 格式
            try:
                if not day_key.startswith('W') or 'D' not in day_key:
                    print(f'⚠️  跳过格式错误的 day_key: {day_key}')
                    continue

                parts = day_key.replace('W', '').split('D')
                if len(parts) != 2:
                    print(f'⚠️  跳过无法解析的 day_key: {day_key}')
                    continue

                week, day = int(parts[0]), int(parts[1])

                # 验证范围
                if not (1 <= week <= 50 and 1 <= day <= 6):
                    print(f'⚠️  跳过超出范围的 day_key: {day_key} (W{week}D{day})')
                    continue

            except (ValueError, IndexError) as e:
                print(f'⚠️  解析 day_key 失败: {day_key}, 错误: {e}')
                continue

            # 查找课表
            for item in schedule:
                if item['week'] == week and item['day'] == day:
                    mt = item.get('morning_theory', '')
                    if mt:
                        new = self.create_cards_from_day(week, day, mt)
                        all_new.extend(new)
                    break

        return all_new
