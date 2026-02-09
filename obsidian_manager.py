#!/usr/bin/env python3
"""
Obsidian笔记管理工具
功能：整理笔记、创建索引、生成MOC、清理重复
"""

import json
from pathlib import Path
from datetime import datetime
import re

class ObsidianManager:
    def __init__(self, vault_path: str = None):
        if vault_path is None:
            self.vault = Path(__file__).parent / "obsidian-vault"
        else:
            self.vault = Path(vault_path)

        self.daily_dir = self.vault / "00-Daily"
        self.concepts_dir = self.vault / "01-Concepts"
        self.projects_dir = self.vault / "02-Projects"
        self.quizzes_dir = self.vault / "03-Quizzes"
        self.reviews_dir = self.vault / "04-Reviews"
        self.moc_dir = self.vault / "99-MOC"

    def scan_vault(self):
        """扫描vault中所有笔记"""
        print("🔍 扫描Obsidian笔记库...")
        print("=" * 50)

        # 统计各类笔记
        stats = {
            "daily": [],
            "concepts": [],
            "projects": [],
            "quizzes": [],
            "reviews": [],
            "others": [],
            "duplicates": []
        }

        all_files = list(self.vault.rglob("*.md"))
        # 排除模板和隐藏文件
        all_files = [f for f in all_files
                     if ".templates" not in str(f)
                     and ".obsidian" not in str(f)
                     and f.name != "README.md"]

        for file in all_files:
            rel_path = file.relative_to(self.vault)

            if "00-Daily" in str(file):
                stats["daily"].append(file)
            elif "01-Concepts" in str(file):
                stats["concepts"].append(file)
            elif "02-Projects" in str(file):
                stats["projects"].append(file)
            elif "03-Quizzes" in str(file):
                stats["quizzes"].append(file)
            elif "04-Reviews" in str(file):
                stats["reviews"].append(file)
            else:
                stats["others"].append(file)

        # 检测重复文件
        seen = {}
        for file in all_files:
            name = file.name
            if name in seen and name != "README.md":
                stats["duplicates"].append((seen[name], file))
            else:
                seen[name] = file

        # 打印统计
        print(f"📅 日记: {len(stats['daily'])} 篇")
        print(f"💡 概念: {len(stats['concepts'])} 篇")
        print(f"🚀 项目: {len(stats['projects'])} 篇")
        print(f"📝 测验: {len(stats['quizzes'])} 篇")
        print(f"📊 周回顾: {len(stats['reviews'])} 篇")
        print(f"📄 其他: {len(stats['others'])} 篇")

        if stats["duplicates"]:
            print(f"\n⚠️  发现 {len(stats['duplicates'])} 组重复文件:")
            for f1, f2 in stats["duplicates"]:
                print(f"   - {f1.name} @ {f1.parent} 和 {f2.parent}")

        return stats

    def extract_links(self, file: Path):
        """提取文件中的所有[[链接]]"""
        content = file.read_text(encoding="utf-8")
        # 匹配 [[链接]] 和 [[链接|别名]]
        links = re.findall(r'\[\[([^\]|]+)(?:\|[^\]]+)?\]\]', content)
        return set(links)

    def build_backlink_index(self):
        """构建反向链接索引"""
        print("\n🔗 构建反向链接索引...")
        print("=" * 50)

        index = {}  # target -> [sources]

        for md_file in self.vault.rglob("*.md"):
            if ".templates" in str(md_file) or ".obsidian" in str(md_file):
                continue

            links = self.extract_links(md_file)
            for link in links:
                if link not in index:
                    index[link] = []
                index[link].append(md_file)

        # 打印被引用最多的概念
        sorted_links = sorted(index.items(), key=lambda x: len(x[1]), reverse=True)
        print("\n📚 最常被引用的概念:")
        for link, sources in sorted_links[:10]:
            print(f"   [[{link}]] - 被 {len(sources)} 个文件引用")

        return index

    def generate_moc(self, topic: str, concept_files: list):
        """生成主题MOC（Map of Content）"""
        moc_file = self.moc_dir / f"{topic}.md"

        # 收集所有子概念
        concepts = []
        for file in concept_files:
            name = file.stem
            # 提取描述
            content = file.read_text(encoding="utf-8")
            desc = ""
            for line in content.split("\n")[:20]:
                if "一句话定义" in line or "定义" in line or "概述" in line:
                    desc = line.strip("*# ")
                    break

            concepts.append({
                "name": name,
                "file": str(file.relative_to(self.vault)),
                "desc": desc
            })

        # 生成MOC内容
        content = f"""---
type: moc
created: {datetime.now().strftime("%Y-%m-%d")}
tags: [moc, {topic}]
aliases: ["{topic}索引"]
---

# {topic} 知识索引 (MOC)

## 📖 概念笔记

"""

        for c in concepts:
            content += f"- **[[{c['name']}]]** - {c['desc'][:50]}...\n"

        content += f"""
## 🔗 相关主题

-

## 📚 学习资源

-

## 📝 学习进度

- 开始时间: -
- 完成度: 0%

---
*最后更新: {datetime.now().strftime("%Y-%m-%d %H:%M")}*
"""

        self.moc_dir.mkdir(parents=True, exist_ok=True)
        moc_file.write_text(content, encoding="utf-8")
        print(f"✅ 创建MOC: {moc_file}")

    def cleanup_duplicates(self):
        """清理重复文件"""
        print("\n🧹 清理重复文件...")
        print("=" * 50)

        # 根目录下的重复日记
        root_dailies = list(self.vault.glob("2026-*.md"))
        moved = []

        for file in root_dailies:
            target = self.daily_dir / file.name
            if target.exists():
                # 比较内容
                root_content = file.read_text(encoding="utf-8")
                target_content = target.read_text(encoding="utf-8")
                if len(root_content) > len(target_content):
                    # 根目录的更新，覆盖
                    target.write_text(root_content, encoding="utf-8")
                    print(f"📝 更新: {file.name}")
                file.unlink()
                moved.append(file.name)
            else:
                # 移动到正确位置
                file.rename(target)
                moved.append(file.name)

        if moved:
            print(f"✅ 整理了 {len(moved)} 个日记文件")
        else:
            print("✓ 没有需要清理的重复文件")

    def generate_dashboard(self):
        """生成学习仪表盘"""
        dashboard = self.vault / "🏠 Dashboard.md"

        # 统计数据
        daily_notes = list(self.daily_dir.glob("*.md"))
        concepts = list(self.concepts_dir.glob("*.md"))

        # 获取最新日记
        latest_daily = sorted(daily_notes, key=lambda x: x.name)[-1] if daily_notes else None

        content = f"""---
type: dashboard
cssclass: dashboard
---

# 🏠 学习仪表盘

> ML/DL 50周学习之路

---

## 📅 今日学习

{f"[[{latest_daily.stem}]]" if latest_daily else "> 还没有创建今日笔记"}

---

## 📊 学习统计

| 指标 | 数值 |
|------|------|
| 📅 学习天数 | {len(daily_notes)} |
| 💡 概念笔记 | {len(concepts)} |
| 🚀 完成项目 | 0 |
| 📝 周回顾 | 0 |

---

## 📚 快速导航

### 知识领域
- [[线性代数]] - 数学基础
- [[微积分]] - 优化理论基础
- [[概率统计]] - 机器学习数学基础
- [[经典ML]] - 传统机器学习算法
- [[深度学习]] - 神经网络
- [[Transformer]] - 现代DL架构

### 历史记录
- [[00-Daily]] - 所有日记
- [[04-Reviews]] - 周回顾

---

## 🎯 当前阶段

**Phase 0** - 数学直觉 + NumPy/Pandas + sklearn入门

### 本周计划
- [ ] 3Blue1Brown 线性代数本质 第1-8集
- [ ] NumPy 基础操作
- [ ] Pandas 数据处理入门

---

*最后更新: {datetime.now().strftime("%Y-%m-%d %H:%M")}*
"""

        dashboard.write_text(content, encoding="utf-8")
        print(f"✅ 创建仪表盘: {dashboard}")

    def run_cleanup(self):
        """执行完整清理流程"""
        print("\n" + "=" * 50)
        print("🗂️  Obsidian笔记整理工具")
        print("=" * 50)

        # 1. 扫描
        stats = self.scan_vault()

        # 2. 构建反向链接索引
        self.build_backlink_index()

        # 3. 清理重复
        self.cleanup_duplicates()

        # 4. 生成仪表盘
        self.generate_dashboard()

        # 5. 创建MOC目录
        self.moc_dir.mkdir(parents=True, exist_ok=True)

        # 6. 为主要主题生成MOC
        print("\n📚 生成主题索引...")
        # 线性代数相关
        la_concepts = [f for f in stats["concepts"]
                       if any(k in f.name.lower() for k in ["线性", "向量", "矩阵", "变换", "行列式"])]
        if la_concepts:
            self.generate_moc("线性代数", la_concepts)

        print("\n" + "=" * 50)
        print("✅ 整理完成！")
        print("=" * 50)


def main():
    import sys

    manager = ObsidianManager()

    if len(sys.argv) == 1:
        manager.run_cleanup()
    else:
        cmd = sys.argv[1]
        if cmd == "scan":
            manager.scan_vault()
        elif cmd == "links":
            manager.build_backlink_index()
        elif cmd == "cleanup":
            manager.cleanup_duplicates()
        elif cmd == "moc":
            if len(sys.argv) > 2:
                topic = sys.argv[2]
                manager.generate_moc(topic, [])
            else:
                print("用法: python obsidian_manager.py moc <主题名>")
        elif cmd == "dashboard":
            manager.generate_dashboard()
        else:
            print(f"未知命令: {cmd}")
            print("可用命令: scan, links, cleanup, moc, dashboard")


if __name__ == "__main__":
    main()
