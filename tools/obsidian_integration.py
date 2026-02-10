#!/usr/bin/env python3
"""
Obsidian 集成模块
功能: 将学习数据写入 Obsidian Vault，支持 Zettelkasten 风格的知识管理
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
import re

# 导入核心模块
from ml_tutor import MLTutor, QUIZ_BANK


class ObsidianIntegration:
    """Obsidian 集成类"""

    def __init__(self, vault_path: str = None):
        """
        初始化 Obsidian 集成

        Args:
            vault_path: Obsidian Vault 路径，默认为项目下的 obsidian-vault/
        """
        if vault_path is None:
            # 默认在项目目录下创建 vault
            self.vault_path = Path(__file__).parent.parent / "obsidian-vault"
        else:
            self.vault_path = Path(vault_path).expanduser()

        self.tutor = MLTutor()

        # 初始化目录结构
        self._init_vault_structure()

    def _init_vault_structure(self):
        """创建 Obsidian Vault 目录结构"""
        dirs = [
            self.vault_path / "00-Daily",
            self.vault_path / "01-Concepts",
            self.vault_path / "02-Projects",
            self.vault_path / "03-Quizzes",
            self.vault_path / "04-Reviews",
            self.vault_path / "05-Code-Snippets",
            self.vault_path / "99-Resources",
            self.vault_path / ".templates",
        ]

        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)

        # 创建配置文件
        self._create_templates()

    def _create_templates(self):
        """创建 Obsidian 模板文件"""
        template_dir = self.vault_path / ".templates"

        # 每日日记模板
        daily_template = """---
date: {{date}}
week: {{week}}
day: {{day}}
phase: {{phase}}
tags: [daily/week-{{week}}, phase-{{phase}}]
cssclass: daily-note
---

# 📅 第{{week}}周·{{day_name}} | Phase {{phase}} {{phase_name}}

## 📋 今日计划

### 🌅 上午·理论 (60-90min)
{{morning_content}}

### 🌆 下午·实践 (90-120min)
{{afternoon_content}}

## 📦 今日交付
{{deliverables}}

## 📝 学习笔记
<!-- 在这里记录今天的学习笔记 -->

## 💡 代码片段
\`\`\`python
# 今天学到的代码
\`\`\`

## 🔗 相关概念
<!-- Obsidian 会自动链接到相关概念 -->

## ✅ 完成情况
- [ ] 上午理论
- [ ] 下午实践
- [ ] 今日交付

## 📌 明日预告
{{next_day_preview}}

---

## 元数据
- 学习时长: ___ 小时
- 完成度: ___ %
- 心情: 😊 😐 😫
"""

        (template_dir / "Daily Note.md").write_text(daily_template, encoding="utf-8")

        # 概念笔记模板
        concept_template = """---
type: concept
created: {{date}}
phase: {{phase}}
tags: [{{tags}}]
aliases: [{{aliases}}]
---

# {{concept_title}}

## 🎯 一句话定义
<!-- 一句话解释这个概念 -->

## 🧠 直觉理解
<!-- 用日常或编程类比来理解 -->

## 📐 数学表达
<!-- 核心公式，用 $$...$$ 包裹 LaTeX -->

## 💻 代码实现
\`\`\`python
# 代码示例
\`\`\`

## 🔗 在ML/DL中的位置
- **上游依赖**:
- **下游应用**:

## 💬 面试常见问法
1.
2.
3.

## 📚 学习资源
- [ ] 视频:
- [ ] 文章:
- [ ] 练习:

---

# 🔗 反向链接
<!-- 这里会自动显示链接到这个笔记的其他笔记 -->
"""

        (template_dir / "Concept Note.md").write_text(concept_template, encoding="utf-8")

        # 项目笔记模板
        project_template = """---
type: project
week: {{week}}
status: {{status}}
tags: [project, {{project_type}}]
start_date: {{date}}
---

# {{project_title}}

## 📋 项目概述
<!-- 项目简介 -->

## 🎯 学习目标
-
-
-

## 📁 项目结构
\`\`\`
ml-learning/
└── projects/
    └── week{{week:02d}}-{{project_slug}}/
\`\`\`

## 🔧 技术栈
- Python:
- 库:

## 📝 实施步骤
### 1. 数据准备
- [ ] 加载数据
- [ ] 探索性分析

### 2. 特征工程
- [ ]

### 3. 模型训练
- [ ]

### 4. 评估与优化
- [ ]

## ✅ 验收标准
- [ ]

## 📊 结果记录
<!-- 记录最终结果、指标 -->

## 💡 心得体会
<!-- 学到了什么 -->

## 🔗 相关概念

## 🐛 遇到的问题与解决
| 问题 | 解决方案 |
|------|----------|
|      |          |
"""

        (template_dir / "Project Note.md").write_text(project_template, encoding="utf-8")

        # 测验笔记模板
        quiz_template = """---
type: quiz
date: {{date}}
topic: {{topic}}
score: {{score}}/{{total}}
tags: [quiz, {{topic}}]
---

# 📝 测验: {{topic}}

## 📊 成绩: {{score}}/{{total}} ({{percentage}}%)

## ❓ 题目

### 1. {{question1}}
**你的答案**:
<!-- 写下你的答案 -->

**正确答案**: {{answer1}}
{{#if correct1}}✅ 正确{{/if}}
{{#if incorrect1}}❌ 错误{{/if}}

### 2. {{question2}}
**你的答案**:

**正确答案**: {{answer2}}

### 3. {{question3}}
**你的答案**:

**正确答案**: {{answer3}}

### 4. {{question4}}
**你的答案**:

**正确答案**: {{answer4}}

### 5. {{question5}}
**你的答案**:

**正确答案**: {{answer5}}

## 💡 需要复习的知识点
<!-- 记录答错的题目对应的知识点 -->

## 📚 下一步行动
- [ ] 复习错题相关概念
- [ ] 做相关练习

---

## 🔗 相关笔记
"""

        (template_dir / "Quiz Note.md").write_text(quiz_template, encoding="utf-8")

        # 周回顾模板
        review_template = """---
type: weekly-review
week: {{week}}
start_date: {{start_date}}
end_date: {{end_date}}
completion_rate: {{completion_rate}}%
tags: [review/week-{{week}}]
---

# 📊 第{{week}}周回顾

> {{date_range}}

## 📈 完成情况

| 指标 | 数值 |
|------|------|
| ✅ 已完成 | {{completed}}/6 天 |
| ⏭️ 跳过 | {{skipped}} 天 |
| ⬜ 待完成 | {{pending}} 天 |
| 📊 完成率 | {{completion_rate}}% |

## 📚 本周核心概念

{{concepts_list}}

## 📝 自测题

{{quiz_questions}}

## ⚠️ 待补强内容

{{weak_points}}

## 💡 本周心得

<!-- 写下这周的学习心得 -->

## 🎯 下周计划

- [ ] {{next_week_preview}}

## 📊 学习数据

```dataview
TABLE date, week, day, status
FROM "00-Daily"
WHERE week = {{week}}
SORT date ASC
```

---

## 🔗 相关资源
"""

        (template_dir / "Weekly Review.md").write_text(review_template, encoding="utf-8")

    def sanitize_filename(self, name: str) -> str:
        """清理文件名，移除非法字符"""
        # 移除或替换非法字符
        name = re.sub(r'[<>:"/\\|?*]', '', name)
        # 限制长度
        if len(name) > 100:
            name = name[:100]
        return name.strip()

    def slugify(self, text: str) -> str:
        """将文本转换为 URL 友好的 slug"""
        text = text.lower()
        # 移除特殊字符，保留中文、字母、数字、连字符
        text = re.sub(r'[^\w\u4e00-\u9fff\s-]', '', text)
        # 空格替换为连字符
        text = re.sub(r'[-\s]+', '-', text)
        return text.strip('-')

    # ========== 日记相关 ==========

    def create_daily_note(self, date: str = None) -> str:
        """
        创建每日学习笔记

        Args:
            date: 日期字符串 (YYYY-MM-DD)，默认为今天

        Returns:
            创建的文件路径
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        # 获取今日计划
        plan = self.tutor.get_today_plan()
        schedule_item = plan.get("schedule_item", {})

        # 获取明天预览
        tomorrow_plan = self._get_tomorrow_preview()

        # 准备模板变量
        template_vars = {
            "date": date,
            "week": plan["week"],
            "day": plan["day"],
            "day_name": plan["day_name"],
            "phase": plan["phase"],
            "phase_name": plan["phase_name"],
            "morning_content": schedule_item.get("morning_theory", "暂无内容"),
            "afternoon_content": schedule_item.get("afternoon_practice", "暂无内容"),
            "deliverables": schedule_item.get("deliverables", ""),
            "next_day_preview": tomorrow_plan,
        }

        # 生成 Markdown 内容
        content = self._render_daily_note(template_vars)

        # 写入文件
        filename = f"{date}.md"
        filepath = self.vault_path / "00-Daily" / filename
        filepath.write_text(content, encoding="utf-8")

        return str(filepath)

    def _render_daily_note(self, vars: Dict) -> str:
        """渲染每日日记"""
        return f"""---
date: {vars['date']}
week: {vars['week']}
day: {vars['day']}
phase: {vars['phase']}
tags: [daily/week-{vars['week']}, phase-{vars['phase']}]
cssclass: daily-note
---

# 📅 第{vars['week']}周·{vars['day_name']} | Phase {vars['phase']} {vars['phase_name']}

## 📋 今日计划

### 🌅 上午·理论 (60-90min)
{vars['morning_content']}

### 🌆 下午·实践 (90-120min)
{vars['afternoon_content']}

## 📦 今日交付
{vars['deliverables']}

## 📝 学习笔记
<!-- 在这里记录今天的学习笔记 -->

## 💡 代码片段
```python
# 今天学到的代码
```

## 🔗 相关概念
<!-- Obsidian 会自动链接到相关概念 -->

## ✅ 完成情况
- [ ] 上午理论
- [ ] 下午实践
- [ ] 今日交付

## 📌 明日预告
{vars['next_day_preview']}

---

## 元数据
- 学习时长: ___ 小时
- 完成度: ___ %
- 心情: 😊 😐 😫
"""

    def _get_tomorrow_preview(self) -> str:
        """获取明天的预告"""
        current_week = self.tutor.tracker["current_week"]
        current_day = self.tutor.tracker["current_day"]

        # 计算明天
        if current_day < 6:
            next_day = current_day + 1
            next_week = current_week
        else:
            next_day = 1
            next_week = current_week + 1

        # 获取明天的课表
        schedule = self.tutor._load_schedule()
        for item in schedule:
            if item["week"] == next_week and item["day"] == next_day:
                theory = item.get("morning_theory", "")
                # 简化显示
                return theory[:50] + "..." if len(theory) > 50 else theory

        return "暂无"

    # ========== 概念笔记相关 ==========

    def create_concept_note(self, concept_name: str, content: Dict = None) -> str:
        """
        创建概念笔记

        Args:
            concept_name: 概念名称
            content: 概念内容字典，包含定义、直觉、数学、代码等

        Returns:
            创建的文件路径
        """
        if content is None:
            content = {}

        # 获取当前阶段信息
        phase = self.tutor.tracker["phase"]

        # 默认内容
        default_content = {
            "definition": f"# {concept_name}\n\n## 🎯 一句话定义\n<!-- 一句话解释这个概念 -->\n\n",
            "intuition": "## 🧠 直觉理解\n<!-- 用日常或编程类比来理解 -->\n\n",
            "math": "## 📐 数学表达\n<!-- 核心公式 -->\n\n",
            "code": "## 💻 代码实现\n```python\n# 代码示例\n```\n\n",
            "position": "## 🔗 在ML/DL中的位置\n- **上游依赖**:\n- **下游应用**:\n\n",
            "interview": "## 💬 面试常见问法\n1.\n2.\n3.\n\n",
            "resources": "## 📚 学习资源\n- [ ] 视频:\n- [ ] 文章:\n",
        }

        # 合并内容
        for key, value in content.items():
            if key in default_content:
                default_content[key] = value

        # 生成文件名（清理特殊字符，但保留原始概念名用于内容）
        safe_filename = self.sanitize_filename(concept_name)
        filename = f"{safe_filename}.md"
        filepath = self.vault_path / "01-Concepts" / filename

        # 如果文件已存在，不覆盖
        if filepath.exists():
            return str(filepath)

        # 生成标签
        tags = self._generate_tags_for_concept(concept_name, phase)

        # 组合内容
        markdown = f"""---
type: concept
created: {datetime.now().strftime("%Y-%m-%d")}
phase: {phase}
tags: [{tags}]
aliases: [{concept_name}]
---

# {concept_name}

{default_content['definition']}
{default_content['intuition']}
{default_content['math']}
{default_content['code']}
{default_content['position']}
{default_content['interview']}
{default_content['resources']}

---

# 🔗 反向链接
<!-- 这里会自动显示链接到这个笔记的其他笔记 -->
"""

        filepath.write_text(markdown, encoding="utf-8")
        return str(filepath)

    def _generate_tags_for_concept(self, concept_name: str, phase: int) -> str:
        """为概念生成标签"""
        # 基础标签
        phase_tags = {
            0: "math,basics",
            1: "ml,classical",
            2: "dl,basics",
            3: "dl,transformer",
            4: "llm,application",
            5: "project,career"
        }

        base_tag = phase_tags.get(phase, "general")

        # 根据概念名称添加特定标签
        concept_lower = concept_name.lower()
        specific_tags = []

        if any(x in concept_lower for x in ["向量", "矩阵", "线性", "特征值", "秩"]):
            specific_tags.append("linear-algebra")
        if any(x in concept_lower for x in ["导数", "梯度", "偏导", "微分"]):
            specific_tags.append("calculus")
        if any(x in concept_lower for x in ["回归", "分类", "聚类"]):
            specific_tags.append("ml-algorithm")
        if any(x in concept_lower for x in ["神经网络", "激活", "损失"]):
            specific_tags.append("neural-network")
        if any(x in concept_lower for x in ["卷积", "池化", "CNN"]):
            specific_tags.append("cnn")
        if any(x in concept_lower for x in ["attention", "transformer", "bert"]):
            specific_tags.append("transformer")

        all_tags = [base_tag] + specific_tags
        return ",".join(all_tags)

    def create_moc_note(self, topic: str, subconcepts: List[str]) -> str:
        """
        创建 MOC (Map of Content) 笔记

        Args:
            topic: 主题名称 (如 "线性代数")
            subconcepts: 子概念列表

        Returns:
            创建的文件路径
        """
        filename = f"{topic}.md"
        filepath = self.vault_path / "01-Concepts" / filename

        # 生成子概念链接
        concept_links = "\n".join([
            f"- [[{name}]]" for name in subconcepts
        ])

        content = f"""---
type: moc
tags: [moc, {self.slugify(topic)}]
---

# {topic}

> 这个页面是 {topic} 知识点的索引（Map of Content）

## 📚 子概念

{concept_links}

## 🔗 相关资源

## 📝 学习笔记

---

## 📊 学习进度

| 概念 | 状态 | 掌握程度 |
|------|------|----------|
{chr(10).join([f"| [[{name}]] | ⬜ | ⭐⭐⭐ |" for name in subconcepts[:10]])}
"""

        filepath.write_text(content, encoding="utf-8")
        return str(filepath)

    # ========== 测验相关 ==========

    def create_quiz_note(self, topic: str, questions: List[Dict], score: int = None, total: int = None) -> str:
        """
        创建测验笔记

        Args:
            topic: 测验主题
            questions: 题目列表
            score: 得分 (可选)
            total: 总分 (可选)

        Returns:
            创建的文件路径
        """
        date_str = datetime.now().strftime("%Y%m%d")
        filename = f"Quiz-{self.slugify(topic)}-{date_str}.md"
        filepath = self.vault_path / "03-Quizzes" / filename

        # 计算百分比
        if score is not None and total is not None:
            percentage = (score / total) * 100
            score_section = f"## 📊 成绩: {score}/{total} ({percentage:.1f}%)\n\n"
        else:
            score_section = "## 📊 成绩: 未评分\n\n"

        # 生成题目
        questions_section = "## ❓ 题目\n\n"
        for i, q in enumerate(questions[:5], 1):
            questions_section += f"### {i}. {q['question']}\n\n"
            questions_section += "**你的答案**:\n<!-- 写下你的答案 -->\n\n"
            questions_section += f"**正确答案**: {q['answer']}\n\n"
            questions_section += "---\n\n"

        content = f"""---
type: quiz
date: {datetime.now().strftime("%Y-%m-%d")}
topic: {topic}
tags: [quiz, {self.slugify(topic)}]
---

# 📝 测验: {topic}

{score_section}
{questions_section}
## 💡 需要复习的知识点
<!-- 记录答错的题目对应的知识点 -->

## 📚 下一步行动
- [ ] 复习错题相关概念
- [ ] 做相关练习

---

## 🔗 相关笔记
"""

        filepath.write_text(content, encoding="utf-8")
        return str(filepath)

    # ========== 周回顾相关 ==========

    def create_weekly_review(self, week: int) -> str:
        """
        创建周回顾笔记

        Args:
            week: 周数

        Returns:
            创建的文件路径
        """
        review_data = self.tutor.generate_weekly_review(week)

        filename = f"Week-{week:02d}-Review.md"
        filepath = self.vault_path / "04-Reviews" / filename

        # 计算日期范围
        # 假设从 2026-02-10 开始
        start_date = datetime(2026, 2, 10) + timedelta(weeks=week-1)
        end_date = start_date + timedelta(days=6)
        date_range = f"{start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}"

        # 生成概念列表
        concepts_list = "\n".join([
            f"- {concept}" for concept in review_data.get("concepts", [])
        ])

        # 生成题目
        quiz_questions = "\n".join([
            f"### {i}. {q['question']}\n"
            for i, q in enumerate(review_data.get("quiz_questions", [])[:5], 1)
        ])

        # 生成薄弱点
        weak_points = "\n".join([
            f"- [ ] {point}" for point in review_data.get("weak_points", [])
        ])

        content = f"""---
type: weekly-review
week: {week}
start_date: {start_date.strftime("%Y-%m-%d")}
end_date: {end_date.strftime("%Y-%m-%d")}
completion_rate: {review_data['completion_rate']:.1f}%
tags: [review/week-{week}]
---

# 📊 第{week}周回顾

> {date_range}

## 📈 完成情况

| 指标 | 数值 |
|------|------|
| ✅ 已完成 | {review_data['completed']}/6 天 |
| ⏭️ 跳过 | {review_data['skipped']} 天 |
| ⬜ 待完成 | {review_data['pending']} 天 |
| 📊 完成率 | {review_data['completion_rate']:.1f}% |

## 📚 本周核心概念

{concepts_list}

## 📝 自测题

{quiz_questions}

## ⚠️ 待补强内容

{weak_points}

## 💡 本周心得

<!-- 写下这周的学习心得 -->

## 🎯 下周计划

- [ ] 第{week+1}周的学习内容

## 📊 学习数据

```dataview
TABLE file.ctime as date, week, day, status
FROM "00-Daily"
WHERE week = {week}
SORT date ASC
```

---

## 🔗 相关资源
"""

        filepath.write_text(content, encoding="utf-8")
        return str(filepath)

    # ========== 项目笔记相关 ==========

    def create_project_note(self, project_id: str, project_info: Dict = None) -> str:
        """
        创建项目笔记

        Args:
            project_id: 项目ID (如 "titanic-eda")
            project_info: 项目信息字典

        Returns:
            创建的文件路径
        """
        # 从 tracker 获取项目信息
        projects = self.tutor.tracker.get("projects", {})
        tracker_info = projects.get(project_id, {})
        week = tracker_info.get("week", 1)

        if project_info is None:
            project_info = {}

        # 项目标题
        project_names = {
            "titanic-eda": "Titanic EDA - 探索性数据分析",
            "numpy-lr": "NumPy 线性回归从零实现",
            "spam-classifier": "垃圾邮件分类器",
            "customer-churn": "客户流失预测",
            "numpy-neural-net": "NumPy 手写神经网络",
            "mnist-cnn-99": "MNIST CNN 达到99%准确率",
            "minigpt": "miniGPT 从零实现",
            "bert-classification": "BERT 文本分类",
            "rag-qa-system": "RAG 文档问答系统",
            "mlops-pipeline": "端到端 MLOps 流水线",
        }

        title = project_info.get("title", project_names.get(project_id, project_id))
        slug = self.slugify(title)

        filename = f"{slug}.md"
        filepath = self.vault_path / "02-Projects" / filename

        content = f"""---
type: project
week: {week}
status: {tracker_info.get('status', 'not_started')}
tags: [project, week-{week}]
start_date: {datetime.now().strftime("%Y-%m-%d")}
---

# {title}

## 📋 项目概述
<!-- 项目简介 -->

## 🎯 学习目标
-
-
-

## 📁 项目结构
\`\`\`
ml-learning/
└── projects/
    └── week{week:02d}-{project_id}/
        ├── data/
        ├── notebooks/
        ├── src/
        └── README.md
\`\`\`

## 🔧 技术栈
- Python: 3.10+
- 核心库:
- 可选库:

## 📝 实施步骤

### 1. 数据准备
- [ ] 加载数据
- [ ] 探索性分析 (EDA)
- [ ] 数据清洗

### 2. 特征工程
- [ ] 特征选择
- [ ] 特征转换

### 3. 模型训练
- [ ] 基线模型
- [ ] 模型调优

### 4. 评估与优化
- [ ] 交叉验证
- [ ] 性能评估

## ✅ 验收标准
- [ ]

## 📊 结果记录
<!-- 记录最终结果、指标 -->

| 指标 | 目标 | 实际 |
|------|------|------|
|      |      |      |

## 💡 心得体会
<!-- 学到了什么 -->

## 🔗 相关概念
- [[相关概念1]]
- [[相关概念2]]

## 🐛 遇到的问题与解决
| 问题 | 解决方案 |
|------|----------|
|      |          |

## 📚 参考资源
"""

        filepath.write_text(content, encoding="utf-8")
        return str(filepath)

    # ========== 进度仪表盘相关 ==========

    def update_progress_dashboard(self):
        """更新总进度仪表盘"""
        status = self.tutor.get_status()

        filepath = self.vault_path / "📊 Progress.md"

        # 获取最近的测验记录
        quiz_scores = self.tutor.tracker.get("quiz_scores", [])
        recent_quizzes = quiz_scores[-5:] if quiz_scores else []

        # 获取项目状态
        projects = self.tutor.tracker.get("projects", {})

        content = f"""---
type: dashboard
updated: {datetime.now().strftime("%Y-%m-%d %H:%M")}
---

# 📊 ML/DL 学习进度

> 最后更新: {datetime.now().strftime("%Y-%m-%d %H:%M")}

## 🎯 当前状态

| 指标 | 值 |
|------|-----|
| 📅 当前周 | 第{status['current_week']}周 · {status['day_name']} |
| 🏷️ 当前阶段 | Phase {status['phase']} - {status['phase_name']} |
| 📊 总进度 | {status['progress']:.1f}% |
| 🔥 连续学习 | {status['streak']}天 |
| ✅ 已完成 | {status['total_completed']}/300 天 |
| 🧪 测验次数 | {status.get('quiz_count', 0)}次 |

## 📁 项目进度

```dataview
TABLE project, week, status
FROM "02-Projects"
WHERE type = "project"
SORT week ASC
```

## 📝 最近测验

{self._format_recent_quizzes(recent_quizzes)}

## 📅 本周计划

### 今天
- [[00-Daily/{datetime.now().strftime('%Y-%m-%d')}|今天的学习笔记]]

### 本周概览
```dataview
TABLE file.ctime as date, day_name as day, morning_theory as 上午, afternoon_practice as 下午
FROM "00-Daily"
WHERE week = {status['current_week']}
SORT date ASC
```

## 🔗 快速链接

- [[00-Daily|📅 日记]]
- [[01-Concepts|📚 概念笔记]]
- [[02-Projects|🚀 项目]]
- [[03-Quizzes|📝 测验]]
- [[04-Reviews|📊 周回顾]]

## 📊 阶段进度

### Phase 0: 数学直觉 + 工具链 (W1-3)
```dataview
LIST
FROM "01-Concepts"
WHERE contains(tags, "phase-0")
```

### Phase 1: 经典ML (W4-12)
```dataview
LIST
FROM "01-Concepts"
WHERE contains(tags, "phase-1")
```

### Phase 2: DL基础 (W13-20)
```dataview
LIST
FROM "01-Concepts"
WHERE contains(tags, "phase-2")
```
"""

        filepath.write_text(content, encoding="utf-8")
        return str(filepath)

    def _format_recent_quizzes(self, quizzes: List[Dict]) -> str:
        """格式化最近的测验记录"""
        if not quizzes:
            return "暂无测验记录\n"

        lines = ["| 日期 | 主题 | 得分 |", "|------|------|------|"]
        for q in quizzes[-5:]:
            date = q.get("date", "")[:10]
            topic = q.get("topic", "")
            score = f"{q.get('score', 0)}/{q.get('total', 0)}"
            lines.append(f"| {date} | {topic} | {score} |")

        return "\n".join(lines) + "\n"

    # ========== 批量初始化 ==========

    def init_all_weekly_notes(self, max_weeks: int = 50):
        """初始化所有周的计划笔记（可选）"""
        print("⏳ 正在初始化周计划笔记...")
        for week in range(1, min(max_weeks + 1, 53)):
            # 这里可以预创建周计划
            pass
        print(f"✅ 初始化完成 (共 {max_weeks} 周)")

    def init_concept_mocs(self):
        """初始化主要知识领域的 MOC 笔记"""
        mocs = {
            "线性代数": ["向量", "矩阵", "线性变换", "特征值与特征向量", "行列式", "秩"],
            "微积分": ["导数", "偏导数", "梯度", "链式法则", "泰勒展开"],
            "概率统计": ["期望", "方差", "概率分布", "贝叶斯定理", "假设检验"],
            "经典ML": ["线性回归", "逻辑回归", "决策树", "SVM", "朴素贝叶斯", "K-means", "PCA"],
            "深度学习": ["神经网络", "激活函数", "损失函数", "反向传播", "优化器"],
            "CNN": ["卷积", "池化", "卷积神经网络", "LeNet", "ResNet"],
            "RNN": ["循环神经网络", "LSTM", "GRU", "Seq2Seq"],
            "Attention": ["自注意力", "多头注意力", "Scaled Dot-Product"],
            "Transformer": ["Transformer", "BERT", "GPT", "T5"],
            "LLM应用": ["Prompt Engineering", "RAG", "Fine-tuning", "LoRA"],
        }

        print("⏳ 正在初始化 MOC 笔记...")
        for topic, subconcepts in mocs.items():
            self.create_moc_note(topic, subconcepts)
        print(f"✅ 初始化完成 (共 {len(mocs)} 个 MOC)")


# ========== CLI 入口点 ==========

def main():
    """CLI 入口"""
    import sys

    # 可选：指定 vault 路径
    vault_path = sys.argv[2] if len(sys.argv) > 2 else None
    obsidian = ObsidianIntegration(vault_path)

    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"

    if cmd == "daily":
        filepath = obsidian.create_daily_note()
        print(f"✅ 每日笔记已创建: {filepath}")

    elif cmd == "concept":
        if len(sys.argv) > 2:
            concept_name = sys.argv[2]
            filepath = obsidian.create_concept_note(concept_name)
            print(f"✅ 概念笔记已创建: {filepath}")
        else:
            print("用法: python obsidian_integration.py concept <概念名>")

    elif cmd == "quiz":
        # quiz <主题>
        topic = sys.argv[2] if len(sys.argv) > 2 else None
        quiz_data = obsidian.tutor.generate_quiz(topic, 5)
        filepath = obsidian.create_quiz_note(
            quiz_data["topic"],
            quiz_data["questions"]
        )
        print(f"✅ 测验笔记已创建: {filepath}")

    elif cmd == "review":
        week = int(sys.argv[2]) if len(sys.argv) > 2 else obsidian.tutor.tracker["current_week"]
        filepath = obsidian.create_weekly_review(week)
        print(f"✅ 周回顾已创建: {filepath}")

    elif cmd == "project":
        if len(sys.argv) > 2:
            project_id = sys.argv[2]
            filepath = obsidian.create_project_note(project_id)
            print(f"✅ 项目笔记已创建: {filepath}")
        else:
            print("用法: python obsidian_integration.py project <项目ID>")

    elif cmd == "dashboard":
        filepath = obsidian.update_progress_dashboard()
        print(f"✅ 进度仪表盘已更新: {filepath}")

    elif cmd == "init-mocs":
        obsidian.init_concept_mocs()

    elif cmd == "init":
        print("🚀 正在初始化 Obsidian Vault...")
        print(f"📁 Vault 位置: {obsidian.vault_path}")
        obsidian.init_concept_mocs()
        obsidian.update_progress_dashboard()
        print("✅ 初始化完成！")
        print(f"\n💡 在 Obsidian 中打开此目录: {obsidian.vault_path}")

    else:
        print("Obsidian 集成工具")
        print("\n用法:")
        print("  python obsidian_integration.py daily          # 创建今日学习笔记")
        print("  python obsidian_integration.py concept <名称> # 创建概念笔记")
        print("  python obsidian_integration.py quiz [主题]    # 创建测验笔记")
        print("  python obsidian_integration.py review [周数]  # 创建周回顾")
        print("  python obsidian_integration.py project <ID>   # 创建项目笔记")
        print("  python obsidian_integration.py dashboard      # 更新进度仪表盘")
        print("  python obsidian_integration.py init-mocs      # 初始化MOC笔记")
        print("  python obsidian_integration.py init           # 完整初始化")
        print(f"\n当前 Vault 路径: {obsidian.vault_path}")


if __name__ == "__main__":
    main()
