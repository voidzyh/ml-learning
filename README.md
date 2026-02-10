# ML/DL 50 周系统学习

> 软工科班生的 ML/DL 自学之路 — 从数学直觉到可部署项目，边学边练。

## 学习路线

| Phase | 周数 | 核心内容 |
|-------|------|---------|
| 0 | W1-3 | 数学直觉 + NumPy/Pandas + sklearn 入门 |
| 1 | W4-12 | 经典机器学习 |
| 2 | W13-20 | 深度学习基础 |
| 3 | W21-32 | Transformer + 现代 DL |
| 4 | W33-42 | LLM 应用 + RAG + MLOps |
| 5 | W43-50 | 毕业项目 + 求职 |

## 核心项目

| 周 | 项目 | 说明 |
|----|------|------|
| W3 | Titanic EDA | 数据探索与可视化入门 |
| W10 | 客户流失预测 | 经典 ML 全流程 |
| W13 | NumPy 神经网络 | 从零手写前馈网络 |
| W20 | MNIST CNN (>=99%) | 卷积网络实战 |
| W22 | miniGPT | 最小 GPT 实现 |
| W29 | 推荐系统 Web | 端到端推荐服务 |
| W35 | RAG 问答系统 | 检索增强生成 |
| W41 | MLOps 流水线 | 模型部署与监控 |
| W43-45 | 毕业项目 | 综合能力展示 |

## 当前进度

| 项目 | 状态 |
|------|------|
| 当前位置 | **第 1 周 · 周三** (Phase 0 数学直觉 + NumPy/Pandas) |
| 总进度 | **2 / 300 天** (0.7%) |
| 连续学习 | 2 天 |
| 已完成项目 | 0 / 17 |

### 已完成天数

| 天 | 内容 | 完成时间 |
|----|------|----------|
| W1D1 (周一) | 3B1B 线性代数第1-4集 · 向量/线性组合/矩阵乘法 | 2026-02-09 |
| W1D2 (周二) | 3B1B 第5-8集 · 行列式/逆矩阵/列空间/零空间/非方阵变换/点积 | 2026-02-10 |

### 核心项目进度

| 周 | 项目 | 状态 |
|----|------|------|
| W3 | ⭐ Titanic EDA | ⬜ 未开始 |
| W10 | ⭐ 客户流失预测 | ⬜ 未开始 |
| W13 | ⭐ NumPy 神经网络 | ⬜ 未开始 |
| W20 | ⭐ MNIST CNN ≥99% | ⬜ 未开始 |
| W22 | ⭐ miniGPT | ⬜ 未开始 |
| W29 | ⭐ 推荐系统 Web | ⬜ 未开始 |
| W35 | ⭐ RAG 问答系统 | ⬜ 未开始 |
| W41 | ⭐ MLOps 流水线 | ⬜ 未开始 |
| W43-45 | ⭐ 毕业项目 | ⬜ 未开始 |

## 系统特性

- **逐日课表驱动** — 50 周 × 6 天，每天有明确的理论 + 实践安排
- **间隔复习 (SM-2)** — 基于艾宾浩斯遗忘曲线，自动调度复习卡片
- **Obsidian 知识库** — 概念笔记、周回顾、测验、代码片段一体化管理
- **Claude Code 驱动** — 中文自然语言交互，所有命令对用户透明

## 快速开始

```bash
# 1. 克隆项目
git clone <repo-url> && cd ml-learning

# 2. 初始化目录结构
bash setup.sh

# 3. 安装依赖
pip install openpyxl pandas numpy scikit-learn torch jupyter matplotlib seaborn

# 4. 将课表 Excel 放入 data/ 目录
#    - ML_DL_逐日课表_软工科班版.xlsx
#    - B站ML_DL优质资源清单.xlsx

# 5. 启动 Claude Code 开始学习
claude
```

## 使用方式

通过 Claude Code 用中文自然语言交互，无需手动执行命令：

| 说 | 做什么 |
|----|--------|
| 今天学什么 | 查看今日学习计划 |
| 完成 / 打卡 | 标记今日完成，自动创建复习卡片 |
| 复习 | 开始间隔复习流程（Claude 提问 → 用户回答 → 自动评分） |
| 复习统计 | 查看复习数据面板 |
| 学习分析 | 掌握度分布 + 复习量预测 + 记忆保持率 |
| 进度 | 查看总进度仪表盘 |
| 本周计划 | 查看本周概览 |
| 讲解 + 概念 | 结构化讲解（定义→类比→公式→代码→面试） |
| 考考我 + 主题 | 生成自测题 |
| 查资源 + 主题 | 搜索 B 站推荐视频 |

## 间隔复习系统

基于 SM-2 算法（SuperMemo-2）实现：

- 完成当天学习后自动从课表提取概念，创建复习卡片
- 每次复习评分 0-5，算法自动计算下次复习间隔
- 评分 >= 3：间隔递增（1 天 → 6 天 → interval × EF）
- 评分 < 3：重置为 1 天（需要重学）
- 每日复习量上限 50 张，优先过期最久 + 难度最高的卡片

## 项目结构

```
ml-learning/
├── ml_tutor.py              # 学习助手主脚本
├── setup.sh                 # 初始化脚本
├── CLAUDE.md                # Claude Code 指令文档
├── data/                    # 课表 Excel
├── tools/
│   ├── learning_system.py       # 统一 CLI 入口
│   ├── spaced_repetition.py     # SM-2 间隔复习引擎
│   ├── obsidian_integration.py  # Obsidian 笔记集成
│   └── obsidian_manager.py      # Obsidian 管理工具
├── progress/
│   ├── tracker.json             # 学习进度数据
│   ├── review_cards.json        # 复习卡片（SM-2 状态）
│   └── knowledge-gaps.md        # 薄弱知识点追踪
├── projects/                # 核心项目工作目录
│   ├── week03-titanic/
│   ├── week10-churn/
│   ├── week13-numpy-nn/
│   ├── week20-mnist-cnn/
│   ├── week22-minigpt/
│   └── ...
├── obsidian-vault/          # Obsidian 知识库
│   ├── 00-Daily/                # 日志笔记
│   ├── 01-Concepts/             # 概念笔记
│   ├── 03-Quizzes/              # 自测题库
│   ├── 04-Reviews/              # 周回顾
│   └── 99-MOC/                  # 知识地图
└── code/                    # 代码片段和练习
```

## 博客文章

（学习过程中持续更新）
