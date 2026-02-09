# ML/DL 学习 Vault

> 使用 Obsidian 进行 ML/DL 知识管理的个人知识库

## 📁 目录结构

```
obsidian-vault/
├── 00-Daily/          # 每日学习日记
├── 01-Concepts/       # 概念笔记 (Zettelkasten)
├── 02-Projects/       # 项目笔记
├── 03-Quizzes/        # 测验记录
├── 04-Reviews/        # 周回顾
├── 05-Code-Snippets/  # 代码片段
├── 99-Resources/      # 资源索引
├── .templates/        # Obsidian 模板
└── 📊 Progress.md     # 总进度仪表盘
```

## 🚀 快速开始

### 1. 在 Obsidian 中打开此 Vault

打开 Obsidian → 打开文件夹 → 选择 `obsidian-vault` 目录

### 2. 推荐插件

| 插件 | 用途 | 说明 |
|------|------|------|
| Dataview | 数据查询 | 用于生成进度统计表格 |
| Templates | 模板管理 | 快速插入预设模板 |
| Calendar | 日历视图 | 可视化查看每日学习记录 |
| Obsidian Git | 版本控制 | 备份和同步笔记 |

### 3. 设置 Dataview

在设置中启用 Dataview 插件后，仪表盘中的数据表格才会正常显示。

## 📝 使用方式

### 方式1: 通过 Python 脚本（推荐）

```bash
# 创建今日学习笔记
python3 ../obsidian_integration.py daily

# 创建概念笔记
python3 ../obsidian_integration.py concept <概念名>

# 创建测验笔记
python3 ../obsidian_integration.py quiz <主题>

# 创建周回顾
python3 ../obsidian_integration.py review <周数>

# 更新进度仪表盘
python3 ../obsidian_integration.py dashboard
```

### 方式2: 直接在 Obsidian 中编辑

所有笔记都是标准 Markdown 格式，可以直接在 Obsidian 中编辑。

## 🔗 链接系统

### 概念链接

- `[[概念名]]` — 链接到其他概念笔记
- `![[概念名]]` — 嵌入其他笔记内容
- Obsidian 会自动显示"反向链接"（谁链接到了当前笔记）

### 标签系统

- `#daily/week-X` — 每日笔记标签
- `#phase-X` — 阶段标签 (0-5)
- `#quiz` — 测验标签
- `#project` — 项目标签

## 📊 进度追踪

总进度在 `📊 Progress.md` 中实时更新，包含：

- 当前学习状态
- 项目完成进度
- 最近测验记录
- 本周计划概览

## 💡 工作流示例

### 每日学习

1. 运行 `python3 obsidian_integration.py daily` 创建今日笔记
2. 在 Obsidian 中打开今日笔记
3. 完成学习后，填写笔记内容和完成情况
4. 遇到新概念时，创建概念笔记并建立链接

### 概念学习

1. 创建概念笔记（脚本或手动）
2. 填写：定义 → 直觉 → 数学 → 代码 → 应用
3. 链接到相关概念（上下游依赖）
4. 定期回顾和更新

### 周回顾

1. 运行 `python3 obsidian_integration.py review` 生成本周回顾
2. 查看完成情况和自测题
3. 记录心得体会
4. 规划下周学习重点

## 🔧 自定义

### 修改模板

模板文件在 `.templates/` 目录下，可以根据需要修改：

- `Daily Note.md` — 每日日记模板
- `Concept Note.md` — 概念笔记模板
- `Project Note.md` — 项目笔记模板
- `Quiz Note.md` — 测验模板
- `Weekly Review.md` — 周回顾模板

### 添加新主题 MOC

编辑 `obsidian_integration.py` 中的 `init_concept_mocs()` 方法，添加新的知识领域。

## 📚 参考资源

- [Obsidian 官方文档](https://help.obsidian.md/)
- [Zettelkasten 方法](https://zettelkasten.de/)
- [Dataview 文档](https://blacksmithgu.github.io/obsidian-dataview/)
