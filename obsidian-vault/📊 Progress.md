---
type: dashboard
updated: 2026-02-09 21:07
---

# 📊 ML/DL 学习进度

> 最后更新: 2026-02-09 21:07

## 🎯 当前状态

| 指标 | 值 |
|------|-----|
| 📅 当前周 | 第1周 · 周二 |
| 🏷️ 当前阶段 | Phase 0 - 数学直觉 + NumPy/Pandas + sklearn入门 |
| 📊 总进度 | 0.3% |
| 🔥 连续学习 | 1天 |
| ✅ 已完成 | 1/300 天 |
| 🧪 测验次数 | 1次 |

## 📁 项目进度

```dataview
TABLE project, week, status
FROM "02-Projects"
WHERE type = "project"
SORT week ASC
```

## 📝 最近测验

| 日期 | 主题 | 得分 |
|------|------|------|
| 2026-02-09 | linear-algebra | 4.0/5 |


## 📅 本周计划

### 今天
- [[00-Daily/2026-02-09|今天的学习笔记]]

### 本周概览
```dataview
TABLE file.ctime as date, day_name as day, morning_theory as 上午, afternoon_practice as 下午
FROM "00-Daily"
WHERE week = 1
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
