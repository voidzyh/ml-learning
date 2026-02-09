---
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
