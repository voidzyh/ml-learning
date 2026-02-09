#!/bin/bash
# ML/DL 50周学习系统 — 一键初始化脚本
# 用法: cd ml-learning && bash setup.sh

set -e

echo "🚀 正在初始化 ML/DL 学习项目..."

# 创建目录结构
dirs=(
  "data"
  "progress/weekly-reviews"
  "notes/concepts"
  "notes/paper-notes"
  "projects"
  "code/snippets"
  "code/templates"
  "code/exercises"
  "blog/drafts"
  "resources/cheatsheets"
  "resources/mindmaps"
)

for d in "${dirs[@]}"; do
  mkdir -p "$d"
  echo "  📁 $d"
done

# 初始化 tracker.json
if [ ! -f "progress/tracker.json" ]; then
cat > progress/tracker.json << 'JSON'
{
  "start_date": null,
  "current_week": 1,
  "current_day": 1,
  "streak": 0,
  "total_completed_days": 0,
  "total_skipped_days": 0,
  "phase": 0,
  "days": {},
  "projects": {
    "titanic-eda": { "status": "not_started", "github": "", "week": 3 },
    "numpy-lr": { "status": "not_started", "github": "", "week": 4 },
    "spam-classifier": { "status": "not_started", "github": "", "week": 7 },
    "customer-churn": { "status": "not_started", "github": "", "week": 10 },
    "kaggle-competition-1": { "status": "not_started", "github": "", "week": 12 },
    "numpy-neural-net": { "status": "not_started", "github": "", "week": 13 },
    "transfer-learning": { "status": "not_started", "github": "", "week": 17 },
    "imdb-sentiment": { "status": "not_started", "github": "", "week": 18 },
    "mnist-cnn-99": { "status": "not_started", "github": "", "week": 20 },
    "minigpt": { "status": "not_started", "github": "", "week": 22 },
    "bert-classification": { "status": "not_started", "github": "", "week": 24 },
    "recommendation-web": { "status": "not_started", "github": "", "week": 29 },
    "rag-qa-system": { "status": "not_started", "github": "", "week": 35 },
    "llm-lora-finetune": { "status": "not_started", "github": "", "week": 36 },
    "mlops-pipeline": { "status": "not_started", "github": "", "week": 41 },
    "capstone-project": { "status": "not_started", "github": "", "week": 45 },
    "kaggle-featured": { "status": "not_started", "github": "", "week": 47 }
  },
  "blogs": [],
  "quiz_scores": [],
  "knowledge_gaps": []
}
JSON
echo "  📊 progress/tracker.json 已初始化"
fi

# 初始化 knowledge-gaps.md
if [ ! -f "progress/knowledge-gaps.md" ]; then
cat > progress/knowledge-gaps.md << 'MD'
# 薄弱知识点追踪

> 记录学习过程中遇到的难点和薄弱环节，定期回顾和补强。

## 待补强

| 日期 | 知识点 | 所属阶段 | 难度 | 状态 |
|------|--------|---------|------|------|
|      |        |         |      |      |

## 已补强

（从上面移到这里）
MD
echo "  📝 progress/knowledge-gaps.md 已初始化"
fi

# 创建 PyTorch 训练模板
cat > code/templates/pytorch-training-loop.py << 'PY'
"""
PyTorch 标准训练循环模板
用法: 根据你的任务修改 Model / Dataset / 超参数
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
# import wandb  # 取消注释以启用W&B记录

# ─── 超参数 ───
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── 模型定义 ───
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: 定义你的网络层
        pass

    def forward(self, x):
        # TODO: 定义前向传播
        pass


# ─── 训练一个epoch ───
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)

        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)
        # 分类任务: 计算准确率
        _, predicted = output.max(1)
        correct += predicted.eq(batch_y).sum().item()
        total += batch_y.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# ─── 评估 ───
@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
        output = model(batch_x)
        loss = criterion(output, batch_y)

        total_loss += loss.item() * batch_x.size(0)
        _, predicted = output.max(1)
        correct += predicted.eq(batch_y).sum().item()
        total += batch_y.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# ─── 主训练循环 ───
def main():
    model = Model().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # TODO: 创建你的DataLoader
    # train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    best_val_acc = 0
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        # scheduler.step()

        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  ✅ 新最佳! Val Acc: {val_acc:.4f}")

    print(f"\n🏆 最佳验证准确率: {best_val_acc:.4f}")

if __name__ == "__main__":
    main()
PY
echo "  🐍 code/templates/pytorch-training-loop.py"

# 创建 sklearn Pipeline 模板
cat > code/templates/sklearn-pipeline.py << 'PY'
"""
Scikit-learn 完整Pipeline模板
包含: 预处理 + 特征工程 + 模型训练 + 评估
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


def build_pipeline(num_features: list, cat_features: list) -> Pipeline:
    """构建完整的预处理+模型Pipeline"""

    # 数值特征处理: 填充缺失值 + 标准化
    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # 类别特征处理: 填充缺失值 + One-Hot编码
    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    # 组合预处理器
    preprocessor = ColumnTransformer([
        ("num", num_transformer, num_features),
        ("cat", cat_transformer, cat_features),
    ])

    # 完整Pipeline: 预处理 → 模型
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(n_estimators=100, random_state=42)),
    ])

    return pipeline


def main():
    # TODO: 加载你的数据
    # df = pd.read_csv("your_data.csv")
    # X = df.drop("target", axis=1)
    # y = df["target"]

    # TODO: 指定特征类型
    # num_features = ["age", "income", "score"]
    # cat_features = ["gender", "city", "plan"]

    # 构建Pipeline
    # pipe = build_pipeline(num_features, cat_features)

    # 交叉验证
    # scores = cross_val_score(pipe, X, y, cv=5, scoring="f1")
    # print(f"5-Fold F1: {scores.mean():.4f} ± {scores.std():.4f}")

    # 训练 + 评估
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # pipe.fit(X_train, y_train)
    # y_pred = pipe.predict(X_test)
    # print(classification_report(y_test, y_pred))
    pass

if __name__ == "__main__":
    main()
PY
echo "  🐍 code/templates/sklearn-pipeline.py"

# 创建 .gitignore
cat > .gitignore << 'GIT'
# 数据文件（太大不上传）
data/*.xlsx
data/*.csv
*.h5
*.pth
*.onnx

# Python
__pycache__/
*.pyc
.ipynb_checkpoints/
*.egg-info/
venv/
.env

# IDE
.vscode/
.idea/

# 系统文件
.DS_Store
Thumbs.db

# W&B
wandb/
GIT
echo "  📄 .gitignore"

# 创建 README.md
cat > README.md << 'README'
# 🧠 ML/DL 50周系统学习

> 软工科班生的ML/DL自学之路 — 从零到可部署，边学边练。

## 学习路线

| Phase | 周数 | 核心内容 |
|-------|------|---------|
| 0 | W1-3 | 数学直觉 + Python工具链 |
| 1 | W4-12 | 经典机器学习 |
| 2 | W13-20 | 深度学习基础 |
| 3 | W21-32 | Transformer + 现代DL |
| 4 | W33-42 | LLM应用 + MLOps |
| 5 | W43-50 | 毕业项目 + 求职 |

## 核心项目

（学习过程中持续更新）

## 使用 Claude Code 学习

```bash
cd ml-learning
claude    # 启动Claude Code

# 常用指令
/init     # 首次初始化
/today    # 查看今日计划
/done     # 完成今日学习
/status   # 查看总进度
/quiz ML  # 知识自测
/explain [概念]  # 深入讲解
```

## 博客文章

（学习过程中持续更新）
README
echo "  📄 README.md"

echo ""
echo "✅ 初始化完成！"
echo ""
echo "📌 接下来的步骤："
echo "  1. 把三个Excel文件放到 data/ 目录"
echo "  2. cd ml-learning && claude"
echo "  3. 输入 /init 开始学习之旅！"
echo ""
