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
