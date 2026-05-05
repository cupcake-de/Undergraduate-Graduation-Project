"""
数据采样对比实验 — XGBoost 默认参数评估
采样方法：原始数据(不采样) + 过采样 + 欠采样
评估指标：Accuracy / Precision / Recall / F1 / ROC-AUC / PR-AUC
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from collections import Counter

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)

from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, NearMiss
from imblearn.combine import SMOTETomek

from xgboost import XGBClassifier

# ============================================================
# 1. 读取数据
# ============================================================
df = pd.read_csv(r"C:\Users\86158\Desktop\ch5模型构建5.5\原始数据.csv", encoding="utf-8-sig")
target_col = df.columns[-1]  # "是否为谣言"
# 丢弃文本列（第一列"事件名称"及其他非数值列）
num_cols = df.select_dtypes(include="number").columns.tolist()
feat_cols = [c for c in num_cols if c != target_col]
X = df[feat_cols]
y = df[target_col]

print("=" * 70)
print(" 原始数据概览")
print("=" * 70)
print(f"  样本总数 : {len(df)}")
print(f"  特征数量 : {X.shape[1]}")
print(f"  特征列表 : {list(X.columns)}")
print(f"  类别分布 : {Counter(y)}")
print(f"  少数类占比: {y.mean():.4%}")
print(f"  不平衡比 : {Counter(y)[0] / Counter(y)[1]:.1f}:1")
print()

# ============================================================
# 2. 定义采样方法
# ============================================================
samplers = {
    "原始数据(不采样)": None,
    # --- 过采样 ---
    "RandomOverSampler": RandomOverSampler(random_state=42),
    "SMOTE": SMOTE(random_state=42, k_neighbors=5),
    "BorderlineSMOTE": BorderlineSMOTE(random_state=42, k_neighbors=5),
    "ADASYN": ADASYN(random_state=42, n_neighbors=5),
    # --- 欠采样 ---
    "RandomUnderSampler": RandomUnderSampler(random_state=42),
    "NearMiss-1": NearMiss(version=1),
    "NearMiss-2": NearMiss(version=2),
    "TomekLinks": TomekLinks(),
    # --- 混合采样 ---
    "SMOTETomek": SMOTETomek(random_state=42),
}

# ============================================================
# 3. 采样后数据报告
# ============================================================
print("=" * 70)
print(" 采样后数据报告")
print("=" * 70)
print(f"{'采样方法':<22} {'总样本':>8} {'多数类':>8} {'少数类':>8} {'少数类占比':>10} {'不平衡比':>10}")
print("-" * 70)

for name, sampler in samplers.items():
    if sampler is None:
        X_s, y_s = X, y
    else:
        X_s, y_s = sampler.fit_resample(X, y)
    cnt = Counter(y_s)
    ratio = cnt[0] / max(cnt[1], 1)
    print(f"{name:<22} {len(y_s):>8} {cnt[0]:>8} {cnt[1]:>8} {cnt[1]/len(y_s):>10.4%} {ratio:>10.1f}:1")
print()

# ============================================================
# 4. XGBoost 默认参数 + 5折交叉验证评估
# ============================================================
print("=" * 70)
print(" XGBoost 默认参数 — 5折交叉验证评估")
print("=" * 70)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# XGBoost 默认参数
xgb_params = dict(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.3,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    verbosity=0,
)

results = []

for name, sampler in samplers.items():
    accs, precs, recs, f1s, roc_aucs, pr_aucs = [], [], [], [], [], []

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # 仅在训练集上采样
        if sampler is not None:
            X_train_s, y_train_s = sampler.fit_resample(X_train, y_train)
        else:
            X_train_s, y_train_s = X_train, y_train

        # 标准化（仅用于少数需要距离计算的采样器兼容性，XGBoost 本身不强制要求）
        scaler = StandardScaler()
        X_train_s = pd.DataFrame(
            scaler.fit_transform(X_train_s), columns=X.columns
        )
        X_test_s = pd.DataFrame(
            scaler.transform(X_test), columns=X.columns
        )

        model = XGBClassifier(**xgb_params)
        model.fit(X_train_s, y_train_s)

        y_pred = model.predict(X_test_s)
        y_prob = model.predict_proba(X_test_s)[:, 1]

        accs.append(accuracy_score(y_test, y_pred))
        precs.append(precision_score(y_test, y_pred, zero_division=0))
        recs.append(recall_score(y_test, y_pred, zero_division=0))
        f1s.append(f1_score(y_test, y_pred, zero_division=0))
        roc_aucs.append(roc_auc_score(y_test, y_prob))
        pr_aucs.append(average_precision_score(y_test, y_prob))

    results.append({
        "采样方法": name,
        "Accuracy": np.mean(accs),
        "Precision": np.mean(precs),
        "Recall": np.mean(recs),
        "F1-Score": np.mean(f1s),
        "ROC-AUC": np.mean(roc_aucs),
        "PR-AUC": np.mean(pr_aucs),
    })

# ============================================================
# 5. 输出评估结果表格
# ============================================================
res_df = pd.DataFrame(results)
print()
print(res_df.to_string(index=False, float_format="%.4f"))
print()

# ============================================================
# 6. 标注最佳结果
# ============================================================
print("=" * 70)
print(" 各指标最佳采样方法")
print("=" * 70)
for metric in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"]:
    best_idx = res_df[metric].idxmax()
    best_name = res_df.loc[best_idx, "采样方法"]
    best_val = res_df.loc[best_idx, metric]
    print(f"  {metric:<12} → {best_name:<22} ({best_val:.4f})")
print()

# ============================================================
# 7. 保存结果
# ============================================================
output_path = r"C:\Users\86158\Desktop\ch5模型构建5.5\采样对比评估结果.csv"
res_df.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"评估结果已保存至: {output_path}")
