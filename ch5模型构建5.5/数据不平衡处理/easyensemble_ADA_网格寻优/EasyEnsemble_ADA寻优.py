"""
EasyEnsemble + AdaBoost 默认参数评估
5折交叉验证
基模型: AdaBoost (全部默认参数)
"""

import warnings
warnings.filterwarnings("ignore")

import time
import tracemalloc
import psutil

import pandas as pd
import numpy as np
from collections import Counter

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)

from imblearn.ensemble import EasyEnsembleClassifier

# ============================================================
# 1. 读取数据
# ============================================================
df = pd.read_csv(r"C:\Users\86158\Desktop\ch5模型构建5.5\原始数据.csv", encoding="utf-8-sig")
target_col = df.columns[-1]
num_cols = df.select_dtypes(include="number").columns.tolist()
feat_cols = [c for c in num_cols if c != target_col]
X = df[feat_cols].values
y = df[target_col].values

print("=" * 72)
print(" EasyEnsemble + AdaBoost 默认参数评估")
print("=" * 72)
print(f"  样本总数 : {len(df)}, 特征数: {X.shape[1]}")
print(f"  类别分布 : {dict(sorted(Counter(y).items()))}")
print(f"  不平衡比 : {Counter(y)[0] / Counter(y)[1]:.1f}:1")

# ============================================================
# 2. 默认参数说明
# ============================================================
# EasyEnsembleClassifier 默认: n_estimators=10
# AdaBoostClassifier 默认: n_estimators=50, learning_rate=1.0, estimator=DecisionTreeClassifier(max_depth=1)
print(f"\n 使用默认参数:")
print(f"    EasyEnsemble n_estimators : 10 (默认)")
print(f"    AdaBoost n_estimators     : 50 (默认)")
print(f"    AdaBoost learning_rate    : 1.0 (默认)")
print(f"    DecisionTree max_depth    : 1 (默认)")

# ============================================================
# 3. 5折交叉验证评估
# ============================================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print(f"\n{'─' * 72}")
print(f" 5折交叉验证评估 (默认参数)...")
print(f"{'─' * 72}")

accs, precs, recs, f1s, roc_aucs, pr_aucs = [], [], [], [], [], []

for train_idx, test_idx in cv.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    ee = EasyEnsembleClassifier(
        estimator=AdaBoostClassifier(random_state=42),
        random_state=42,
        n_jobs=-1,
    )
    ee.fit(X_train, y_train)

    y_pred = ee.predict(X_test)
    y_prob = ee.predict_proba(X_test)[:, 1]

    accs.append(accuracy_score(y_test, y_pred))
    precs.append(precision_score(y_test, y_pred, zero_division=0))
    recs.append(recall_score(y_test, y_pred, zero_division=0))
    f1s.append(f1_score(y_test, y_pred, zero_division=0))
    roc_aucs.append(roc_auc_score(y_test, y_prob))
    pr_aucs.append(average_precision_score(y_test, y_prob))

metrics = {
    "Accuracy": np.mean(accs),
    "Precision": np.mean(precs),
    "Recall": np.mean(recs),
    "F1-Score": np.mean(f1s),
    "ROC-AUC": np.mean(roc_aucs),
    "PR-AUC": np.mean(pr_aucs),
}

# ============================================================
# 4. 推理时长 & 内存占用
# ============================================================
print(f"  正在计算推理时长 (repeat=3, 冷启动排除)...")
infer_times = []
for trial_run in range(3):
    train_idx, test_idx = list(cv.split(X, y))[0]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    ee_run = EasyEnsembleClassifier(
        estimator=AdaBoostClassifier(random_state=42),
        random_state=42,
        n_jobs=-1,
    )
    ee_run.fit(X_train, y_train)

    if trial_run == 0:
        _ = ee_run.predict(X_test)

    start = time.perf_counter()
    _ = ee_run.predict(X_test)
    infer_times.append((time.perf_counter() - start) / len(X_test))

single_sample_time = float(np.median(infer_times))

print(f"  正在计算内存占用...")
ee_full = EasyEnsembleClassifier(
    estimator=AdaBoostClassifier(random_state=42),
    random_state=42,
    n_jobs=-1,
)

tracemalloc.start()
_ = ee_full.fit(X, y)
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

mem_delta = peak / (1024 * 1024)
total_mem = psutil.virtual_memory().total / (1024 * 1024)
mem_ratio = mem_delta / total_mem * 100

# ============================================================
# 5. 输出结果
# ============================================================
print()
print("=" * 72)
print(" 最终结果")
print("=" * 72)
print(f" 模型: EasyEnsemble + AdaBoost (全部默认参数)")
print()
print(f" 参数配置:")
print(f"    EasyEnsemble n_estimators : 10 (默认)")
print(f"    AdaBoost n_estimators     : 50 (默认)")
print(f"    AdaBoost learning_rate    : 1.0 (默认)")
print(f"    DecisionTree max_depth    : 1 (默认)")
print()
print(f" 评估指标 (5折交叉验证均值):")
print(f"    Accuracy   : {metrics['Accuracy']:.4f}")
print(f"    Precision  : {metrics['Precision']:.4f}")
print(f"    Recall     : {metrics['Recall']:.4f}")
print(f"    F1-Score   : {metrics['F1-Score']:.4f}")
print(f"    ROC-AUC    : {metrics['ROC-AUC']:.4f}")
print(f"    PR-AUC     : {metrics['PR-AUC']:.4f}")
print()
print(f" 性能指标:")
print(f"    单样本推理时长 (从第2轮计, repeat=3): {single_sample_time:.6f} 秒")
print(f"    内存占用 (peak - baseline)         : {mem_delta:.2f} MB")
print(f"    内存占比                             : {mem_ratio:.4f}%")
print("=" * 72)

# ============================================================
# 6. 保存结果
# ============================================================
output_dir = r"C:\Users\86158\Desktop\ch5模型构建5.5\数据不平衡处理\easyensemble_ADA_网格寻优"

# 6.1 CSV
result_row = {
    "参数配置": "默认",
    "EasyEnsemble_n_estimators": 10,
    "AdaBoost_n_estimators": 50,
    "AdaBoost_learning_rate": 1.0,
    "DecisionTree_max_depth": 1,
    "Accuracy": round(metrics["Accuracy"], 4),
    "Precision": round(metrics["Precision"], 4),
    "Recall": round(metrics["Recall"], 4),
    "F1-Score": round(metrics["F1-Score"], 4),
    "ROC-AUC": round(metrics["ROC-AUC"], 4),
    "PR-AUC": round(metrics["PR-AUC"], 4),
    "单样本推理时长(秒)": round(single_sample_time, 6),
    "内存占用(MB)": round(mem_delta, 2),
    "内存占比(%)": round(mem_ratio, 4),
}
result_df = pd.DataFrame([result_row])
csv_path = f"{output_dir}\\EasyEnsemble_ADA寻优结果.csv"
result_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"\n结果已保存: {csv_path}")

# 6.2 Markdown
md_content = f"""# EasyEnsemble + AdaBoost 默认参数评估报告

## 1. 实验配置

| 项目 | 内容 |
|---|---|
| 采样方法 | EasyEnsemble |
| 基分类器 | AdaBoost (默认参数) |
| 参数调优 | 无 (全部默认) |
| 交叉验证 | 5折 Stratified KFold (random_state=42) |

## 2. 默认参数

| 参数 | 默认值 |
|---|---|
| EasyEnsemble n_estimators | 10 |
| AdaBoost n_estimators | 50 |
| AdaBoost learning_rate | 1.0 |
| DecisionTree max_depth | 1 |

## 3. 评估指标（5折交叉验证均值）

| 指标 | 值 |
|---|---|
| Accuracy | **{metrics['Accuracy']:.4f}** |
| Precision | **{metrics['Precision']:.4f}** |
| Recall | **{metrics['Recall']:.4f}** |
| F1-Score | **{metrics['F1-Score']:.4f}** |
| ROC-AUC | **{metrics['ROC-AUC']:.4f}** |
| PR-AUC | **{metrics['PR-AUC']:.4f}** |

## 4. 性能指标

| 指标 | 值 |
|---|---|
| 单样本推理时长（从第2轮计, repeat=3） | **{single_sample_time:.6f} 秒** |
| 内存占用（peak - baseline） | **{mem_delta:.2f} MB** |
| 内存占比 | **{mem_ratio:.4f}%** |

## 5. 结论

- 使用全部默认参数，无需任何调参
- Recall = **{metrics['Recall']:.4f}**，PR-AUC = **{metrics['PR-AUC']:.4f}**
- 作为基线对比，可与其他调优方法（网格搜索、Optuna）的结果进行对比

---
*报告生成时间: 自动生成*
"""

md_path = f"{output_dir}\\EasyEnsemble_ADA寻优结果.md"
with open(md_path, "w", encoding="utf-8") as f:
    f.write(md_content)
print(f"Markdown 报告已保存: {md_path}")
