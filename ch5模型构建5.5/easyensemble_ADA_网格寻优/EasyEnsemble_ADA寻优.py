"""
EasyEnsemble + AdaBoost 参数寻优
约束: Recall >= 0.7
优化目标: PR-AUC 最大
5折交叉验证
基模型: AdaBoost (默认)
"""

import warnings
warnings.filterwarnings("ignore")

import time
import tracemalloc
import psutil

import pandas as pd
import numpy as np
from collections import Counter
from itertools import product

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
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
print(" EasyEnsemble + AdaBoost 参数寻优")
print("=" * 72)
print(f"  样本总数 : {len(df)}, 特征数: {X.shape[1]}")
print(f"  类别分布 : {dict(sorted(Counter(y).items()))}")
print(f"  不平衡比 : {Counter(y)[0] / Counter(y)[1]:.1f}:1")

# ============================================================
# 2. 参数搜索空间
# ============================================================
n_estimators_ee = [5, 10, 15, 20]
n_estimators_ada = [25, 50, 75, 100]
learning_rates = [0.01, 0.1, 0.5, 1.0]
max_depths = [1, 2, 3]  # AdaBoost 基学习器深度

param_combinations = list(product(n_estimators_ee, n_estimators_ada, learning_rates, max_depths))
print(f"\n参数搜索空间: {len(param_combinations)} 种组合")
print(f"  EasyEnsemble n_estimators : {n_estimators_ee}")
print(f"  AdaBoost n_estimators    : {n_estimators_ada}")
print(f"  AdaBoost learning_rate    : {learning_rates}")
print(f"  DecisionTree max_depth    : {max_depths}")

# ============================================================
# 3. 5折交叉验证网格搜索
# ============================================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_results = []

total = len(param_combinations)
print(f"\n{'─' * 72}")
print(f" 开始网格搜索 ({total} 组 x 5折)...")
print(f"{'─' * 72}")

for idx, (n_est_ee, n_est_ada, lr_ada, max_d) in enumerate(param_combinations):
    accs, precs, recs, f1s, roc_aucs, pr_aucs = [], [], [], [], [], []

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # EasyEnsemble + AdaBoost (默认基模型)
        ee = EasyEnsembleClassifier(
            n_estimators=n_est_ee,
            estimator=AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=max_d),
                n_estimators=n_est_ada,
                learning_rate=lr_ada,
                random_state=42,
            ),
            sampling_strategy="auto",
            replacement=False,
            n_jobs=-1,
            random_state=42,
            verbose=0,
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

    mean_recall = np.mean(recs)
    mean_prauc = np.mean(pr_aucs)

    all_results.append({
        "n_estimators_ee": n_est_ee,
        "n_estimators_ada": n_est_ada,
        "learning_rate_ada": lr_ada,
        "max_depth_tree": max_d,
        "Accuracy": np.mean(accs),
        "Precision": np.mean(precs),
        "Recall": mean_recall,
        "F1-Score": np.mean(f1s),
        "ROC-AUC": np.mean(roc_aucs),
        "PR-AUC": mean_prauc,
    })

    if (idx + 1) % 30 == 0 or (idx + 1) == total:
        print(f"  进度: {idx + 1}/{total} ({100*(idx+1)/total:.1f}%)  |  "
              f"Recall >= 0.7: {sum(1 for r in all_results if r['Recall'] >= 0.7)} 组")

# ============================================================
# 4. 选择满足约束且 PR-AUC 最优的结果
# ============================================================
valid_results = [r for r in all_results if r["Recall"] >= 0.7]
constraint_satisfied = len(valid_results) > 0

if constraint_satisfied:
    best = max(valid_results, key=lambda r: r["PR-AUC"])
    constraint_status = f"[PASS] Recall = {best['Recall']:.4f} >= 0.7"
else:
    best = max(all_results, key=lambda r: r["PR-AUC"])
    constraint_status = f"[WARN] Recall < 0.7, best Recall = {best['Recall']:.4f}"

print(f"\n{'─' * 72}")
print(f" 最优参数 (约束: Recall >= 0.7, 优化: PR-AUC)")
print(f"{'─' * 72}")

# ============================================================
# 5. 使用最优参数重新训练，计算推理时长和内存
# ============================================================
print(f"\n{'─' * 72}")
print(f" 最优参数精细评估 (含推理时长 & 内存占用)")
print(f"{'─' * 72}")

best_ee = EasyEnsembleClassifier(
    n_estimators=best["n_estimators_ee"],
    estimator=AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=best["max_depth_tree"]),
        n_estimators=best["n_estimators_ada"],
        learning_rate=best["learning_rate_ada"],
        random_state=42,
    ),
    sampling_strategy="auto",
    replacement=False,
    n_jobs=-1,
    random_state=42,
    verbose=0,
)

# 推理时长（第2轮开始计时）
print("  正在计算推理时长 (repeat=3, 冷启动排除)...")
infer_times = []
for trial in range(3):
    train_idx, test_idx = list(cv.split(X, y))[0]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    best_ee_trial = EasyEnsembleClassifier(
        n_estimators=best["n_estimators_ee"],
        estimator=AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=best["max_depth_tree"]),
            n_estimators=best["n_estimators_ada"],
            learning_rate=best["learning_rate_ada"],
            random_state=42,
        ),
        sampling_strategy="auto",
        replacement=False,
        n_jobs=-1,
        random_state=42,
        verbose=0,
    )
    best_ee_trial.fit(X_train, y_train)

    if trial == 0:
        _ = best_ee_trial.predict(X_test)  # 第1轮: 冷启动

    start = time.perf_counter()
    _ = best_ee_trial.predict(X_test)
    infer_times.append((time.perf_counter() - start) / len(X_test))

single_sample_time = float(np.median(infer_times))

# 内存占用
print("  正在计算内存占用...")
process = psutil.Process()
mem_before = process.memory_info().rss / (1024 * 1024)

tracemalloc.start()
_ = best_ee.fit(X, y)
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

mem_delta = peak / (1024 * 1024)
total_mem = psutil.virtual_memory().total / (1024 * 1024)
mem_ratio = mem_delta / total_mem * 100

# ============================================================
# 6. 输出最终结果
# ============================================================
print()
print("=" * 72)
print(" 最终结果")
print("=" * 72)
print(f" 约束检验: {constraint_status}")
print()
print(f" 最优参数:")
print(f"    EasyEnsemble n_estimators : {best['n_estimators_ee']}")
print(f"    AdaBoost n_estimators     : {best['n_estimators_ada']}")
print(f"    AdaBoost learning_rate    : {best['learning_rate_ada']}")
print(f"    DecisionTree max_depth    : {best['max_depth_tree']}")
print()
print(f" 评估指标 (5折交叉验证均值):")
print(f"    Accuracy   : {best['Accuracy']:.4f}")
print(f"    Precision  : {best['Precision']:.4f}")
print(f"    Recall     : {best['Recall']:.4f}")
print(f"    F1-Score   : {best['F1-Score']:.4f}")
print(f"    ROC-AUC    : {best['ROC-AUC']:.4f}")
print(f"    PR-AUC     : {best['PR-AUC']:.4f}")
print()
print(f" 性能指标:")
print(f"    单样本推理时长 (从第2轮计, repeat=3): {single_sample_time:.6f} 秒")
print(f"    内存占用 (peak - baseline)         : {mem_delta:.2f} MB")
print(f"    内存占比                             : {mem_ratio:.4f}%")
print("=" * 72)

# ============================================================
# 7. 保存结果
# ============================================================
output_dir = r"C:\Users\86158\Desktop\ch5模型构建5.5\数据不平衡处理"

full_df = pd.DataFrame(all_results)
full_df = full_df.sort_values("PR-AUC", ascending=False).reset_index(drop=True)
full_path = f"{output_dir}\\EasyEnsemble_ADA_GridSearch_All.csv"
full_df.to_csv(full_path, index=False, encoding="utf-8-sig")
print(f"\n完整搜索结果已保存: {full_path}")

best_row = {
    "约束满足": "是" if constraint_satisfied else "否（输出PR-AUC最优）",
    "EasyEnsemble_n_estimators": best["n_estimators_ee"],
    "AdaBoost_n_estimators": best["n_estimators_ada"],
    "AdaBoost_learning_rate": best["learning_rate_ada"],
    "DecisionTree_max_depth": best["max_depth_tree"],
    "Accuracy": best["Accuracy"],
    "Precision": best["Precision"],
    "Recall": best["Recall"],
    "F1-Score": best["F1-Score"],
    "ROC-AUC": best["ROC-AUC"],
    "PR-AUC": best["PR-AUC"],
    "单样本推理时长(秒)": single_sample_time,
    "内存占用(MB)": mem_delta,
    "内存占比(%)": mem_ratio,
}
best_df = pd.DataFrame([best_row])
best_path = f"{output_dir}\\EasyEnsemble_ADA寻优结果.csv"
best_df.to_csv(best_path, index=False, encoding="utf-8-sig")
print(f"最优结果已保存: {best_path}")

if valid_results:
    valid_df = pd.DataFrame(valid_results).sort_values("PR-AUC", ascending=False).reset_index(drop=True)
    print(f"\n满足 Recall >= 0.7 的参数组合: {len(valid_df)} 组")
    print(valid_df[["n_estimators_ee","n_estimators_ada","learning_rate_ada","max_depth_tree",
                    "Accuracy","Precision","Recall","F1-Score","ROC-AUC","PR-AUC"]].to_string(index=False))
else:
    print(f"\n无满足 Recall >= 0.7 的参数组合，已输出 PR-AUC 最优结果")
