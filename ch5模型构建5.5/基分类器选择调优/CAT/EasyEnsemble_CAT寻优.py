"""
EasyEnsemble + CatBoost 参数寻优 (Optuna)
约束: Recall >= 0.7
优化目标: PR-AUC 最大
5折交叉验证
基模型: CatBoost
"""

import warnings
warnings.filterwarnings("ignore")

import time
import tracemalloc
import psutil

import pandas as pd
import numpy as np
from collections import Counter

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)

from imblearn.ensemble import EasyEnsembleClassifier
from catboost import CatBoostClassifier

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
print(" EasyEnsemble + CatBoost 参数寻优 (Optuna)")
print("=" * 72)
print(f"  样本总数 : {len(df)}, 特征数: {X.shape[1]}")
print(f"  类别分布 : {dict(sorted(Counter(y).items()))}")
print(f"  不平衡比 : {Counter(y)[0] / Counter(y)[1]:.1f}:1")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ============================================================
# 2. Optuna 目标函数
# ============================================================
N_TRIALS = 150

def objective(trial):
    n_est_ee = trial.suggest_int("n_estimators_ee", 5, 20, step=5)
    depth = trial.suggest_int("depth", 3, 9)
    lr = trial.suggest_float("learning_rate", 0.01, 0.5, log=True)
    iters = trial.suggest_int("iterations", 50, 200, step=50)

    accs, precs, recs, f1s, roc_aucs, pr_aucs = [], [], [], [], [], []

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        ee = EasyEnsembleClassifier(
            n_estimators=n_est_ee,
            estimator=CatBoostClassifier(
                depth=depth,
                learning_rate=lr,
                iterations=iters,
                random_seed=42,
                verbose=False,
                allow_writing_files=False,
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

    if mean_recall < 0.7:
        trial.set_user_attr("recall", mean_recall)
        trial.set_user_attr("constraint_met", False)
        return -999.0

    trial.set_user_attr("recall", mean_recall)
    trial.set_user_attr("accuracy", np.mean(accs))
    trial.set_user_attr("precision", np.mean(precs))
    trial.set_user_attr("f1", np.mean(f1s))
    trial.set_user_attr("roc_auc", np.mean(roc_aucs))
    trial.set_user_attr("constraint_met", True)

    return mean_prauc

# ============================================================
# 3. 运行 Optuna 寻优
# ============================================================
print(f"\n参数搜索空间 (Optuna): {N_TRIALS} 轮")
print(f"  EasyEnsemble n_estimators : [5, 10, 15, 20]")
print(f"  CatBoost depth         : [3, 9]")
print(f"  CatBoost learning_rate : [0.01, 0.5] (log-scale)")
print(f"  CatBoost iterations    : [50, 200] (step=50)")

print(f"\n{'─' * 72}")
print(f" 开始 Optuna 寻优 ({N_TRIALS} 轮 x 5折)...")
print(f"{'─' * 72}")

study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42),
)

study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

# ============================================================
# 4. 提取最优结果
# ============================================================
valid_trials = [t for t in study.trials if t.value > -900 and t.user_attrs.get("constraint_met", False)]

print(f"\n总试验数: {len(study.trials)}, 满足约束: {len(valid_trials)} 组")

if valid_trials:
    best_trial = max(valid_trials, key=lambda t: t.value)
    constraint_satisfied = True
    constraint_status = f"[PASS] Recall = {best_trial.user_attrs['recall']:.4f} >= 0.7"
else:
    best_trial = max(study.trials, key=lambda t: t.value)
    constraint_satisfied = False
    constraint_status = f"[WARN] Recall < 0.7, best Recall = {best_trial.user_attrs.get('recall', 0):.4f}"

best_params = best_trial.params
print(f"\n{'─' * 72}")
print(f" 最优参数 (约束: Recall >= 0.7, 优化: PR-AUC)")
print(f"{'─' * 72}")

# ============================================================
# 5. 使用最优参数精细评估
# ============================================================
print(f"\n{'─' * 72}")
print(f" 最优参数精细评估 (含推理时长 & 内存占用)")
print(f"{'─' * 72}")

best_ee = EasyEnsembleClassifier(
    n_estimators=best_params["n_estimators_ee"],
    estimator=CatBoostClassifier(
        depth=best_params["depth"],
        learning_rate=best_params["learning_rate"],
        iterations=best_params["iterations"],
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
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
for trial_run in range(3):
    train_idx, test_idx = list(cv.split(X, y))[0]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    best_ee_run = EasyEnsembleClassifier(
        n_estimators=best_params["n_estimators_ee"],
        estimator=CatBoostClassifier(
            depth=best_params["depth"],
            learning_rate=best_params["learning_rate"],
            iterations=best_params["iterations"],
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
        ),
        sampling_strategy="auto",
        replacement=False,
        n_jobs=-1,
        random_state=42,
        verbose=0,
    )
    best_ee_run.fit(X_train, y_train)

    if trial_run == 0:
        _ = best_ee_run.predict(X_test)

    start = time.perf_counter()
    _ = best_ee_run.predict(X_test)
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
print(f"    EasyEnsemble n_estimators : {best_params['n_estimators_ee']}")
print(f"    CatBoost depth         : {best_params['depth']}")
print(f"    CatBoost learning_rate : {best_params['learning_rate']:.4f}")
print(f"    CatBoost iterations    : {best_params['iterations']}")
print()
print(f" 评估指标 (5折交叉验证均值):")
print(f"    Accuracy   : {best_trial.user_attrs.get('accuracy', 0):.4f}")
print(f"    Precision  : {best_trial.user_attrs.get('precision', 0):.4f}")
print(f"    Recall     : {best_trial.user_attrs.get('recall', 0):.4f}")
print(f"    F1-Score   : {best_trial.user_attrs.get('f1', 0):.4f}")
print(f"    ROC-AUC    : {best_trial.user_attrs.get('roc_auc', 0):.4f}")
print(f"    PR-AUC     : {best_trial.value:.4f}")
print()
print(f" 性能指标:")
print(f"    单样本推理时长 (从第2轮计, repeat=3): {single_sample_time:.6f} 秒")
print(f"    内存占用 (peak - baseline)         : {mem_delta:.2f} MB")
print(f"    内存占比                             : {mem_ratio:.4f}%")
print("=" * 72)

# ============================================================
# 7. 保存结果
# ============================================================
output_dir = r"C:\Users\86158\Desktop\ch5模型构建5.5\基分类器选择调优\CAT"

# 7.1 所有试验结果
all_trials_data = []
for t in study.trials:
    if t.value > -900:
        all_trials_data.append({
            "n_estimators_ee": t.params.get("n_estimators_ee", None),
            "depth": t.params.get("depth", None),
            "learning_rate": t.params.get("learning_rate", None),
            "iterations": t.params.get("iterations", None),
            "Accuracy": t.user_attrs.get("accuracy", None),
            "Precision": t.user_attrs.get("precision", None),
            "Recall": t.user_attrs.get("recall", None),
            "F1-Score": t.user_attrs.get("f1", None),
            "ROC-AUC": t.user_attrs.get("roc_auc", None),
            "PR-AUC": t.value if t.user_attrs.get("constraint_met", False) else None,
            "constraint_met": t.user_attrs.get("constraint_met", False),
        })

all_df = pd.DataFrame(all_trials_data)
all_df = all_df.dropna(subset=["PR-AUC"]).sort_values("PR-AUC", ascending=False).reset_index(drop=True)
all_path = f"{output_dir}\\EasyEnsemble_CAT_GridSearch_All.csv"
all_df.to_csv(all_path, index=False, encoding="utf-8-sig")
print(f"\n完整搜索结果已保存: {all_path}")

# 7.2 最优结果 CSV
best_row = {
    "约束满足": "是" if constraint_satisfied else "否（输出PR-AUC最优）",
    "EasyEnsemble_n_estimators": best_params["n_estimators_ee"],
    "CatBoost_depth": best_params["depth"],
    "CatBoost_learning_rate": round(best_params["learning_rate"], 4),
    "CatBoost_iterations": best_params["iterations"],
    "Accuracy": round(best_trial.user_attrs.get("accuracy", 0), 4),
    "Precision": round(best_trial.user_attrs.get("precision", 0), 4),
    "Recall": round(best_trial.user_attrs.get("recall", 0), 4),
    "F1-Score": round(best_trial.user_attrs.get("f1", 0), 4),
    "ROC-AUC": round(best_trial.user_attrs.get("roc_auc", 0), 4),
    "PR-AUC": round(best_trial.value, 4),
    "单样本推理时长(秒)": round(single_sample_time, 6),
    "内存占用(MB)": round(mem_delta, 2),
    "内存占比(%)": round(mem_ratio, 4),
}
best_df = pd.DataFrame([best_row])
best_path = f"{output_dir}\\EasyEnsemble_CAT寻优结果.csv"
best_df.to_csv(best_path, index=False, encoding="utf-8-sig")
print(f"最优结果已保存: {best_path}")

# 7.3 Markdown 报告
md_content = f"""# EasyEnsemble + CatBoost Optuna 寻优结果报告

## 1. 实验配置

| 项目 | 内容 |
|---|---|
| 采样方法 | EasyEnsemble |
| 基分类器 | CatBoost |
| 优化算法 | Optuna (TPE Sampler, seed=42) |
| 试验轮数 | {N_TRIALS} |
| 约束条件 | Recall >= 0.7 |
| 优化目标 | PR-AUC 最大 |
| 交叉验证 | 5折 Stratified KFold (random_state=42) |

## 2. 参数搜索空间

| 参数 | 范围 |
|---|---|
| EasyEnsemble n_estimators | [5, 10, 15, 20] |
| CatBoost depth | [3, 9] |
| CatBoost learning_rate | [0.01, 0.5] (log-scale) |
| CatBoost iterations | [50, 100, 150, 200] |

## 3. 最优参数

| 参数 | 最优值 |
|---|---|
| EasyEnsemble n_estimators | **{best_params['n_estimators_ee']}** |
| CatBoost depth | **{best_params['depth']}** |
| CatBoost learning_rate | **{best_params['learning_rate']:.4f}** |
| CatBoost iterations | **{best_params['iterations']}** |

## 4. 约束检验

**{constraint_status}**

## 5. 评估指标（5折交叉验证均值）

| 指标 | 值 |
|---|---|
| Accuracy | **{best_trial.user_attrs.get('accuracy', 0):.4f}** |
| Precision | **{best_trial.user_attrs.get('precision', 0):.4f}** |
| Recall | **{best_trial.user_attrs.get('recall', 0):.4f}** |
| F1-Score | **{best_trial.user_attrs.get('f1', 0):.4f}** |
| ROC-AUC | **{best_trial.user_attrs.get('roc_auc', 0):.4f}** |
| PR-AUC | **{best_trial.value:.4f}** |

## 6. 性能指标

| 指标 | 值 |
|---|---|
| 单样本推理时长（从第2轮计, repeat=3） | **{single_sample_time:.6f} 秒** |
| 内存占用（peak - baseline） | **{mem_delta:.2f} MB** |
| 内存占比 | **{mem_ratio:.4f}%** |

## 7. 搜索统计

| 统计项 | 值 |
|---|---|
| 总试验数 | {len(study.trials)} |
| 满足约束（Recall >= 0.7） | {len(valid_trials)} 组 |
| 满足约束比例 | {len(valid_trials) / len(study.trials):.1%} |

## 8. 满足约束的参数组合 Top-10（按 PR-AUC 降序）

"""
if valid_trials:
    top_valid = sorted(valid_trials, key=lambda t: t.value, reverse=True)[:10]
    md_content += "| 排名 | EE_n_est | depth | LR | iterations | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |\n"
    md_content += "|---|---|---|---|---|---|---|---|---|---|---|\n"
    for rank, t in enumerate(top_valid, 1):
        attrs = t.user_attrs
        md_content += (f"| {rank} | {t.params['n_estimators_ee']} | {t.params['depth']} | "
                       f"{t.params['learning_rate']:.4f} | {t.params['iterations']} | "
                       f"{attrs['accuracy']:.4f} | {attrs['precision']:.4f} | {attrs['recall']:.4f} | "
                       f"{attrs['f1']:.4f} | {attrs['roc_auc']:.4f} | **{t.value:.4f}** |\n")
else:
    md_content += "无满足约束的参数组合\n"

md_content += f"""
## 9. 结论

- 共搜索 **{len(study.trials)}** 种参数组合，其中 **{len(valid_trials)}** 组满足 Recall >= 0.7 约束
- 最优配置 PR-AUC 达到 **{best_trial.value:.4f}**，对应 Recall = **{best_trial.user_attrs.get('recall', 0):.4f}**
- 最优基分类器配置：CatBoost(depth={best_params['depth']}, learning_rate={best_params['learning_rate']:.4f}, iterations={best_params['iterations']})

---
*报告生成时间: 基于 Optuna {optuna.__version__} 自动生成*
"""

md_path = f"{output_dir}\\EasyEnsemble_CAT寻优结果.md"
with open(md_path, "w", encoding="utf-8") as f:
    f.write(md_content)
print(f"Markdown 报告已保存: {md_path}")
