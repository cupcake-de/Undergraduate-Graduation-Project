# EasyEnsemble + AdaBoost 默认参数评估报告

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
| Accuracy | **0.6368** |
| Precision | **0.0191** |
| Recall | **0.7818** |
| F1-Score | **0.0373** |
| ROC-AUC | **0.7623** |
| PR-AUC | **0.0415** |

## 4. 性能指标

| 指标 | 值 |
|---|---|
| 单样本推理时长（从第2轮计, repeat=3） | **0.000045 秒** |
| 内存占用（peak - baseline） | **1.64 MB** |
| 内存占比 | **0.0102%** |

## 5. 结论

- 使用全部默认参数，无需任何调参
- Recall = **0.7818**，PR-AUC = **0.0415**
- 作为基线对比，可与其他调优方法（网格搜索、Optuna）的结果进行对比

---
*报告生成时间: 自动生成*
