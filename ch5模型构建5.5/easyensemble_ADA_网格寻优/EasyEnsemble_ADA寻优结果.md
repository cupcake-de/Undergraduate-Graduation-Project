1.3.1.最优参数
参数	值
EasyEnsemble n_estimators	5
AdaBoost n_estimators	100
AdaBoost learning_rate	0.01
DecisionTree max_depth	3
1.3.2.评估指标（5折均值）
指标	AdaBoost
Accuracy	0.5579
Precision	0.0158
Recall	0.7818 ✅
F1-Score	0.0309
ROC-AUC	0.7509
PR-AUC	0.0663
1.3.3.性能指标
指标	AdaBoost	XGBoost
单样本推理时长	0.000055 秒	0.000063 秒
内存占用	1.66 MB	1.65 MB
内存占比	0.0103%	0.0103%