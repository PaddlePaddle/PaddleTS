## 算法指标

### 时序预测

数据集: [ETTH1](https://paddle-model-ecology.bj.bcebos.com/paddlex/data/Etth1.tar) （在测试集test.csv上的评测结果）

| ID | model |  输入序列长度 | 预测序列长度 | 精度（mse/mae） |
|-----|-----|--------|----| --- |
| 01 | DLinear | 96 | 96 | 0.382/0.394 |
| 02| RLinear | 96 | 96  | 0.384/0.392 |
| 03 | NLinear | 96 | 96 | 0.386/0.392 |
| 04 | PatchTST | 96 | 96 | 0.385/0.397 |
| 05 | Non-stationary | 96 | 96 | 0.600/0.515 |
| 06 | TimesNet | 96 | 96 | 0.417/0.431 |
| 07 | TiDE | 720 | 96 | 0.405/0.412 |

### 时序异常检测

数据 [PSM](https://paddle-model-ecology.bj.bcebos.com/paddlex/data/ts_anomaly_examples.tar)

| ID | model |  输入序列长度 | f1/recall/precision |
| 01 | TimesNet_ad | 100 | 96.56/94.80/98.37|
| 02 | DLinear_ad | 100 | 96.41/93.96/0.9898|
| 03 | PatchTST_ad | 100 | 94.57/90.70/98.78 |
| 04 | Non-stationary_ad | 100 |93.51/ 88.95/98.55|
| 05 | AutoEncoder_ad | 100 |  91.25/84.36/99.36 |


### 时序分类

数据：
UWaveGestureLibrary：
[训练](https://paddlets.bj.bcebos.com/classification/UWaveGestureLibrary_TRAIN.csv)
[评测](https://paddlets.bj.bcebos.com/classification/UWaveGestureLibrary_TEST.csv)

| ID | model |  acc |
| 01 | TimesNet_cls | 87.5|







