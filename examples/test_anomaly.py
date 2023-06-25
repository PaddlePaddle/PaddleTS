import paddle
import numpy as np
from paddlets.transform import StandardScaler
from paddlets.models.anomaly import AutoEncoder, MTADGAT
from paddlets.datasets.repository import get_dataset
from paddlets.utils import plot_anoms
from paddlets.metrics import F1, Precision, Recall

# 固定随机随机种子，保证训练结果可复现
seed = 2022
paddle.seed(seed)
np.random.seed(seed)

import pdb
pdb.set_trace()
# 据集拆分
ts_data = get_dataset('NAB_TEMP')
train_tsdata, test_tsdata = ts_data.split(0.15)
# 标准化
scaler = StandardScaler('value')
scaler.fit(train_tsdata)  # 提取trainset的统计信息
train_data_scaled = scaler.transform(train_tsdata)
test_data_scaled = scaler.transform(test_tsdata)

# #建模与训练
model = MTADGAT(
    in_chunk_len=2,  # 样本数据窗口大小
    max_epochs=100  # 最大epoch设为100
)
model.fit(train_data_scaled)

# 预测
pred_label = model.predict(test_data_scaled)
lable_name = pred_label.target.data.columns[0]
# 计算评估指标 f1, precision, recall
f1 = F1()(test_tsdata, pred_label)
precision = Precision()(test_tsdata, pred_label)
recall = Recall()(test_tsdata, pred_label)
print('f1: ', f1[lable_name])
print('precision: ', precision[lable_name])
print('recall: ', recall[lable_name])

import pdb
pdb.set_trace()
# plot_anoms(origin_data=test_tsdata, predict_data=pred_label, feature_name="value")

# 模型重载
# model.save('./anomaly_ae_model')

# from paddlets.models.model_loader import load
# loaded_model = load('./anomaly_ae_model')
# pred_label = loaded_model.predict(test_data_scaled)
# pred_score = loaded_model.predict_score(test_data_scaled)

# AutoEncoder
# 0.48688
