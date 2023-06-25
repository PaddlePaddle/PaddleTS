import numpy as np

import paddle

from paddlets.datasets.repository import get_dataset
from paddlets.metrics.metrics import MAE
from paddlets.models.forecasting import NBEATSModel
from paddlets.metrics import MSE, MAE
from paddlets.transform.sklearn_transforms import StandardScaler
from paddlets.datasets.repository._data_config import ETTm1Dataset
from paddlets.automl.autots import AutoTS

# 固定随机随机种子，保证训练结果可复现
seed = 2022
paddle.seed(seed)
np.random.seed(seed)

dataset_name = 'ETTm1'
dataset_config = dataset_name + 'Dataset'
save_path = 'output/NBeats_forecasting_scaler_all_ETT'

target_col = eval(dataset_config).load_param['target_cols']
observerd_cols = eval(dataset_config).load_param['observed_cov_cols']

ts_data = get_dataset(dataset_name)

# analysis data: summary and fft
print('Summary:', ts_data.summary())
# from paddlets.analysis import FFT
# fft = FFT()
# res = fft(ts_data, columns=target_col)
# fft.plot().savefig(f'{dataset_name}_fft.png')
# OR
# from paddlets.analysis import AnalysisReport
# customized_config = {"fft":{"norm":False,"fs":1}}
# report = AnalysisReport(ts_data, ["summary","fft"], customized_config)
# report.export_docx_report()

# preprocess:
scaler = StandardScaler(
    [target_col] +
    observerd_cols)  # do scale across all columns/seperate scale
# scaler = StandardScaler(['MT_320']+ts_data._observed_cov._data.columns.to_list()) # do scale across all columns
ts_data = scaler.fit_transform(ts_data)
# ts_train_dataset, ts_val_test_dataset = ts_data.split('2014-12-27 23:00:00') # 96 prediction length
# ts_train_dataset, ts_val_dataset = ts_data.split('2018-06-25 19:45:00') # 96 prediction length
ts_train_dataset, ts_val_test_dataset = ts_data.split(
    '2017-09-09 19:45:00')  # 96 prediction length # 13936
ts_val_dataset, ts_test_dataset = ts_val_test_dataset.split(0.5)

# model = NBEATSModel(in_chunk_len=96, out_chunk_len=96, max_epochs=50)
# model.fit(train_tsdataset=ts_train_dataset, )
# AUTOTS model
# model = AutoTS(NBEATSModel, in_chunk_len=96, out_chunk_len=48)
# model.fit(ts_train_dataset, gpu_resource=1)

# pred on the val data span
# predicted_dataset = model.predict(ts_train_dataset) # does the predict process like this?


## Plot the result
def plot_result(predicted_dataset):
    ts_data.plot().get_figure().savefig('origin.png')
    ts_train_dataset.plot('MT_320').get_figure().savefig('train_only.png')
    ts_val_dataset.plot(
        'MT_320', labels=['VAL']).get_figure().savefig('val_only.png')
    ts_val_test_dataset.plot(
        add_data=recursive_pred_data,
        labels=['Pred']).get_figure().savefig('predict_ETT_backtest.png')
    predicted_dataset.plot(
        columns='MT_320').get_figure().savefig('pred_only.png')


# Calculate the results
def calculate_metric(predicted_dataset):
    mse = MSE(mode='normal')(ts_test_dataset, recursive_pred_data)
    mae = MAE(mode='normal')(ts_val_dataset, predicted_dataset)
    print("mse: {}; mae: {}".format(mse, mae))


def save_model(path, autots):
    if not autots:
        model.save('output/NBeats_forecasting_scaler_all_ETT')
    else:
        # AutoTS save
        best_estimator = model.best_estimator()
        best_estimator.save(path="./autots_best_estimator")


from paddlets.models.model_loader import load
model = load(save_path)
# predicted_dataset = model.predict(ts_train_dataset) # does the predict process like this?
#如果我们想要预测的长度大于模型初始化时候指定的 out_chunk_len 长度：
# recursive_pred_data = model.recursive_predict(ts_val_dataset, predict_length=13920)

# Backtest
from paddlets.utils import backtest
score, preds_data = backtest(
    data=ts_test_dataset, model=model, return_predicts=True)
ts_val_test_dataset.plot(
    add_data=preds_data,
    labels='backtest').get_figure().savefig('backtest_ETT.png')

import pdb
pdb.set_trace()
