import argparse
import os
import shutil
import numpy as np

import paddle
import paddle.nn.functional as F

from paddlets.utils import backtest
from paddlets.models.model_loader import load
from paddlets.datasets.repository import get_dataset
from paddlets.models.forecasting import TimesNetModel
from paddlets.models.forecasting.utils import M4Meta
from paddlets.metrics import SMAPE
from paddlets.transform.sklearn_transforms import StandardScaler
from paddlets.datasets.repository._data_config import ETTm2Dataset
from paddlets.utils import Type1Decay

import os
import pandas
import warnings
import numpy as np

from paddlets.models.forecasting.utils import M4Summary
from paddlets.datasets import UnivariateDataset

warnings.filterwarnings('ignore')


def test(args, model, label_len, window_sampling_limit):
    ts_train_dataset = [get_dataset(dataset_name + 'train')]
    ts_val_dataset = [get_dataset(dataset_name + 'test')]
    train_dataset = UnivariateDataset(
        ts_train_dataset, seq_len, args.pred_len, label_len,
        window_sampling_limit)  # (96, 48, 48, 480)
    valid_dataset = UnivariateDataset(ts_val_dataset, seq_len, args.pred_len,
                                      label_len, window_sampling_limit)

    x, _ = train_dataset.last_insample_window()
    y = valid_dataset.timeseries
    x = paddle.to_tensor(x).cast('float32')
    x = x.unsqueeze(axis=-1)

    folder_path = './test_results/' + 'M4' + '/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    model._network.eval()
    with paddle.no_grad():
        B, _, C = x.shape
        dec_inp = paddle.zeros(shape=(B, args.pred_len, C)).astype(
            dtype='float32')
        dec_inp = paddle.concat(
            x=[x[:, -label_len:, :], dec_inp], axis=1).astype(dtype='float32')
        outputs = paddle.zeros(shape=(B, args.pred_len, C)).astype(
            dtype='float32')
        id_list = np.arange(0, B, 1)
        id_list = np.append(id_list, B)
        for i in range(len(id_list) - 1):
            outputs[id_list[i]:id_list[i + 1], :, :] = model._network(
                x[id_list[i]:id_list[i + 1]], None)
            if id_list[i] % 1000 == 0:
                print(id_list[i])
        outputs = outputs[:, -args.pred_len:, 0:]
        outputs = outputs.detach().cpu().numpy()
        preds = outputs
        trues = y
        x = x.detach().cpu().numpy()

    print('test shape:', preds.shape)
    folder_path = './m4_results/' + 'TimesNetModel' + '/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    forecasts_df = pandas.DataFrame(
        preds[:, :, (0)], columns=[f'V{i + 1}' for i in range(args.pred_len)])
    forecasts_df.index = valid_dataset.M4ids[:preds.shape[0]]
    forecasts_df.index.name = 'id'
    forecasts_df.set_index(forecasts_df.columns[0], inplace=True)
    forecasts_df.to_csv(folder_path + args.dataset_name + '_forecast.csv')

    file_path = './m4_results/' + 'TimesNetModel' + '/'
    if 'M4Weekly_forecast.csv' in os.listdir(
            file_path) and 'M4Monthly_forecast.csv' in os.listdir(
                file_path) and 'M4Yearly_forecast.csv' in os.listdir(
                    file_path) and 'M4Daily_forecast.csv' in os.listdir(
                        file_path) and 'M4Hourly_forecast.csv' in os.listdir(
                            file_path
                        ) and 'M4Quarterly_forecast.csv' in os.listdir(
                            file_path):
        m4_summary = M4Summary(
            file_path,
            '.', )
        smape_results, owa_results, mape, mase = m4_summary.evaluate()
        print('smape:', smape_results)
        print('mape:', mape)
        print('mase:', mase)
        print('owa:', owa_results)
    else:
        print(
            'After all 6 tasks are finished, you can calculate the averaged index'
        )
    return


# 固定随机随机种子，保证训练结果可复现
seed = 2022
paddle.seed(seed)
np.random.seed(seed)

# for script running
parser = argparse.ArgumentParser(description='TimesNet')
parser.add_argument(
    '--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument(
    '--dataset_name',
    type=str,
    default='ETTh1',
    choices=[
        'ETTh1', 'ETTm1', 'ETTm2', 'ETTh2', 'M4Hourly', 'M4Weekly',
        'M4Monthly', 'M4Daily', 'M4Yearly', 'M4Quarterly'
    ],
    help='The name of hourly dataset')
args = parser.parse_args()

dataset_name = args.dataset_name
seq_len = 96
label_len = 0
d_model = 32
batch_size = 32
enc_in = c_out = 7
drop_last = True
learning_rate = 0.0001
add_transformed_datastamp = True
need_date_in_network = True
loss_fn = F.mse_loss
training = False
eval_metrics = ["mae", "mse"]

exp = f'timesnet_{args.dataset_name}_{args.pred_len}_nopretrain'
dataset_config = dataset_name + 'Dataset'
save_path = 'output/' + exp

if 'M4' in dataset_name:
    ts_train_dataset = get_dataset(dataset_name + 'train')
    ts_test_dataset = None
    ts_val_dataset = get_dataset(dataset_name + 'test')
    args.pred_len = M4Meta.horizons_map[args.dataset_name[
        2:]]  # pred_len is set internally
    seq_len = 2 * args.pred_len
    enc_in = c_out = 1
    label_len = args.pred_len
    frequency_map = M4Meta.frequency_map[args.dataset_name[2:]]
    batch_size = 16
    learning_rate = 0.001
    add_transformed_datastamp = False
    loss_fn = SMAPE().metric_fn
    history_size = M4Meta.history_size[args.dataset_name[2:]]
    window_sampling_limit = int(history_size * args.pred_len)
    drop_last = False
    eval_metrics = ['smape']

elif 'ETTh' in dataset_name:
    ts_data = get_dataset(dataset_name)
    ts_train_dataset, ts_val_dataset, ts_test_dataset = get_dataset(
        dataset_name, {
            'train': [0, 12 * 30 * 24],
            'val': [
                12 * 30 * 24 - seq_len,
                12 * 30 * 24 + 4 * 30 * 24,
            ],
            'test': [
                12 * 30 * 24 + 4 * 30 * 24 - seq_len,
                12 * 30 * 24 + 8 * 30 * 24
            ]
        })
    d_model = 16
    # data transform
    scaler = StandardScaler(ts_data._target._data.columns._data)
    scaler.fit(ts_train_dataset)
    ts_train_dataset = scaler.transform(ts_train_dataset)
    ts_val_dataset = scaler.transform(ts_val_dataset)
    ts_test_dataset = scaler.transform(ts_test_dataset)

elif 'ETTm' in dataset_name:
    ts_data = get_dataset(dataset_name)
    ts_train_dataset, ts_val_dataset, ts_test_dataset = get_dataset(
        dataset_name, {
            'train': [0, 12 * 30 * 24 * 4],
            'val': [
                12 * 30 * 24 * 4 - seq_len,
                12 * 30 * 24 * 4 + 4 * 30 * 24 * 4,
            ],
            'test': [
                12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - seq_len,
                12 * 30 * 24 * 4 + 8 * 30 * 24 * 4
            ]
        })
    # data transform
    scaler = StandardScaler(ts_data._target._data.columns._data)
    scaler.fit(ts_train_dataset)
    ts_train_dataset = scaler.transform(ts_train_dataset)
    ts_val_dataset = scaler.transform(ts_val_dataset)
    ts_test_dataset = scaler.transform(ts_test_dataset)

model = TimesNetModel(task_name='short_term_forecast', in_chunk_len=seq_len, out_chunk_len=args.pred_len, label_len=label_len, \
                    add_transformed_datastamp=add_transformed_datastamp, window_sampling_limit=window_sampling_limit, \
                    enc_in=enc_in, c_out=1, batch_size=batch_size,  d_model=d_model, loss_fn=loss_fn, \
                    lrSched=Type1Decay(learning_rate=learning_rate, last_epoch=0), optimizer_params=dict(learning_rate=learning_rate), \
                    need_date_in_network=need_date_in_network, drop_last=drop_last, eval_metrics=eval_metrics)
if training:
    model.fit(ts_train_dataset, valid_tsdataset=ts_val_dataset)
    if os.path.exists(save_path):
        os.remove(save_path)
        os.remove(save_path + '_model_meta')
        os.remove(save_path + '_network_statedict')
    model.save(save_path)
else:
    # Load model and plot
    model = load(save_path)
    test(args, model, label_len, window_sampling_limit)

# score, preds_data= backtest(data=ts_test_dataset, model=model, return_predicts=True, stride=1)
# print(f'{dataset_name}_{exp}: mse is { sum(score[0].values())/7}; mae is {sum(score[1].values())/7}')
# import pdb; pdb.set_trace()

# ts_test_dataset.plot(columns=['OT'], add_data=preds_data, labels='backtest').get_figure().savefig(f'backtest_{args.dataset_name}_timesnet_OT-new.png')

# export CUDA_VISIBLE_DEVICES=4;python examples/exp.py  --dataset_name M4Monthly
