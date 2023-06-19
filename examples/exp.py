from itertools import count
import os
import shutil
import pandas
import warnings
import argparse
import numpy as np
from paddlets.metrics.metrics import MAE
from sklearn.metrics import precision_recall_fscore_support

import paddle
import paddle.nn.functional as F

from paddlets.datasets import UnivariateDataset
from paddlets.datasets.repository import get_dataset
from paddlets.models.model_loader import load
from paddlets.models.data_adapter import DataAdapter
from paddlets.models.forecasting import TimesNetModel
from paddlets.models.forecasting.utils import M4Meta, adjustment, M4Summary
from paddlets.metrics import SMAPE, F1, Precision, Recall, ACC, MSE, MAPE
from paddlets.transform.sklearn_transforms import StandardScaler
from paddlets.utils import backtest
from paddlets.utils import Type1Decay

warnings.filterwarnings('ignore')


def short_term_test(args, model, label_len, window_sampling_limit,
                    dataset_name, seq_len):
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


def anomaly_test(args, model, dataset_name, seq_len, batch_size,
                 anomaly_ratio):
    ts_train_dataset = get_dataset(dataset_name + 'train')  # csv
    ts_test_dataset = get_dataset(dataset_name + 'test')

    scaler = StandardScaler(ts_train_dataset.observed_cov._data.columns._data)
    scaler.fit(ts_train_dataset)
    ts_train_dataset = scaler.transform(ts_train_dataset)
    ts_test_dataset = scaler.transform(ts_test_dataset)

    data_adapter = DataAdapter()
    train_dataset = data_adapter.to_sample_dataset(
        ts_train_dataset,
        in_chunk_len=seq_len,
        out_chunk_len=args.pred_len,
        skip_chunk_len=0,
        add_transformed_datastamp=False,
        sampling_stride=1)
    train_loader = data_adapter.to_paddle_dataloader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    test_dataset = data_adapter.to_sample_dataset(
        ts_test_dataset,
        in_chunk_len=seq_len,
        out_chunk_len=args.pred_len,
        skip_chunk_len=0,
        add_transformed_datastamp=False,
        sampling_stride=1)

    test_loader = data_adapter.to_paddle_dataloader(
        test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    attens_energy = []
    folder_path = './test_results/' + 'TimesNetModel' + '/' + dataset_name
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    model._network.eval()
    anomaly_criterion = paddle.nn.functional.mse_loss
    with paddle.no_grad():
        for i, data in enumerate(train_loader):
            batch_x = data['observed_cov_numeric'].astype(dtype='float32')
            outputs = model._network(batch_x)
            score = paddle.mean(
                x=anomaly_criterion(
                    batch_x, outputs, reduction='none'),
                axis=-1)  # 误差mse的平均数
            score = score.detach().cpu().numpy()
            attens_energy.append(score)
    attens_energy = np.concatenate(
        attens_energy, axis=0).reshape(-1)  # 每个batch误差值组成的一维数组
    train_energy = np.array(attens_energy)

    attens_energy = []
    test_labels = []
    for i, data in enumerate(test_loader):  # past_target is the label
        batch_x, batch_y = data['observed_cov_numeric'], data['past_target']
        batch_x = batch_x.astype(dtype='float32')
        outputs = model._network(batch_x)
        score = paddle.mean(
            x=anomaly_criterion(
                batch_x, outputs, reduction='none'), axis=-1)
        score = score.detach().cpu().numpy()
        attens_energy.append(score)
        test_labels.append(batch_y)
    attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
    test_energy = np.array(attens_energy)
    combined_energy = np.concatenate(
        [train_energy, test_energy], axis=0)  # 训练集和测试集上的所有异常误差联结

    threshold = np.percentile(combined_energy,
                              100 - anomaly_ratio)  # 找到幅度位于99%位置的数值
    print('Threshold :', threshold)
    pred = (test_energy > threshold).astype(int)
    test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
    test_labels = np.array(test_labels)
    gt = test_labels.astype(int)
    print('pred:   ', pred.shape)
    print('gt:     ', gt.shape)
    gt, pred = adjustment(gt, pred)
    pred = np.array(pred)
    gt = np.array(gt)
    print('pred: ', pred.shape)
    print('gt:   ', gt.shape)
    accuracy = ACC().metric_fn(gt, pred)
    # precision, recall, f_score = Precision().metric_fn(gt, pred), Recall().metric_fn(gt, pred), F1().metric_fn(gt, pred)
    precision, recall, f_score, support = precision_recall_fscore_support(
        gt, pred, average='binary')
    print(
        'Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} '
        .format(accuracy, precision, recall, f_score))
    f = open('result_anomaly_detection.txt', 'a')
    f.write(
        'Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} '.
        format(accuracy, precision, recall, f_score))
    f.write('\n')
    f.write('\n')
    f.close()
    return


def imputation_test(ts_test_dataset, args, seq_len, batch_size, dataset_name,
                    exp):
    data_adapter = DataAdapter()

    test_dataset = data_adapter.to_sample_dataset(
        ts_test_dataset,
        in_chunk_len=seq_len,
        out_chunk_len=args.pred_len,
        skip_chunk_len=0,
        add_transformed_datastamp=True,
        sampling_stride=1)

    test_loader = data_adapter.to_paddle_dataloader(
        test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    preds = []
    trues = []
    masks = []
    folder_path = './test_results/' + dataset_name + '/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    model._network.eval()
    with paddle.no_grad():
        for i, data in enumerate(test_loader):
            batch_x = data['past_target'].astype(dtype='float32')
            batch_x_mark = data['past_transformed_datastamp'].astype(
                dtype='float32')

            B, T, N = batch_x.shape

            mask = paddle.rand(shape=(B, T, N))
            mask[mask <= args.mask_rate] = 0
            mask[mask > args.mask_rate] = 1
            inp = paddle.where(
                mask == 0,
                paddle.zeros(
                    shape=batch_x.shape, dtype='float32'),
                batch_x)

            outputs = model._network(inp, batch_x_mark, mask)
            outputs = outputs.detach().cpu().numpy()
            pred = outputs
            true = batch_x.detach().cpu().numpy()
            preds.append(pred)
            trues.append(true)
            masks.append(mask.detach().cpu())
    preds = np.concatenate(preds, 0)
    trues = np.concatenate(trues, 0)
    masks = np.concatenate(masks, 0)
    print('test shape:', preds.shape, trues.shape)
    folder_path = './results/' + dataset_name + '/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    mae, mse = MAE().metric_fn(preds[masks == 0],
                               trues[masks == 0]), MSE().metric_fn(
                                   preds[masks == 0], trues[masks == 0])
    print(f'{dataset_name}_{exp}_{args.mask_rate}: mse:{mse}, mae:{mae}')
    f = open('result_imputation.txt', 'a')
    f.write(dataset_name + '  \n')
    f.write(f'{dataset_name}_{exp}_{args.mask_rate}: ' +
            'mse:{}, mae:{}'.format(mse, mae))
    f.write('\n')
    f.write('\n')
    f.close()
    np.save(folder_path + 'metrics.npy', np.array([mae, mse]))
    np.save(folder_path + 'pred.npy', preds)
    np.save(folder_path + 'true.npy', trues)
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
    '--training', type=int, default=1, help='Whether include training process')
parser.add_argument(
    '--mask_rate',
    type=float,
    default=None,
    help='The mask ration in imputation.')
parser.add_argument(
    '--task_name',
    type=str,
    default=None,
    choices=[
        'long_term_forecast', 'anomaly_detection', 'short_term_forecast',
        'imputation', 'classification'
    ],
    help='The name of task')
parser.add_argument(
    '--dataset_name',
    type=str,
    default='ETTh1',
    choices=[
        'ETTh1', 'ETTm1', 'ETTm2', 'ETTh2', 'Weather', 'Exchange', 'ILI',
        'ECL', 'Traffic', 'M4Hourly', 'M4Weekly', 'M4Monthly', 'M4Daily',
        'M4Yearly', 'M4Quarterly', 'SMD', 'MSL', 'SMAP', 'SWAT', 'PSM',
        'EthanolConcentration', 'FaceDetection', 'Handwriting', 'Heartbeat',
        'JapaneseVowels', 'PEMS-SF', 'SelfRegulationSCP1',
        'SelfRegulationSCP2', 'SpokenArabicDigits', 'UWaveGestureLibrary'
    ],
    help='The name of hourly dataset')
args = parser.parse_args()

dataset_name = args.dataset_name
c_out = 7
seq_len = 96
label_len = 0
d_ff = d_model = 32
enc_in = c_out = 7
e_layers = 2
top_k = 5
batch_size = 32
max_epochs = 10
drop_last = True
learning_rate = 0.0001
add_transformed_datastamp = True
need_date_in_network = True
loss_fn = F.mse_loss
training = args.training
eval_metrics = ["mae", "mse"]
window_sampling_limit = None

# short time forecast
if 'M4' in dataset_name:
    ts_train_dataset = get_dataset(dataset_name + 'train')
    ts_test_dataset = None
    ts_val_dataset = get_dataset(dataset_name + 'test')
    args.pred_len = M4Meta.horizons_map[args.dataset_name[
        2:]]  # pred_len is set internally
    seq_len = 2 * args.pred_len
    drop_last = False
    need_date_in_network = False
    enc_in = c_out = 1
    label_len = args.pred_len
    frequency_map = M4Meta.frequency_map[args.dataset_name[2:]]
    batch_size = 16
    learning_rate = 0.001
    add_transformed_datastamp = False
    loss_fn = SMAPE().metric_fn
    history_size = M4Meta.history_size[args.dataset_name[2:]]
    window_sampling_limit = int(history_size * args.pred_len)
    eval_metrics = ['smape']
    # task_name='short_term_forecast'

elif 'SMD' in dataset_name:
    enc_in = c_out = 38
    d_model = d_ff = 64
    anomaly_ratio = 0.5
    batch_size = 128
    seq_len = 100
    drop_last = False
    args.pred_len = 0
    # task_name = 'anomaly_detection'
    need_date_in_network = False
    add_transformed_datastamp = False

    ts_train_dataset = get_dataset(dataset_name + 'train')  # csv
    ts_test_dataset = get_dataset(dataset_name + 'test')
    data_len = ts_train_dataset.observed_cov.to_numpy().shape[0]

    scaler = StandardScaler(ts_train_dataset.observed_cov._data.columns._data)
    scaler.fit(ts_train_dataset)
    ts_train_dataset = scaler.transform(ts_train_dataset)
    ts_test_dataset = scaler.transform(ts_test_dataset)
    ts_val_dataset = ts_train_dataset.split(int(0.8 * data_len))[1]
    # print(np.abs(ts_test_dataset._observed_cov._data.to_numpy()).mean()) # 0.6506949867979888
    # print(np.abs(ts_train_dataset._observed_cov._data.to_numpy()).mean()) #  0.6069401903755253
elif 'SMAP' in dataset_name:
    enc_in = c_out = 25
    d_model = d_ff = 128
    anomaly_ratio = 1
    batch_size = 128
    seq_len = 100
    max_epochs = 3
    drop_last = False
    args.pred_len = 0
    need_date_in_network = False
    add_transformed_datastamp = False
    e_layers = 3
    top_k = 3

    ts_train_dataset = get_dataset(dataset_name + 'train')  # csv
    ts_test_dataset = get_dataset(dataset_name + 'test')

    scaler = StandardScaler(ts_train_dataset.observed_cov._data.columns._data)
    scaler.fit(ts_train_dataset)
    ts_train_dataset = scaler.transform(ts_train_dataset)
    ts_test_dataset = scaler.transform(ts_test_dataset)
    ts_val_dataset = ts_test_dataset

elif 'MSL' in dataset_name:
    enc_in = c_out = 55
    d_model = 8
    d_ff = 16
    anomaly_ratio = 1
    batch_size = 128
    seq_len = 100
    max_epochs = 1
    drop_last = False
    args.pred_len = 0
    # task_name = 'anomaly_detection'
    need_date_in_network = False
    add_transformed_datastamp = False
    e_layers = 1
    top_k = 3

    ts_train_dataset = get_dataset(dataset_name + 'train')  # csv
    ts_test_dataset = get_dataset(dataset_name + 'test')

    scaler = StandardScaler(ts_train_dataset.observed_cov._data.columns._data)
    scaler.fit(ts_train_dataset)
    ts_train_dataset = scaler.transform(ts_train_dataset)
    ts_test_dataset = scaler.transform(ts_test_dataset)
    ts_val_dataset = ts_test_dataset

elif 'PSM' in dataset_name:
    enc_in = c_out = 25
    d_model = 64
    d_ff = 64
    anomaly_ratio = 1
    batch_size = 128
    seq_len = 100
    max_epochs = 3
    drop_last = False
    args.pred_len = 0
    # task_name = 'anomaly_detection'
    need_date_in_network = False
    add_transformed_datastamp = False
    e_layers = 2
    top_k = 3

    ts_train_dataset = get_dataset(dataset_name + 'train')  # csv
    ts_test_dataset = get_dataset(dataset_name + 'test')

    scaler = StandardScaler(ts_train_dataset.observed_cov._data.columns._data)
    scaler.fit(ts_train_dataset)
    ts_train_dataset = scaler.transform(ts_train_dataset)
    ts_test_dataset = scaler.transform(ts_test_dataset)
    ts_val_dataset = ts_test_dataset

elif 'SWAT' in dataset_name:
    enc_in = c_out = 51
    d_model = d_ff = 64

    anomaly_ratio = 1
    batch_size = 128
    seq_len = 100
    max_epochs = 3
    drop_last = False
    args.pred_len = 0
    # task_name = 'anomaly_detection'
    need_date_in_network = False
    add_transformed_datastamp = False
    e_layers = 3
    top_k = 3

    ts_train_dataset = get_dataset(dataset_name + 'train')  # csv
    ts_test_dataset = get_dataset(dataset_name + 'test')

    scaler = StandardScaler(ts_train_dataset.observed_cov._data.columns._data)
    scaler.fit(ts_train_dataset)
    ts_train_dataset = scaler.transform(ts_train_dataset)
    ts_test_dataset = scaler.transform(ts_test_dataset)
    ts_val_dataset = ts_test_dataset

# long term forecast
elif 'Traffic' in dataset_name:
    enc_in = c_out = 862
    d_ff = d_model = 512

    ts_data = get_dataset(dataset_name)
    dataset_length = len(ts_data._target)
    num_train = int(dataset_length * 0.7)
    num_test = int(dataset_length * 0.2)
    num_vali = dataset_length - num_train - num_test
    ts_train_dataset, ts_val_dataset, ts_test_dataset = get_dataset(
        dataset_name, {
            'train': [0, num_train],
            'val': [
                num_train - seq_len,
                num_train + num_vali,
            ],
            'test': [dataset_length - num_test - seq_len, dataset_length]
        })
    scaler = StandardScaler(ts_data._target._data.columns._data)
    scaler.fit(ts_train_dataset)
    ts_train_dataset = scaler.transform(ts_train_dataset)
    ts_val_dataset = scaler.transform(ts_val_dataset)
    ts_test_dataset = scaler.transform(ts_test_dataset)

elif 'Weather' in dataset_name:
    enc_in = c_out = 21
    d_ff = d_model = 32

    if args.task_name == 'imputation':
        top_k = 3
        learning_rate = 0.001
        d_ff = 64
        d_model = 64
        batch_size = 16
        c_out = enc_in = 21
        e_layers = 2
        args.pred_len = 0

    ts_data = get_dataset(dataset_name)
    dataset_length = len(ts_data._target)
    num_train = int(dataset_length * 0.7)  # 
    num_test = int(dataset_length * 0.2)
    num_vali = dataset_length - num_train - num_test
    ts_train_dataset, ts_val_dataset, ts_test_dataset = get_dataset(
        dataset_name, {
            'train': [0, num_train],
            'val': [
                num_train - seq_len,
                num_train + num_vali,
            ],
            'test': [dataset_length - num_test - seq_len, dataset_length]
        })
    scaler = StandardScaler(ts_data._target._data.columns._data)
    scaler.fit(ts_train_dataset)
    ts_train_dataset = scaler.transform(ts_train_dataset)
    ts_val_dataset = scaler.transform(ts_val_dataset)
    ts_test_dataset = scaler.transform(ts_test_dataset)

elif 'ILI' in dataset_name:
    enc_in = c_out = 7
    d_ff = d_model = 768
    args.pred_len = 24
    seq_len = 36

    ts_data = get_dataset(dataset_name)
    dataset_length = len(ts_data._target)
    num_train = int(dataset_length * 0.7)  # 
    num_test = int(dataset_length * 0.2)
    num_vali = dataset_length - num_train - num_test
    ts_train_dataset, ts_val_dataset, ts_test_dataset = get_dataset(
        dataset_name, {
            'train': [0, num_train],
            'val': [
                num_train - seq_len,
                num_train + num_vali,
            ],
            'test': [dataset_length - num_test - seq_len, dataset_length]
        })
    scaler = StandardScaler(ts_data._target._data.columns._data)
    scaler.fit(ts_train_dataset)
    ts_train_dataset = scaler.transform(ts_train_dataset)
    ts_val_dataset = scaler.transform(ts_val_dataset)
    ts_test_dataset = scaler.transform(ts_test_dataset)

elif 'ECL' in dataset_name:
    enc_in = c_out = 321
    d_ff = 512
    d_model = 256
    seq_len = 36
    top_k = 5
    e_layers = 2

    if args.task_name == 'imputation':
        top_k = 3
        learning_rate = 0.001
        d_ff = 64
        d_model = 64
        batch_size = 16
        c_out = enc_in = 321
        e_layers = 2
        args.pred_len = 0

    ts_data = get_dataset(dataset_name)
    dataset_length = len(ts_data._target)
    num_train = int(dataset_length * 0.7)  # 
    num_test = int(dataset_length * 0.2)
    num_vali = dataset_length - num_train - num_test
    ts_train_dataset, ts_val_dataset, ts_test_dataset = get_dataset(
        dataset_name, {
            'train': [0, num_train],
            'val': [
                num_train - seq_len,
                num_train + num_vali,
            ],
            'test': [dataset_length - num_test - seq_len, dataset_length]
        })
    scaler = StandardScaler(ts_data._target._data.columns._data)
    scaler.fit(ts_train_dataset)
    ts_train_dataset = scaler.transform(ts_train_dataset)
    ts_val_dataset = scaler.transform(ts_val_dataset)
    ts_test_dataset = scaler.transform(ts_test_dataset)

elif 'ETTh' in dataset_name:
    if args.task_name == 'imputation':
        top_k = 3
        learning_rate = 0.001
        d_ff = 32
        d_model = 16
        batch_size = 16
        c_out = enc_in = 7
        e_layers = 2
        args.pred_len = 0

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
    if args.task_name == 'imputation':
        top_k = 3
        learning_rate = 0.001
        d_ff = 64
        d_model = 64
        batch_size = 16
        c_out = enc_in = 7
        e_layers = 2
        args.pred_len = 0

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

model = TimesNetModel(task_name=args.task_name, in_chunk_len=seq_len, out_chunk_len=args.pred_len, label_len=label_len, \
                    add_transformed_datastamp=add_transformed_datastamp, window_sampling_limit=window_sampling_limit, \
                    enc_in=enc_in, c_out=c_out, batch_size=batch_size,  d_model=d_model, d_ff=d_ff, loss_fn=loss_fn, \
                    lrSched=Type1Decay(learning_rate=learning_rate, last_epoch=0), optimizer_params=dict(learning_rate=learning_rate), \
                    need_date_in_network=need_date_in_network, drop_last=drop_last, eval_metrics=eval_metrics,
                    max_epochs=max_epochs, e_layers=e_layers, top_k=top_k, mask_rate=args.mask_rate)

exp = f'timesnet_{args.dataset_name}_{args.pred_len}_nopretrain'
save_path = 'output/' + exp

if training:
    model.fit(ts_train_dataset, valid_tsdataset=ts_val_dataset)
    if os.path.exists(save_path):
        os.remove(save_path)
        os.remove(save_path + '_model_meta')
        os.remove(save_path + '_network_statedict')
    model.save(save_path)

    if args.task_name == 'short_term_forecast':
        short_term_test(args, model, label_len, window_sampling_limit,
                        dataset_name, seq_len)
    elif args.task_name == "long_term_forecast":
        score, preds_data = backtest(
            data=ts_test_dataset, model=model, return_predicts=True, stride=1)
        print(
            f'{dataset_name}_{exp}: mse is { sum(score[0].values())/enc_in}; mae is {sum(score[1].values())/enc_in}'
        )
        # ts_test_dataset.plot(columns=['OT'], add_data=preds_data, labels='backtest').get_figure().savefig(f'backtest_{args.dataset_name}_timesnet_OT-new.png')
    elif args.task_name == 'anomaly_detection':
        anomaly_test(args, model, dataset_name, seq_len, batch_size,
                     anomaly_ratio)
    elif args.task_name == "imputation":
        imputation_test(ts_test_dataset, args, seq_len, batch_size,
                        dataset_name, exp)
else:
    # Load model and plot
    model = load(save_path)
    if args.task_name == 'short_term_forecast':
        short_term_test(args, model, label_len, window_sampling_limit,
                        dataset_name, seq_len)
    elif args.task_name == "long_term_forecast":
        score, preds_data = backtest(
            data=ts_test_dataset, model=model, return_predicts=True, stride=1)
        print(
            f'{dataset_name}_{exp}: mse is { sum(score[0].values())/enc_in}; mae is {sum(score[1].values())/enc_in}'
        )
    elif args.task_name == 'anomaly_detection':
        anomaly_test(args, model, dataset_name, seq_len, batch_size,
                     anomaly_ratio)
    elif args.task_name == "imputation":
        imputation_test(ts_test_dataset, args, seq_len, batch_size,
                        dataset_name, exp)
