from itertools import count
import os
import pandas
import numpy as np
from paddlets.metrics.metrics import MAE
from sklearn.metrics import precision_recall_fscore_support

import paddle

from paddlets.datasets import UnivariateDataset
from paddlets.datasets.repository import get_dataset
from paddlets.models.data_adapter import DataAdapter
from paddlets.models.forecasting.M4_utils import adjustment, M4Summary
from paddlets.metrics import ACC, MSE
from paddlets.transform.sklearn_transforms import StandardScaler


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


def imputation_test(args, ts_test_dataset, seq_len, batch_size, dataset_name,
                    exp, model):
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


def classification_test(dataset_name, ts_test_dataset, train_tsdataset, model):
    _, test_loader = model._init_fit_dataloaders(train_tsdataset,
                                                 ts_test_dataset)

    preds = []
    trues = []
    folder_path = './test_results/' + dataset_name + '/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    model._network.eval()
    with paddle.no_grad():
        for i, (batch_x, label, padding_mask) in enumerate(test_loader[0]):
            batch_x = batch_x.cast('float32')
            padding_mask = padding_mask.cast('float32')

            outputs = model._network(batch_x, padding_mask)

            preds.append(outputs.detach())
            trues.append(label)

    preds = paddle.concat(preds, 0)
    trues = paddle.concat(trues, 0)
    print('test shape:', preds.shape, trues.shape)

    probs = paddle.nn.functional.softmax(
        preds
    )  # (total_samples, num_classes) est. prob. for each class and sample
    predictions = paddle.argmax(
        probs, axis=1).cpu(
        ).numpy()  # (total_samples,) int class index for each sample
    trues = trues.flatten().cpu().numpy()
    accuracy = ACC().metric_fn(predictions, trues)

    # result save
    folder_path = './results/' + dataset_name + '/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    print('accuracy:{}'.format(accuracy))
    f = open("result_classification.txt", 'a')
    f.write(dataset_name + "  \n")
    f.write('accuracy:{}'.format(accuracy))
    f.write('\n')
    f.write('\n')
    f.close()
    return
