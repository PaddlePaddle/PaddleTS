import os
import warnings
warnings.filterwarnings("ignore")
import argparse
import numpy as np

import paddle
import paddle.nn.functional as F
import paddle.nn as nn

from paddlets.utils.config import Config
from paddlets.utils.manager import MODELS
from paddlets.utils import backtest
from paddlets.datasets.repository import get_dataset
from paddlets.models.model_loader import load
from paddlets.models.forecasting.M4_utils import M4Meta, adjustment, M4Summary
from paddlets.metrics import SMAPE, F1, Precision, Recall, ACC, MSE, MAPE, MAE
from paddlets.transform.sklearn_transforms import StandardScaler
from paddlets.utils import backtest
from paddlets.utils import Type1Decay
from paddlets.utils import short_term_test, anomaly_test, imputation_test, classification_test


def parse_args():
    parser = argparse.ArgumentParser(description='Model training')
    # Common params
    parser.add_argument("--config", help="The path of config file.", type=str)
    parser.add_argument(
        '--device',
        help='Set the device place for training model.',
        default='gpu',
        choices=['cpu', 'gpu', 'xpu', 'npu', 'mlu'],
        type=str)
    parser.add_argument(
        '--save_dir',
        help='The directory for saving the model snapshot.',
        type=str,
        default='./output/')
    parser.add_argument(
        '--do_eval',
        help='Whether to do evaluation after training.',
        action='store_true')

    # Runntime params
    parser.add_argument('--epoch', help='Iterations in training.', type=int)
    parser.add_argument(
        '--batch_size', help='Mini batch size of one gpu or cpu. ', type=int)
    parser.add_argument('--learning_rate', help='Learning rate.', type=float)

    # Other params
    parser.add_argument(
        '--seed',
        help='Set the random seed in training.',
        default=2022,
        type=int)
    parser.add_argument(
        "--precision",
        default="fp32",
        type=str,
        choices=["fp32", "fp16"],
        help="Use AMP (Auto mixed precision) if precision='fp16'. If precision='fp32', the training is normal."
    )
    parser.add_argument(
        "--amp_level",
        default="O1",
        type=str,
        choices=["O1", "O2"],
        help="Auto mixed precision level. Accepted values are “O1” and “O2”: O1 represent mixed precision, the input \
                data type of each operator will be casted by white_list and black_list; O2 represent Pure fp16, all operators \
                parameters and input data will be casted to fp16, except operators in black_list, don’t support fp16 kernel \
                and batchnorm. Default is O1(amp).")
    parser.add_argument(
        '--training',
        type=bool,
        default=1,
        help='Whether include training process')
    parser.add_argument(
        '--mask_rate', type=float, help='The mask ration in imputation.')

    return parser.parse_args()


def main(args):
    assert args.config is not None, \
        'No configuration file specified, please set --config'
    paddle.seed(args.seed)
    np.random.seed(args.seed)

    cfg = Config(
        args.config,
        learning_rate=args.learning_rate,
        epoch=args.epoch,
        batch_size=args.batch_size,
        opts={"mask_rate": args.mask_rate,
              "training": args.training})

    batch_size = cfg.batch_size
    dataset = cfg.dataset
    predict_len = cfg.predict_len
    seq_len = cfg.seq_len
    label_len = cfg.dic.get("label_len")
    enc_in = cfg.model['model_cfg'].get('enc_in', None)
    window_sampling_limit = cfg.model['model_cfg'].get('window_sampling_limit',
                                                       None)
    epoch = cfg.epoch
    anomaly_ratio = cfg.dic.get("anomaly_ratio", None)
    split = dataset.get('split', None)
    do_eval = cfg.dic.get('do_eval', True)
    task_name = cfg.model.get('model_cfg').get('task_name', None)
    eval_metrics = cfg.dic.get('eval_metrics', None)
    optimizer_params = dict(
        learning_rate=cfg.dic['lr_scheduler']['lr_cfg']['learning_rate'])
    lrSched = eval(cfg.dic['lr_scheduler']['name'])(
        **cfg.dic.get('lr_scheduler').get('lr_cfg'))
    mask_rate = cfg.dic.get("mask_rate")

    if eval_metrics is None:
        eval_metrics = ["mae", "mse"]
        Warning(
            "The eval_metrics is not set in the config file, set to [\"mae\", \"mse\"] by default"
        )

    if task_name == 'anomaly_detection':
        assert anomaly_ratio is not None, 'Anomaly ratio is None but should be set when the task is anomaly_detection.'

    assert task_name is not None, "task_name is needed to be set in model.model_cfg.task_name"

    # loss 
    if cfg.dic.get('loss', None) is None or cfg.dic.get('loss', None) == 'mse':
        loss = F.mse_loss
    elif cfg.dic.get('loss', None) == 'smape':
        loss = SMAPE().metric_fn
    elif cfg.dic.get('loss', None) == 'CE':
        loss = nn.CrossEntropyLoss()

    setting = dataset['name'] + '_' + str(seq_len) + '_' + str(
        predict_len) + '/'
    print(setting)
    save_path = args.save_dir + setting

    if split:
        ts_train, ts_val, ts_test = get_dataset(dataset['name'], split,
                                                seq_len)
    else:
        if 'M4' in dataset['name']:
            ts_train = get_dataset(dataset['name'] + 'train', split, seq_len)
            ts_val = get_dataset(dataset['name'] + 'test', split, seq_len)
            ts_test = None
        elif task_name == 'anomaly_detection' or task_name == 'classification':
            ts_train = get_dataset(dataset['name'] + 'train', split, seq_len)
            ts_test = get_dataset(dataset['name'] + 'test', split, seq_len)
            ts_val = ts_test
            if 'SMD' in dataset['name']:
                ts_val = ts_train.split(
                    int(0.8 * ts_train.observed_cov.to_numpy().shape[0]))[1]
        else:
            assert do_eval == False, 'if not split test data, please set do_eval False'
            ts_train = get_dataset(dataset['name'], split, seq_len)

    model = MODELS.components_dict[cfg.model['name']](
        in_chunk_len=seq_len,
        out_chunk_len=cfg.predict_len,
        label_len=label_len,
        batch_size=batch_size,
        max_epochs=epoch,
        loss_fn=loss,
        lrSched=lrSched,
        optimizer_params=optimizer_params,
        eval_metrics=eval_metrics,
        mask_rate=mask_rate,
        **cfg.model['model_cfg'])

    if not (task_name == 'classification' or
            task_name == 'short_term_forecast'):
        scaler = StandardScaler()
        scaler.fit(ts_train)
        ts_train = scaler.transform(ts_train)
        ts_val = scaler.transform(ts_val)

        if ts_test is not None:
            ts_test = scaler.transform(ts_test)

    if args.training:
        if do_eval:
            model.fit(ts_train, ts_val)
        else:
            model.fit(ts_train)

        if os.path.exists(save_path):
            os.remove(save_path)
            os.remove(save_path + '_model_meta')
            os.remove(save_path + '_network_statedict')
        model.save(save_path)

        if do_eval:
            if task_name == 'short_term_forecast':
                short_term_test(args, model, label_len, window_sampling_limit,
                                dataset['name'], seq_len)
            elif task_name == "long_term_forecast":
                score = backtest(
                    data=ts_test,
                    model=model,
                    predict_window=predict_len,
                    stride=1)
                print(
                    f'{setting}: mse is { sum(score[0].values())/enc_in}; mae is {sum(score[1].values())/enc_in}'
                )
            elif task_name == 'anomaly_detection':
                anomaly_test(args, model, dataset['name'], seq_len, batch_size,
                             anomaly_ratio)
            elif task_name == "imputation":
                imputation_test(ts_test, args, seq_len, batch_size,
                                dataset['name'], setting, model)
            elif task_name == 'classification':
                classification_test(dataset['name'], ts_test, ts_train, model)

    else:
        model = load(save_path)

        if task_name == 'short_term_forecast':
            short_term_test(args, model, label_len, window_sampling_limit,
                            dataset['name'], seq_len)
        elif task_name == "long_term_forecast":
            score = backtest(
                data=ts_test,
                model=model,
                predict_window=predict_len,
                stride=1)
            print(
                f'{setting}: mse is { sum(score[0].values())/enc_in}; mae is {sum(score[1].values())/enc_in}'
            )
        elif task_name == 'anomaly_detection':
            anomaly_test(args, model, dataset['name'], seq_len, batch_size,
                         anomaly_ratio)
        elif task_name == "imputation":
            imputation_test(ts_test, args, seq_len, batch_size,
                            dataset['name'], setting, model)
        elif task_name == 'classification':
            classification_test(dataset['name'], ts_test, ts_train, model)


if __name__ == '__main__':
    args = parse_args()
    main(args)
