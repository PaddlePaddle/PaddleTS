import os
import numpy as np
import random
import argparse
import warnings

import paddle
from paddlets.utils.config import Config
from paddlets.models.model_loader import load
from paddlets.datasets.repository import get_dataset
from paddlets.utils.manager import MODELS
from paddlets.logger import Logger

logger = Logger(__name__)
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description='Time Series Forecasting')
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
        '--checkpoints',
        help='model checkpoints for eval.',
        type=str,
        default=None)
    # Runntime params
    parser.add_argument(
        '--predict_len', help='output length in training.', type=int)
    parser.add_argument(
        '--batch_size', help='Mini batch size of one gpu or cpu. ', type=int)
    # Other params
    parser.add_argument(
        '--seed',
        help='Set the random seed in training.',
        default=42,
        type=int)
    parser.add_argument(
        '--opts', help='Update the key-value pairs of all options.', nargs='+')

    return parser.parse_args()


def main(args):
    paddle.set_device(args.device)
    seed = args.seed
    paddle.seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    assert args.config is not None, \
        'No configuration file specified, please set --config'

    assert args.checkpoints is not None, \
        'No checkpoints dictionary specified, please set --checkpoints'

    cfg = Config(
        args.config,
        predict_len=args.predict_len,
        batch_size=args.batch_size,
        opts=args.opts)

    batch_size = cfg.batch_size
    dataset = cfg.dataset
    predict_len = cfg.predict_len
    seq_len = cfg.seq_len
    epoch = cfg.epoch
    split = dataset.get('split', None)
    logger.info(cfg.__dict__)

    ts_val, ts_test = None, None
    if dataset['name'] == 'TSDataset':
        import pandas as pd
        from paddlets import TSDataset
        dataset_root = dataset.get('dataset_root', None)

        if not os.path.exists(dataset_root):
            raise FileNotFoundError('there is not `dataset_root`: {}.'.format(
                dataset_root))

        if cfg.dic.get('info_params', None) is None:
            raise ValueError("`info_params` is necessary, but it is None.")
        else:
            info_params = cfg.dic['info_params']
            if info_params.get('time_col', None) is None:
                raise ValueError("`time_col` is necessary, but it is None.")
            if info_params.get('target_cols', None):
                target_cols = info_params['target_cols'] if info_params[
                    'target_cols'] != [''] else None
                info_params['target_cols'] = target_cols

        if dataset.get('val_path', False):
            if os.path.exists(dataset['val_path']):
                df = pd.read_csv(dataset['val_path'])
                ts_val = TSDataset.load_from_dataframe(df, **info_params)
    else:
        info_params = cfg.dic.get('info_params', None)
        if split:
            _, ts_val, _ = get_dataset(dataset['name'], split, seq_len,
                                       info_params)
        else:
            ts_val = get_dataset(dataset['name'], split, seq_len, info_params)

    weight_path = args.checkpoints
    if 'best_model' in weight_path:
        weight_path = weight_path.split('best_model')[0]

    if cfg.model['name'] == 'PPTimes':
        from paddlets.ensemble.base import EnsembleBase
        model = EnsembleBase.load(weight_path + '/')

    elif cfg.model['name'] == 'XGBoost':
        from paddlets.models.ml_model_wrapper import make_ml_model
        from xgboost import XGBRegressor
        model = make_ml_model(
            in_chunk_len=seq_len,
            out_chunk_len=predict_len,
            sampling_stride=1,
            model_class=XGBRegressor,
            use_skl_gridsearch=False,
            model_init_params=cfg.model['model_cfg'])
        model = model.load(weight_path + '/checkpoints')
    else:
        model = load(weight_path + '/checkpoints')

    if dataset.get('scale', False):
        logger.info('start scaling...')
        if not os.path.exists(os.path.join(weight_path, 'scaler.pkl')):
            raise FileNotFoundError('there is not `scaler`: {}.'.format(
                os.path.join(weight_path, 'scaler.pkl')))
        import joblib
        scaler = joblib.load(os.path.join(weight_path, 'scaler.pkl'))
        ts_val = scaler.transform(ts_val)

    if cfg.dataset.get('time_feat', 'False'):
        logger.info('generate times feature')
        from paddlets.transform import TimeFeatureGenerator
        if dataset.get('use_holiday', False):
            time_feature_generator = TimeFeatureGenerator(feature_cols=[
                'minuteofhour', 'hourofday', 'dayofmonth', 'dayofweek',
                'dayofyear', 'monthofyear', 'weekofyear', 'holidays'
            ])
            if dataset['name'] != 'TSDataset':
                ts_all = get_dataset(dataset['name'])
                ts_all = time_feature_generator.fit_transform(ts_all)
                ts_val._known_cov = ts_all._known_cov[split['val'][0] -
                                                      seq_len:split['val'][1]]
            else:
                if ts_val is not None:
                    ts_val = time_feature_generator.fit_transform(ts_val)

        else:
            time_feature_generator = TimeFeatureGenerator(feature_cols=[
                'hourofday', 'dayofmonth', 'dayofweek', 'dayofyear'
            ])
            ts_val = time_feature_generator.fit_transform(ts_val)

    logger.info('start evalution...')
    metric = model.eval(ts_val)
    logger.info(metric)


if __name__ == '__main__':
    args = parse_args()
    main(args)
