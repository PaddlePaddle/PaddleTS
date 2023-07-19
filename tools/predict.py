import os
import numpy as np
import random
import argparse
import warnings

import paddle
import pandas as pd
from paddlets import TSDataset

from paddlets.utils.config import Config
from paddlets.models.model_loader import load
from paddlets.datasets.repository import get_dataset
from paddlets.utils.manager import MODELS
from paddlets.utils import backtest
from paddlets.logger import Logger

logger = Logger(__name__)
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description='Time Series Forecasting')
    # Common params
    parser.add_argument("--config", help="The path of config file.", type=str)
    parser.add_argument(
        '--csv_path', help='input test csv format file in predict.', type=str)
    parser.add_argument(
        '--device',
        help='Set the device place for training model.',
        default='gpu',
        choices=['cpu', 'gpu'],
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
    assert args.csv_path is not None, \
        'No predicted file specified, please set --csv_path'
    assert args.checkpoints is not None, \
        'No checkpoints dictionary specified, please set --checkpoints'

    cfg = Config(args.config, opts=args.opts)

    dataset = cfg.dataset
    split = dataset.get('split', None)
    logger.info(cfg.__dict__)

    if cfg.dic.get('info_params', None) is None:
        raise ValueError("`info_params` is necessary, but it is None.")
    else:
        info_params = cfg.dic['info_params']
        if info_params.get('time_col', None) is None:
            raise ValueError("`time_col` is necessary, but it is None.")

    df = pd.read_csv(args.csv_path)
    ts_test = TSDataset.load_from_dataframe(df, **info_params)

    weight_path = args.checkpoints
    if 'best_model' in weight_path:
        weight_path = weight_path.split('best_model')[0]

    if cfg.model['name'] == 'PPTimes':
        from paddlets.ensemble import WeightingEnsembleForecaster
        estimators = []
        for model_name, model_cfg in cfg.model['model_cfg'].items():
            model_cfg = Config(
                model_cfg,
                seq_len=cfg.seq_len,
                predict_len=cfg.predict_len,
                batch_size=cfg.batch_size,
                opts=args.opts)
            logger.info(model_cfg.model)
            one_model = MODELS.components_dict[model_name]
            params = model_cfg.model['model_cfg']
            params['in_chunk_len'] = cfg.seq_len
            params['out_chunk_len'] = cfg.predict_len
            params['batch_size'] = cfg.batch_size
            params['max_epochs'] = cfg.epoch

            estimators.append((one_model, params))

        model = WeightingEnsembleForecaster(
            in_chunk_len=cfg.seq_len,
            out_chunk_len=cfg.predict_len,
            skip_chunk_len=0,
            estimators=estimators,
            mode='mean')
        model = model.load(weight_path + '/')
    else:
        model = load(weight_path + 'checkpoints')

    if dataset.get('scale', False):
        logger.info('start scaling...')
        if not os.path.exists(os.path.join(weight_path, 'scaler.pkl')):
            raise FileNotFoundError('there is not `scaler`: {}.'.format(
                os.path.join(weight_path, 'scaler.pkl')))
        import joblib
        scaler = joblib.load(weight_path + 'scaler.pkl')
        ts_test = scaler.transform(ts_test)

    if cfg.dataset.get('time_feat', 'False'):
        logger.info('generate times feature')
        from paddlets.transform import TimeFeatureGenerator
        if dataset.get('use_holiday', False):
            time_feature_generator = TimeFeatureGenerator(
                feature_cols=[
                    'minuteofhour', 'hourofday', 'dayofmonth', 'dayofweek',
                    'dayofyear', 'monthofyear', 'weekofyear', 'holidays'
                ],
                extend_points=model._out_chunk_len + 1)
            if dataset['name'] != 'Dataset':
                ts_all = get_dataset(dataset['name'])
                ts_all = time_feature_generator.fit_transform(ts_all)
                ts_test._known_cov = ts_all._known_cov[split['val'][0]:split[
                    'val'][1]]
            else:
                if ts_test is not None:
                    ts_test = time_feature_generator.fit_transform(ts_test)

        else:
            time_feature_generator = TimeFeatureGenerator(
                feature_cols=[
                    'hourofday', 'dayofmonth', 'dayofweek', 'dayofyear'
                ],
                extend_points=model._out_chunk_len + 1)
            ts_test = time_feature_generator.fit_transform(ts_test)

    logger.info('start to predit...')
    result = model.predict(ts_test)

    if dataset.get('scale', 'False'):
        result = scaler.inverse_transform(result)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    result.to_dataframe().to_csv(os.path.join(args.save_dir, 'result.csv'))
    logger.info('save result to {}'.format(
        os.path.join(args.save_dir, 'result.csv')))


if __name__ == '__main__':
    args = parse_args()
    main(args)
