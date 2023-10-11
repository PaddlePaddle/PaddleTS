import os
import paddlets
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
        choices=['cpu', 'gpu', 'xpu'],
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
        '--seed', help='Set the random seed in training.', default=42, type=int)
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
        if cfg.task == 'longforecast' and info_params.get('time_col',
                                                          None) is None:
            raise ValueError("`time_col` is necessary, but it is None.")
        if info_params.get('target_cols', None):
            if isinstance(info_params['target_cols'], str):
                info_params['target_cols'] = info_params['target_cols'].split(',')
        if info_params.get('static_cov_cols', None):
            info_params['static_cov_cols'] = None

    df = pd.read_csv(args.csv_path)
    ts_test = TSDataset.load_from_dataframe(df, **info_params)

    weight_path = args.checkpoints
    if 'best_model' in weight_path:
        weight_path = weight_path.split('best_model')[0]

    if cfg.model['name'] == 'PP-TS':
        from paddlets.ensemble.base import EnsembleBase
        model = EnsembleBase.load(weight_path + '/')
    else:
        model = load(weight_path + '/checkpoints')

    if dataset.get('scale', False):
        logger.info('start scaling...')
        if not os.path.exists(os.path.join(weight_path, 'scaler.pkl')):
            raise FileNotFoundError('there is not `scaler`: {}.'.format(
                os.path.join(weight_path, 'scaler.pkl')))
        import joblib
        scaler = joblib.load(os.path.join(weight_path, 'scaler.pkl'))
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
            if dataset['name'] != 'TSDataset':
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

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if cfg.task == 'longforecast':
        logger.info('start to predit...')
        import paddle.inference as paddle_infer
        
        config = paddle_infer.Config(os.path.join(weight_path, "checkpoints.pdmodel"), os.path.join(weight_path,"checkpoints.pdiparams"))
        predictor = paddle_infer.create_predictor(config)
        input_names = predictor.get_input_names()
        print(f"input_name: f{input_names}")
        import json
        with open(os.path.join(weight_path,"./checkpoints_model_meta")) as f:
            json_data = json.load(f)
            print(json_data)

        from paddlets.utils.utils import build_ts_infer_input
        input_data = build_ts_infer_input(ts_test, os.path.join(weight_path, "./checkpoints_model_meta"))
        for key, value in json_data['input_data'].items():
            input_handle1 = predictor.get_input_handle(key)
            #set batch_size=1
            value[0] = 1
            input_handle1.reshape(value)
            input_handle1.copy_from_cpu(input_data[key])
        
        predictor.run()
        output_names = predictor.get_output_names()
        output_handle = predictor.get_output_handle(output_names[0])
        output_data = output_handle.copy_to_cpu()
        
        past_target_index = ts_test.get_target().data.index
        freq = past_target_index.freqstr
        future_target_index = pd.date_range(
            past_target_index[-1] + past_target_index.freq,
            periods=output_data.shape[1],
            freq=freq)
        future_target = pd.DataFrame(
            np.reshape(
                output_data[0], newshape=[output_data.shape[1], -1]),
            index=future_target_index,
            columns=ts_test.get_target().data.columns)
        output = TSDataset.load_from_dataframe(future_target, freq=freq)

        if dataset.get('scale', 'False'):
            result = scaler.inverse_transform(output)

        result.to_dataframe().to_csv(os.path.join(args.save_dir, 'result.csv'))


if __name__ == '__main__':
    args = parse_args()
    main(args)
