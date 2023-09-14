import os
import numpy as np
import random
import argparse
import warnings

import paddle
from paddlets.utils.config import Config
from paddlets.datasets.repository import get_dataset
from paddlets.transform.sklearn_transforms import StandardScaler
from paddlets.utils.manager import MODELS
from paddlets.metrics import MSE, MAE
from paddlets.utils import backtest
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
    # Runntime params
    parser.add_argument('--seq_len', help='input length in training.', type=int)
    parser.add_argument(
        '--predict_len', help='output length in training.', type=int)
    parser.add_argument('--epoch', help='Iterations in training.', type=int)
    parser.add_argument(
        '--batch_size', help='Mini batch size of one gpu or cpu. ', type=int)
    parser.add_argument('--learning_rate', help='Learning rate.', type=float)

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

    cfg = Config(
        args.config,
        learning_rate=args.learning_rate,
        epoch=args.epoch,
        seq_len=args.seq_len,
        predict_len=args.predict_len,
        batch_size=args.batch_size,
        opts=args.opts)

    batch_size = cfg.batch_size
    dataset = cfg.dataset
    predict_len = cfg.predict_len
    seq_len = cfg.seq_len
    epoch = cfg.epoch
    split = dataset.get('split', None)
    do_eval = cfg.dic.get('do_eval', True)
    sampling_stride = cfg.dic.get('sampling_stride', 1)
    logger.info(cfg.__dict__)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    else:
        if not os.path.isdir(args.save_dir):
            os.remove(args.save_dir)
            os.makedirs(args.save_dir, exist_ok=True)
        else:
            for f in os.listdir(args.save_dir):
                if f.startswith('checkpoint') or f.startswith(
                        'paddlets-ensemble') or f == 'scaler.pkl':
                    os.remove(os.path.join(args.save_dir, f))
                if f == 'best_model':
                    import shutil
                    shutil.rmtree(os.path.join(args.save_dir, f))

    ts_val = None
    ts_test = None
    if dataset['name'] == 'TSDataset':
        import pandas as pd
        from paddlets import TSDataset
        dataset_root = dataset.get('dataset_root', None)

        if not os.path.exists(dataset_root):
            raise FileNotFoundError('there is not `dataset_root`: {}.'.format(
                dataset_root))
        df = pd.read_csv(dataset['train_path'])

        if cfg.dic.get('info_params', None) is None:
            raise ValueError("`info_params` is necessary, but it is None.")
        else:
            info_params = cfg.dic['info_params']
            if cfg.task == 'longforecast' and info_params.get('time_col',
                                                              None) is None:
                raise ValueError("`time_col` is necessary, but it is None.")
            if info_params.get('target_cols', None):
                # target_cols = info_params['target_cols'] if info_params[
                #     'target_cols'] != [''] else None
                if isinstance(info_params['target_cols'], str):
                    info_params['target_cols'] = info_params['target_cols'].split(',')

        if cfg.task == 'anomaly':
            if info_params.get('feature_cols', None):
                if isinstance(info_params['feature_cols'], str):
                    info_params['feature_cols'] = info_params['feature_cols'].split(',')
            else:
                cols = df.columns.values.tolist()
                if info_params.get('label_col', None) and info_params['label_col'] in cols:
                    cols.remove(info_params['label_col'])
                if info_params.get('time_col', None) and info_params['time_col'] in cols:
                    cols.remove(info_params['time_col'])
                info_params['feature_cols'] = cols
            
            info_params_train = info_params.copy()
            info_params_train.pop("label_col", None)
            ts_train = TSDataset.load_from_dataframe(df, **info_params_train)
        elif cfg.task == 'classification':
            if info_params.get('target_cols', None) is None:
                cols = df.columns.values.tolist()
                if info_params.get('time_col', None) and info_params['time_col'] in cols:
                    cols.remove(info_params['time_col'])
                if info_params.get('group_id', None) and info_params['group_id'] in cols:
                    cols.remove(info_params['group_id'])
                if info_params.get('static_cov_cols', None) and info_params['static_cov_cols'] in cols:
                    cols.remove(info_params['static_cov_cols'])
                info_params['target_cols'] = cols
            ts_train = TSDataset.load_from_dataframe(df, **info_params)
        else:
            ts_train = TSDataset.load_from_dataframe(df, **info_params)

        if dataset.get('val_path', False):
            if os.path.exists(dataset['val_path']):
                df = pd.read_csv(dataset['val_path'])
                ts_val = TSDataset.load_from_dataframe(df, **info_params)
        if dataset.get('test_path', False):
            if os.path.exists(dataset['test_path']):
                df = pd.read_csv(dataset['test_path'])
                ts_test = TSDataset.load_from_dataframe(df, **info_params)
    else:
        info_params = cfg.dic.get('info_params', None)
        if split:
            ts_train, ts_val, ts_test = get_dataset(dataset['name'], split,
                                                    seq_len, info_params)
        else:
            ts_train = get_dataset(dataset['name'], split, seq_len, info_params)

    if cfg.model['name'] == 'PP-TS':
        from paddlets.ensemble import WeightingEnsembleForecaster
        estimators = []
        for model_name, model_cfg in cfg.model['model_cfg']['Ensemble'].items():
            model_cfg = Config(
                model_cfg,
                seq_len=seq_len,
                predict_len=predict_len,
                batch_size=batch_size,
                opts=args.opts)
            logger.info(model_cfg.model)

            params = dict()
            params['in_chunk_len'] = seq_len
            params['out_chunk_len'] = predict_len

            if model_name == 'XGBoost':
                from paddlets.models.ml_model_wrapper import SklearnModelWrapper
                from xgboost import XGBRegressor
                params['model_init_params'] = model_cfg.model['model_cfg']
                params['sampling_stride'] = sampling_stride
                params['model_class'] = XGBRegressor
                estimators.append((SklearnModelWrapper, params))
            else:
                one_model = MODELS.components_dict[model_name]
                params = model_cfg.model['model_cfg']
                params['batch_size'] = batch_size
                params['max_epochs'] = epoch
                estimators.append((one_model, params))

        model = WeightingEnsembleForecaster(
            in_chunk_len=seq_len,
            out_chunk_len=predict_len,
            skip_chunk_len=0,
            estimators=estimators,
            mode='mean')

    elif cfg.model['name'] == 'XGBoost':
        from paddlets.models.ml_model_wrapper import make_ml_model
        from xgboost import XGBRegressor
        #sample_len = len(ts_train._target.data) - seq_len - predict_len
        # max sample = 1000
        model = make_ml_model(
            in_chunk_len=seq_len,
            out_chunk_len=predict_len,
            sampling_stride=sampling_stride,
            model_class=XGBRegressor,
            use_skl_gridsearch=False,
            model_init_params=cfg.model['model_cfg'])

    else:
        model = MODELS.components_dict[cfg.model['name']](
            in_chunk_len=seq_len,
            out_chunk_len=predict_len,
            batch_size=batch_size,
            sampling_stride=sampling_stride,
            max_epochs=epoch,
            **cfg.model['model_cfg'])

    if dataset.get('scale', False):
        logger.info('start scaling...')
        scaler = StandardScaler()
        scaler.fit(ts_train)
        ts_train = scaler.transform(ts_train)
        if ts_val is not None:
            ts_val = scaler.transform(ts_val)
        if ts_test is not None:
            ts_test = scaler.transform(ts_test)
        import joblib
        joblib.dump(scaler, os.path.join(args.save_dir, 'scaler.pkl'))

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
                ts_train._known_cov = ts_all._known_cov[split['train'][0]:split[
                    'train'][1]]
                if ts_val is not None:
                    ts_val._known_cov = ts_all._known_cov[split['val'][
                        0] - seq_len:split['val'][1]]
                if ts_test is not None:
                    ts_test._known_cov = ts_all._known_cov[split['test'][
                        0] - seq_len:split['test'][1]]
            else:
                ts_train = time_feature_generator.fit_transform(ts_train)
                if ts_val is not None:
                    ts_val = time_feature_generator.fit_transform(ts_val)
                if ts_test is not None:
                    ts_test = time_feature_generator.fit_transform(ts_test)

        else:
            time_feature_generator = TimeFeatureGenerator(feature_cols=[
                'hourofday', 'dayofmonth', 'dayofweek', 'dayofyear'
            ])
            ts_train = time_feature_generator.fit_transform(ts_train)
            if ts_val is not None:
                ts_val = time_feature_generator.fit_transform(ts_val)
            if ts_test is not None:
                ts_test = time_feature_generator.fit_transform(ts_test)

    if cfg.task == 'longforecast':

        logger.info('start training...')
        model.fit(ts_train, ts_val)

        logger.info('search best model...')
        if cfg.model['name'] == 'PP-TS' and ts_val is not None:
            model.search_best(ts_val)

        logger.info('save best model...')
        if cfg.model['name'] == 'PP-TS':
            model.save(args.save_dir + '/')
        else:
            model.save(args.save_dir + '/checkpoints/')

        logger.info('done.')

    elif cfg.task == 'classification':
        logger.info('start training...')

        y_label = []
        for dataset in ts_train:
            y_label.append(dataset.static_cov[info_params['static_cov_cols']])
            dataset.static_cov = None
        ts_y = np.array(y_label)

        ts_val_y = None
        if ts_val is not None:
            y_label = []
            for dataset in ts_val:
                y_label.append(dataset.static_cov[info_params[
                    'static_cov_cols']])
                dataset.static_cov = None
            ts_val_y = np.array(y_label)

        model.fit(ts_train, ts_y, ts_val, ts_val_y)
        model.save(args.save_dir + '/checkpoints/')

    elif cfg.task == 'anomaly':
        model.fit(ts_train, ts_val)
        model.save(args.save_dir + '/checkpoints/')


if __name__ == '__main__':
    args = parse_args()
    main(args)
