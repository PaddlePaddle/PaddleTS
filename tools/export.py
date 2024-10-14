import os
import paddlets
import numpy as np
import shutil
import random
import argparse
import warnings
import joblib
import tarfile

import paddle
from paddlets.utils.config import Config
from paddlets.models.model_loader import load
from paddlets.datasets.repository import get_dataset
from paddlets.utils.manager import MODELS
from paddlets.logger import Logger
from paddlets.utils.download import get_weights_path_from_url, uncompress_file_tar

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
        help='The directory for saving the exported inference model',
        type=str,
        default='./output/inference_model')
    parser.add_argument(
        '--checkpoints',
        help='model checkpoints for eval.',
        type=str,
        default=None)

    return parser.parse_args()


def export(args, model=None):
    paddle.set_device(args.device)
    if model is None:
        assert args.checkpoints is not None, \
            'No checkpoints dictionary specified, please set --checkpoints'
        weight_path = args.checkpoints
        if weight_path.startswith(("http://", "https://")):
            weight_path = get_weights_path_from_url(weight_path)
        else:
            if tarfile.is_tarfile(weight_path):
                weight_path = uncompress_file_tar(weight_path)
        if 'best_model' in weight_path:
            weight_path = weight_path.split('best_model')[0]
        save_path = args.save_dir
    else:
        weight_path = args.save_dir
        save_path = os.path.join(args.save_dir, 'inference')

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if os.path.exists(os.path.join(weight_path, 'config.yaml')):
        cfg = Config(os.path.join(weight_path, 'config.yaml'))
    else:
        cfg = Config(args.config)

    if cfg.dic.get('info_params', None) is None:
        raise ValueError("`info_params` is necessary, but it is None.")
    else:
        info_params = cfg.dic['info_params']
    if cfg.task == 'longforecast':
        if info_params.get('time_col', None) is None:
            raise ValueError("`time_col` is necessary, but it is None.")
        if info_params.get('target_cols', None):
            if isinstance(info_params['target_cols'], str):
                info_params['target_cols'] = info_params['target_cols'].split(
                    ',')
        if info_params.get('static_cov_cols', None):
            info_params['static_cov_cols'] = None
    elif cfg.task == 'anomaly':
        info_params.pop("label_col", None)
        if info_params.get('feature_cols', None):
            if isinstance(info_params['feature_cols'], str):
                info_params['feature_cols'] = info_params['feature_cols'].split(
                    ',')
        else:
            cols = df.columns.values.tolist()
            if info_params.get('time_col',
                               None) and info_params['time_col'] in cols:
                cols.remove(info_params['time_col'])
            info_params['feature_cols'] = cols
    elif cfg.task == 'classification':
        info_params.pop("static_cov_cols", None)
        if info_params.get('target_cols', None) is None:
            cols = df.columns.values.tolist()
            if info_params.get('time_col',
                               None) and info_params['time_col'] in cols:
                cols.remove(info_params['time_col'])
            if info_params.get('group_id',
                               None) and info_params['group_id'] in cols:
                cols.remove(info_params['group_id'])
            if info_params.get('static_cov_cols',
                               None) and info_params['static_cov_cols'] in cols:
                cols.remove(info_params['static_cov_cols'])
            info_params['target_cols'] = cols
        else:
            if isinstance(info_params['target_cols'], str):
                info_params['target_cols'] = info_params['target_cols'].split(
                    ',')

    data_info = {}
    data_info['time_feat'] = cfg.dataset['time_feat']
    data_info['holiday'] = cfg.dataset.get('use_holiday', False)
    data_info['info_params'] = info_params

    if os.path.exists(os.path.join(weight_path, 'scaler.pkl')):
        shutil.copyfile(
            os.path.join(weight_path, 'scaler.pkl'),
            os.path.join(save_path, 'scaler.pkl'))
        data_info['scale'] = True
    else:
        data_info['scale'] = False

    if cfg.model['name'] == 'PP-TS':
        from paddlets.ensemble.base import EnsembleBase
        if model is None:
            model = EnsembleBase.load(weight_path + '/')
        model.save(
            save_path,
            network_model=True,
            dygraph_to_static=True,
            model_name=cfg.dic.get('pdx_model_name', None))
    else:
        if model is None:
            model = load(weight_path + '/checkpoints')
        model.save(
            save_path + '/inference',
            network_model=True,
            dygraph_to_static=True,
            data_info=data_info,
            model_name=cfg.dic.get('pdx_model_name', None))


if __name__ == '__main__':
    args = parse_args()
    export(args)
