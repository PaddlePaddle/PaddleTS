import os
import paddlets
import numpy as np
import shutil
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
        help='The directory for saving the exported inference model',
        type=str,
        default='./output/inference_model')
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

    assert args.checkpoints is not None, \
        'No checkpoints dictionary specified, please set --checkpoints'

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    cfg = Config(
        args.config,
        batch_size=args.batch_size,
        opts=args.opts)

    weight_path = args.checkpoints
    if 'best_model' in weight_path:
        weight_path = weight_path.split('best_model')[0]

    if os.path.exists(os.path.join(weight_path, 'scaler.pkl')):
        shutil.copyfile(os.path.join(weight_path,'scaler.pkl'), os.path.join(args.save_dir,'scaler.pkl'))
    
    if cfg.model['name'] == 'PP-TS':
        from paddlets.ensemble.base import EnsembleBase
        model = EnsembleBase.load(weight_path + '/')
        model.save(args.save_dir, network_model=True, dygraph_to_static=True)
    else:
        model = load(weight_path + '/checkpoints')
        model.save(args.save_dir + '/checkpoints', network_model=True, dygraph_to_static=True)


if __name__ == '__main__':
    args = parse_args()
    main(args)
