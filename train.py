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
    parser.add_argument(
        '--time_feat',
        help='Whether to do evaluation after training.',
        action='store_true')
    # Runntime params
    parser.add_argument('--seq_len', help='input length in training.', type=int)
    parser.add_argument('--predict_len', help='output length in training.', type=int)
    parser.add_argument('--epoch', help='Iterations in training.', type=int)
    parser.add_argument(
        '--batch_size', help='Mini batch size of one gpu or cpu. ', type=int)
    parser.add_argument('--learning_rate', help='Learning rate.', type=float)

    # Other params
    parser.add_argument(
        '--seed',
        help='Set the random seed in training.',
        default=42,
        type=int)
    parser.add_argument(
        '--iters',
        help='Set the iters in training.',
        default=1,
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
    
    for iter in range(args.iters):
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
        seq_len=cfg.seq_len
        epoch = cfg.epoch
        split = dataset.get('split', None)
        do_eval = cfg.dic.get('do_eval', True)
        setting = cfg.model['name']+'_' + dataset['name'] + '_' + str(seq_len) + '_' + str(predict_len) + '_' + str(iter) + '/'
        logger.info(cfg.__dict__)
        logger.info('========='+setting+'===========')
        
        ts_val, ts_test = None, None    
        if split:
            ts_train, ts_val, ts_test = get_dataset(dataset['name'], split, seq_len)
        else:
            assert do_eval == False, 'if not split test data, please set do_eval False'
            ts_train = get_dataset(dataset['name'], split, seq_len)

        model = MODELS.components_dict[cfg.model['name']](
            in_chunk_len=seq_len,
            out_chunk_len=cfg.predict_len,
            batch_size=batch_size,
            max_epochs=epoch,
            **cfg.model['model_cfg']
        )

        scaler = StandardScaler()
        scaler.fit(ts_train)
        ts_train = scaler.transform(ts_train)
        ts_val = scaler.transform(ts_val)
        ts_test = scaler.transform(ts_test)
        
        if args.time_feat: 
            logger.info('generate times feature')
            from paddlets.transform import TimeFeatureGenerator
            if dataset.get('use_holiday', False):
                ts_all = get_dataset(dataset['name'])
                time_feature_generator = TimeFeatureGenerator(feature_cols=['minuteofhour','hourofday','dayofmonth','dayofweek','dayofyear', 'monthofyear', 
                                                                        'weekofyear', 'holidays'])           
                ts_all = time_feature_generator.fit_transform(ts_all)
                ts_train._known_cov = ts_all._known_cov[split['train'][0]:split['train'][1]]
                ts_val._known_cov = ts_all._known_cov[split['val'][0] - seq_len:split['val'][1]]
                ts_test._known_cov = ts_all._known_cov[split['test'][0] - seq_len:split['test'][1]]   
                    
            else:
                time_feature_generator = TimeFeatureGenerator(feature_cols=['hourofday','dayofmonth','dayofweek','dayofyear'])
                ts_train = time_feature_generator.fit_transform(ts_train)
                ts_val = time_feature_generator.fit_transform(ts_val)
                ts_test = time_feature_generator.fit_transform(ts_test)
        
        logger.info('start training...')
        model.fit(ts_train, ts_val)
        
        logger.info('start backtest...')
        if do_eval:
            metrics_score = backtest(
                data=ts_test,
                model=model,
                predict_window=predict_len,
                stride=1,
                metric=[MSE(), MAE()],
            )
            logger.info(setting + f"{metrics_score}")
            for metric in metrics_score.keys():
                logger.info(f"{metric}: {np.mean([v for v in metrics_score[metric].values()])}")



if __name__ == '__main__':
    args = parse_args()
    main(args)
