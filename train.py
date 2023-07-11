import os
import numpy as np
import random
import argparse
import warnings

import paddle
from paddlets.utils.config import Config
from paddlets.datasets.repository import get_dataset
from paddlets.transform.sklearn_transforms import StandardScaler
from paddlets.utils.manager import TasktoModel
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
        '--task',
        required=True,
        help='Set the task for training model.',
        default='longforecast',
        choices=['longforecast', 'shortforecast', 'classification', 'imputation', 'anomaly'],
        type=str)
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
        '--do_train',
        help='Whether to train the model.',
        action='store_false')
    parser.add_argument(
        '--do_eval',
        help='Whether to do evaluation after training.',
        action='store_false')
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
    
    cfg = Config(
        args.config,
        learning_rate=args.learning_rate,
        epoch=args.epoch,
        seq_len=args.seq_len,
        predict_len=args.predict_len,
        batch_size=args.batch_size,
        opts=args.opts)

    dataset = cfg.dataset
    seq_len=cfg.seq_len
    split = dataset.get('split', None)
    logger.info(cfg.__dict__)
    
    ts_val, ts_test = None, None    
    if split:
        ts_train, ts_val, ts_test = get_dataset(dataset['name'], split, seq_len)
    else:
        ts_train, ts_y = get_dataset(dataset['name']+'_Train', split, seq_len)
        ts_test, ts_test_y = get_dataset(dataset['name']+'_Test', split, seq_len)
    
    model = TasktoModel[args.task].components_dict[cfg.model['name']](
        in_chunk_len=seq_len,
        out_chunk_len=cfg.predict_len,
        batch_size=cfg.batch_size,
        max_epochs=cfg.epoch,
        **cfg.model['model_cfg']
    )
    
    if cfg.dataset.get('scale', True):
        scaler = StandardScaler()
        scaler.fit(ts_train)
        ts_train = scaler.transform(ts_train)
        ts_val = scaler.transform(ts_val)
        ts_test = scaler.transform(ts_test)
        logger.info('data scale done!')
    

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
    
    for iter in range(args.iters):
        logger.info('start training...')
        if args.task == 'longforecast':
            model.fit(ts_train, ts_val)
            if args.do_eval:
                logger.info('start backtest...')
                metrics_score = backtest(
                    data=ts_test,
                    model=model,
                    predict_window=cfg.predict_len,
                    stride=1,
                    metric=[MSE(), MAE()],
                )
                setting = cfg.model['name']+'_' + dataset['name'] + '_' + str(seq_len) + '_' + str(cfg.predict_len) + '_' + str(iter) + '/'
                logger.info(setting + f"{metrics_score}")
                for metric in metrics_score.keys():
                    logger.info(f"{metric}: {np.mean([v for v in metrics_score[metric].values()])}")
        
        elif args.task == 'classification':
            model.fit(ts_train, ts_y, ts_test, ts_test_y)
            if args.do_eval:
                from sklearn.metrics import accuracy_score, f1_score
                preds = model.predict_proba(ts_test)
                score = accuracy_score(ts_test_y, np.argmax(preds,axis=1))
                f1 = f1_score(ts_test_y, np.argmax(preds,axis=1), average="macro")
                logger.info(f"acc: {score}, f1: {f1}")


if __name__ == '__main__':
    args = parse_args()
    main(args)
