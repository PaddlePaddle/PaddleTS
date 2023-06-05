import warnings
warnings.filterwarnings("ignore")
import argparse
from paddlets.utils.config import Config
from paddlets.datasets.repository import get_dataset
from paddlets.transform.sklearn_transforms import StandardScaler
from paddlets.utils.manager import MODELS
from paddlets.metrics import MSE, MAE
from paddlets.utils import backtest

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
        default=None,
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

    return parser.parse_args()

def main(args):
    assert args.config is not None, \
        'No configuration file specified, please set --config'
    cfg = Config(
        args.config,
        learning_rate=args.learning_rate,
        epoch=args.epoch,
        batch_size=args.batch_size)

    batch_size = cfg.batch_size
    dataset = cfg.dataset
    predict_len = cfg.predict_len
    seq_len=cfg.seq_len
    epoch = cfg.epoch
    split = dataset.get('split', None)
    do_eval = cfg.dic.get('do_eval', True)

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
    if do_eval:
        ts_val = scaler.transform(ts_val)
        ts_test = scaler.transform(ts_test)
        model.fit(ts_train, ts_val)
    else:
        model.fit(ts_train)
    
    setting = dataset['name'] + '_' + str(seq_len) + '_' + str(predict_len) + '/'
    print(setting)
    model.save(args.save_dir +setting)
    
    if do_eval:
        metrics_mse_score = backtest(
            data=ts_test,
            model=model,
            predict_window=predict_len,
            stride=1,
            metric=[MSE(), MAE()],
        )
        print(f"{metrics_mse_score}")

if __name__ == '__main__':
    args = parse_args()
    main(args)
