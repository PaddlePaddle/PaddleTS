import numpy as np
from typing import List, Dict, Any, Callable, Optional

import paddle
from paddle import nn
import paddle.nn.functional as F
import paddle.nn.initializer as paddle_init
from paddle.optimizer import Optimizer

from paddlets.datasets import TSDataset
from paddlets.models.forecasting.dl.paddle_base_impl import PaddleBaseModelImpl
from paddlets.models.forecasting.dl._patchtst.revin import RevIN
from paddlets.models.common.callbacks import Callback
from paddlets.logger import raise_if, raise_if_not, raise_log, Logger
from paddlets.utils import manager, param_init

zeros_ = paddle_init.Constant(value=0.)
ones_ = paddle_init.Constant(value=1.)

logger = Logger(__name__)


class _RLinearModule(paddle.nn.Layer):
    """
    The RLinear implementation based on PaddlePaddle.

    The original article refers to
    Zhe Li, Shiyi Qi, "Revisiting Long-term Time Series Forecasting: An Investigation on Linear Mapping"
    (https://arxiv.org/pdf/2305.10721.pdf)
    """

    def __init__(self, c_in=7, seq_len=96, pred_len=96, dropout=0.0, revin=1, individual=False, pretrain=None):
        super(_RLinearModule, self).__init__()
        self.Linear = paddle.nn.LayerList(sublayers=[paddle.nn.Linear(
            in_features=seq_len, out_features=pred_len) for
            _ in range(c_in)]
            ) if individual else paddle.nn.Linear(in_features=
            seq_len, out_features=pred_len)
        self.dropout = paddle.nn.Dropout(p=dropout)
        self.rev = RevIN(c_in) if revin else None
        self.individual = individual
        self.pretrain = pretrain
        self.init_weight()

    def forward(self, x):
        x = x['past_target']
        x = self.rev(x, 'norm') if self.rev else x
        x = self.dropout(x)
         
        if self.individual:
            pred = paddle.zeros_like(x)
            for idx, proj in enumerate(self.Linear):
                pred[:, :, (idx)] = proj(x[:, :, (idx)])
        else:
            x = x
            perm_0 = list(range(x.ndim))
            perm_0[1] = 2
            perm_0[2] = 1
            x = self.Linear(x.transpose(perm=perm_0))
            perm_1 = list(range(x.ndim))
            perm_1[1] = 2
            perm_1[2] = 1
            pred = x.transpose(perm=perm_1)
        pred = self.rev(pred, 'denorm') if self.rev else pred
        return pred
    
    def init_weight(self):
        if self.pretrain:
            para_state_dict = paddle.load(self.pretrain)
            model_state_dict = self.model.state_dict()
            keys = model_state_dict.keys()
            num_params_loaded = 0
            for k in keys:
                if k not in para_state_dict:
                    logger.warning("{} is not in pretrained model".format(k))
                elif list(para_state_dict[k].shape) != list(model_state_dict[k]
                                                            .shape):
                    logger.warning(
                        "[SKIP] Shape of pretrained params {} doesn't match.(Pretrained: {}, Actual: {})"
                        .format(k, para_state_dict[k].shape, model_state_dict[k]
                                .shape))
                else:
                    model_state_dict[k] = para_state_dict[k]
                    num_params_loaded += 1
            self.model.set_dict(model_state_dict)
            logger.info("There are {}/{} variables loaded into {}.".format(
                num_params_loaded,
                len(model_state_dict), self.model.__class__.__name__))
        else:
            for layer in self.sublayers():
                if isinstance(layer, nn.Conv2D):
                    std = layer._kernel_size[0] * layer._kernel_size[
                        1] * layer._out_channels
                    std //= layer._groups
                    param_init.normal_init(layer.weight, std=std)
                elif isinstance(layer, (nn.BatchNorm, nn.SyncBatchNorm)):
                    param_init.constant_init(layer.weight, value=1.0)
                    param_init.constant_init(layer.bias, value=0.0)
                elif isinstance(layer, nn.Linear):
                    param_init.th_linear_fill(layer)
                    #param_init.kaiming_normal_init(layer.weight)
                    # if layer.bias is not None:
                    #     param_init.uniform_init(layer.bias,  low=-0.102, high=0.102)

    
@manager.MODELS.add_component
class RLinearModel(PaddleBaseModelImpl):
    """
    Implementation of PatchTST model.

    Args:
        in_chunk_len(int): The size of the loopback window, i.e., the number of time steps feed to the model.
        out_chunk_len(int): The size of the forecasting horizon, i.e., the number of time steps output by the model.
        generic_architecture(bool, Optional): Boolean value indicating whether the generic architecture of N-BEATS is used. \
                    If not, the interpretable architecture outlined in the paper (consisting of one trend and one seasonality stack \
                    with appropriate waveform generator functions).
        num_stacks(int, Optional): The number of stacks that make up the whole model. Only used if `generic_architecture` is set to `True`.
        num_blocks(Union[int, List[int]], Optional): The number of blocks making up each stack. \
                    If a list is passed, it must have a length equal to `num_stacks` and every entry in that list corresponds to the corresponding stack.\
                    If an integer is passed, every stack will have the same number of blocks.
        num_layers(int, Optional): The number of fully connected layers preceding the final forking layers in each block of every stack. \
                    Only used if `generic_architecture` is set to `True`.
        layer_widths(Union[int, List[int]], Optional): Determines the number of neurons that make up each fully connected layer in each block of every stack. If a list is passed, it must have a length equal to `num_stacks` and every entry in that list corresponds to the layer width of the corresponding stack. If an integer is passed, every stack will have blocks with FC layers of the same width.
        expansion_coefficient_dim(int, Optional): The dimensionality of the waveform generator parameters, also known as expansion coefficients. Only used if `generic_architecture` is set to `True`.
        trend_polynomial_degree(int, Optional): The degree of the polynomial used as waveform generator in trend stacks. Only used if `generic_architecture` is set to `False`.
        skip_chunk_len(int, Optional): Optional, the number of time steps between in_chunk and out_chunk for a single sample. The skip chunk is neither used as a feature (i.e. X) nor a label (i.e. Y) for a single sample. By default it will NOT skip any time steps.
        sampling_stride(int, optional): sampling intervals between two adjacent samples.
        loss_fn(Callable, Optional): loss function.
        optimizer_fn(Callable, Optional): optimizer algorithm.
        optimizer_params(Dict, Optional): optimizer parameters.
        eval_metrics(List[str], Optional): evaluation metrics of model.
        callbacks(List[Callback], Optional): customized callback functions.
        batch_size(int, Optional): number of samples per batch.
        max_epochs(int, Optional): max epochs during training.
        verbose(int, Optional): verbosity mode.
        patience(int, Optional): number of epochs with no improvement after which learning rate wil be reduced.
        seed(int, Optional): global random seed.
    """

    def __init__(self,
                 in_chunk_len: int,
                 out_chunk_len: int,
                 c_in: int=3,
                 dropout: float=0.0, 
                 individual: bool=False,
                 skip_chunk_len: int=0,
                 sampling_stride: int=1,
                 loss_fn: Callable[..., paddle.Tensor]=F.mse_loss,
                 optimizer_fn: Callable[..., Optimizer]=paddle.optimizer.Adam,
                 optimizer_params: Dict[str, Any]=dict(learning_rate=1e-4),
                 use_revin: bool=False,
                 revin_params: Dict[str, Any]=dict(
                     eps=1e-5, affine=True),
                 eval_metrics: List[str]=[],
                 callbacks: List[Callback]=[],
                 batch_size: int=32,
                 max_epochs: int=10,
                 verbose: int=1,
                 patience: int=10,
                 seed: int=0,
                 pretrain=None):
        self.c_in = c_in
        self.dropout = dropout
        self.individual = individual
        self._use_revin = use_revin
        self._revin_params = revin_params
        self.pretrain = pretrain

        super(RLinearModel, self).__init__(
            in_chunk_len=in_chunk_len,
            out_chunk_len=out_chunk_len,
            skip_chunk_len=skip_chunk_len,
            sampling_stride=sampling_stride,
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
            optimizer_params=optimizer_params,
            eval_metrics=eval_metrics,
            callbacks=callbacks,
            batch_size=batch_size,
            max_epochs=max_epochs,
            verbose=verbose,
            patience=patience,
            seed=seed, )


    def _check_tsdataset(self, tsdataset: TSDataset):
        """ 
        Rewrite _check_tsdataset to fit the specific model.
        For NBeats, all data variables are expected to be float32.
        """
        for column, dtype in tsdataset.dtypes.items():
            raise_if_not(
                np.issubdtype(dtype, np.floating),
                f"nbeats variables' dtype only supports [float16, float32, float64], " \
                f"but received {column}: {dtype}."
            )
        super(RLinearModel, self)._check_tsdataset(tsdataset)

    def _update_fit_params(
            self,
            train_tsdataset: List[TSDataset],
            valid_tsdataset: Optional[List[TSDataset]]=None) -> Dict[str, Any]:
        """
        Infer parameters by TSDataset automatically.

        Args:
            train_tsdataseet(List[TSDataset]): list of train dataset
            valid_tsdataset(List[TSDataset], optional): list of validation dataset
        
        Returns:
            Dict[str, Any]: model parameters
        """
        fit_params = {
            "target_dim": train_tsdataset[0].get_target().data.shape[1],
            "known_cov_dim": 0,
            "observed_cov_dim": 0
        }
        if train_tsdataset[0].get_known_cov() is not None:
            fit_params["known_cov_dim"] = train_tsdataset[0].get_known_cov(
            ).data.shape[1]
        if train_tsdataset[0].get_observed_cov() is not None:
            fit_params["observed_cov_dim"] = train_tsdataset[
                0].get_observed_cov().data.shape[1]
        return fit_params

    def _init_network(self) -> paddle.nn.Layer:
        """
        Init network.

        Returns:
            paddle.nn.Layer
        """
        return _RLinearModule(
            c_in = self.c_in,
            seq_len=self._in_chunk_len, 
            pred_len=self._out_chunk_len,
            dropout=self.dropout, 
            individual=self.individual, 
            pretrain = self.pretrain
            )