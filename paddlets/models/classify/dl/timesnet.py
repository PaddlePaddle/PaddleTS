import numpy as np
from typing import List, Dict, Any, Callable, Optional, Tuple

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddlets.utils import param_init, manager
from paddle.optimizer import Optimizer
import paddle.nn.initializer as paddle_init

from paddlets.datasets import TSDataset
from paddlets.models.classify.dl.paddle_base import PaddleBaseClassifier
from paddlets.models.classify.dl.adapter.data_adapter import ClassifyDataAdapter
from paddlets.models.forecasting.dl.revin import revin_norm
from paddlets.models.common.callbacks import Callback
from paddlets.models.base_model._timesnet.embedding import DataEmbedding
from paddlets.models.base_model._timesnet.timesblock import TimesBlock
from paddlets.logger import raise_if, raise_if_not, raise_log, Logger

zeros_ = paddle_init.Constant(value=0.)
ones_ = paddle_init.Constant(value=1.)

logger = Logger(__name__)


class _TimesNet(nn.Layer):
    """
    The TimesNet implementation based on PaddlePaddle.

    Haixu Wu, Tengge Hu, et al. "TIMESNET: TEMPORAL 2D-VARIATION MODELING FOR GENERAL TIME SERIES ANALYSIS"
    (https://openreview.net/pdf?id=ju_Uqw384Oq)
    """

    def __init__(
            self,
            in_chunk_len: int,
            out_chunk_len: int,
            e_layers: int=2,
            c_in: int=7,
            d_model: int=32,
            embed: str='timeF',  # [timeF, fixed, learned]
            freq: str='h',
            dropout: float=0.1,
            c_out: int=7,
            d_ff: int=32,
            top_k: int=5,
            num_class: int=3,
            num_kernels: int=6,
            pretrain: str=None):
        super(_TimesNet, self).__init__()
        self.seq_len = in_chunk_len
        self.pred_len = out_chunk_len
        self.model = nn.LayerList(sublayers=[
            TimesBlock(
                in_chunk_len=in_chunk_len,
                out_chunk_len=out_chunk_len,
                d_model=d_model,
                d_ff=d_ff,
                top_k=top_k,
                num_kernels=num_kernels) for _ in range(e_layers)
        ])
        self.enc_embedding = DataEmbedding(c_in, d_model, embed, freq, dropout)
        self.layer = e_layers
        self.layer_norm = nn.LayerNorm(
            normalized_shape=d_model,
            epsilon=1e-05,
            weight_attr=None,
            bias_attr=None)
        self.act = nn.functional.gelu
        self.dropout = nn.Dropout(p=dropout)
        self.projection = nn.Linear(
            in_features=d_model * in_chunk_len, out_features=num_class)
        self.pretrain = pretrain
        self.init_weight()

    def init_weight(self):
        if self.pretrain:
            para_state_dict = paddle.load(self.pretrain)
            model_state_dict = self.state_dict()
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
            self.set_dict(model_state_dict)
            logger.info("There are {}/{} variables loaded into {}.".format(
                num_params_loaded,
                len(model_state_dict), self.__class__.__name__))
        else:
            for layer in self.sublayers():
                if isinstance(layer, nn.LayerNorm):
                    zeros_(layer.bias)
                    ones_(layer.weight)
                elif isinstance(layer, nn.Linear):
                    param_init.th_linear_fill(layer)
                elif isinstance(layer, nn.Embedding):
                    param_init.normal_init(layer.weight, mean=0.0, std=1.0)

    def classification(self, x_enc, x_mark_enc):
        enc_out = self.enc_embedding(x_enc, None)
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        output = self.act(enc_out)
        output = self.dropout(output)
        output = output * x_mark_enc.unsqueeze(axis=-1)
        output = output.reshape([output.shape[0], -1])
        output = self.projection(output)
        return output

    def forward(self, x):
        x_enc = x["features"].cast('float32')
        times_mark = x.get("pad_mask", None)
        dec_out = self.classification(x_enc, times_mark)
        return dec_out


@manager.MODELS.add_component
class TimesNet_CLS(PaddleBaseClassifier):
    """
    Implementation of TimesNet model.

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

    def __init__(
            self,
            in_chunk_len: int,  # 96
            out_chunk_len: int,  # 96 192 336 720
            sampling_stride: int=1,
            e_layers: int=2,
            c_in: int=7,
            d_model: int=32,
            embed: str='timeF',  # [timeF, fixed, learned]
            freq: str='h',
            dropout: float=0.1,
            c_out: int=7,
            d_ff: int=32,
            top_k: int=5,
            num_class: int=2,
            num_kernels: int=6,
            window_sampling_limit: int=None,
            use_revin: bool=False,
            revin_params: Dict[str, Any]=dict(
                eps=1e-5, affine=True),
            loss_fn: Callable[..., paddle.Tensor]=F.cross_entropy,  # 
            optimizer_fn: Callable[..., Optimizer]=paddle.optimizer.Adam,  #
            optimizer_params: Dict[str, Any]=dict(learning_rate=1e-4),  #
            eval_metrics: List[str]=[],
            callbacks: List[Callback]=[],
            batch_size: int=32,  # 
            max_epochs: int=10,
            verbose: int=1,
            patience: int=10,
            seed: int=0,
            renorm: bool=None):

        super(TimesNet_CLS, self).__init__(
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
        self._in_chunk_len = in_chunk_len
        self._out_chunk_len = out_chunk_len
        self._e_layers = e_layers
        self._c_in = c_in
        self._d_model = d_model
        self._num_class = num_class
        self._embed = embed
        self._freq = freq
        self._dropout = dropout
        self._c_out = c_out
        self._d_ff = d_ff
        self._top_k = top_k
        self._num_kernels = num_kernels
        self._window_sampling_limit = window_sampling_limit
        self._use_revin = use_revin
        self._revin_params = revin_params
        self._renorm = renorm

    def _update_fit_params(self,
                           train_tsdatasets: List[TSDataset],
                           train_labels: np.ndarray,
                           valid_tsdatasets: List[TSDataset],
                           valid_labels: np.ndarray) -> Dict[str, Any]:
        """Infer parameters by TSdataset automatically.

        Args:
            train_tsdataset(TSDataset): Train set.
            train_labels:(np.ndarray) : The train data class labels
            valid_tsdataset(TSDataset|None): Eval set, used for early stopping.
            valid_labels:(np.ndarray) : The valid data class labels
        Returns:
            Dict[str, Any]: model parameters.
        """
        max_len = train_tsdatasets[0].get_target().data.shape[0]
        for tsdataset in train_tsdatasets:
            if tsdataset.get_target().data.shape[0] > max_len:
                max_len = tsdataset.get_target().data.shape[0]
            #print(tsdataset.get_target().data.shape[1])

        if valid_tsdatasets is not None:
            for tsdataset in valid_tsdatasets:
                if tsdataset.get_target().data.shape[0] > max_len:
                    max_len = tsdataset.get_target().data.shape[0]

        fit_params = {
            "feature_dim": train_tsdatasets[0].get_target().data.shape[1],
            "input_lens": max_len,
        }

        return fit_params

    @revin_norm  # aligned? not used seemly
    def _init_network(self) -> nn.Layer:
        """
        Init network.

        Returns:
            nn.Layer
        """
        return _TimesNet(
            in_chunk_len=self._fit_params["input_lens"],
            out_chunk_len=self._out_chunk_len,
            e_layers=self._e_layers,
            c_in=self._fit_params["feature_dim"],
            d_model=self._d_model,
            embed=self._embed,
            freq=self._freq,
            dropout=self._dropout,
            c_out=self._c_out,
            d_ff=self._d_ff,
            top_k=self._top_k,
            num_class=self._n_classes,
            num_kernels=self._num_kernels)
