from typing import List, Dict, Any, Callable, Optional
import numpy as np
from math import ceil
import numpy as np
from einops import repeat

import paddle
from paddle import nn
from paddle.optimizer import Optimizer
import paddle.nn.functional as F
import paddle.nn.initializer as paddle_init

from paddlets.utils import param_init, manager
from paddlets.datasets import TSDataset
from paddlets.models.forecasting.dl._crossformer.encoder import Encoder
from paddlets.models.forecasting.dl._crossformer.decoder import Decoder
from paddlets.models.forecasting.dl._crossformer.embedding import DSW_embedding
from paddlets.models.forecasting.dl.paddle_base_impl import PaddleBaseModelImpl
from paddlets.models.forecasting.dl.revin import revin_norm
from paddlets.models.common.callbacks import Callback
from paddlets.logger import raise_if, raise_if_not, raise_log, Logger

zeros_ = paddle_init.Constant(value=0.)
ones_ = paddle_init.Constant(value=1.)
logger = Logger(__name__)


class CrossformerModule(paddle.nn.Layer):
    """
    The Crossformer implementation based on PaddlePaddle.

    The original article refers to
    Yunhao Zhang, Junchi Yan, et al. "Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting"
    (https://openreview.net/forum?id=vSVLM2j9eie)
    """

    def __init__(self, c_in, seq_len, pred_len, seg_len, win_size=4,
        factor=10, d_model=512, d_ff=1024, n_heads=8, e_layers=3, dropout=
        0.0, baseline=False, pretrain=None):
        super(CrossformerModule, self).__init__()
        self.data_dim = c_in
        self.in_len = seq_len
        self.out_len = pred_len
        self.seg_len = seg_len
        self.merge_win = win_size
        self.baseline = baseline
        self.pad_in_len = ceil(1.0 * seq_len / seg_len) * seg_len
        self.pad_out_len = ceil(1.0 * pred_len / seg_len) * seg_len
        self.in_len_add = self.pad_in_len - self.in_len
        self.enc_value_embedding = DSW_embedding(seg_len, d_model)

        x = paddle.randn(shape=[1, c_in, self.pad_in_len // seg_len, d_model])
        self.enc_pos_embedding = paddle.create_parameter(shape=[1, c_in, self.pad_in_len // seg_len, d_model],
                                    dtype=str(x.numpy().dtype),
                                    default_initializer=paddle.nn.initializer.Assign(x))
        
        self.pre_norm = paddle.nn.LayerNorm(normalized_shape=d_model,
            epsilon=1e-05, weight_attr=None, bias_attr=None)
        self.encoder = Encoder(e_layers, win_size, d_model, n_heads, d_ff,
            block_depth=1, dropout=dropout, in_seg_num=self.pad_in_len //
            seg_len, factor=factor)
        
        x = paddle.randn(shape=[1, c_in, self.pad_out_len // seg_len, d_model])
        self.dec_pos_embedding = paddle.create_parameter(shape=[1,c_in, self.pad_out_len // seg_len, d_model],
                                    dtype=str(x.numpy().dtype),
                                    default_initializer=paddle.nn.initializer.Assign(x))
        
        self.decoder = Decoder(seg_len, e_layers + 1, d_model, n_heads,
            d_ff, dropout, out_seg_num=self.pad_out_len // seg_len, factor=
            factor)
        self.pretrain = pretrain
        self.init_weight()

    def forward(self, x):
        x_seq = x['past_target']
        if self.baseline:
            base = x_seq.mean(axis=1, keepdim=True)
        else:
            base = 0
        batch_size = x_seq.shape[0]
        if self.in_len_add != 0:
            x_seq = paddle.concat(x=(x_seq[:, :1, :].expand(shape=[-1, self
                .in_len_add, -1]), x_seq), axis=1)
        x_seq = self.enc_value_embedding(x_seq)
        x_seq += self.enc_pos_embedding
        x_seq = self.pre_norm(x_seq)
        enc_out = self.encoder(x_seq)
        dec_in = repeat(self.dec_pos_embedding,
            'b ts_d l d -> (repeat b) ts_d l d', repeat=batch_size)
        predict_y = self.decoder(dec_in, enc_out)
        return base + predict_y[:, :self.out_len, :]

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


@manager.MODELS.add_component
class Crossformer(PaddleBaseModelImpl):
    """
    Implementation of NBeats model.

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
                 c_in: int,
                 factor: int,
                 seg_len: int=24,
                 win_size: int=4,
                 dropout: float=0.0,
                 d_model: int=512, 
                 d_ff: int=1024, 
                 n_heads: int=8, 
                 e_layers: int=3,
                 pretrain = None,
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
                 seed: int=0):
        self.c_in = c_in
        self.factor = factor
        self.seg_len = seg_len
        self.win_size = win_size
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_head = n_heads
        self.e_layers= e_layers
        self.dropout = dropout
        self._use_revin = use_revin
        self._revin_params = revin_params
        self.pretrain = pretrain

        super(Crossformer, self).__init__(
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
        super(Crossformer, self)._check_tsdataset(tsdataset)

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

    @revin_norm
    def _init_network(self) -> paddle.nn.Layer:
        """
        Init network.

        Returns:
            paddle.nn.Layer
        """
        return CrossformerModule(
            c_in=self.c_in,
            seq_len=self._in_chunk_len, 
            pred_len=self._out_chunk_len,
            seg_len = self.seg_len,
            factor=self.factor,
            win_size=self.win_size,
            d_model=self.d_model, 
            d_ff=self.d_ff,
            n_heads=self.n_head, 
            e_layers=self.e_layers,
            dropout=self.dropout,
            pretrain=self.pretrain)