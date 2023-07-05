from typing import List, Dict, Any, Callable, Optional, NewType, Tuple, Union
import numpy as np

import paddle
from paddle import nn
from paddle.optimizer import Optimizer
import paddle.nn.functional as F
import paddle.nn.initializer as paddle_init

from paddlets.datasets import TSDataset
from paddlets.utils import param_init, manager
from paddlets.models.forecasting.dl._patchtst.revin import RevIN
from paddlets.models.forecasting.dl.paddle_base_impl import PaddleBaseModelImpl
from paddlets.models.common.callbacks import Callback
from paddlets.logger import raise_if, raise_if_not, raise_log, Logger

zeros_ = paddle_init.Constant(value=0.)
ones_ = paddle_init.Constant(value=1.)

logger = Logger(__name__)


class ResidualBlock(paddle.nn.Layer):

    def __init__(self, in_features, hid_features, out_features,
        dropout_prob=0.3, layer_norm=True):
        super(ResidualBlock, self).__init__()
        self.fc1 = paddle.nn.Linear(in_features=in_features, out_features=
            hid_features)
        self.relu1 = paddle.nn.ReLU()
        self.fc2 = paddle.nn.Linear(in_features=hid_features, out_features=
            out_features)
        self.dropout = paddle.nn.Dropout(p=dropout_prob)
        self.norm = paddle.nn.LayerNorm(normalized_shape=out_features,
            epsilon=1e-05, weight_attr=None, bias_attr=None)
        self.layer_norm = layer_norm
        self.shortcut = paddle.nn.Identity()
        if in_features != out_features:
            self.shortcut = paddle.nn.Linear(in_features=in_features,
                out_features=out_features)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out += self.shortcut(residual)
        if self.layer_norm:
            out = self.norm(out)
        return out


class TiDEEncoder(paddle.nn.Layer):

    def __init__(self, in_features, hid_features, out_features, drop_prob,
        num_blocks=1, layer_norm=True):
        super(TiDEEncoder, self).__init__()
        self.blocks = paddle.nn.LayerList()
        for i in range(num_blocks):
            self.blocks.append(ResidualBlock(in_features, hid_features,
                out_features, drop_prob, layer_norm))
            in_features = out_features

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class TiDEDenseDecoder(paddle.nn.Layer):

    def __init__(self, in_features, hid_features, out_features, drop_prob,
        num_blocks=1, layer_norm=True):
        super(TiDEDenseDecoder, self).__init__()
        self.blocks = paddle.nn.LayerList()
        for i in range(num_blocks):
            self.blocks.append(ResidualBlock(in_features, hid_features,
                out_features, drop_prob, layer_norm))
            in_features = out_features

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class TiDETemporalDecoder(paddle.nn.Layer):

    def __init__(self, in_features, hid_features, out_features, drop_prob,
        layer_norm=True):
        super(TiDETemporalDecoder, self).__init__()
        self.residual = ResidualBlock(in_features, hid_features,
            out_features, drop_prob, layer_norm)

    def forward(self, x):
        return self.residual(x)


class _TiDETModule(paddle.nn.Layer):
    """
    The TiDE implementation based on PaddlePaddle.

    The original article refers to
    Abhimanyu Das, Weihao Kong, et al. "Long-term Forecasting with TiDE: Time-series Dense Encoder"
    (https://arxiv.org/pdf/2304.08424.pdf)
    """

    def __init__(self, c_in=7, seq_len=96, pred_len=720, c_in_time_feat=4, time_feat_size=4,
        time_hidden_size=64,hidden_size=256, num_encoder_layers=2, num_decoder_layers=2, decoder_output_dim=8, temporal_decoder_hidden=128,
        drop_prob=0.3 ,layer_norm =True, revin=True, pretrain=None):  
        super(_TiDETModule, self).__init__()

        self.c_in = c_in
        self.seq_len = seq_len
        self.dynCov_shape = [seq_len + pred_len, c_in_time_feat]
        self.pred_len =pred_len
        self.time_feat_size = time_feat_size
        self.time_hidden_size = time_hidden_size
        self.hidden_size = hidden_size
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers =  num_decoder_layers
        self.decoder_output_dim =  decoder_output_dim
        self.temporal_decoder_hidden =  temporal_decoder_hidden
        self.drop_prob = drop_prob
        self.layer_norm = layer_norm
        self.revin =  revin
        self.pretrain = pretrain
        self.RevIN = RevIN(self.c_in)
        self.concat_shape = self.seq_len  + self.dynCov_shape[0
                    ] * self.time_feat_size 
        self.featproj = ResidualBlock(self.dynCov_shape[1], self.
            time_hidden_size, self.time_feat_size, self.drop_prob, self.layer_norm)
        self.encoder = TiDEEncoder(self.concat_shape, self.hidden_size,
            self.hidden_size, self.drop_prob, self.num_encoder_layers, self
            .layer_norm)
        self.denseDecoder = TiDEDenseDecoder(self.hidden_size, self.
            hidden_size, self.decoder_output_dim * self.pred_len, self.
            drop_prob, self.num_decoder_layers, self.layer_norm)
        self.temporalDecoder = TiDETemporalDecoder(self.time_feat_size + self.
            decoder_output_dim, self.temporal_decoder_hidden, 
            1, self.drop_prob, self.layer_norm)
        self.linear = paddle.nn.Linear(in_features=self.seq_len,
            out_features=self.pred_len)
        
        self.init_weight()

    def forward(self, x):
        batch_x = x["past_target"]
        batch_size = batch_x.shape[0]
        times_mark = x.get("known_cov_numeric", None)
        proj_feature = self.featproj(times_mark) # time_encoder
        proj_feature_x, proj_feature_y = proj_feature[:,:self.seq_len,:], proj_feature[:,self.seq_len:,:]

        if self.revin:
            batch_x = self.RevIN(batch_x, mode='norm')

        proj_feature_past = paddle.repeat_interleave(proj_feature_x.reshape([batch_size, -1, 1]), self.c_in, 2) # [1, 2880]
        proj_feature_furt = paddle.repeat_interleave(proj_feature_y.reshape([batch_size, -1, 1]), self.c_in, 2)  # [1, 384]
        encoder_input = paddle.concat(x=(batch_x, proj_feature_past, proj_feature_furt), axis=1) # [1, 3984, 7]
        encoded = self.encoder(encoder_input.transpose([0, 2, 1]))
        denseDecoded = self.denseDecoder(encoded) 

        denseDecoded = denseDecoded.reshape([batch_size, self.c_in, self.pred_len,
            self.decoder_output_dim]) 
        denseDecoded = paddle.concat(x=(denseDecoded, proj_feature_furt.transpose([0,2,1]).reshape([
            batch_size, self.c_in, self.pred_len, self.time_feat_size])), axis=3)
        temporalDecoded = self.temporalDecoder(denseDecoded).squeeze(-1)  
        res_lookback = self.linear(batch_x.transpose([0, 2, 1])) 
        pred = temporalDecoded + res_lookback

        if self.revin:
            pred = self.RevIN(pred.transpose([0,2,1]), mode='denorm')
            return pred
        else:
            return pred.transpose([0,2,1])
    
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
                    if isinstance(layer, nn.Linear):
                        param_init.th_linear_fill(layer)



@manager.MODELS.add_component
class TiDE(PaddleBaseModelImpl):
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
                 c_in_time_feat=4, 
                 time_feat_size=4,
                 time_hidden_size=64, 
                 hidden_size=256, 
                 num_encoder_layers=2, 
                 num_decoder_layers=2, 
                 decoder_output_dim=8, 
                 temporal_decoder_hidden=128,
                 drop_prob=0.3,
                 layer_norm =True, 
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
        self.c_in_time_feat = c_in_time_feat
        self.time_feat_size = time_feat_size
        self.time_hidden_size = time_hidden_size
        self.hidden_size = hidden_size
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.decoder_output_dim = decoder_output_dim
        self.temporal_decoder_hidden = temporal_decoder_hidden
        self.drop_prob = drop_prob 
        self.layer_norm = layer_norm
        self._use_revin = use_revin
        self._revin_params = revin_params
        self.pretrain = pretrain

        super(TiDE, self).__init__(
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
        super(TiDE, self)._check_tsdataset(tsdataset)

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
        return _TiDETModule(
            c_in = self.c_in,
            seq_len=self._in_chunk_len, 
            pred_len=self._out_chunk_len,
            c_in_time_feat= self._fit_params["known_cov_dim"], 
            time_feat_size=self.time_feat_size,
            time_hidden_size=self.time_hidden_size, 
            hidden_size=self.hidden_size, 
            num_encoder_layers=self.num_encoder_layers, 
            num_decoder_layers=self.num_decoder_layers, 
            decoder_output_dim=self.decoder_output_dim, 
            temporal_decoder_hidden=self.temporal_decoder_hidden,
            drop_prob=self.drop_prob,
            layer_norm =self.layer_norm, 
            revin=self._use_revin,
            pretrain = self.pretrain
            )