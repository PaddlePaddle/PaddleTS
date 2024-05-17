from typing import List, Dict, Any, Callable, Optional
import numpy as np

import paddle
from paddle import nn
from paddle.optimizer import Optimizer
import paddle.nn.functional as F
import paddle.nn.initializer as paddle_init

from paddlets.utils import param_init, manager
from paddlets.datasets import TSDataset
from paddlets.models.utils import to_tsdataset
from paddlets.models.anomaly.dl.anomaly_base import AnomalyBaseModel
from paddlets.models.anomaly.dl import utils as U
from paddlets.models.forecasting.dl.revin import revin_norm
from paddlets.models.common.callbacks import Callback
from paddlets.models.forecasting.dl._nonstationary.EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from paddlets.models.forecasting.dl._nonstationary.MHSA_Family import DSAttention, AttentionLayer
from paddlets.models.forecasting.dl._nonstationary.Embed import DataEmbedding
from paddlets.logger import raise_if, raise_if_not, raise_log, Logger

zeros_ = paddle_init.Constant(value=0.)
ones_ = paddle_init.Constant(value=1.)

logger = Logger(__name__)


class Projector(paddle.nn.Layer):
    """
    MLP to learn the De-stationary factors
    """

    def __init__(self,
                 enc_in,
                 seq_len,
                 hidden_dims,
                 hidden_layers,
                 output_dim,
                 kernel_size=3):
        super(Projector, self).__init__()
        padding = 1 if paddle.__version__ >= '1.5.0' else 2
        self.series_conv = paddle.nn.Conv1D(
            in_channels=seq_len,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode='circular',
            bias_attr=False)
        layers = [
            paddle.nn.Linear(
                in_features=2 * enc_in, out_features=hidden_dims[0]),
            paddle.nn.ReLU()
        ]
        for i in range(hidden_layers - 1):
            layers += [
                paddle.nn.Linear(
                    in_features=hidden_dims[i],
                    out_features=hidden_dims[i + 1]), paddle.nn.ReLU()
            ]
        layers += [
            paddle.nn.Linear(
                in_features=hidden_dims[-1],
                out_features=output_dim,
                bias_attr=False)
        ]
        self.backbone = paddle.nn.Sequential(*layers)

    def forward(self, x, stats):
        batch_size = x.shape[0]
        x = self.series_conv(x)
        x = paddle.concat(x=[x, stats], axis=1)
        x = x.reshape([batch_size, -1])
        y = self.backbone(x)
        return y


class Nonstationary_Transformer_Module(paddle.nn.Layer):
    """
    Non-stationary Transformer implementation based on PaddlePaddle.

    The original article refers to
    Yong Liu, Haixu Wu, et al. "Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting."
    (https://openreview.net/pdf?id=ucNDIDRNjjv)
    """

    def __init__(self,
                 c_in=7,
                 seq_len=96,
                 pred_len=720,
                 label_len=48,
                 e_layers=2,
                 d_layers=1,
                 n_heads=8,
                 d_model=512,
                 d_ff=2048,
                 dropout=0.05,
                 embed='timeF',
                 freq='h',
                 factor=1,
                 p_hidden_dims=[256, 256],
                 activation='gelu',
                 p_hidden_layers=2,
                 output_attention=False,
                 pretrain=None):
        super(Nonstationary_Transformer_Module, self).__init__()
        self.pred_len = pred_len
        self.c_in = c_in
        c_out = c_in
        self.seq_len = seq_len
        self.label_len = label_len
        self.output_attention = output_attention
        self.enc_embedding = DataEmbedding(c_in, d_model, embed, freq, dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        DSAttention(
                            False,
                            factor,
                            attention_dropout=dropout,
                            output_attention=output_attention),
                        d_model,
                        n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation) for l in range(e_layers)
            ],
            norm_layer=paddle.nn.LayerNorm(
                normalized_shape=d_model,
                epsilon=1e-05,
                weight_attr=None,
                bias_attr=None))

        self.tau_learner = Projector(
            enc_in=c_in,
            seq_len=seq_len,
            hidden_dims=p_hidden_dims,
            hidden_layers=p_hidden_layers,
            output_dim=1)
        self.delta_learner = Projector(
            enc_in=c_in,
            seq_len=seq_len,
            hidden_dims=p_hidden_dims,
            hidden_layers=p_hidden_layers,
            output_dim=seq_len)
        self.projection = paddle.nn.Linear(
            in_features=d_model, out_features=c_out, bias_attr=True)
        self.pretrain = pretrain
        self.init_weight()

    def forward(self, x):
        x_enc = x["observed_cov_numeric"].cast('float32')
        # init forecast tensor
        x_raw = x_enc.clone().detach()
        mean_enc = x_enc.mean(axis=1, keepdim=True).detach()
        x_enc = x_enc - mean_enc
        std_enc = paddle.sqrt(x=paddle.var(
            x=x_enc, axis=1, keepdim=True, unbiased=False) + 1e-05).detach()
        x_enc = x_enc / std_enc

        tau = self.tau_learner(x_raw, std_enc).exp()
        delta = self.delta_learner(x_raw, mean_enc)
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(
            enc_out, attn_mask=None, tau=tau, delta=delta)

        dec_out = self.projection(enc_out)
        dec_out = dec_out * std_enc + mean_enc

        return dec_out, x["observed_cov_numeric"].cast('float32')

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
                if isinstance(layer, nn.Conv2D):
                    std = layer._kernel_size[0] * layer._kernel_size[
                        1] * layer._out_channels
                    std //= layer._groups
                    param_init.normal_init(layer.weight, std=std)
                elif isinstance(layer, (nn.BatchNorm, nn.SyncBatchNorm)):
                    param_init.constant_init(layer.weight, value=1.0)
                    param_init.constant_init(layer.bias, value=0.0)
                elif isinstance(layer, nn.LayerNorm):
                    zeros_(layer.bias)
                    ones_(layer.weight)
                elif isinstance(layer, nn.Linear):
                    param_init.th_linear_fill(layer)


@manager.MODELS.add_component
class NonStationary_AD(AnomalyBaseModel):
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
                 out_chunk_len: int=0,
                 c_in: int=7,
                 factor: int=3,
                 label_len: int=48,
                 p_hidden_dims: List[int]=[],
                 p_hidden_layers: int=2,
                 d_model: int=512,
                 d_ff: int=512,
                 use_revin: bool=False,
                 revin_params: Dict[str, Any]=dict(
                     eps=1e-5, affine=True),
                 pretrain=None,
                 skip_chunk_len: int=0,
                 sampling_stride: int=1,
                 loss_fn: Callable[..., paddle.Tensor]=F.mse_loss,
                 optimizer_fn: Callable[..., Optimizer]=paddle.optimizer.Adam,
                 anomaly_ratio: float=1,
                 threshold: Optional[float]=None,
                 threshold_coeff: float=1.0,
                 threshold_fn: Callable[..., float]=U.get_threshold,
                 anomaly_score_fn: Callable[..., List[float]]=None,
                 pred_adjust: bool=True,
                 pred_adjust_fn: Callable[..., np.ndarray]=U.result_adjust,
                 optimizer_params: Dict[str, Any]=dict(learning_rate=1e-3),
                 eval_metrics: List[str]=[],
                 callbacks: List[Callback]=[],
                 batch_size: int=32,
                 max_epochs: int=100,
                 verbose: int=1,
                 patience: int=10,
                 seed: Optional[int]=None,
                 **kwargs):
        self.c_in = c_in
        self.factor = factor
        self.label_len = label_len
        self.p_hidden_layers = p_hidden_layers
        self.p_hidden_dims = p_hidden_dims
        self.d_model = d_model
        self.d_ff = d_ff
        self.pretrain = pretrain
        self._anomaly_ratio = anomaly_ratio
        self._use_revin = use_revin
        self._revin_params = revin_params
        self._criterion = paddle.nn.functional.mse_loss

        super(NonStationary_AD, self).__init__(
            in_chunk_len=in_chunk_len,
            sampling_stride=sampling_stride,
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
            threshold=threshold,
            threshold_coeff=threshold_coeff,
            threshold_fn=threshold_fn,
            anomaly_score_fn=anomaly_score_fn,
            pred_adjust=pred_adjust,
            pred_adjust_fn=pred_adjust_fn,
            optimizer_params=optimizer_params,
            eval_metrics=eval_metrics,
            callbacks=callbacks,
            batch_size=batch_size,
            max_epochs=max_epochs,
            verbose=verbose,
            patience=patience,
            seed=seed, )

    def _update_fit_params(
            self,
            train_tsdataset: TSDataset,
            valid_tsdataset: Optional[TSDataset]=None) -> Dict[str, Any]:
        """Infer parameters by TSdataset automatically.

        Args:
            train_tsdataset(TSDataset): Train dataset.
            valid_tsdataset(TSDataset|None): Validation dataset.

        Returns:
            Dict[str, Any]: model parameters.
        """
        fit_params = {
            "observed_dim": train_tsdataset.get_observed_cov().data.shape[1]
        }
        return fit_params

    @revin_norm
    def _init_network(self) -> paddle.nn.Layer:
        """
        Init network.

        Returns:
            paddle.nn.Layer
        """
        return Nonstationary_Transformer_Module(
            c_in=self._fit_params['observed_dim'],
            seq_len=self._in_chunk_len,
            pred_len=self._in_chunk_len,
            factor=self.factor,
            label_len=self.label_len,
            d_model=self.d_model,
            d_ff=self.d_ff,
            p_hidden_dims=self.p_hidden_dims,
            p_hidden_layers=self.p_hidden_layers,
            pretrain=self.pretrain)

    def fit(self,
            train_tsdataset: TSDataset,
            valid_tsdataset: Optional[TSDataset]=None):
        """Train a neural network stored in self._network, 
            Using train_dataloader for training data and valid_dataloader for validation.

        Args: 
            train_tsdataset(TSDataset): Train set. 
            valid_tsdataset(TSDataset|None): Eval set, used for early stopping.
        """
        self._check_tsdataset(train_tsdataset)
        if valid_tsdataset is not None:
            self._check_tsdataset(valid_tsdataset)
        self._fit_params = self._update_fit_params(train_tsdataset,
                                                   valid_tsdataset)
        train_dataloader, valid_dataloaders = self._init_fit_dataloaders(
            train_tsdataset, valid_tsdataset)
        self._fit(train_dataloader, valid_dataloaders)

        # Get threshold
        if self._threshold is None:
            dataloader, valid_dataloaders = self._init_fit_dataloaders(
                train_tsdataset, valid_tsdataset, shuffle=False)
            self._threshold = self._get_threshold(dataloader, valid_dataloaders)

    @to_tsdataset(scenario="anomaly_label")
    def predict(self, tsdataset: TSDataset, **predict_kwargs) -> TSDataset:
        """Get anomaly label on a batch. the result are output as tsdataset.

        Args:
            tsdataset(TSDataset): Data to be predicted.
            **predict_kwargs: Additional arguments for `_predict`.

        Returns:
            TSDataset.
        """
        boundary = (len(tsdataset._observed_cov.data) - 1)
        dataloader = self._init_predict_dataloader(tsdataset,
                                                   (boundary, boundary))
        anomaly_score = self._get_anomaly_score(dataloader, **predict_kwargs)
        anomaly_score = np.concatenate(anomaly_score, axis=0).reshape(-1)
        anomaly_label = (anomaly_score >= self._threshold) + 0
        # adjust pred 

        return anomaly_label

    def _get_threshold(self,
                       train_dataloader: TSDataset,
                       val_dataloader: Optional[TSDataset]=None):
        """Get the threshold value to judge anomaly.
        
        Args:
            anomaly_score(np.ndarray): 
            
        Returns:
            float: Thresold value.
        """
        raise_if(train_dataloader is None,
                 f" Please pass in train_tsdataset to calculate the threshold.")
        logger.info(f"calculate threshold...")
        self._threshold = self._threshold_fn(
            self._network,
            train_dataloader,
            val_dataloader,
            anomaly_ratio=self._anomaly_ratio,
            criterion=self._criterion)
        logger.info(f"threshold is {self._threshold}")
        return self._threshold

    def _get_loss(self, y_pred: paddle.Tensor,
                  y_true: paddle.Tensor) -> np.ndarray:
        """Get the loss for anomaly label and anomaly score.

        Note:
            This function could be overrided by the subclass if necessary.

        Args:
            y_pred(paddle.Tensor): Estimated feature values.
            y_true(paddle.Tensor): Ground truth (correct) feature values.

        Returns:
            np.ndarray.

        """
        anomaly_criterion = paddle.nn.functional.mse_loss
        loss = paddle.mean(
            x=anomaly_criterion(
                y_true, y_pred, reduction='none'), axis=-1)
        return loss.numpy()
