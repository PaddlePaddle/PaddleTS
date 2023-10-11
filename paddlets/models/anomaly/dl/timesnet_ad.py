#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from typing import List, Dict, Any, Callable, Optional
import collections

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddlets.utils import param_init, manager
from paddle.optimizer import Optimizer
import paddle.nn.initializer as paddle_init

from paddlets.datasets import TSDataset
from paddlets.models.utils import to_tsdataset
from paddlets.models.anomaly.dl.anomaly_base import AnomalyBaseModel
from paddlets.models.common.callbacks import Callback
from paddlets.models.anomaly.dl import utils as U
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

        self.projection = nn.Linear(
            in_features=d_model, out_features=c_in, bias_attr=True)
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
                        .format(k, para_state_dict[k].shape, model_state_dict[
                            k].shape))
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

    def anomaly_detection(self, x_enc):
        means = x_enc.mean(axis=1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = paddle.sqrt(x=paddle.var(
            x=x_enc, axis=1, keepdim=True, unbiased=False) + 1e-05)
        x_enc /= stdev
        enc_out = self.enc_embedding(x_enc, None)
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        dec_out = self.projection(enc_out)
        dec_out = dec_out * stdev[:, (0), :].unsqueeze(axis=1).tile(
            repeat_times=[1, self.pred_len + self.seq_len, 1])
        dec_out = dec_out + means[:, (0), :].unsqueeze(axis=1).tile(
            repeat_times=[1, self.pred_len + self.seq_len, 1])
        return dec_out

    def forward(self, x):
        x_enc = x["observed_cov_numeric"].cast('float32')
        dec_out = self.anomaly_detection(x_enc)
        return dec_out, x["observed_cov_numeric"].cast('float32')


@manager.MODELS.add_component
class TimesNet_AD(AnomalyBaseModel):
    """Auto encoder network for anomaly detection.

    Args:
        in_chunk_len(int): The size of the loopback window, i.e. the number of time steps feed to the model.
        sampling_stride(int): Sampling intervals between two adjacent samples.
        loss_fn(Callable[..., paddle.Tensor]): Loss function.
        optimizer_fn(Callable[..., Optimizer]): Optimizer algorithm.
        threshold_fn(Callable[..., float]|None): The method to get anomaly threshold.
        q(float): The parameter used to calculate the quantile which range is [0, 100].
        threshold(float|None): The threshold to judge anomaly.
        anomaly_score_fn(Callable[..., List[float]]|None): The method to get anomaly score.
        pred_adjust(bool): Whether to adjust the pred label according to the real label.
        pred_adjust_fn(Callable[..., np.ndarray]|None): The method to adjust pred label.
        optimizer_params(Dict[str, Any]): Optimizer parameters.
        eval_metrics(List[str]): Evaluation metrics of model.
        callbacks(List[Callback]): Customized callback functions.
        batch_size(int): Number of samples per batch.
        max_epochs(int): Max epochs during training.
        verbose(int): Verbosity mode.
        patience(int): Number of epochs to wait for improvement before terminating.
        seed(int|None): Global random seed.

        ed_type(str): The type of encoder and decoder.
        activation(Callable[..., paddle.Tensor]): The activation function for the hidden layers.
        last_layer_activation(Callable[..., paddle.Tensor]): The activation function for the last layer.
        hidden_config(List[int]|None): The ith element represents the number of neurons in the ith hidden layer.
        kernel_size(int): Kernel size for Conv1D.
        dropout_rate(float): Dropout regularization parameter.
        use_bn(bool): Whether to use batch normalization.
        embedding_size(int): The size of each embedding vector.
        pooling: Whether to use average pooling to aggregate embeddings, if False, concat each embedding.

    Attributes:
        _in_chunk_len(int): The size of the loopback window, i.e. the number of time steps feed to the model.
        _sampling_stride(int): Sampling intervals between two adjacent samples.
        _loss_fn(Callable[..., paddle.Tensor]): Loss function.
        _optimizer_fn(Callable[..., Optimizer]): Optimizer algorithm.
        _threshold_fn(Callable[..., float]|None)): The method to get anomaly threshold.
        _q(float): The parameter used to calculate the quantile which range is [0, 100].
        _threshold(float|None): The threshold to judge anomaly.
        _anomaly_score_fn(Callable[..., List[float]]|None): The method to get anomaly score.
        _pred_adjust(bool): Whether to adjust the pred label according to the real label.
        _pred_adjust_fn(Callable[..., np.ndarray]|None): The method to adjust pred label.
        _optimizer_params(Dict[str, Any]): Optimizer parameters.
        _eval_metrics(List[str]): Evaluation metrics of model.
        _callbacks(List[Callback]): Customized callback functions.
        _batch_size(int): Number of samples per batch.
        _max_epochs(int): Max epochs during training.
        _verbose(int): Verbosity mode.
        _patience(int): Number of epochs to wait for improvement before terminating.
        _seed(int|None): Global random seed.
        _stop_training(bool): Training status.
        _ed_type(str): The type of encoder and decoder.
        _activation(Callable[..., paddle.Tensor]): The activation function for the hidden layers.
        _last_layer_activation(Callable[..., paddle.Tensor]): The activation function for the last layer.
        _hidden_config(List[int]|None): The ith element represents the number of neurons in the ith hidden layer.
        _kernel_size(int): Kernel size for Conv1D.
        _dropout_rate(float): Dropout regularization parameter.
        _use_bn(bool): Whether to use batch normalization.
        _embedding_size(int): The size of each embedding vector.
        _pooling(bool): Whether to use average pooling to aggregate embeddings, if False, concat each embedding.
    """

    def __init__(
            self,
            in_chunk_len: int,  # 96
            out_chunk_len: int=0,  # 96 192 336 720
            e_layers: int=2,
            c_in: int=7,
            d_model: int=32,
            embed: str='timeF',  # [timeF, fixed, learned]
            freq: str='h',
            dropout: float=0.1,
            c_out: int=7,
            d_ff: int=32,
            top_k: int=5,
            num_kernels: int=6,
            window_sampling_limit: int=None,
            use_revin: bool=False,
            revin_params: Dict[str, Any]=dict(
                eps=1e-5, affine=True),
            skip_chunk_len: int=0,
            sampling_stride: int=1,  # 采样间隔，每个样本之间的差距
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
            ed_type: str='MLP',
            activation: Callable[..., paddle.Tensor]=paddle.nn.ReLU,
            last_layer_activation: Callable[...,
                                            paddle.Tensor]=paddle.nn.Identity,
            use_bn: bool=False,
            hidden_config: List[int]=None,
            kernel_size: int=3,
            dropout_rate: float=0.2,
            embedding_size: int=16,
            pooling: bool=False, 
            renorm: bool=None,
            **kwargs):

        self._e_layers = e_layers
        self._c_in = c_in
        self._d_model = d_model
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
        self._hidden_config = (hidden_config if hidden_config else [32, 16])
        self._use_bn = use_bn
        self._kernel_size = kernel_size
        self._ed_type = ed_type
        self._activation = activation
        self._last_layer_activation = last_layer_activation
        self._dropout_rate = dropout_rate
        self._embedding_size = embedding_size
        self._pooling = pooling
        self._anomaly_ratio = anomaly_ratio
        self._criterion = paddle.nn.functional.mse_loss

        super(TimesNet_AD, self).__init__(
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

    def _init_network(self) -> paddle.nn.Layer:
        """Setup the network.

        Returns:
            paddle.nn.Layer.
        """
        return _TimesNet(
            in_chunk_len=self._in_chunk_len,
            out_chunk_len=0,
            e_layers=self._e_layers,
            c_in=self._fit_params["observed_dim"],
            d_model=self._d_model,
            embed=self._embed,
            freq=self._freq,
            dropout=self._dropout,
            d_ff=self._d_ff,
            top_k=self._top_k,
            num_kernels=self._num_kernels)

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
            self._threshold = self._get_threshold(
                dataloader, valid_dataloaders)

    @to_tsdataset(scenario="anomaly_label")
    def predict(self, tsdataset: TSDataset, **predict_kwargs) -> TSDataset:
        """Get anomaly label on a batch. the result are output as tsdataset.

        Args:
            tsdataset(TSDataset): Data to be predicted.
            **predict_kwargs: Additional arguments for `_predict`.

        Returns:
            TSDataset.
        """
        boundary = (len(tsdataset._observed_cov.data) - 1 )
        dataloader = self._init_predict_dataloader(tsdataset, (boundary, boundary))
        anomaly_score = self._get_anomaly_score(dataloader, **predict_kwargs)
        anomaly_score = np.concatenate(anomaly_score, axis=0).reshape(-1)
        anomaly_label = (anomaly_score >= self._threshold) + 0
        # adjust pred 
        
        return anomaly_label
            
    def _get_threshold(self, 
            train_dataloader: TSDataset,
            val_dataloader: Optional[TSDataset]=None) :
        """Get the threshold value to judge anomaly.
        
        Args:
            anomaly_score(np.ndarray): 
            
        Returns:
            float: Thresold value.
        """
        raise_if(
            train_dataloader is None,
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
