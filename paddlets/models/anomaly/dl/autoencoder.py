#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from typing import List, Dict, Any, Callable, Optional

from paddle.optimizer import Optimizer
import paddle.nn.functional as F
import numpy as np
import paddle

from paddlets.models.anomaly.dl.anomaly_base import AnomalyBaseModel
from paddlets.models.anomaly.dl._ed.ed import MLP, CNN
from paddlets.models.common.callbacks import Callback
from paddlets.models.anomaly.dl import utils as U
from paddlets.datasets import TSDataset
from paddlets.logger import raise_if, raise_if_not

       
class _AEBlock(paddle.nn.Layer):
    """AE Network structure.

    Args:
        in_chunk_len(int): The size of the loopback window, i.e. the number of time steps feed to the model.
        feature_dim(int): The numer of feature.
        mlp_hidden_config(List[int]): The ith element represents the number of neurons in the ith hidden layer for mlp.
        cnn_hidden_config(List[int]): The ith element represents the number of neurons in the ith hidden layer for cnn.
        activation(Callable[..., paddle.Tensor]): The activation function for the hidden layers.
        last_layer_activation(Callable[..., paddle.Tensor]): The activation function for the last layer.
        kernel_size(int): Kernel size for Conv1D.
        dropout_rate(float): Dropout regularization parameter.
        use_bn(bool): Whether to use batch normalization.
        
    Attributes:
        _nn(paddle.nn.Sequential): Dynamic graph LayerList.
    """
    def __init__(
        self,
        in_chunk_len: int,
        ed_type: str,
        feature_dim: int,
        hidden_config: List[int],
        activation: Callable[..., paddle.Tensor],
        last_layer_activation: Callable[..., paddle.Tensor],
        kernel_size: int,
        dropout_rate: float,
        use_bn: bool
    ):
        super(_AEBlock, self).__init__()
        raise_if_not(
            ed_type in ("MLP", "CNN"), 
            "`ae_type` must be either 'MLP' or 'CNN'"
        )
        raise_if(
            np.any(np.array(hidden_config) <= 0),
            f"hidden_config must be > 0, got {hidden_config}."
        )
        if ed_type == 'MLP':
            self._encoder = MLP(in_chunk_len, feature_dim, hidden_config, \
                               activation, last_layer_activation, dropout_rate, use_bn)
            self._decoder = MLP(hidden_config[-1], feature_dim, hidden_config[::-1][1:] + [in_chunk_len], \
                               activation, last_layer_activation, dropout_rate, use_bn)
        elif ed_type == 'CNN':
            for i in range(len(hidden_config)):
                out_chunk_len = in_chunk_len - kernel_size + 1
                raise_if(
                    out_chunk_len < 1,
                    "Conv1d output size must be greater than or equal to 1, "
                    "Please choose a smaller `kernel_size` or bigger `in_chunk_len`"
                )
                in_chunk_len = out_chunk_len
            self._encoder = CNN(feature_dim, hidden_config, activation, last_layer_activation, kernel_size, \
                                dropout_rate, use_bn, is_encoder=True)
            self._decoder = CNN(hidden_config[-1], hidden_config[::-1][1:] + [feature_dim], \
                                activation, last_layer_activation, kernel_size, dropout_rate, use_bn, is_encoder=False)
    def forward(
        self, 
        X: Dict[str, paddle.Tensor]
    ) -> paddle.Tensor:
        """Forward.

        Args: 
            X(paddle.Tensor): Dict of feature tensor.

        Returns:
            paddle.Tensor: Output of model.
        """        
        X = paddle.transpose(X["observed_cov_numeric"], perm=[0, 2, 1])
        X = self._encoder(X)
        X = self._decoder(X)
        
        return paddle.transpose(X, perm=[0, 2, 1])
        

class AutoEncoder(AnomalyBaseModel):
    """Auto encoder network for anomaly detection.

    Args:
        in_chunk_len(int): The size of the loopback window, i.e. the number of time steps feed to the model.
        sampling_stride(int): Sampling intervals between two adjacent samples.
        loss_fn(Callable[..., paddle.Tensor]): Loss function.
        optimizer_fn(Callable[..., Optimizer]): Optimizer algorithm.
        thresold_fn(Callable[..., float]|None): The method to get anomaly thresold.
        thresold(float|None): The thresold to judge anomaly.
        anomaly_score_fn(Callable[..., List[float]]|None): The method to get anomaly score.
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

    Attributes:
        _in_chunk_len(int): The size of the loopback window, i.e. the number of time steps feed to the model.
        _sampling_stride(int): Sampling intervals between two adjacent samples.
        _loss_fn(Callable[..., paddle.Tensor]): Loss function.
        _optimizer_fn(Callable[..., Optimizer]): Optimizer algorithm.
        _thresold_fn(Callable[..., float]|None)): The method to get anomaly thresold.
        _thresold(float|None): The thresold to judge anomaly.
        _anomaly_score_fn(Callable[..., List[float]]|None): The method to get anomaly score.
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
    """
    def __init__(
        self,
        in_chunk_len: int,
        sampling_stride: int = 1,
        loss_fn: Callable[..., paddle.Tensor] = F.mse_loss,
        optimizer_fn: Callable[..., Optimizer] = paddle.optimizer.Adam,
        thresold_fn: Callable[..., float] = U.max,
        thresold: Optional[float] = None,
        thresold_coeff: float = 1.0,
        anomaly_score_fn: Callable[..., List[float]] = None,
        optimizer_params: Dict[str, Any] = dict(learning_rate=1e-3),
        eval_metrics: List[str] = [], 
        callbacks: List[Callback] = [], 
        batch_size: int = 32,
        max_epochs: int = 100,
        verbose: int = 1,
        patience: int = 10,
        seed: Optional[int] = None,

        ed_type: str = 'MLP',
        activation: Callable[..., paddle.Tensor] = paddle.nn.ReLU,
        last_layer_activation: Callable[..., paddle.Tensor] = paddle.nn.Identity,
        use_bn: bool = False,
        hidden_config: List[int] = None,
        kernel_size: int = 3,
        dropout_rate: float = 0.2,

    ):
        self._hidden_config = (
            hidden_config if hidden_config else [32, 16]
        )
        self._use_bn = use_bn
        self._kernel_size = kernel_size
        self._ed_type = ed_type
        self._activation = activation
        self._last_layer_activation = last_layer_activation
        self._dropout_rate = dropout_rate

        super(AutoEncoder, self).__init__(
            in_chunk_len=in_chunk_len, 
            sampling_stride=sampling_stride,
            loss_fn=loss_fn, 
            optimizer_fn=optimizer_fn,
            thresold=thresold,
            thresold_coeff=thresold_coeff,
            thresold_fn=thresold_fn,
            anomaly_score_fn=anomaly_score_fn,
            optimizer_params=optimizer_params, 
            eval_metrics=eval_metrics, 
            callbacks=callbacks, 
            batch_size=batch_size, 
            max_epochs=max_epochs, 
            verbose=verbose, 
            patience=patience, 
            seed=seed,
        )
        
    def _update_fit_params(
        self,
        train_tsdataset: TSDataset,
        valid_tsdataset: Optional[TSDataset] = None
    ) -> Dict[str, Any]:
        """Infer parameters by TSdataset automatically.

        Args:
            train_tsdataset(TSDataset): train dataset.
            valid_tsdataset(TSDataset|None): validation dataset.

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
        return _AEBlock(
            self._in_chunk_len,
            self._ed_type,
            self._fit_params["observed_dim"],
            self._hidden_config,
            self._activation,
            self._last_layer_activation,
            self._kernel_size,
            self._dropout_rate,
            self._use_bn
        )
