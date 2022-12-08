#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from typing import List, Dict, Any, Callable, Optional

from paddle.optimizer import Optimizer
import paddle.nn.functional as F
import numpy as np
import paddle

from paddlets.models.anomaly.dl.anomaly_base import AnomalyBaseModel
from paddlets.models.anomaly.dl._ed.ed import MLP, CNN, LSTM
from paddlets.models.common.callbacks import Callback
from paddlets.models.anomaly.dl import utils as U
from paddlets.datasets import TSDataset
from paddlets.logger import raise_if, raise_if_not


class stack(paddle.nn.Layer):
    """stack structure.
    
    Args:
        in_chunk_dim(int): The size of the loopback window, i.e. the number of time steps feed to the model.
        hidden_config(List[int]): The ith element represents the number of neurons in the ith hidden layer.
        feature_dim(int): The numer of feature.
        is_encoder(bool): Encoder or Decoder.
        base_nn(str): base network for stack. 
        use_bn(bool): Whether to use batch normalization.
        use_drop(bool): Whether to use dropout.
        dropout_rate(float): probability of an element to be zeroed.
        kernel_size(int): Size of the convolving kernel.
        rnn_num_layers(int): Number of recurrent layers.
        direction(str): If True, becomes a bidirectional LSTM. Default: False.
        activation(Callable[..., paddle.Tensor]): The activation function for the hidden layers.
        last_layer_activation(Callable[..., paddle.Tensor]): The activation function for the last layers.

    Attributes:
        _nn(paddle.nn.Sequential): Dynamic graph LayerList.
    
    """
    def __init__(
        self,
        in_chunk_dim: int, 
        hidden_config: List[int], 
        feature_dim: int, 
        is_encoder: bool = True, 
        base_nn: str = 'MLP',
        use_bn: bool = True, 
        use_drop: bool = True, 
        dropout_rate: float = 0.5,
        kernel_size: int = 1, 
        rnn_num_layers: int = 1, 
        direction: str = 'forward',
        activation: Callable[..., paddle.Tensor] = paddle.nn.ReLU6, 
        last_layer_activation: Callable[..., paddle.Tensor] = paddle.nn.ReLU6,
                ):
        super(stack, self).__init__()
        if not is_encoder:
            hidden_config = [int(i) for i in reversed(hidden_config)]
        if  base_nn=='MLP':
            self._nn = MLP(input_dim=feature_dim, feature_dim=in_chunk_dim, hidden_config=hidden_config,
                           activation=activation, last_layer_activation=last_layer_activation,
                           dropout_rate=dropout_rate, use_bn=use_bn, use_drop=use_drop)
        elif base_nn=='LSTM':
            self._nn = LSTM(input_dim=feature_dim, hidden_config=hidden_config,
                            num_layers=rnn_num_layers, direction=direction, activation=activation,
                            last_layer_activation=last_layer_activation, dropout_rate=dropout_rate, 
                            use_drop=use_drop)
        elif base_nn=='CNN':
            self._nn = CNN(input_dim=feature_dim, hidden_config=hidden_config, activation=activation,
                           last_layer_activation=last_layer_activation, kernel_size=kernel_size,
                           use_drop=use_drop, dropout_rate=dropout_rate, use_bn=use_bn, 
                           is_encoder=True, data_format='NLC')
    def forward(self, x):
        return self._nn(x)


class _VAEBlock(paddle.nn.Layer):
    """VAE Network structure.

    Args:
        in_chunk_dim(int): The size of the loopback window, i.e. the number of time steps feed to the model.
        hidden_config(List[int]): The ith element represents the number of neurons in the ith hidden layer.
        feature_dim(int): The numer of feature.
        base_en(str): base nn in encoder.
        base_de(str): base nn in decoder.
        use_bn(bool): Whether to use batch normalization.
        use_drop(bool): Whether to use dropout.
        dropout_rate(float): probability of an element to be zeroed.
        kernel_size(int): Size of the convolving kernel.
        rnn_num_layers(int): Number of recurrent layers.
        direction(str): If True, becomes a bidirectional LSTM. Default: False.
        activation(Callable[..., paddle.Tensor]): The activation function for the hidden layers.
        last_layer_activation(Callable[..., paddle.Tensor]): The activation function for the last layers.
        stdev(int): param for reparameterize.

    Attributes:
        _nn(paddle.nn.Sequential): Dynamic graph LayerList.
    """
    def __init__(
        self, 
        in_chunk_dim: int, 
        hidden_config: List[int], 
        feature_dim: int, 
        base_en: str = 'MLP', 
        base_de: str = 'MLP', 
        use_bn: bool = True, 
        use_drop: bool = True, 
        dropout_rate: float = 0.5,
        kernel_size: int = 1, 
        rnn_num_layers: int = 1, 
        direction: str = 'forward',
        activation: Callable[..., paddle.Tensor] = paddle.nn.ReLU6, 
        last_layer_activation: Callable[..., paddle.Tensor] = paddle.nn.ReLU6,
        stdev: float = 0.1,
                ):
        super(_VAEBlock, self).__init__()
        raise_if_not(
            base_en in ("MLP", "CNN", "LSTM"), 
            "`base_en` must be in 'MLP', 'CNN', 'LSTM'."
        )
        raise_if_not(
            base_de in ("MLP", "CNN", "LSTM"), 
            "`base_de` must be in 'MLP', 'CNN', 'LSTM'."
        )
        raise_if(
            len(hidden_config) <= 0,
            f"length of hidden_config must be > 0, got {hidden_config}."
        )
        raise_if(
            np.any(np.array(hidden_config) <= 0),
            f"anyone value in hidden_config must be > 0, got {hidden_config}."
        )
        self.stdev = stdev
        self.de_hidden_config = [int(i) for i in reversed(hidden_config)]
        self.encoder = stack(in_chunk_dim, hidden_config, feature_dim, is_encoder=True, base_nn=base_en,
                             use_bn=use_bn, use_drop=use_drop,dropout_rate=dropout_rate, kernel_size=kernel_size, 
                             rnn_num_layers=rnn_num_layers, direction=direction, activation=paddle.nn.ReLU6, 
                             last_layer_activation=paddle.nn.ReLU6)
        self.decoder = stack(in_chunk_dim, hidden_config, feature_dim, is_encoder=False, base_nn=base_de,
                             use_bn=use_bn, use_drop=use_drop, dropout_rate=dropout_rate, kernel_size=kernel_size, 
                             rnn_num_layers=rnn_num_layers, direction=direction, activation=paddle.nn.ReLU6, 
                             last_layer_activation=paddle.nn.ReLU6)
        # reparameterize
        self.mu = paddle.nn.Linear(hidden_config[-1], feature_dim)
        self.logvar = paddle.nn.Linear(hidden_config[-1], feature_dim)
        # reconstructed
        self.reconstructed = paddle.nn.Linear(hidden_config[0], feature_dim)

    def reparameterize(
        self, 
        mu: paddle.Tensor, 
        logvar: paddle.Tensor
    ):
        """The reparameterisation trick allows us to backpropagate through the encoder.

        Args:
            mu(paddle.tensor): output from self.mu.
            logvar(paddle.tensor): output from self.logvar.

        Return:
            mu(paddle.tensor): Generator from model.
        """
        if self.training:
            std = paddle.exp(0.5 * logvar)
            eps = paddle.randn(std.shape, ) * self.stdev
            return eps * std + mu
        else:
            return mu

    def forward(self, x):
        """Forward.

        Args:
            X(paddle.Tensor): Dict of feature tensor.

        Returns:
            paddle.Tensor: Output of model.
        """
        # encoder
        x = x["observed_cov_numeric"]
        h = self.encoder(x)
        # reparameterize
        mu_ = self.mu(h)
        logvar_ = self.logvar(h)
        z = self.reparameterize(mu_, logvar_)
        # decoder
        z = self.decoder(z)
        recon = self.reconstructed(z)
        return [recon, mu_, logvar_, x] 


class VAE(AnomalyBaseModel):
    """VAE network for anomaly detection.

    Args:
        in_chunk_len(int): The size of the loopback window, i.e. the number of time steps feed to the model.
        sampling_stride(int): Sampling intervals between two adjacent samples.
        loss_fn(Callable[..., paddle.Tensor]): Loss function.
        optimizer_fn(Callable[..., Optimizer]): Optimizer algorithm.
        threshold_fn(Callable[..., float]|None): The method to get anomaly threshold.
        threshold_coeff(float): The coefficient of threshold.
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
        hidden_config(List[int]|None):The ith element represents the number of neurons in the ith hidden layer.
        base_en(str): The type of encoder.
        base_de(str): The type of decoder.
        use_bn:Whether to use batch normalization.
        use_drop: Whether to use dropout.
        dropout_rate(float):probability of an element to be zeroed.
        kernel_size(int): Size of the convolving kernel.
        rnn_num_layers(int): Number of recurrent layers.
        direction(str):If True, becomes a bidirectional LSTM. Default: False.
        activation(Callable[..., paddle.Tensor]): The activation function for the hidden layers.
        last_layer_activation(Callable[..., paddle.Tensor]): The activation function for the last layers.
        stdev(int): param for reparameterize.

    Attributes:
        _in_chunk_len(int): The size of the loopback window, i.e. the number of time steps feed to the model.
        _sampling_stride(int): Sampling intervals between two adjacent samples.
        _loss_fn(Callable[..., paddle.Tensor]): Loss function.
        _optimizer_fn(Callable[..., Optimizer]): Optimizer algorithm.
        _threshold_fn(Callable[..., float]|None): The method to get anomaly threshold.
        _threshold_coeff(float): The coefficient of threshold.
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
        _hidden_config(List[int]|None):The ith element represents the number of neurons in the ith hidden layer.
        _base_en(str): The type of encoder.
        _base_de(str): The type of decoder.
        _use_bn(bool):Whether to use batch normalization.
        _use_drop(bool): Whether to use dropout.
        _dropout_rate(float):probability of an element to be zeroed.
        _kernel_size(int): Size of the convolving kernel.
        _rnn_num_layers(int): Number of recurrent layers.
        _direction(str): If True, becomes a bidirectional LSTM. Default: False.
        _activation(Callable[..., paddle.Tensor]): The activation function for the hidden layers.
        _last_layer_activation(Callable[..., paddle.Tensor]): The activation function for the last layers.
        _stdev(int): param for reparameterize.
    """
    def __init__(
        self,
        in_chunk_len: int,
        sampling_stride: int = 1,
        loss_fn: Callable[..., paddle.Tensor] = U.smooth_l1_loss_vae,
        optimizer_fn: Callable[..., Optimizer] = paddle.optimizer.Adam,
        threshold_fn: Callable[..., float] = U.percentile,
        threshold: Optional[float] = None,
        threshold_coeff: float = 1.0,
        anomaly_score_fn: Callable[..., List[float]] = None,
        pred_adjust: bool = False,
        pred_adjust_fn: Callable[..., np.ndarray] = U.result_adjust,
        optimizer_params: Dict[str, Any] = dict(learning_rate=1e-4),
        eval_metrics: List[str] = [], 
        callbacks: List[Callback] = [], 
        batch_size: int = 32,
        max_epochs: int = 100,
        verbose: int = 1,
        patience: int = 10,
        seed: Optional[int] = None,
        hidden_config: List[int]= [32, 16], 
        base_en: str = 'MLP', 
        base_de: str = 'MLP', 
        use_bn: bool = True, 
        use_drop: bool = True, 
        dropout_rate: float = 0.5,
        kernel_size: int = 1, 
        rnn_num_layers: int = 1, 
        direction: str = 'forward',
        activation: Callable[..., paddle.Tensor] = paddle.nn.ReLU6, 
        last_layer_activation: Callable[..., paddle.Tensor] = paddle.nn.ReLU6,
        stdev: float = 0.1, 
        ):
        self._hidden_config = hidden_config
        self._in_chunk_len = in_chunk_len
        self._base_en = base_en
        self._base_de = base_de
        self._use_bn = use_bn
        self._use_drop = use_drop
        self._dropout_rate = dropout_rate
        self._kernel_size = kernel_size
        self._rnn_num_layers = rnn_num_layers
        self._direction = direction
        self._activation = activation
        self._last_layer_activation = last_layer_activation
        self._stdev = stdev

        super(VAE, self).__init__(
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
        return _VAEBlock(
        self._in_chunk_len,
        self._hidden_config,
        self._fit_params["observed_dim"],
        self._base_en,
        self._base_de,
        self._use_bn,
        self._use_drop,
        self._dropout_rate,
        self._kernel_size,
        self._rnn_num_layers,
        self._direction,
        self._activation,
        self._last_layer_activation,
        self._stdev
        )
    
    def _train_batch(
        self, 
        X: Dict[str, paddle.Tensor]
    ) -> Dict[str, Any]:
        """Trains one batch of data.

        Args:
            X(Dict[str, paddle.Tensor]): Dict of feature tensor.
            y(paddle.Tensor): Target tensor.

        Returns:
            Dict[str, Any]: Dict of logs.
        """
        output = self._network(X)
        loss = self._compute_loss(output)
        loss.backward()
        self._optimizer.step()
        self._optimizer.clear_grad()
        batch_logs = {
            "batch_size": output[-1].shape[0],
            "loss": loss.item()
        }
        return batch_logs

    def _predict_batch(
        self, 
        X: paddle.Tensor
    ) -> np.ndarray:
        """Predict one batch of data.

        Args: 
            X(paddle.Tensor): Feature tensor.

        Returns:
            np.ndarray: Prediction results.
        """
        recon, mu_, logvar_, x = self._network(X)
        return x.numpy(), recon.numpy()

    def _predict(
        self, 
        dataloader: paddle.io.DataLoader
    ) -> np.ndarray:
        """Predict function core logic.

        Args:
            dataloader(paddle.io.DataLoader): Data to be predicted.

        Returns:
            np.ndarray.
        """
        self._network.eval()
        loss_list = []
        for batch_nb, data in enumerate(dataloader):
            recon, mu_, logvar_, x = self._network(data)
            loss = self._get_loss(recon, x)
            loss_list.extend(loss)

        return np.array(loss_list)
    
    def _compute_loss(
        self, 
        output: List[paddle.Tensor], 
    ) -> paddle.Tensor:
        """Compute the loss.

        Args:
            output(list[paddle.Tensor]): Model ouput.

        Returns:
            paddle.Tensor: Loss value.
        """
        return self._loss_fn(output)
