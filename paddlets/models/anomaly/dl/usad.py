#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
This implementation is based on the article `USAD : UnSupervised Anomaly Detection on Multivariate Time Series <https://dl.acm.org/doi/pdf/10.1145/3394486.3403392>`_ .

Base model features
    Based on adversely trained autoencoders.

"""

from typing import List, Dict, Any, Callable, Optional
import collections

from paddle.optimizer import Optimizer
import paddle.nn.functional as F
import numpy as np
import paddle

from paddlets.models.data_adapter import DataAdapter
from paddlets.models.anomaly.dl.anomaly_base import AnomalyBaseModel
from paddlets.models.anomaly.dl._ed.ed import MLP, CNN
from paddlets.models.common.callbacks import Callback
from paddlets.models.anomaly.dl import utils as U
from paddlets.datasets import TSDataset
from paddlets.logger import raise_if, raise_if_not
from paddlets.models.utils import to_tsdataset

       
class _USAModule(paddle.nn.Layer):
    """USAD Network structure.

    Args:
        in_chunk_len(int): The size of the loopback window, i.e. the number of time steps feed to the model.
        ed_type(str): The type of encoder and decoder ("MLP" or "CNN").
        fit_params(dict): The parameters for fitting, including dimensions and dict sizes of variables.
        hidden_config(List[int]): The ith element represents the number of neurons in the ith hidden layer.
        activation(Callable[..., paddle.Tensor]): The activation function for the hidden layers.
        last_layer_activation(Callable[..., paddle.Tensor]): The activation function for the last layer.
        kernel_size(int): Kernel size for Conv1D.
        dropout_rate(float): Dropout regularization parameter.
        use_bn(bool): Whether to use batch normalization.
        embedding_size(int): The size of each one-dimension embedding vector.
        pooling(bool): Whether to use average pooling to aggregate embeddings, if False, concat each embedding.

    """
    def __init__(
        self,
        in_chunk_len: int,
        ed_type: str,
        fit_params: Dict[str, Any],
        hidden_config: List[int],
        activation: Callable[..., paddle.Tensor],
        last_layer_activation: Callable[..., paddle.Tensor],
        kernel_size: int,
        dropout_rate: float,
        use_bn: bool,
        embedding_size: int,
        pooling: bool,
        flatten: bool,
    ):
        super().__init__()
        raise_if_not(
            ed_type in ("MLP", "CNN"), 
            "`ed_type` must be either 'MLP' or 'CNN'"
        )
        raise_if(
            np.any(np.array(hidden_config) <= 0),
            f"hidden_config must be > 0, got {hidden_config}."
        )
        # embedding cate feature
        self._pooling = pooling
        self._cat_size = len(fit_params['observed_cat_cols'])
        self._cat_dim = 0
        self._num_dim = fit_params['observed_num_dim']
        self._in_chunk_len = in_chunk_len
        self._flatten = flatten if ed_type == 'MLP' else False
        if fit_params['observed_cat_cols']:
            observed_cat_cols = fit_params['observed_cat_cols']
            self._observed_cat_emb = []
            for _, col_size in observed_cat_cols.items():
                self._observed_cat_emb.append(paddle.nn.Embedding(col_size, embedding_size))
            if pooling:
                self._cat_dim = embedding_size
            else:
                self._cat_dim = embedding_size * len(observed_cat_cols)
                
        feature_dim = self._num_dim + self._cat_dim
        self._feature_dim = feature_dim
        if ed_type == 'MLP':
            if self._flatten:
                in_chunk_len = in_chunk_len * feature_dim
                use_bn = False
            self._encoder = MLP(in_chunk_len, feature_dim, hidden_config, \
                               activation, activation, dropout_rate, use_bn)
            self._decoder1 = MLP(hidden_config[-1], feature_dim, hidden_config[::-1][1:] + [in_chunk_len], \
                               activation, last_layer_activation, dropout_rate, use_bn)
            self._decoder2 = MLP(hidden_config[-1], feature_dim, hidden_config[::-1][1:] + [in_chunk_len], \
                               activation, last_layer_activation, dropout_rate, use_bn)
        elif ed_type == 'CNN':
            for _ in range(len(hidden_config)):
                out_chunk_len = in_chunk_len - kernel_size + 1
                raise_if(
                    out_chunk_len < 1,
                    "Conv1d output size must be greater than or equal to 1, "
                    "Please choose a smaller `kernel_size` or bigger `in_chunk_len`"
                )
                in_chunk_len = out_chunk_len
            self._encoder = CNN(feature_dim, hidden_config, activation, last_layer_activation, kernel_size, \
                                dropout_rate, use_bn, is_encoder=True)
            self._decoder1 = CNN(hidden_config[-1], hidden_config[::-1][1:] + [feature_dim], \
                                activation, last_layer_activation, kernel_size, dropout_rate, use_bn, is_encoder=False)
            self._decoder2 = CNN(hidden_config[-1], hidden_config[::-1][1:] + [feature_dim], \
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
        x = paddle.transpose(X["observed_cov_numeric"], perm=[0, 2, 1])
        if self._cat_size > 0:
            feature_cat = []
            observed_cat = paddle.transpose(X["observed_cov_categorical"], perm=[0, 2, 1])
            for i in range(self._cat_size):
                feature_cat.append(self._observed_cat_emb[i](observed_cat[:, i]))
            if self._pooling:
                feature_cat = paddle.stack(feature_cat, axis=-1).mean(axis=-1)
            else:
                feature_cat = paddle.concat(feature_cat, axis=-1)
            feature_cat = paddle.transpose(feature_cat, perm=[0, 2, 1])
            x = paddle.concat([x, feature_cat], axis=-2)
        
        if self._flatten:
            x = paddle.reshape(x, [paddle.shape(x)[0], -1])

        z = self._encoder(x)
        w1 = self._decoder1(z)
        w2 = self._decoder2(z)
        w3 = self._decoder2(self._encoder(w1))
    
        return x, w1, w2, w3
        

class USAD(AnomalyBaseModel):
    """USAD model for anomaly detection.

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
        flatten(bool): Whether to flatten the in_chunk_len and feature_dim.

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
        _flatten(bool): Whether to flatten the in_chunk_len and feature_dim.
    """
    def __init__(
        self,
        in_chunk_len: int,
        sampling_stride: int = 1,
        loss_fn: Callable[..., paddle.Tensor] = F.mse_loss,
        optimizer_fn: Callable[..., Optimizer] = paddle.optimizer.Adam,
        threshold_fn: Callable[..., float] = U.percentile,
        q: float = 100,
        threshold: Optional[float] = None,
        threshold_coeff: float = 1.0,
        anomaly_score_fn: Callable[..., List[float]] = None,
        pred_adjust: bool = False,
        pred_adjust_fn: Callable[..., np.ndarray] = U.result_adjust,
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
        last_layer_activation: Callable[..., paddle.Tensor] = paddle.nn.Sigmoid,
        use_bn: bool = False,
        hidden_config: List[int] = None,
        kernel_size: int = 3,
        dropout_rate: float = 0.2,
        embedding_size: int = 16,
        pooling: bool = False,
        flatten: bool = True,
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
        self._embedding_size = embedding_size
        self._pooling = pooling
        self._q = q
        self._flatten = flatten

        super().__init__(
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
        train_df = train_tsdataset.to_dataframe()
        observed_cat_cols = collections.OrderedDict()
        observed_num_cols = []
        observed_train_tsdataset = train_tsdataset.get_observed_cov()
        observed_dtypes = dict(observed_train_tsdataset.dtypes)
        for col in observed_train_tsdataset.columns:
            if np.issubdtype(observed_dtypes[col], np.integer):
                observed_cat_cols[col] = len(train_df[col].unique())
            else:
                observed_num_cols.append(col)
        
        fit_params = {
            "observed_cat_cols": observed_cat_cols,
            "observed_num_dim": len(observed_num_cols),
            "observed_cat_dim": len(observed_cat_cols),
        }
        return fit_params
        
    def _init_network(self) -> paddle.nn.Layer:
        """Setup the network.

        Returns:
            paddle.nn.Layer.
        """
        return _USAModule(
            self._in_chunk_len,
            self._ed_type,
            self._fit_params,
            self._hidden_config,
            self._activation,
            self._last_layer_activation,
            self._kernel_size,
            self._dropout_rate,
            self._use_bn,
            self._embedding_size,
            self._pooling,
            self._flatten,
        )
    
    def _get_threshold(
        self,
        anomaly_score: np.ndarray
    ) -> float:
        """Get the threshold value to judge anomaly.
        
        Args:
            anomaly_score(np.ndarray): 
            
        Returns:
            float: Thresold value.
        """
        return self._threshold_fn(anomaly_score, self._q)

    def _init_optimizer(self) -> Optimizer:
        """Setup optimizer.

        Returns:
            Optimizer.
        """
        opt1 = self._optimizer_fn(
            **self._optimizer_params,
            parameters=(self._network._encoder.parameters() + self._network._decoder1.parameters())
        )
        opt2 = self._optimizer_fn(
            **self._optimizer_params,
            parameters=(self._network._encoder.parameters() + self._network._decoder2.parameters())
        )
        return (opt1, opt2)

    def _predict(
        self, 
        dataloader: paddle.io.DataLoader,
        alpha: float = .5,
        beta: float = .5, 
    ) -> np.ndarray:
        """Predict function core logic.

        Args:
            dataloader(paddle.io.DataLoader): Data to be predicted.
            alpha(float): Alpha parameter for usad anomaly score.
            beta(float): Beta parameter for uasd anomaly score.

        Returns:
            np.ndarray.
        """
        self._network.eval()
        loss_list = []
        for _, data in enumerate(dataloader):
            y_true, w1, _, w3 = self._network(data)
            loss = alpha * self._get_loss(w1, y_true) + beta * self._get_loss(w3, y_true)
            loss_list.extend(loss)
            
        return np.array(loss_list)

    def _get_loss(
        self,
        y_pred: paddle.Tensor,
        y_true: paddle.Tensor
    ) -> np.ndarray:
        """Get the loss for anomaly label and anomaly score.

        Note:
            This function could be overrided by the subclass if necessary.

        Args:
            y_pred(paddle.Tensor): Estimated feature values.
            y_true(paddle.Tensor): Ground truth (correct) feature values.

        Returns:
            np.ndarray.

        """
        if self._flatten and self._ed_type == 'MLP':
            loss = paddle.mean(paddle.square(paddle.subtract(y_pred, y_true)), axis=1, keepdim=False)
        else:
            loss = paddle.mean(paddle.square(paddle.subtract(y_pred, y_true)), axis=[1, 2], keepdim=False)
        return loss.numpy()

    def _train_epoch(
        self, 
        train_loader: paddle.io.DataLoader
    ):
        """Trains one epoch of the network in self._network.

        Args: 
            train_loader(paddle.io.DataLoader): Training dataloader.
        """
        self._network.train()
        for batch_idx, data in enumerate(train_loader):
            self._callback_container.on_batch_begin(batch_idx)
            batch_logs = self._train_batch(data, batch_idx)
            self._callback_container.on_batch_end(batch_idx, batch_logs)
        epoch_logs = {"lr": self._optimizer[0].get_lr()}
        self._history._epoch_metrics.update(epoch_logs)
    
    def _train_batch(
        self, 
        X: Dict[str, paddle.Tensor],
        batch_idx: int
    ) -> Dict[str, Any]:
        """Trains one batch of data.

        Args:
            X(Dict[str, paddle.Tensor]): Dict of feature tensor.
            batch_idx(int): Index of current batch.

        Returns:
            Dict[str, Any]: Dict of logs.
        """
        batch_idx = batch_idx + 1
        y_true, w1, w2, w3 = self._network(X)
        loss1 = 1 / batch_idx * self._compute_loss(y_true, w1) + (1 - 1/batch_idx) * self._compute_loss(y_true, w3)
        loss1.backward()
        self._optimizer[0].step()
        self._optimizer[0].clear_grad()

        y_true, w1, w2, w3 = self._network(X)
        loss2 = 1 / batch_idx * self._compute_loss(y_true, w2) - (1 - 1/batch_idx) * self._compute_loss(y_true, w3)
        loss2.backward()
        self._optimizer[1].step()
        self._optimizer[1].clear_grad()

        batch_logs = {
            "batch_size": y_true.shape[0],
            "loss": loss1.item() + loss2.item()
        }
        return batch_logs

    def _predict_batch(
        self, 
        X: Dict[str, paddle.Tensor]
    ) -> np.ndarray:
        """Predict one batch of data.

        Args: 
            X(paddle.Tensor): Feature tensor.

        Returns:
            y_true(np.ndarray): Origin data(features).
            y_pred(np.ndarray): Prediction results.
        """
        y_true, y_pred, _, _ = self._network(X)
        return y_true.numpy(), y_pred.numpy()
