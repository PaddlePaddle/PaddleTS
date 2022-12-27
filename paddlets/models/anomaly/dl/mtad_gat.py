#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
This implementation is based on the article `Multivariate Time-series Anomaly Detection via Graph Attention Network <https://arxiv.org/pdf/2009.02040.pdf>`_ .

Some codes refer to `https://github.com/ML4ITS/mtad-gat-pytorch`.

Base model features
    The author proposes a framework for multivariate time series anomaly detection named MTAD-GAT, which mainly includes a 1D convolution layer, two parallel GAT layers, a GRU layer, a full connection layer and an automatic encoder-decoder layer.
    
    1D conv layer: Feature extraction of input data.
    
    Two parallel GAT layers: Extract the features of spatial dimension and temporal dimension respectively.
    
    A GRU layer: Fusion the features of 1D conv and the two parallel GAT.
    
    A full connection layer: Implementation of anomaly detection based on forecasting method.
    
    An automatic encoder-decoder layer: Implementation of anomaly detection based on reconstruction method.
"""

from typing import List, Dict, Any, Callable, Optional
import collections

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle.optimizer import Optimizer

from paddlets.models.anomaly.dl.anomaly_base import AnomalyBaseModel
from paddlets.models.anomaly.dl._mtad_gat import FeatOrTempAttention
from paddlets.models.anomaly.dl._mtad_gat import ConvLayer, GRULayer
from paddlets.models.anomaly.dl._mtad_gat import Reconstruction, Forecasting
from paddlets.models.common.callbacks import Callback
from paddlets.models.anomaly.dl import utils as U
from paddlets.models.utils import to_tsdataset
from paddlets.datasets import TSDataset
from paddlets.logger import raise_if, raise_if_not

       
class _MTADGATBlock(paddle.nn.Layer):
    """MTADGAT Network structure.

    Args:
        in_chunk_len(int): The size of the loopback window, i.e. the number of time steps feed to the model.
        fit_params(dict): The parameters for fitting, including dimensions and dict sizes of variables.
        target_dims(Optional[List[int]]): The target dim index for forecasting and reconstruction model.
        kernel_size(int): Kernel size for Conv1D.
        feat_gat_embed_dim(Optional[int]): Output dimension of linear transformation in feat-oriented GAT layer.
        time_gat_embed_dim(Optional[int]): Output dimension of linear transformation in time-oriented GAT layer.
        use_gatv2(bool): Whether to use the modified attention mechanism of GATv2 instead of standard GAT.
        use_bias(bool): Whether to include a bias term in the attention layer.
        gru_n_layers(int): Number of layers in the GRU layer.
        gru_hid_size(int): Hidden size in the GRU layer.
        forecast_n_layers(int): Number of layers in the FC-based Forecasting Model.
        forecast_hid_size(int): Hidden size in the FC-based Forecasting Model.
        recon_n_layers(int): Number of layers in the GRU-based Reconstruction Model.
        recon_hid_size(int): Hidden size in the GRU-based Reconstruction Model.
        dropout(float): Dropout regularization parameter.
        alpha(float): The negative slope used in the LeakyReLU activation function.
       
    Attributes:
        _in_chunk_len(int): The size of the loopback window, i.e. the number of time steps feed to the model.
        _num_dim(int): The numerical feature dims.
        _out_dim(int): The target dim for forecasting and reconstruction model.
        _conv(paddle.nn.Layer): The 1D conv layer.
        _feature_gat(paddle.nn.Layer): The feat-oriented GAT layer.
        _temporal_gat(paddle.nn.Layer): The time-oriented GAT layer.
        _gru(paddle.nn.Layer): The gru layer.
        _forec_model(paddle.nn.Layer): The FC-based Forecasting Model.
        _recon_model(paddle.nn.Layer): The GRU-based Reconstruction Model.
    """
    def __init__(
        self,
        in_chunk_len: int,
        fit_params: Dict[str, Any],
        target_dims: Optional[List[int]], 
        kernel_size: int,
        feat_gat_embed_dim: Optional[int],
        time_gat_embed_dim: Optional[int],
        use_gatv2: bool,
        use_bias: bool,
        gru_n_layers: int,
        gru_hid_size: int,
        forecast_n_layers: int,
        forecast_hid_size: int,
        recon_n_layers: int,
        recon_hid_size: int,
        dropout: float,
        alpha: float,
    ):
        super(_MTADGATBlock, self).__init__()
        
        raise_if(
            in_chunk_len < 2,
            f"In mtad_gat model, the in_chunk_len must >= 2, got {in_chunk_len}."
        )         
        raise_if(
            target_dims is not None and (np.any(np.array(target_dims) < 0) or len(target_dims) == 0),
            f"target_dims must be > 0, got {target_dims}."
        )
        
        self._in_chunk_len = in_chunk_len
        self._num_dim = fit_params['observed_num_dim']
        self._out_dim = self._num_dim
        if target_dims is not None:
            self._out_dim = len(target_dims)
                
        self._conv = ConvLayer(self._num_dim, kernel_size)
        self._feature_gat = FeatOrTempAttention(self._num_dim, in_chunk_len - 1, dropout, alpha, feat_gat_embed_dim, use_gatv2, use_bias, 'feature')
        self._temporal_gat = FeatOrTempAttention(self._num_dim, in_chunk_len - 1, dropout, alpha, time_gat_embed_dim, use_gatv2, use_bias, 'temporal')
        self._gru = GRULayer(3 * self._num_dim, gru_hid_size, gru_n_layers, dropout)
        self._forec_model = Forecasting(gru_hid_size, forecast_hid_size, self._out_dim, forecast_n_layers, dropout)
        self._recon_model = Reconstruction(in_chunk_len - 1, gru_hid_size, recon_hid_size, self._out_dim, recon_n_layers, dropout)
        
    def forward(
        self, 
        X: Dict[str, paddle.Tensor]
    ) -> paddle.Tensor:
        """Forward.

        Args: 
            X(Dict[str, paddle.Tensor]): Dict of feature tensor.

        Returns:
            paddle.Tensor: Output of model.
        """
        #x: [batch_size, in_chunk_len, feature_dim]
        x = X["observed_cov_numeric"][:, :self._in_chunk_len - 1, :]
        x = self._conv(x)
        h_feat = self._feature_gat(x)
        h_temp = self._temporal_gat(x)
        
        #x: [batch_size, in_chunk_len, 3 * feature_dim]
        h_cat = paddle.concat([x, h_feat, h_temp], axis=2)
        
        _, h_end = self._gru(h_cat)
        # Extracting from last layer
        h_end = h_end[-1,:, :]
        # Hidden state for last timestamp
        h_end = h_end.reshape((x.shape[0], -1))
        
        #forecasting-based model
        preds = self._forec_model(h_end)
        #reconstruction-based model
        recons = self._recon_model(h_end)
        
        return preds, recons
    
    
class MTADGAT(AnomalyBaseModel):
    """Multivariate Time-series Anomaly Detection via Graph Attention Network.

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
        
        target_dims(Optional[List[int]]): The target dim index for forecasting and reconstruction model.
        kernel_size(int): Kernel size for Conv1D.
        feat_gat_embed_dim(Optional[int]): Output dimension of linear transformation in feat-oriented GAT layer.
        time_gat_embed_dim(Optional[int]): Output dimension of linear transformation in time-oriented GAT layer.
        use_gatv2(bool): Whether to use the modified attention mechanism of GATv2 instead of standard GAT.
        use_bias(bool): Whether to include a bias term in the attention layer.
        gru_n_layers(int): Number of layers in the GRU layer.
        gru_hid_size(int): Hidden size in the GRU layer.
        forecast_n_layers(int): Number of layers in the FC-based Forecasting Model.
        forecast_hid_size(int): Hidden size in the FC-based Forecasting Model.
        recon_n_layers(int): Number of layers in the GRU-based Reconstruction Model.
        recon_hid_size(int): Hidden size in the GRU-based Reconstruction Model.
        dropout(float): Dropout regularization parameter.
        alpha(float): The negative slope used in the LeakyReLU activation function.

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
        _target_dims(Optional[List[int]]): The target dim index for forecasting and reconstruction model.
        _kernel_size(int): Kernel size for Conv1D.
        _feat_gat_embed_dim(Optional[int]): Output dimension of linear transformation in feat-oriented GAT layer.
        _time_gat_embed_dim(Optional[int]): Output dimension of linear transformation in time-oriented GAT layer.
        _use_gatv2(bool): Whether to use the modified attention mechanism of GATv2 instead of standard GAT.
        _use_bias(bool): Whether to include a bias term in the attention layer.
        _gru_n_layers(int): Number of layers in the GRU layer.
        _gru_hid_size(int): Hidden size in the GRU layer.
        _forecast_n_layers(int): Number of layers in the FC-based Forecasting Model.
        _forecast_hid_size(int): Hidden size in the FC-based Forecasting Model.
        _recon_n_layers(int): Number of layers in the GRU-based Reconstruction Model.
        _recon_hid_size(int): Hidden size in the GRU-based Reconstruction Model.
        _dropout(float): Dropout regularization parameter.
        _alpha(float): The negative slope used in the LeakyReLU activation function.
    """
    def __init__(
        self,
        in_chunk_len: int,
        sampling_stride: int = 1,
        loss_fn: Callable[..., paddle.Tensor] = F.mse_loss,
        optimizer_fn: Callable[..., Optimizer] = paddle.optimizer.Adam,
        threshold_fn: Callable[..., float] = U.epsilon_th,
        q: float = 100,
        threshold: Optional[float] = None,
        threshold_coeff: float = 1.0,
        anomaly_score_fn: Callable[..., List[float]] = None,
        pred_adjust: bool = True,
        pred_adjust_fn: Callable[..., np.ndarray] = U.result_adjust,
        optimizer_params: Dict[str, Any] = dict(learning_rate=1e-3),
        callbacks: List[Callback] = [], 
        batch_size: int = 256,
        max_epochs: int = 100,
        verbose: int = 1,
        patience: int = 10,
        seed: Optional[int] = None,

        target_dims: Optional[List[int]] = None,
        kernel_size: int = 7,
        feat_gat_embed_dim: Optional[int] = None,
        time_gat_embed_dim: Optional[int] = None,
        use_gatv2: bool = False,
        use_bias: bool = False,
        gru_n_layers: int = 1,
        gru_hid_size: int = 150,
        forecast_n_layers: int = 1,
        forecast_hid_size: int = 150,
        recon_n_layers: int = 1,
        recon_hid_size: int = 150,
        dropout: float = 0.2,
        alpha: float = 0.2,
    ):
        
        self._target_dims = target_dims
        self._kernel_size = kernel_size
        self._dropout = dropout
        self._q = q
        self._feat_gat_embed_dim = feat_gat_embed_dim
        self._time_gat_embed_dim = time_gat_embed_dim
        self._use_gatv2 = use_gatv2
        self._use_bias = use_bias
        self._gru_n_layers = gru_n_layers
        self._gru_hid_size = gru_hid_size
        self._forecast_n_layers = forecast_n_layers
        self._forecast_hid_size = forecast_hid_size
        self._recon_n_layers = recon_n_layers
        self._recon_hid_size = recon_hid_size
        self._dropout = dropout
        self._alpha = alpha
       
        super(MTADGAT, self).__init__(
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
        train_tsdataset.sort_columns()
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
        }
        return fit_params
        
    def _init_network(self) -> paddle.nn.Layer:
        """Setup the network.

        Returns:
            paddle.nn.Layer.
        """
        return _MTADGATBlock(
            self._in_chunk_len,
            self._fit_params,
            self._target_dims,
            self._kernel_size,
            self._feat_gat_embed_dim,
            self._time_gat_embed_dim,
            self._use_gatv2,
            self._use_bias,
            self._gru_n_layers,
            self._gru_hid_size,
            self._forecast_n_layers,
            self._forecast_hid_size,
            self._recon_n_layers,
            self._recon_hid_size,
            self._dropout,
            self._alpha,
        )
       
    def _train_batch(
        self, 
        X: Dict[str, paddle.Tensor], 
    ) -> Dict[str, Any]:
        """Trains one batch of data.

        Args:
            X(Dict[str, paddle.Tensor]): Dict of feature tensor.
            y(paddle.Tensor): Target tensor.

        Returns:
            Dict[str, Any]: Dict of logs.
        """
        self._optimizer.clear_grad()
        preds, recons = self._network(X)
        total_loss, forecast_loss, recon_loss = self._compute_loss(X, preds, recons)
        total_loss.backward() 
        self._optimizer.step()
        batch_logs = {
            "batch_size": X['observed_cov_numeric'].shape[0],
            "loss": total_loss.item(),       
        }
        return batch_logs
    
    def _predict_epoch(
        self, 
        name: str, 
        loader: paddle.io.DataLoader
    ):
        """Predict an epoch and update metrics.

        Args:
            name(str): Name of the validation set.
            loader(paddle.io.DataLoader): DataLoader with validation set.
        """
        self._network.eval()
        list_y_pred, list_y_true = [], []
        loss_list = []
        for batch_idx, data in enumerate(loader):
            total_loss = self._predict_batch(data)
            loss_list.append(total_loss.item())
        metrics_logs = {name + '_loss': np.mean(loss_list)}
        self._history._epoch_metrics.update(metrics_logs)
        self._network.train()
    
    def _predict_batch(
        self, 
        X: Dict[str, paddle.Tensor]
    ) -> np.ndarray:
        """Predict one batch of data.

        Args: 
            X(Dict[str, paddle.Tensor]): Feature tensor.

        Returns:
            total_loss(paddle.Tensor): Total loss.
        """
        preds, recons = self._network(X)
        total_loss, forecast_loss, recon_loss = self._compute_loss(X, preds, recons)
        return total_loss
    
    def _compute_loss(
        self, 
        X: Dict[str, paddle.Tensor],
        preds: paddle.Tensor,
        recons: paddle.Tensor
    ) -> paddle.Tensor:
        """Compute the loss.

        Args:
            X(Dict[str, paddle.Tensor]): Feature tensor.
            preds(paddle.Tensor): Result by Forecasting-Based Model.
            recons(paddle.Tensor): Result by Reconstruction-Based Model.

        Returns:
            paddle.Tensor: Loss value.
        """
        x = X["observed_cov_numeric"][:, :self._in_chunk_len - 1, :]
        y = X["observed_cov_numeric"][:, self._in_chunk_len - 1:, :]
     
        if self._target_dims is not None:
            x = paddle.to_tensor(x.numpy()[:, :, self._target_dims])
            y = paddle.to_tensor(y.numpy()[:, :, self._target_dims])

        if preds.ndim == 3:
            preds = preds.squeeze(1)
        if y.ndim == 3:
            y = y.squeeze(1)
        
        forecast_loss = paddle.sqrt(self._loss_fn(y, preds))
        recon_loss = paddle.sqrt(self._loss_fn(x, recons))
        total_loss = forecast_loss + recon_loss
        
        return total_loss, forecast_loss, recon_loss
      
    def _predict(
        self, 
        dataloader: paddle.io.DataLoader
    ) -> np.ndarray:
        """Predict function core logic.

        Args:
            dataloader(paddle.io.DataLoader): Data to be predicted.

        Returns:
            anomaly_scores(np.ndarray): The anomaly scores.
        """
        self._network.eval()
        pred_list, recon_list, true_list = [], [], []
        for batch_nb, data in enumerate(dataloader):
            # get pred result
            preds, _ = self._network(data)
            x = data["observed_cov_numeric"][:, :self._in_chunk_len - 1, :]
            y = data["observed_cov_numeric"][:, self._in_chunk_len - 1:, :]
            recon_x = paddle.concat([x[:, 1:, :], y, x[:, 0:1, :]], axis=1)
            
            X = {'observed_cov_numeric': recon_x}
            # get recon result
            _, recons = self._network(X)
            pred_list.append(preds.numpy())
            recon_list.append(recons[:, -1, :].numpy())
            
            # the true value
            y_value = y.reshape((-1, y.shape[-1])).numpy()
            if self._target_dims is not None:
                y_value = y_value[:, self._target_dims]
            true_list.append(y_value)
        
        pred_list = np.concatenate(pred_list, axis=0)
        recon_list = np.concatenate(recon_list, axis=0)
        true_list = np.concatenate(true_list, axis=0)
              
        anomaly_scores = np.zeros_like(true_list)
        for i in range(pred_list.shape[1]):  
            a_score = np.sqrt((pred_list[:, i] - true_list[:, i]) ** 2) + np.sqrt(
                (recon_list[:, i] - true_list[:, i]) ** 2)         
            anomaly_scores[:, i] = a_score       
        anomaly_scores = np.mean(anomaly_scores, 1)
        
        return anomaly_scores
