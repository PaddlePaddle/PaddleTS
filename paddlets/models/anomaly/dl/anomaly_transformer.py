#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
This implementation is based on the article `Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy <https://arxiv.org/pdf/2110.02642.pdf>`_ .

Base model features
    Basic architecture: A network with stacking anomaly Attention. 
    
    Anomaly Attention have two branch,  one is self attention,  another one is Gaussian kernel.
    
    The backward is for computing of the the prior-association loss  and the series-association loss, distinguishing anomaly data in raw data. 

"""

from typing import List, Dict, Any, Callable, Optional

from paddle.optimizer import Optimizer
import paddle.nn.functional as F
import pandas as pd
import numpy as np
import paddle
import time

from paddlets.models.anomaly.dl.anomaly_base import AnomalyBaseModel
from paddlets.models.common.callbacks import Callback
from paddlets.models.anomaly.dl.utils import to_tsdataset
from paddlets.models.anomaly.dl import utils as U
from paddlets.datasets import TSDataset
from paddlets.logger import raise_if, raise_if_not
from paddlets.models.anomaly.dl._anomaly_transformer.encoder import Encoder, EncoderLayer
from paddlets.models.anomaly.dl._anomaly_transformer.attention import AttentionLayer, AnomalyAttention
from paddlets.models.anomaly.dl._anomaly_transformer.embedding import DataEmbedding


class _Anomaly(paddle.nn.Layer):
    """
    Anomaly transformer Network structure.
    
    Args:
        win_size(int): The size of the loopback window, i.e. the number of time steps feed to the model.
        enc_in(int): The number of feature in model input.
        c_out(int): The number of feature in model output.
        d_model(int): The expected feature size for the input of the anomaly transformer.
        n_heads(int): The number of heads in multi-head attention.
        e_layers(int): The number of attentionLayer layers to be stacked.
        d_ff(int): The Number of channels for FFN layers.
        dropout(float): Dropout regularization parameter.
        activation(Callable[..., paddle.Tensor]): The activation function for AnomalyAttention.
        output_attention(float): Whether to output series, prior and sigma.
        
    Attributes:
        _nn(paddle.nn.Sequential): Dynamic graph LayerList. 
    """
    def __init__(
        self, 
        win_size: int, 
        enc_in: int, 
        c_out: int, 
        d_model: int = 512, 
        n_heads: int = 8, 
        e_layers: int = 3, 
        d_ff: int = 512,
        dropout: float = 0.0, 
        activation: Callable[..., paddle.Tensor] = F.gelu, 
        output_attention: bool = True
    ):
        super(_Anomaly, self).__init__()
        self.output_attention = output_attention
        self.embedding = DataEmbedding(enc_in, d_model, dropout)
        self.encoder = Encoder(
            [EncoderLayer
             (AttentionLayer
                  (AnomalyAttention(win_size, False, attention_dropout=dropout, output_attention=output_attention),
                   d_model, n_heads
                  ),
                  d_model, d_ff, dropout=dropout, activation=activation
             ) for l in range(e_layers)
            ],
            norm_layer=paddle.nn.LayerNorm(d_model)
        )
        self.projection = paddle.nn.Linear(d_model, c_out, bias_attr=True)  #bias

    def forward(
        self, 
        x: Dict[str, paddle.Tensor]
   ) -> paddle.Tensor:
        """Anomaly transformer Forward.

        Args: 
            X(paddle.Tensor): Dict of feature tensor.

        Returns:
            paddle.Tensor: Output of model.
        """ 
        x = x["observed_cov_numeric"]
        enc_out = self.embedding(x)
        enc_out, series, prior, sigmas = self.encoder(enc_out)
        enc_out = self.projection(enc_out)
        if self.output_attention:
            return enc_out, series, prior, sigmas
        else:
            return enc_out  # [B, L, D]


class AnomalyTransformer(AnomalyBaseModel):
    """
    Anomaly Transformer network for anomaly detection.
    
    Args:
        in_chunk_len(int): The size of the loopback window, i.e. the number of time steps feed to the model.
        sampling_stride(int): Sampling intervals between two adjacent samples.
        loss_fn(Callable[..., paddle.Tensor]): Loss function.
        optimizer_fn(Callable[..., Optimizer]): Optimizer algorithm.
        threshold_fn(Callable[..., float]|None): The method to get anomaly threshold.
        criterion(Callable[..., paddle.Tensor]): Loss function for for the reconstruction loss.
        threshold(float|None): The threshold to judge anomaly.
        threshold_coeff(float): The coefficient of threshold.
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
        temperature(int|float): A parameter to adjust series loss and prior loss. 
        k(int): The optimization is to enlarge the association discrepancy.
        anormly_ratio(int|float): The Proportion of Anomaly data in train set and test set.
        d_model(int): The expected feature size for the input of the anomaly transformer.
        n_heads(int): The number of heads in multi-head attention.
        e_layers(int): The number of attentionLayer layers to be stacked.
        d_ff(int): The Number of channels for FFN layers.
        dropout(float): Dropout regularization parameter.
        activation(Callable[..., paddle.Tensor]): The activation function for AnomalyAttention.
        output_attention(float): Whether to output series, prior and sigma.

    Attributes:
        _in_chunk_len(int): The size of the loopback window, i.e. the number of time steps feed to the model.
        _sampling_stride(int): Sampling intervals between two adjacent samples.
        _loss_fn(Callable[..., paddle.Tensor]): Loss function.
        _optimizer_fn(Callable[..., Optimizer]): Optimizer algorithm.
        _threshold_fn(Callable[..., float]|None): The method to get anomaly threshold.
        _criterion(Callable[..., paddle.Tensor]): Loss function for for the reconstruction loss.
        _threshold(float|None): The threshold to judge anomaly.
        _threshold_coeff(float): The coefficient of threshold.
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
        _temperature(int|float): A parameter to adjust series loss and prior loss.
        _k(int): The optimization is to enlarge the association discrepancy.
        _anormly_ratio(int|float): The Proportion of Anomaly data in train set and test set.
        _d_model(int): The expected feature size for the input of the anomaly transformer.
        _n_heads(int): The number of heads in multi-head attention.
        _e_layers(int): The number of attentionLayer layers to be stacked.
        _d_ff(int): The Number of channels for FFN layers.
        _dropout(float): Dropout regularization parameter.
        _activation(Callable[..., paddle.Tensor]): The activation function for AnomalyAttention.
        _output_attention(float): whether to output series, prior and sigma.
        _adjust_lr(function): Dynamic Learning Rate Adjustment.
    """
    def __init__(
        self,
        in_chunk_len: int,
        sampling_stride: int = 1,
        loss_fn: Callable[..., paddle.Tensor] = U.series_prior_loss,
        optimizer_fn: Callable[..., Optimizer] = paddle.optimizer.Adam,
        threshold_fn: Callable[..., float] = U.anomaly_get_threshold,
        criterion: Callable[..., paddle.Tensor] = paddle.nn.MSELoss(),
        threshold: Optional[float] = None,
        threshold_coeff: float = 1.0,
        anomaly_score_fn: Callable[..., List[float]] = None,
        pred_adjust: bool = True,
        pred_adjust_fn: Callable[..., np.ndarray] = U.result_adjust,
        optimizer_params: Dict[str, Any] = dict(learning_rate=1e-4),
        eval_metrics: List[str] = [], 
        callbacks: List[Callback] = [], 
        batch_size: int = 32,          
        max_epochs: int = 100,
        verbose: int = 1,
        patience: int = 10,
        seed: Optional[int] = None,
        temperature: int = 50,  
        k: int = 3,
        anormly_ratio = 1,
        d_model: int = 512, 
        n_heads: int = 8, 
        e_layers: int = 3, 
        d_ff: int = 512,
        dropout: float = 0.0, 
        activation: Callable[..., paddle.Tensor] = F.gelu, 
        output_attention = True,
        ):
        self.in_chunk_len = in_chunk_len
        self._d_model = d_model
        self._n_heads = n_heads
        self._e_layers = e_layers
        self._d_ff = d_ff
        self._dropout = dropout
        self._activation = activation
        self._output_attention = output_attention 
        self._adjust_lr = U.adjust_learning_rate
        self._threshold_fn = threshold_fn
        self._criterion = criterion
        self._temperature = temperature
        self._k = k
        self._anormly_ratio = anormly_ratio
        
        super(AnomalyTransformer, self).__init__(
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
        return _Anomaly(
                self.in_chunk_len,
                self._fit_params["observed_dim"],
                self._fit_params["observed_dim"],
                self._d_model, 
                self._n_heads, 
                self._e_layers, 
                self._d_ff,
                self._dropout, 
                self._activation, 
                self._output_attention,
                )
    
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
        scores = self._network(X)[0]
        y = X['observed_cov_numeric']
        return y.numpy(), scores.numpy()
    
    @to_tsdataset(scenario="anomaly_label")
    def predict(
        self,
        test_dataset: TSDataset,
        train_dataset: TSDataset = None,
    ) -> TSDataset:
        """Get anomaly label on a batch. the result are output as tsdataset.

        Args:
            train_dataset(TSDataset): Train set.
            test_dataset(TSDataset): Data to be predicted.

        Returns:
            TSDataset.
        """
        raise_if(train_dataset is None, f" Please pass in train_tsdataset to calculate the threshold.")
        train_dataloader = self._init_predict_dataloader(train_dataset)
        test_dataloader  = self._init_predict_dataloader(test_dataset) 
        thre_dataloader  = self._init_predict_dataloader(test_dataset, sampling_stride=self.in_chunk_len)
        self._threshold = self._threshold_fn(self._network, 
                                        train_dataloader, test_dataloader, 
                                        temperature=self._temperature,  
                                        anormly_ratio=self._anormly_ratio,
                                        criterion = self._criterion, 
                                        my_kl_loss = U.my_kl_loss,
                                        win_size=self.in_chunk_len, 
                                        )
        anomaly_score = self._get_anomaly_score(thre_dataloader)
        anomaly_label = []
        for score in anomaly_score: 
            label = 0 if score < self._threshold else 1
            anomaly_label.append(label)
        if test_dataset.target is not None and self._pred_adjust:
            anomaly_label = self._pred_adjust_fn(anomaly_label, test_dataset.target.to_numpy())
        return np.array(anomaly_label)

    def _predict(
        self, 
        dataloader: paddle.io.DataLoader,
    ) -> np.ndarray:
        """Predict function core logic.

        Args:
            dataloader(paddle.io.DataLoader): Data to be predicted.

        Returns:
            np.ndarray.
        """
        self._network.eval()
        attens_energy = []
        for batch_nb, data in enumerate(dataloader):
            y = data['observed_cov_numeric']
            output, series, prior, _ = self._network(data)
            loss = paddle.mean(self._criterion(y, output), axis=-1)
            cri = U.series_prios_energy([output, series, prior, _], loss, 
                                        temperature=self._temperature, win_size=self.in_chunk_len)
            attens_energy.append(cri)
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        return np.array(test_energy)   

    def predict_score(
        self,
        tsdataset: TSDataset,
    ) -> TSDataset:
        """Get anomaly score on a batch. the result are output as tsdataset.

        Args:
            tsdataset(TSDataset): Data to be predicted.

        Returns:
            TSDataset.
        """
        dataloader = self._init_predict_dataloader(tsdataset, sampling_stride=self.in_chunk_len)
        results = self._get_anomaly_score(dataloader)
        # Generate target cols
        target_cols = tsdataset.get_target()
        if target_cols is None:
            target_cols = ["anomaly_score"]
        else:
            target_cols = target_cols.data.columns
            target_cols = target_cols + '_score'
        # Generate target index freq
        target_index = tsdataset.get_observed_cov().data.index
        if isinstance(target_index, pd.RangeIndex):
            freq = target_index.step
        else:
            freq = target_index.freqstr
        results_size = results.size
        raise_if(
            results_size == 0,
            f"There is something wrong, anomaly predict size is 0, you'd better check the tsdataset or the predict logic."
        )
        target_index = target_index[:results_size]# [-results_size:]
        anomaly_target = pd.DataFrame(results, index=target_index, columns=target_cols)
        return TSDataset.load_from_dataframe(anomaly_target, freq=freq)

    def fit(
        self,
        train_tsdataset: TSDataset, 
        valid_tsdataset: Optional[TSDataset] = None
    ):
        """Train a neural network stored in self._network, 
            Using train_dataloader for training data and valid_dataloader for validation.

        Args: 
            train_tsdataset(TSDataset): Train set. 
            valid_tsdataset(TSDataset|None): Eval set, used for early stopping.
        """
        self._fit_params = self._update_fit_params(train_tsdataset, valid_tsdataset)
        train_dataloader, valid_dataloaders = self._init_fit_dataloaders(train_tsdataset, valid_tsdataset)
        self._fit(train_dataloader, valid_dataloaders)
    
    def _fit(
        self, 
        train_dataloader: paddle.io.DataLoader,
        valid_dataloaders: List[paddle.io.DataLoader] = None
    ):
        """Fit function core logic. 

        Args: 
            train_dataloader(paddle.io.DataLoader): Train set. 
            valid_dataloaders(List[paddle.io.DataLoader]|None): Eval set.
        """
        valid_names = [f"val_{k}" for k in range(len(valid_dataloaders))]
        self._metrics, self._metrics_names, \
            self._metric_container_dict =  self._init_metrics(valid_names)
        self._history, self._callback_container = self._init_callbacks()
        self._network = self._init_network()
        self._optimizer = self._init_optimizer()

        # Call the `on_train_begin` method of each callback before the training starts.
        self._callback_container.on_train_begin({"start_time": time.time()})
        for epoch_idx in range(self._max_epochs):
            # Call the `on_epoch_begin` method of each callback before the epoch starts.
            self._callback_container.on_epoch_begin(epoch_idx)
            self._train_epoch(train_dataloader)
            self._adjust_lr(self._optimizer, epoch_idx, self._optimizer_params['learning_rate'])  # update lr
            # Predict for each eval set.
            for eval_name, valid_dataloader in zip(valid_names, valid_dataloaders):
                self._predict_epoch(eval_name, valid_dataloader)
            # Call the `on_epoch_end` method of each callback at the end of the epoch.
            self._callback_container.on_epoch_end(
                epoch_idx, logs=self._history._epoch_metrics
                )
            if self._stop_training:
                break

        # Call the `on_train_end` method of each callback at the end of the training.
        self._callback_container.on_train_end()
        self._network.eval()
    
    def _train_epoch(
        self, 
        train_loader: paddle.io.DataLoader
    ):
        """
        Trains one epoch of the network in self._network.

        Args: 
            train_loader(paddle.io.DataLoader): Training dataloader.
        """
        self._network.train()
        loss1_list = [] 
        for batch_idx, data in enumerate(train_loader):
            self._callback_container.on_batch_begin(batch_idx)
            y = data['observed_cov_numeric']
            batch_logs = self._train_batch(data, y, batch_idx)
            self._callback_container.on_batch_end(batch_idx, batch_logs)
        epoch_logs = {"lr": self._optimizer.get_lr()}
        self._history._epoch_metrics.update(epoch_logs)

    def _train_batch(
        self, 
        X: Dict[str, paddle.Tensor], 
        y: paddle.Tensor,
        batch_idx = None,
    ) -> Dict[str, Any]:
        """Trains one batch of data.

        Args:
            X(Dict[str, paddle.Tensor]): Dict of feature tensor.
            y(paddle.Tensor): Target tensor.

        Returns:
            Dict[str, Any]: Dict of logs.
        """
        self._optimizer.clear_grad()
        output_list = self._network(X)
        loss1, loss2, for_loss_one = self._compute_loss(output_list, y, 
                                     criterion = self._criterion, win_size=self.in_chunk_len, k=self._k)
        # Minimax strategy
        loss1.backward(retain_graph=True)
        loss2.backward()
        self._optimizer.step()
        avg_loss = for_loss_one
        batch_logs = {
            "batch_size": y.shape[0],
            "loss": avg_loss
        }
        return batch_logs 

    def _compute_loss(
        self, 
        y_score: paddle.Tensor, 
        y_true: paddle.Tensor,
        criterion = paddle.nn.MSELoss(),
        win_size = 100, 
        k = 3
    ) -> paddle.Tensor:
        """Compute the loss.

        Note:
            This function could be overrided by the subclass if necessary.

        Args:
            y_score(paddle.Tensor): Estimated target values.
            y_true(paddle.Tensor): Ground truth (correct) target values.
            criterion(Callable[..., paddle.Tensor]): Loss function.
            win_size(int): The size of the loopback window, i.e. the number of time steps feed to the model.
            k(int): The optimization is to enlarge the association discrepancy.

        Returns:
            paddle.Tensor: Loss value.
        """
        return self._loss_fn(y_score, y_true, criterion, win_size, k)
    
