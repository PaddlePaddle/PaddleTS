#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from typing import List, Dict, Any, Callable, Optional, Tuple
from collections import OrderedDict
from copy import deepcopy
import time
import abc
import os
import pickle
import json

from paddle.optimizer import Optimizer
import numpy as np
import paddle

from paddlets.models.common.callbacks import (
    CallbackContainer,
    EarlyStopping,
    Callback,
    History,
)
from paddlets.metrics import (
    MetricContainer, 
    Metric
)
from paddlets.models.data_adapter import DataAdapter
from paddlets.models.utils import check_tsdataset, to_tsdataset, build_network_input_spec
from paddlets.datasets import TSDataset
from paddlets.logger import raise_if, raise_if_not, raise_log, Logger

logger = Logger(__name__)


class AnomalyBaseModel(abc.ABC):
    """PaddleTS deep time series anomaly detection framework, all time series models based on paddlepaddle implementation need to inherit this class.

    Args:
        in_chunk_len(int): The size of the loopback window, i.e. the number of time steps feed to the model.
        sampling_stride(int): Sampling intervals between two adjacent samples.
        loss_fn(Callable[..., paddle.Tensor]|None): Loss function.
        anomaly_score_fn(Callable[..., List[float]]|None): The method to get anomaly score.
        threshold(float|None): The threshold to judge anomaly.
        threshold_fn(Callable[..., float]|None): The method to get anomaly threshold.
        threshold_coeff(float): The coefficient of threshold.
        pred_adjust(bool): Whether to adjust the pred label according to the real label.
        pred_adjust_fn(Callable[..., np.ndarray]|None): The method to adjust pred label.
        optimizer_fn(Callable[..., Optimizer]): Optimizer algorithm.
        optimizer_params(Dict[str, Any]): Optimizer parameters.
        eval_metrics(List[str]): Evaluation metrics of model.
        callbacks(List[Callback]): Customized callback functions.
        batch_size(int): Number of samples per batch.
        max_epochs(int): Max epochs during training.
        verbose(int): Verbosity mode.
        patience(int): Number of epochs to wait for improvement before terminating.
        seed(int|None): Global random seed.

    Attributes:
        _in_chunk_len(int): The size of the loopback window, i.e. the number of time steps feed to the model.
        _sampling_stride(int): Sampling intervals between two adjacent samples.
        _loss_fn(Callable[..., paddle.Tensor]|None): Loss function.
        _anomaly_score_fn(Callable[..., paddle.Tensor]|None): The method to get anomaly score.
        _threshold(float|None): The threshold to judge anomaly.
        _threshold_fn(Callable[..., paddle.Tensor]|None): The method to get anomaly threshold.
        _threshold_coeff(float): The coefficient of threshold.
        _pred_adjust(float): Whether to adjust the pred label according to the real label.
        _pred_adjust_fn(Callable[..., np.ndarray]|None): The method to adjust pred label.
        _optimizer_fn(Callable[..., Optimizer]): Optimizer algorithm.
        _optimizer_params(Dict[str, Any]): Optimizer parameters.
        _eval_metrics(List[str]): Evaluation metrics of model.
        _callbacks(List[Callback]): Customized callback functions.
        _batch_size(int): Number of samples per batch.
        _max_epochs(int): Max epochs during training.
        _verbose(int): Verbosity mode.
        _patience(int): Number of epochs to wait for improvement before terminating.
        _seed(int|None): Global random seed.
        _stop_training(bool) Training status.
        _fit_params(Dict[str, Any]): Infer parameters by TSdataset automatically.
        _network(paddle.nn.Layer): Network structure.
        _optimizer(Optimizer): Optimizer.
        _metrics(List[Metric]): List of metric instance.
        _metrics_names(List[str]): List of metric names.
        _metric_container_dict(Dict[str, MetricContainer]): Dict of metric container.
        _history(History): Callback that records events into a `History` object.
        _callback_container(CallbackContainer): Container holding a list of callbacks.
    """
    def __init__(
        self,
        in_chunk_len: int,
        sampling_stride: int = 1,
        loss_fn: Callable[..., paddle.Tensor] = None,
        anomaly_score_fn: Callable[..., List[float]] = None,
        threshold: Optional[float] = None,
        threshold_fn: Callable[..., float] = None,
        threshold_coeff: float = 1.0,
        pred_adjust: bool = False,
        pred_adjust_fn: Callable[..., np.ndarray] = None,
        optimizer_fn: Callable[..., Optimizer] = paddle.optimizer.Adam,
        optimizer_params: Dict[str, Any] = dict(learning_rate=1e-3),
        eval_metrics: List[str] = [], 
        callbacks: List[Callback] = [], 
        batch_size: int = 128,
        max_epochs: int = 10,
        verbose: int = 1,
        patience: int = 4,
        seed: Optional[int] = None,
    ):
        super(AnomalyBaseModel, self).__init__()
        self._in_chunk_len = in_chunk_len
        self._sampling_stride = sampling_stride
        self._loss_fn = loss_fn
        self._anomaly_score_fn = anomaly_score_fn
        self._threshold = threshold
        self._threshold_fn = threshold_fn
        self._threshold_coeff = threshold_coeff
        self._pred_adjust = pred_adjust
        self._pred_adjust_fn = pred_adjust_fn
        self._optimizer_fn = optimizer_fn
        self._optimizer_params = deepcopy(optimizer_params)
        self._eval_metrics = deepcopy(eval_metrics)
        self._callbacks = deepcopy(callbacks)
        self._batch_size = batch_size
        self._max_epochs = max_epochs
        self._verbose = verbose
        self._patience = patience
        self._stop_training = False
        
        self._fit_params = None
        self._network = None
        self._optimizer = None
        self._metrics = None
        self._metrics_names = None
        self._metric_container_dict = None
        self._history = None
        self._callback_container = None

        # Parameter check.
        self._check_params()
        if seed is not None:
            paddle.seed(seed)
        
    def _check_params(self):
        """Parameter validity verification.

        Check logic:

            batch_size: batch_size must be > 0.

            max_epochs: max_epochs must be > 0.

            verbose: verbose must be > 0.

            patience: patience must be >= 0.
        """
        raise_if(self._batch_size <= 0, f"batch_size must be > 0, got {self._batch_size}.")
        raise_if(self._max_epochs <= 0, f"max_epochs must be > 0, got {self._max_epochs}.")
        raise_if(self._verbose <= 0, f"verbose must be > 0, got {self._verbose}.")
        raise_if(self._patience < 0, f"patience must be >= 0, got {self._patience}.")
        # If user does not specify an evaluation standard, a metric is provided by default.
        if not self._eval_metrics: 
            self._eval_metrics = ["mse"]

    def _check_tsdataset(
        self, 
        tsdataset: TSDataset
    ):
        """Ensure the robustness of input data (consistent feature order), at the same time,
            check whether the data types are compatible. If not, the processing logic is as follows.

        Processing logic:

            1> Check whether the target column is legal.

            2> Prompt that the "know" and "static" will be ignored.
            
            3> Check whether the observed column is legal.

            4> Integer: Convert to np.int64.

            5> Floating: Convert to np.float32.

            6> Missing value: Warning.

            7> Other: Illegal.

        Args:
            tsdataset(TSDataset): Data to be checked.
        """
        target = tsdataset.get_target()
        if target is not None:
            target_dtype = target.dtypes
            raise_if_not(
                np.issubdtype(target_dtype[0], np.integer),
                f"In anomaly detection scenario, the dtype of label must be integer, but recevid {target_dtype[0]}"
            )
        known = tsdataset.get_known_cov()
        if known is not None:
            known_cols = [str(i) for i in known.columns.tolist()]
            msg = f"Input tsdataset contains known cov `{','.join(known_cols)}` " \
                + "which will be ignored in anomaly detection scenario."
            logger.warning(msg)
        static = tsdataset.get_static_cov()
        if static is not None and len(static) != 0:
            static_cols = [str(i) for i in list(static.keys())]
            msg = f"Input tsdataset contains static cov `{','.join(static_cols)}` " \
                + "which will be ignored in anomaly detection scenario."
            logger.warning(msg)
        observed = tsdataset.get_observed_cov()
        raise_if(
            observed is None,
            f"The feature col is empty, please set at least one feature col in anomaly detection scenario."
        )
        illegal_cols = []
        for col, dtype in observed.dtypes.items():
            if not (np.issubdtype(dtype, np.floating) or np.issubdtype(dtype, np.integer)):
                illegal_cols.append(col)
        raise_if(
            len(illegal_cols) > 0,
            f"In anomaly detection scenario, feature dtype only supports [int, float], " \
            f"illegal feature columns: {illegal_cols}"
        )
        valid = False
        for dtype in observed.dtypes:
            if np.issubdtype(dtype, np.floating):
                valid = True
                break
        raise_if(
            not valid,
            f"All the feature dtypes are int, please set at least one float feature col in anomaly detection scenario."
        )

        check_tsdataset(tsdataset)

    def _init_optimizer(self) -> Optimizer:
        """Setup optimizer.

        Returns:
            Optimizer.
        """
        return self._optimizer_fn(
            **self._optimizer_params,
            parameters=self._network.parameters()
        )

    def _init_fit_dataloaders(
        self, 
        train_tsdataset: TSDataset, 
        valid_tsdataset: Optional[TSDataset] = None,
        shuffle: bool = True
    ) -> Tuple[paddle.io.DataLoader, List[paddle.io.DataLoader]]:
        """Generate dataloaders for train and eval set.

        Args: 
            train_tsdataset(TSDataset): Train set.
            valid_tsdataset(TSDataset|None): Eval set.
            shuffle(bool): Shuffle or not.

        Returns:
            paddle.io.DataLoader: Training dataloader.
            List[paddle.io.DataLoader]: List of validation dataloaders..
        """
        data_adapter = DataAdapter()
        train_dataset = data_adapter.to_sample_dataset(
            train_tsdataset,
            in_chunk_len=self._in_chunk_len,
            sampling_stride=self._sampling_stride,
        )
        train_dataloader = data_adapter.to_paddle_dataloader(train_dataset, self._batch_size, shuffle=shuffle)
       
        valid_dataloaders = []
        if valid_tsdataset is not None:
            valid_dataset = data_adapter.to_sample_dataset(
                valid_tsdataset,
                in_chunk_len=self._in_chunk_len,
                sampling_stride=self._sampling_stride,
            )
            valid_dataloader = data_adapter.to_paddle_dataloader(valid_dataset, self._batch_size, shuffle=shuffle)
            valid_dataloaders.append(valid_dataloader)
        return train_dataloader, valid_dataloaders

    def _init_predict_dataloader(
        self, 
        tsdataset: TSDataset,
        sampling_stride: int = 1,
    ) -> paddle.io.DataLoader:
        """Generate dataloaders for data to be predicted.

        Args: 
            tsdataset(TSDataset): Data to be predicted.
            sampling_stride(int): Sampling intervals between two adjacent samples.

        Returns:
            paddle.io.DataLoader: dataloader. 
        """
        self._check_tsdataset(tsdataset)
        data_adapter = DataAdapter()
        dataset = data_adapter.to_sample_dataset(
            tsdataset,
            in_chunk_len=self._in_chunk_len,
            sampling_stride=sampling_stride,
        )
        dataloader = data_adapter.to_paddle_dataloader(dataset, self._batch_size, shuffle=False)
        return dataloader

    def _init_metrics(
        self, 
        eval_names: List[str] 
    ) -> Tuple[List[Metric], List[str], Dict[str, MetricContainer]]:
        """Set attributes relative to the metrics.

        Args:
            eval_names(List[str]): List of eval set names.

        Returns:
            List[Metric]: List of metric instance.
            List[str]: List of metric names.
            Dict[str, MetricContainer]: Dict of metric container.
        """
        metrics = self._eval_metrics
        metric_container_dict = OrderedDict()
        for name in eval_names:
            metric_container_dict.update({
                name: MetricContainer(metrics, prefix=f"{name}_")
            })
        metrics, metrics_names = [], []
        for _, metric_container in metric_container_dict.items():
            metrics.extend(metric_container._metrics)
            metrics_names.extend(metric_container._names)
        return metrics, metrics_names, metric_container_dict

    def _init_callbacks(self) -> Tuple[History, CallbackContainer]:
        """Setup the callbacks functions.

        Returns:
            History: Callback that records events into a `History` object.
            CallbackContainer: Container holding a list of callbacks.
        """
        # Use the last metric in the container as the standard for early stopping.
        early_stopping_metric = (
            self._metrics_names[-1] if len(self._metrics_names) > 0 else None
        )
        # Set callback functions, including history, early stopping, etc..
        history, callbacks = History(self._verbose), [] # nqa
        callbacks.append(history)
        if (early_stopping_metric is not None) and (self._patience > 0):
            early_stopping = EarlyStopping(
                early_stopping_metric=early_stopping_metric,
                is_maximize=self._metrics[-1]._MAXIMIZE, 
                patience=self._patience
            )
            callbacks.append(early_stopping)
        else:
            logger.warning("No early stopping will be performed, last training weights will be used.")

        if self._callbacks:
            callbacks.extend(self._callbacks)
        callback_container = CallbackContainer(callbacks)
        callback_container.set_trainer(self)
        return history, callback_container
    
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
        self._check_tsdataset(train_tsdataset)
        if valid_tsdataset is not None:
            self._check_tsdataset(valid_tsdataset)
        self._fit_params = self._update_fit_params(train_tsdataset, valid_tsdataset)
        train_dataloader, valid_dataloaders = self._init_fit_dataloaders(train_tsdataset, valid_tsdataset)
        self._fit(train_dataloader, valid_dataloaders)
        
        # Get threshold
        if self._threshold is None:
            dataloader, _ = self._init_fit_dataloaders(train_tsdataset, shuffle=False)
            anomaly_score = self._get_anomaly_score(dataloader)
            self._threshold = self._threshold_coeff * self._get_threshold(anomaly_score)
        
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

    def _get_anomaly_score(
        self,
        dataloader: paddle.io.DataLoader,
        **predict_kwargs
    ) -> np.ndarray:
        """Get anomaly score on a batch.

        Args:
            dataloader(paddle.io.DataLoader): Data to be predicted.
            **predict_kwargs: Additional arguments for `_predict`.

        Returns:
            np.ndarray.
        """
        anomaly_score = self._predict(dataloader, **predict_kwargs)
        # Get anomaly score based on predicted loss
        if self._anomaly_score_fn is not None:
            anomaly_score = self._anomaly_score_fn(anomaly_score)
        return anomaly_score
    
    @to_tsdataset(scenario="anomaly_label")
    def predict(
        self,
        tsdataset: TSDataset,
        **predict_kwargs
    ) -> TSDataset:
        """Get anomaly label on a batch. the result are output as tsdataset.

        Args:
            tsdataset(TSDataset): Data to be predicted.
            **predict_kwargs: Additional arguments for `_predict`.

        Returns:
            TSDataset.
        """
        dataloader = self._init_predict_dataloader(tsdataset)
        anomaly_score = self._get_anomaly_score(dataloader, **predict_kwargs)
        anomaly_label = (anomaly_score >= self._threshold) + 0 
        # adjust pred 
        target = tsdataset.get_target()
        if self._pred_adjust and target is not None:
            true_label = np.squeeze(tsdataset.get_target().to_numpy())
            anomaly_label = self._pred_adjust_fn(anomaly_label, true_label[-len(anomaly_label):])
        return anomaly_label
    
    @to_tsdataset(scenario="anomaly_score")
    def predict_score(
        self,
        tsdataset: TSDataset,
        **predict_kwargs
    ) -> TSDataset:
        """Get anomaly score on a batch. the result are output as tsdataset.

        Args:
            tsdataset(TSDataset): Data to be predicted.
            **predict_kwargs: Additional arguments for `_predict`.

        Returns:
            TSDataset.
        """
        dataloader = self._init_predict_dataloader(tsdataset)
        return self._get_anomaly_score(dataloader, **predict_kwargs)
    
    def _predict(
        self, 
        dataloader: paddle.io.DataLoader,
        **predict_kwargs
    ) -> np.ndarray:
        """Predict function core logic.

        Args:
            dataloader(paddle.io.DataLoader): Data to be predicted.
            **predict_kwargs: Additional arguments for for possible use by subclasses.

        Returns:
            np.ndarray.
        """
        self._network.eval()
        loss_list = []
        for _, data in enumerate(dataloader):
            y_pred, y_true = self._network(data)
            loss = self._get_loss(y_pred, y_true)
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
            batch_logs = self._train_batch(data)
            self._callback_container.on_batch_end(batch_idx, batch_logs)
        epoch_logs = {"lr": self._optimizer.get_lr()}
        self._history._epoch_metrics.update(epoch_logs)
    
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
        y_pred, y_true = self._network(X)
        loss = self._compute_loss(y_pred, y_true)
        loss.backward()
        self._optimizer.step()
        self._optimizer.clear_grad()
        batch_logs = {
            "batch_size": y_true.shape[0],
            "loss": loss.item()
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
        for batch_idx, data in enumerate(loader):
            y_pred, y_true = self._predict_batch(data)
            list_y_pred.append(y_pred)
            list_y_true.append(y_true)
        y_pred, y_true = np.vstack(list_y_pred), np.vstack(list_y_true)
        metrics_logs = self._metric_container_dict[name](y_pred, y_true)
        self._history._epoch_metrics.update(metrics_logs)
        self._network.train()
    
    def _predict_batch(
        self, 
        X: Dict[str, paddle.Tensor]
    ) -> np.ndarray:
        """Predict one batch of data.

        Args: 
            X(paddle.Tensor): Feature tensor.

        Returns:
            y_pred(np.ndarray): Prediction results.
            y_true(np.ndarray): Origin data(features).
        """
        y_pred, y_true = self._network(X)
        return y_pred.numpy(), y_true.numpy()

    def _compute_loss(
        self, 
        y_pred: paddle.Tensor, 
        y_true: paddle.Tensor
    ) -> paddle.Tensor:
        """Compute the loss.

        Note:
            This function could be overrided by the subclass if necessary.

        Args:
            y_pred(paddle.Tensor): Estimated target values.
            y_true(paddle.Tensor): Ground truth (correct) target values.

        Returns:
            paddle.Tensor: Loss value.
        """
        return self._loss_fn(y_pred, y_true)
    
    def _get_threshold(
        self,
        anomaly_score: np.ndarray
    ) -> float:
        """Get the threshold value to judge anomaly.
        
        Note:
            This function could be overrided by the subclass if necessary.
            
        Args:
            anomaly_score(np.ndarray): 
            
        Returns:
            float: Thresold value.
        """
        return self._threshold_fn(anomaly_score)

    @abc.abstractmethod
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
        pass

    @abc.abstractmethod
    def _init_network(self) -> paddle.nn.Layer:
        """Setup the network.

        Returns:
            paddle.nn.Layer.
        """
        pass

    def _build_meta(self):
        res = {
            "model_type": "anomaly",
            "ancestor_classname_set": [clazz.__name__ for clazz in self.__class__.mro()],
            "modulename": self.__module__,
            "model_threshold": self._threshold,
            "size": {
                "in_chunk_len": self._in_chunk_len
            },
            "input_data": {}
        }
        for key, value in self._fit_params.items():
            if not isinstance(value, int):
                continue 
            if value != 0:
                res['input_data'][key] = value
        return res
    
    def save(self, path: str, network_model: bool = False, dygraph_to_static: bool = True, batch_size: Optional[int] = None) -> None:
        """
        Saves a AnomalyBaseModel instance to a disk file.

        1> A AnomalyBaseModel (or any child classes inherited from AnomalyBaseModel) instance have a set of member
        variables, they can be divided into 3 categories:
        `pickle-serializable members` (e.g. python built-in type such as int, str, dict, etc.),
        `paddle-related pickle-not-serializable members` (e.g. paddle.nn.Layer, paddle.optimizer.Optimizer),
        `paddle-not-related pickle-not-serializable members`.

        2> To call this method, self._network and self._optimizer must not be None.

        Args:
            path(str): A path string containing a model file name.
            network_model(bool): Save network model structure and parameters separately for Paddle Inference or not, default False.
            dygraph_to_static(bool): Change network from dygraph to static or not, it works when network_model==True, default True.
            batch_size(int): The fixed batch size for the param `input_spec` of network_model save, it works when network_model==True, default None.

        Raises:
            ValueError
        """
        abs_model_path = os.path.abspath(path)
        abs_root_path = os.path.dirname(abs_model_path)
        raise_if_not(
            os.path.exists(abs_root_path),
            "failed to save model, path not exists: %s" % abs_root_path
        )
        raise_if(
            os.path.isdir(abs_model_path),
            "failed to save model, path must be a file, not directory: %s" % abs_model_path
        )
        raise_if(
            os.path.exists(abs_model_path),
            "Failed to save model, target file already exists: %s" % abs_model_path
        )

        raise_if(self._network is None, "failed to save model, model._network must not be None.")
        # raise_if(self._optimizer is None, "failed to save model, model._optimizer must not be None.")

        # path to save other internal files.
        # adding modelname as each internal file name prefix to allow multiple models to be saved at same dir.
        # examples (assume there are 2 models `a` and `b`):
        # a.modelname = "a"
        # a.model_meta_name = "a_model_meta"
        # a.network_statedict = "a_network_statedict"
        # b.modelname = "b"
        # b.model_meta_name = "b_model_meta"
        # b.network_statedict = "b_network_statedict"
        # given above example, adding name prefix avoids conflicts between a.internal files and b.internal files.
        modelname = os.path.basename(abs_model_path)
        internal_filename_map = {
            "model_meta": "%s_%s" % (modelname, "model_meta"),
            "network_statedict": "%s_%s" % (modelname, "network_statedict"),
            "network_model": modelname,
            # currently ignore optimizer.
            # "optimizer_statedict": "%s_%s" % (modelname, "optimizer_statedict"),
        }

        # internal files must not conflict with existing files.
        conflict_files = {*internal_filename_map.values()} - set(os.listdir(abs_root_path))
        raise_if(
            len(conflict_files) < len(internal_filename_map),
            "failed to save model internal files, these files must not exist: %s" % conflict_files
        )

        # start to save
        # 1 save network model and params for paddle inference
        model_meta = self._build_meta()
        if batch_size is not None:
            model_meta['batch_size'] = batch_size
        if network_model:
            self._network.eval()
            input_spec = build_network_input_spec(model_meta)
            try:
                if dygraph_to_static:
                    layer = paddle.jit.to_static(self._network, input_spec=input_spec)
                    paddle.jit.save(layer, os.path.join(abs_root_path, internal_filename_map["network_model"]))
                else:
                    paddle.jit.save(self._network, os.path.join(abs_root_path, internal_filename_map["network_model"]), input_spec=input_spec)
            except Exception as e:
                raise_log(
                    ValueError(
                        "error occurred while saving or dygraph_to_static network_model: %s, err: %s" %
                        (internal_filename_map["network_model"], str(e))
                    )
                )

        # 2 save model meta (e.g. classname)
        try:
            with open(os.path.join(abs_root_path, internal_filename_map["model_meta"]), "w") as f:
                json.dump(model_meta, f, ensure_ascii=False)
        except Exception as e:
            raise_log(
                ValueError("error occurred while saving %s, err: %s" % (internal_filename_map["model_meta"], str(e)))
            )

        # 3 save optimizer state dict (currently ignore optimizer logic.)
        # optimizer_state_dict = self._optimizer.state_dict()
        # try:
        #     paddle.save(
        #         obj=optimizer_state_dict,
        #         path=os.path.join(abs_root_path, internal_filename_map["optimizer_statedict"])
        #     )
        # except Exception as e:
        #     raise_log(
        #         ValueError(
        #             "error occurred while saving %s: %s, err: %s" %
        #             (internal_filename_map["optimizer_statedict"], optimizer_state_dict, str(e))
        #         )
        #     )

        # 4 save network state dict
        network_state_dict = self._network.state_dict()
        try:
            paddle.save(
                obj=network_state_dict,
                path=os.path.join(abs_root_path, internal_filename_map["network_statedict"])
            )
        except Exception as e:
            raise_log(
                ValueError(
                    "error occurred while saving %s: %s, err: %s" %
                    (internal_filename_map["network_statedict"], network_state_dict, str(e))
                )
            )

        # 5 save model
        optimizer = self._optimizer
        network = self._network
        callback_container = self._callback_container

        # _network is inherited from a paddle-related pickle-not-serializable object, so needs to set to None.
        self._network = None
        # _optimizer is inherited from a paddle-related pickle-not-serializable object, so needs to set to None.
        self._optimizer = None
        # _callback_container contains PaddleBaseModel instances, as PaddleBaseModel contains pickle-not-serializable
        # objects `_network` and `_optimizer`, so also needs to set to None.
        self._callback_container = None
        try:
            with open(abs_model_path, "wb") as f:
                pickle.dump(self, f)
        except Exception as e:
            raise_log(ValueError("error occurred while saving %s, err: %s" % (abs_model_path, str(e))))

        # in order to allow a model instance to be saved multiple times, set attrs back.
        self._optimizer = optimizer
        self._network = network
        self._callback_container = callback_container
        return

    @staticmethod
    def load(path: str) -> "AnomalyBaseModel":
        """
        Loads a AnomalyBaseModel from a file.

        As optimizer does not affect the model prediction results, currently optimizer will NOT be loaded.

        Args:
            path(str): A path string containing a model file name.

        Returns:
            AnomalyBaseModel: the loaded AnomalyBaseModel instance.
        """
        abs_path = os.path.abspath(path)
        raise_if_not(os.path.exists(abs_path), "model file does not exist: %s" % abs_path)
        raise_if(os.path.isdir(abs_path), "path must be a file path, not a directory: %s" % abs_path)

        # 1.1 model
        with open(abs_path, "rb") as f:
            model = pickle.load(f)
        raise_if_not(
            isinstance(model, AnomalyBaseModel),
            "loaded model type must be inherited from %s, but actual loaded model type: %s" %
            (AnomalyBaseModel, model.__class__)
        )

        # 1.2 - 1.4 model._network
        model._network = model._init_network()
        raise_if(model._network is None, "model._network must not be None after calling _init_network()")

        modelname = os.path.basename(abs_path)
        network_statedict_filename = "%s_%s" % (modelname, "network_statedict")
        network_statedict_abs_path = os.path.join(os.path.dirname(abs_path), network_statedict_filename)
        network_statedict = paddle.load(network_statedict_abs_path)
        model._network.set_state_dict(network_statedict)

        # 1.5 - 1.7 model._optimizer
        # model._optimizer = model._init_optimizer()
        # raise_if(model._optimizer is None, "model._optimizer must not be None after calling _init_optimizer()")
        #
        # optimizer_statedict_filename = "%s_%s" % (modelname, "optimizer_statedict")
        # optimizer_statedict_abs_path = os.path.join(os.path.dirname(abs_path), optimizer_statedict_filename)
        # optimizer_statedict = paddle.load(optimizer_statedict_abs_path)
        #
        # model._optimizer.set_state_dict(optimizer_statedict)
        return model
