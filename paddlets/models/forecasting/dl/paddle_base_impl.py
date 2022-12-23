#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from typing import List, Dict, Any, Callable, Optional, Tuple, Union
from collections import OrderedDict
from copy import deepcopy
import time
import abc

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
from paddlets.models.forecasting.dl.paddle_base import PaddleBaseModel
from paddlets.models.data_adapter import DataAdapter
from paddlets.models.utils import to_tsdataset, check_tsdataset
from paddlets.datasets import TSDataset
from paddlets.logger import raise_if, Logger

logger = Logger(__name__)


class PaddleBaseModelImpl(PaddleBaseModel, abc.ABC):
    """PaddleTS/PaddleTS deep time series framework, 
        all time series models based on paddlepaddle implementation need to inherit this class.

    Args:
        in_chunk_len(int): The size of the loopback window, i.e. the number of time steps feed to the model.
        out_chunk_len(int): The size of the forecasting horizon, i.e. the number of time steps output by the model.
        skip_chunk_len(int): Optional, the number of time steps between in_chunk and out_chunk for a single sample. 
            The skip chunk is neither used as a feature (i.e. X) nor a label (i.e. Y) for a single sample. 
            By default it will NOT skip any time steps.
        sampling_stride(int): Sampling intervals between two adjacent samples.
        loss_fn(Callable[..., paddle.Tensor]|None): Loss function.
        optimizer_fn(Callable[..., Optimizer]): Optimizer algorithm.
        optimizer_params(Dict[str, Any]): Optimizer parameters.
        eval_metrics(List[str]|List[Metric]): Evaluation metrics of model.
        callbacks(List[Callback]): Customized callback functions.
        batch_size(int): Number of samples per batch.
        max_epochs(int): Max epochs during training.
        verbose(int): Verbosity mode.
        patience(int): Number of epochs to wait for improvement before terminating.
        seed(int|None): Global random seed.

    Attributes:
        _in_chunk_len(int): The size of the loopback window, i.e. the number of time steps feed to the model.
        _out_chunk_len(int): The size of the forecasting horizon, i.e. the number of time steps output by the model.
        _skip_chunk_len(int): Optional, the number of time steps between in_chunk and out_chunk for a single sample. 
            The skip chunk is neither used as a feature (i.e. X) nor a label (i.e. Y) for a single sample. 
            By default it will NOT skip any time steps.
        _sampling_stride(int): Sampling intervals between two adjacent samples.
        _loss_fn(Callable[..., paddle.Tensor]|None): Loss function.
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
        out_chunk_len: int,
        skip_chunk_len: int = 0,
        sampling_stride: int = 1,
        loss_fn: Callable[..., paddle.Tensor] = None,
        optimizer_fn: Callable[..., Optimizer] = paddle.optimizer.Adam,
        optimizer_params: Dict[str, Any] = dict(learning_rate=1e-3),
        eval_metrics: Union[List[str], List[Metric]] = [], 
        callbacks: List[Callback] = [], 
        batch_size: int = 128,
        max_epochs: int = 10,
        verbose: int = 1,
        patience: int = 4,
        seed: Optional[int] = None,
    ):
        super(PaddleBaseModelImpl, self).__init__(
            in_chunk_len=in_chunk_len,
            out_chunk_len=out_chunk_len,
            skip_chunk_len=skip_chunk_len
        )
        self._sampling_stride = sampling_stride
        self._loss_fn = loss_fn
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
            self._eval_metrics = ["mae"]

    def _check_tsdataset(
        self, 
        tsdataset: TSDataset
    ):
        """Ensure the robustness of input data (consistent feature order), at the same time,
            check whether the data types are compatible. If not, the processing logic is as follows.

        Processing logic:

            1> Integer: Convert to np.int64.

            2> Floating: Convert to np.float32.

            3> Missing value: Warning.

            4> Other: Illegal.

        Args:
            tsdataset(TSDataset): Data to be checked.
        """
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
        train_tsdataset: Union[TSDataset, List[TSDataset]], 
        valid_tsdataset: Optional[Union[TSDataset, List[TSDataset]]] = None
    ) -> Tuple[paddle.io.DataLoader, List[paddle.io.DataLoader]]:
        """Generate dataloaders for train and eval set.

        Args: 
            train_tsdataset(Union[TSDataset, List[TSDataset]]): Train set.
            valid_tsdataset(Optional[Union[TSDataset, List[TSDataset]]]): Eval set.

        Returns:
            paddle.io.DataLoader: Training dataloader.
            List[paddle.io.DataLoader]: List of validation dataloaders..
        """
        data_adapter = DataAdapter()
        train_dataset = None
        if isinstance(train_tsdataset, TSDataset):
            train_tsdataset = [train_tsdataset]
        for dataset in train_tsdataset:
            self._check_tsdataset(dataset)
            dataset = data_adapter.to_sample_dataset(
                dataset,
                in_chunk_len=self._in_chunk_len,
                out_chunk_len=self._out_chunk_len,
                skip_chunk_len=self._skip_chunk_len,
                sampling_stride=self._sampling_stride
            )
            if train_dataset is None:
                train_dataset = dataset
            else:
                train_dataset.samples = train_dataset.samples + dataset.samples
        #The design here is to return one dataloader instead of multiple dataloaders, which can ensure the accuracy of shuffle logic
        train_dataloader = data_adapter.to_paddle_dataloader(train_dataset, self._batch_size)
        valid_dataloaders = []
        if valid_tsdataset is not None:
            valid_dataset = None
            if isinstance(valid_tsdataset, TSDataset):
                valid_tsdataset = [valid_tsdataset]
            for dataset in valid_tsdataset:
                self._check_tsdataset(dataset)
                dataset = data_adapter.to_sample_dataset(
                    dataset,
                    in_chunk_len=self._in_chunk_len,
                    out_chunk_len=self._out_chunk_len,
                    skip_chunk_len=self._skip_chunk_len,
                    sampling_stride=self._sampling_stride
                )
                if valid_dataset is None:
                    valid_dataset = dataset
                else:
                    valid_dataset.samples = valid_dataset.samples + dataset.samples
            valid_dataloader = data_adapter.to_paddle_dataloader(valid_dataset, self._batch_size)
            valid_dataloaders.append(valid_dataloader)
        return train_dataloader, valid_dataloaders

    def _init_predict_dataloader(
        self, 
        tsdataset: TSDataset, 
    ) -> paddle.io.DataLoader:
        """Generate dataloaders for data to be predicted.

        Args: 
            tsdataset(TSDataset): Data to be predicted.

        Returns:
            paddle.io.DataLoader: dataloader. 
        """
        self._check_tsdataset(tsdataset)
        boundary = (
            len(tsdataset.get_target().data) - 1 + self._skip_chunk_len + self._out_chunk_len
        )
        data_adapter = DataAdapter()
        dataset = data_adapter.to_sample_dataset(
            tsdataset,
            in_chunk_len=self._in_chunk_len,
            out_chunk_len=self._out_chunk_len,
            skip_chunk_len=self._skip_chunk_len,
            sampling_stride=self._sampling_stride,
            time_window=(boundary, boundary)
        )
        dataloader = data_adapter.to_paddle_dataloader(dataset, self._batch_size)
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
        train_tsdataset: Union[TSDataset, List[TSDataset]], 
        valid_tsdataset: Optional[Union[TSDataset, List[TSDataset]]] = None
    ):
        """Train a neural network stored in self._network, 
            Using train_dataloader for training data and valid_dataloader for validation.

        Args: 
            train_tsdataset(Union[TSDataset, List[TSDataset]]): Train set. 
            valid_tsdataset(Optional[Union[TSDataset, List[TSDataset]]]): Eval set, used for early stopping.
        """
        if isinstance(train_tsdataset, TSDataset):
            train_tsdataset = [train_tsdataset]
        if isinstance(valid_tsdataset, TSDataset):
            valid_tsdataset = [valid_tsdataset]
        self._check_multi_tsdataset(train_tsdataset)
        self._fit_params = self._update_fit_params(train_tsdataset, valid_tsdataset)
        
        if isinstance(valid_tsdataset, list):
            self._check_multi_tsdataset(valid_tsdataset)
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
    
    @to_tsdataset(scenario="forecasting")
    def predict(
        self,
        tsdataset: TSDataset
    ) -> TSDataset:
        """Make predictions on a batch. the result are output as tsdataset.

        Args:
            tsdataset(TSDataset): Data to be predicted.

        Returns:
            TSDataset.
        """
        dataloader = self._init_predict_dataloader(tsdataset)
        return self._predict(dataloader)

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
        results = []
        for batch_nb, data in enumerate(dataloader):
            X, _ = self._prepare_X_y(data)
            output = self._network(X)
            predictions = output.numpy()
            results.append(predictions)
        results = np.vstack(results)
        return results

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
            X, y = self._prepare_X_y(data)
            batch_logs = self._train_batch(X, y)
            self._callback_container.on_batch_end(batch_idx, batch_logs)
        epoch_logs = {"lr": self._optimizer.get_lr()}
        self._history._epoch_metrics.update(epoch_logs)
    
    def _train_batch(
        self, 
        X: Dict[str, paddle.Tensor], 
        y: paddle.Tensor
    ) -> Dict[str, Any]:
        """Trains one batch of data.

        Args:
            X(Dict[str, paddle.Tensor]): Dict of feature tensor.
            y(paddle.Tensor): Target tensor.

        Returns:
            Dict[str, Any]: Dict of logs.
        """
        output = self._network(X)
        loss = self._compute_loss(output, y)
        loss.backward()
        self._optimizer.step()
        self._optimizer.clear_grad()
        batch_logs = {
            "batch_size": y.shape[0],
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
        list_y_true, list_y_score = [], []
        for batch_idx, data in enumerate(loader):
            X, y = self._prepare_X_y(data)
            scores = self._predict_batch(X)
            list_y_true.append(y)
            list_y_score.append(scores)
        y_true, scores = np.vstack(list_y_true), np.vstack(list_y_score)
        metrics_logs = self._metric_container_dict[name](y_true, scores)
        self._history._epoch_metrics.update(metrics_logs)
        self._network.train()
    
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
        scores = self._network(X)
        return scores.numpy()

    def _prepare_X_y(self, 
        X: Dict[str, paddle.Tensor]
    ) -> Tuple[Dict[str, paddle.Tensor], paddle.Tensor]:
        """Split the packet into X, y.

        Note:
            This function could be overrided by the subclass if necessary.

        Args:
            X(Dict[str, paddle.Tensor]): Dict of feature/target tensor.

        Returns:
            X(Dict[str, paddle.Tensor]): Dict of feature tensor. 
            y(paddle.Tensor): Target tensor.
        """
        if "future_target" in X:
            y = X.pop("future_target")
            return X, y
        return X, None

    def _compute_loss(
        self, 
        y_score: paddle.Tensor, 
        y_true: paddle.Tensor
    ) -> paddle.Tensor:
        """Compute the loss.

        Note:
            This function could be overrided by the subclass if necessary.

        Args:
            y_score(paddle.Tensor): Estimated target values.
            y_true(paddle.Tensor): Ground truth (correct) target values.

        Returns:
            paddle.Tensor: Loss value.
        """
        return self._loss_fn(y_score, y_true)

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
        res = super()._build_meta()
        for key, value in self._fit_params.items():
            if not isinstance(value, int):
                continue 
            if value != 0:
                res['input_data'][key] = value
        return res

