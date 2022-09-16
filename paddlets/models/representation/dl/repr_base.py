#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from typing import List, Dict, Any, Callable, Optional, Tuple
from copy import deepcopy
import time
import abc

from paddle.optimizer import Optimizer
import numpy as np
import paddle

from paddlets.models.common.callbacks import (
    CallbackContainer,
    Callback,
    History,
)
from paddlets.models.representation.dl.adapter import ReprDataAdapter
from paddlets.models.utils import check_tsdataset
from paddlets.logger import raise_if, Logger
from paddlets.datasets import TSDataset

logger = Logger(__name__)


class ReprBaseModel(abc.ABC):
    """PaddleTS/PaddleTS deep time series representation framework, 
        all time series models based on paddlepaddle implementation need to inherit this class.

    Args:
        segment_size(int): The size of the loopback window, i.e. the number of time steps feed to the model.
        sampling_stride(int): Sampling intervals between two adjacent samples.
        optimizer_fn(Callable[..., Optimizer]): Optimizer algorithm.
        optimizer_params(Dict[str, Any]): Optimizer parameters.
        callbacks(List[Callback]): Customized callback functions.
        batch_size(int): Number of samples per batch.
        max_epochs(int): Max epochs during training.
        verbose(int): Verbosity mode.
        seed(int|None): Global random seed.

    Attributes:
        _segment_size(int): The size of the loopback window, i.e. the number of time steps feed to the model.
        _sampling_stride(int): Sampling intervals between two adjacent samples.
        _optimizer_fn(Callable[..., Optimizer]): Optimizer algorithm.
        _optimizer_params(Dict[str, Any]): Optimizer parameters.
        _callbacks(List[Callback]): Customized callback functions.
        _batch_size(int): Number of samples per batch.
        _max_epochs(int): Max epochs during training.
        _verbose(int): Verbosity mode.
        _seed(int|None): Global random seed.

        _fit_params(Dict[str, Any]): Infer parameters by TSdataset automatically.
        _network(paddle.nn.Layer): Network structure.
        _optimizer(Optimizer): Optimizer.
        _history(History): Callback that records events into a `History` object.
        _callback_container(CallbackContainer): Container holding a list of callbacks.
    """
    def __init__(
        self,
        segment_size: int,
        sampling_stride: int = 1,
        optimizer_fn: Callable[..., Optimizer] = paddle.optimizer.Adam,
        optimizer_params: Dict[str, Any] = dict(learning_rate=1e-3),
        callbacks: List[Callback] = [], 
        batch_size: int = 128,
        max_epochs: int = 10,
        verbose: int = 1,
        seed: Optional[int] = None,
    ):
        super(ReprBaseModel, self).__init__()
        self._segment_size = segment_size
        self._sampling_stride = sampling_stride
        self._optimizer_fn = optimizer_fn
        self._optimizer_params = deepcopy(optimizer_params)
        self._callbacks = deepcopy(callbacks)
        self._batch_size = batch_size
        self._max_epochs = max_epochs
        self._verbose = verbose
        
        self._fit_params = None
        self._network = None
        self._optimizer = None
        self._history = None
        self._callback_container = None

        self._check_params()
        if seed is not None:
            paddle.seed(seed)
        
    def _check_params(self):
        """Parameter validity verification, the check logic is as follow:

            batch_size: batch_size must be > 0.

            max_epochs: max_epochs must be > 0.

            verbose: verbose must be > 0.
        """
        raise_if(self._batch_size <= 0, f"batch_size must be > 0, got {self._batch_size}.")
        raise_if(self._max_epochs <= 0, f"max_epochs must be > 0, got {self._max_epochs}.")
        raise_if(self._verbose <= 0, f"verbose must be > 0, got {self._verbose}.")

    def _check_tsdataset(self, tsdataset: TSDataset):
        """Ensure the robustness of input data (consistent feature order), at the same time,
            check whether the data types are compatible. If not, the processing logic is as follows:

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

    def _init_fit_dataloader(
        self, 
        train_tsdataset: TSDataset
    ) -> paddle.io.DataLoader:
        """Generate dataloader for train set.

        Args: 
            train_tsdataset(TSDataset): Train set.

        Returns:
            paddle.io.DataLoader: Training dataloader.
        """
        self._check_tsdataset(train_tsdataset)
        data_adapter = ReprDataAdapter()
        train_dataset = data_adapter.to_paddle_dataset(
            train_tsdataset,
            segment_size=self._segment_size,
            sampling_stride=self._sampling_stride,
        )
        return data_adapter.to_paddle_dataloader(train_dataset, self._batch_size)

    def _init_encode_dataloader(
        self, 
        tsdataset: TSDataset
    ) -> paddle.io.DataLoader:
        """Generate dataloader for data to be encoded.

        Args: 
            tsdataset(TSDataset): Data to be encoded.

        Returns:
            paddle.io.DataLoader: dataloader.
        """
        self._check_tsdataset(tsdataset)
        data_adapter = ReprDataAdapter()
        dataset = data_adapter.to_paddle_dataset(
            tsdataset, 
            segment_size=len(tsdataset.get_target().data),
            sampling_stride=self._sampling_stride,
        )
        return data_adapter.to_paddle_dataloader(dataset, self._batch_size)

    def _init_callbacks(self) -> Tuple[History, CallbackContainer]:
        """Setup the callbacks functions.

        Returns:
            History: Callback that records events into a `History` object.
            CallbackContainer: Container holding a list of callbacks.
        """
        # Set callback functions, including history, etc..
        history, callbacks = History(self._verbose), [] # nqa
        callbacks.append(history)
        if self._callbacks:
            callbacks.extend(self._callbacks)
        callback_container = CallbackContainer(callbacks)
        callback_container.set_trainer(self)
        return history, callback_container
    
    def fit(self, train_tsdataset: TSDataset):
        """Train a neural network stored in self._network, 
            Using train_dataloader for training data. 

        Args: 
            train_tsdataset(TSDataset): Train set. 
        """
        self._fit_params = self._update_fit_params(train_tsdataset)
        train_dataloader = self._init_fit_dataloader(train_tsdataset)
        self._fit(train_dataloader)
        
    def _fit(self, train_dataloader: paddle.io.DataLoader):
        """Fit function core logic. 

        Args: 
            train_dataloader(paddle.io.DataLoader): Train set. 
        """
        self._history, self._callback_container = self._init_callbacks()
        self._network = self._init_network()
        self._optimizer = self._init_optimizer()

        # Call the `on_train_begin` method of each callback before the training starts.
        self._callback_container.on_train_begin(
            logs={"start_time": time.time()}
        )
        for epoch_idx in range(self._max_epochs):

            # Call the `on_epoch_begin` method of each callback before the epoch starts.
            self._callback_container.on_epoch_begin(epoch_idx)

            self._train_epoch(train_dataloader)

            # Call the `on_epoch_end` method of each callback at the end of the epoch.
            self._callback_container.on_epoch_end(
                epoch_idx, logs=self._history._epoch_metrics
            )

        # Call the `on_train_end` method of each callback at the end of the training.
        self._callback_container.on_train_end()
        self._network.eval()

    def encode(
        self, 
        tsdataset: TSDataset,
        **encode_params,
    ) -> np.ndarray:
        """Compute representations using the model.

        Args:
             tsdataset(TSDataset): Data to be encoded.
             encode_params: Keyword parameters of encoding functions.

        Returns:
            np.ndarray.
        """
        dataloader = self._init_encode_dataloader(tsdataset)
        return self._encode(dataloader, **encode_params)

    @abc.abstractmethod 
    def _encode(
        self, 
        dataloader: paddle.io.DataLoader,
        **encode_params,
    ) -> TSDataset:
        """Encode function core logic.

        Args:
             dataloader(paddle.io.DataLoader): Data to be encoded.
             encode_params: Keyword parameters of encoding functions.

        Returns:
            np.ndarray.
        """
        pass
    
    def _train_epoch(self, train_loader: paddle.io.DataLoader):
        """Trains one epoch of the network in self._network.

        Args: 
            train_loader(paddle.io.DataLoader): Training dataloader.
        """
        self._network.train()
        for batch_idx, X in enumerate(train_loader):
            self._callback_container.on_batch_begin(batch_idx)
            batch_logs = self._train_batch(X)
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

        Returns:
            Dict[str, Any]: Dict of logs.
        """
        loss = self._compute_loss(X)
        loss.backward()
        self._optimizer.step()
        self._optimizer.clear_grad()
        batch_logs = {
            "batch_size": self._batch_size,
            "loss": loss.item()
        }
        return batch_logs 

    @abc.abstractmethod
    def _compute_loss(
        self,
        X: Dict[str, paddle.Tensor]
    ) -> paddle.Tensor:
        """Compute the loss.

        Args:
            X(Dict[str, paddle.Tensor]): Dict of feature tensor.

        Returns:
            paddle.Tensor: Loss value.
        """
        pass

    @abc.abstractmethod
    def _update_fit_params(
        self, 
        train_tsdataset: TSDataset
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
