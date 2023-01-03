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
from paddle.nn import CrossEntropyLoss
from sklearn.utils import check_random_state
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
from paddlets.models.classify.dl.adapter.data_adapter import ClassifyDataAdapter
from paddlets.models.classify.base import BaseClassifier
from paddlets.datasets import TSDataset
from paddlets.logger import raise_if, raise_if_not, raise_log, Logger

logger = Logger(__name__)


class PaddleBaseClassifier(BaseClassifier):
    """Base class for all paddle deep time series classify models.

    Args:
        loss_fn(Callable[..., paddle.Tensor]|None): Loss function.
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
        _classes_（ndarray）: ndarray of class labels, possibly strings
        _n_class(int) : number of unique labels
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
        loss_fn: Callable[..., paddle.Tensor] = None,
        optimizer_fn: Callable[..., Optimizer] = paddle.optimizer.Adam,
        optimizer_params: Dict[str, Any] = dict(learning_rate=1e-3),
        eval_metrics: List[str] = [], 
        callbacks: List[Callback] = [], 
        batch_size: int = 32,
        max_epochs: int = 10,
        verbose: int = 1,
        patience: int = 4,
        seed: Optional[int] = None,
    ):
        super(PaddleBaseClassifier, self).__init__()
        self._loss_fn = loss_fn
        self._optimizer_fn = optimizer_fn
        self._optimizer_params = deepcopy(optimizer_params)
        self._eval_metrics = deepcopy(eval_metrics)
        self._callbacks = deepcopy(callbacks)
        self._batch_size = batch_size
        self._max_epochs = max_epochs
        self._verbose = verbose
        self._patience = patience
        self._seed = seed
        self._stop_training = False
        
        self._fit_params = None
        self._network = None
        self._optimizer = None
        self._metrics = None
        self._metrics_names = None
        self._metric_container_dict = None
        self._history = None
        self._callback_container = None
        self._classes_ = []
        self._n_class = 0

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

    def _check_tsdatasets(
        self, 
        tsdatasets: List[TSDataset],
        labels: np.ndarray
    ):
        """Ensure the robustness of input data (consistent feature order), at the same time,
            check whether the data types are compatible. If not, the processing logic is as follows.

        Processing logic:

            1> Floating: Convert to np.float32.

            2> Missing value: Warning.

            3> Other: Illegal.

        Args:
            tsdataset(TSDataset): Data to be checked.
            labels:(np.ndarray) : The data class labels

        """
        for i in range(len(tsdatasets)):
            self.check_tsdataset(tsdatasets[i])

    def check_tsdataset(self, tsdataset: TSDataset):
        """Ensure the robustness of input data (consistent feature order), at the same time,
        check whether the data types are compatible. If not, the processing logic is as follows.

            1> Floating: Convert to np.float32.

            2> Missing value: Warning.

            3> Other: Illegal.

        Args:
            tsdataset(TSDataset): Data to be checked.
        """
        new_dtypes = {}
        for column, dtype in tsdataset.dtypes.items():
            if np.issubdtype(dtype, np.floating):
                new_dtypes.update({column: "float32"})
            else:
                msg = f"{dtype} data type not supported, the illegal columns contains: " \
                      + f"{tsdataset.dtypes.index[tsdataset.dtypes == dtype].tolist()}"
                raise_log(TypeError(msg))

            # Check whether the data contains NaN.
            if np.isnan(tsdataset[column]).any() or np.isinf(tsdataset[column]).any():
                msg = f"np.inf or np.NaN, which may lead to unexpected results from the model"
                msg = f"Input `{column}` contains {msg}."
                logger.warning(msg)

        if new_dtypes:
            tsdataset.astype(new_dtypes)

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
        train_tsdatasets: List[TSDataset],
        train_labels: np.ndarray,
        valid_tsdatasets: List[TSDataset] = None,
        valid_labels: np.ndarray = None,
        shuffle: bool = True
    ) -> Tuple[paddle.io.DataLoader, List[paddle.io.DataLoader]]:
        """Generate dataloaders for train and eval set.

        Args: 
            train_tsdatasets(TSDataset): Train set.
            train_labels:(np.ndarray) : The train data class labels
            valid_tsdatasets(TSDataset|None): Eval set.
            valid_labels:(np.ndarray) : The valid data class labels
            shuffle(bool): Shuffle or not.

        Returns:
            paddle.io.DataLoader: Training dataloader.
            List[paddle.io.DataLoader]: List of validation dataloaders..
        """
        self._check_tsdatasets(train_tsdatasets, train_labels)
        data_adapter = ClassifyDataAdapter()
        train_dataset = data_adapter.to_paddle_dataset(
            train_tsdatasets,
            train_labels,
        )
        self._n_classes = train_dataset.n_classes_
        self._classes_ = train_dataset.classes_
        train_dataloader = data_adapter.to_paddle_dataloader(train_dataset, self._batch_size, shuffle=shuffle)

        if valid_tsdatasets is None:
            valid_dataloader = None
        else:
            self._check_tsdatasets(valid_tsdatasets, valid_labels)
            valid_dataset = data_adapter.to_paddle_dataset(
                valid_tsdatasets,
                valid_labels,
            )
            valid_dataloader = data_adapter.to_paddle_dataloader(valid_dataset, self._batch_size, shuffle=shuffle)

        return train_dataloader, valid_dataloader

    def _init_predict_dataloader(
        self, 
        tsdatasets: List[TSDataset],
        labels: np.ndarray = None
    ) -> paddle.io.DataLoader:
        """Generate dataloaders for data to be predicted.

        Args: 
            tsdataset(TSDataset): Data to be predicted.
            labels:(np.ndarray) : The predicted data class labels

        Returns:
            paddle.io.DataLoader: dataloader. 
        """
        self._check_tsdatasets(tsdatasets, labels)
        data_adapter = ClassifyDataAdapter()
        dataset = data_adapter.to_paddle_dataset(
            tsdatasets,
            labels,
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
        train_tsdatasets: List[TSDataset],
        train_labels: np.ndarray,
        valid_tsdatasets: List[TSDataset] = None,
        valid_labels: np.ndarray = None
    ):
        """
        Train a neural network stored in self._network, using train_dataloader for training data and valid_dataloader
        for validation.

        Args: 
            train_tsdataset(TSDataset): Train set.
            train_labels:(np.ndarray) : The train data class labels
            valid_tsdataset(TSDataset|None): Eval set, used for early stopping.
            valid_labels:(np.ndarray) : The valid data class labels
        """
        self._fit_params = self._update_fit_params(train_tsdatasets, train_labels, valid_tsdatasets, valid_labels)
        train_dataloader, valid_dataloader = self._init_fit_dataloaders(train_tsdatasets, train_labels, valid_tsdatasets, valid_labels)
        self._fit(train_dataloader, valid_dataloader)
        
    def _fit(
        self, 
        train_dataloader: paddle.io.DataLoader,
        valid_dataloader: List[paddle.io.DataLoader] = None
    ):
        """Fit function core logic. 

        Args: 
            train_dataloader(paddle.io.DataLoader): Train set. 
            valid_dataloader(paddle.io.DataLoader|None): Eval set.
        """
        valid_names = [] if valid_dataloader is None else ["val_0"]
        self._metrics, self._metrics_names, \
            self._metric_container_dict =  self._init_metrics(valid_names)
        self._history, self._callback_container = self._init_callbacks()
        self._network = self._init_network()
        self._optimizer = self._init_optimizer()
        check_random_state(self._seed)

        # Call the `on_train_begin` method of each callback before the training starts.
        self._callback_container.on_train_begin({"start_time": time.time()})
        for epoch_idx in range(self._max_epochs):

            # Call the `on_epoch_begin` method of each callback before the epoch starts.
            self._callback_container.on_epoch_begin(epoch_idx)
            self._train_epoch(train_dataloader)

            if len(valid_names) > 0:
                self._predict_epoch(valid_names[0], valid_dataloader)

            # Call the `on_epoch_end` method of each callback at the end of the epoch.
            self._callback_container.on_epoch_end(
                epoch_idx, logs=self._history._epoch_metrics
            )
            if self._stop_training:
                break

        # Call the `on_train_end` method of each callback at the end of the training.
        self._callback_container.on_train_end()
        self._network.eval()

    def predict(
        self,
        tsdatasets: List[TSDataset],
    ) -> np.ndarray:
        """Predict labels. the result are output as ndarray.

        Args:
            tsdataset(List[TSDataset]) : Data to be predicted.
        Returns:
            np.ndarray.
        """
        dataloader = self._init_predict_dataloader(tsdatasets)
        probs = self._predict(dataloader)
        # np.save('probs',probs)
        rng = check_random_state(self._seed)
        return np.array(
            [
                self._classes_[int(rng.choice(np.flatnonzero(prob == prob.max())))]
                for prob in probs
            ]
        )
    
    def predict_proba(
        self,
        tsdatasets: List[TSDataset]
    ) -> np.ndarray:
        """Find probability estimates for each class for all cases.

        Args:
            tsdataset(List[TSDataset]) : Data to be predicted.
            labels:(np.ndarray) : The predicted data class labels
        Returns:
            np.ndarray.
        """
        dataloader = self._init_predict_dataloader(tsdatasets)
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

        # check if binary classification
        if results.shape[1] == 1:
            # first column is probability of class 0 and second is of class 1
            probs = np.hstack([1 - results, results])
        results = results / results.sum(axis=1, keepdims=1)

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
            X(Dict[str, paddle.Tensor]): Dict of feature tensor.

        Returns:
            X(Dict[str, paddle.Tensor]): Dict of feature tensor. 
            y(paddle.Tensor): feature tensor.
        """
        y = X['label']
        return X, y

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
        if self._loss_fn == paddle.nn.functional.cross_entropy:
            return paddle.nn.functional.cross_entropy(y_score, y_true, soft_label=True)
        else:
            return self._loss_fn(y_score, y_true)

    def score(
        self,
        tsdatasets: List[TSDataset],
        labels: np.ndarray
    ) -> float:
        """Scores predicted labels against ground truth labels on X.

        Args:
            tsdataset(List[TSDataset]) : Data to be predicted.
            labels:(np.ndarray) : The predicted data class labels
        Returns:
            float, accuracy score of predict(X) vs y
        """
        from sklearn.metrics import accuracy_score
        preds = self.predict(tsdatasets)
        return accuracy_score(labels, preds, normalize=True)

    @abc.abstractmethod
    def _update_fit_params(
        self,
        train_tsdatasets: List[TSDataset],
        train_labels: np.ndarray,
        valid_tsdatasets: List[TSDataset],
        valid_labels: np.ndarray
    ) -> Dict[str, Any]:
        """Infer parameters by TSdataset automatically.

        Args: 
            train_tsdatasets: List[TSDataset],
            train_labels: np.ndarray,
            valid_tsdatasets: List[TSDataset],
            valid_labels: np.ndarray
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
    
    def save(self, path: str) -> None:
        """
        Saves a PaddleBaseClassifier instance to a disk file.

        Args:
            path(str): A path string containing a model file name.

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
        # 1 save optimizer state dict (currently ignore optimizer logic.)
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

        # 2 save network state dict
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

        # 3 save model
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

        # 4 save model meta (e.g. classname)
        model_meta = {
            # ChildModel,PaddleBaseModelImpl,PaddleBaseModel,BaseModel,Trainable,ABC,object
            "ancestor_classname_set": [clazz.__name__ for clazz in self.__class__.mro()],
            "modulename": self.__module__
        }
        try:
            with open(os.path.join(abs_root_path, internal_filename_map["model_meta"]), "w") as f:
                json.dump(model_meta, f, ensure_ascii=False)
        except Exception as e:
            raise_log(
                ValueError("error occurred while saving %s, err: %s" % (internal_filename_map["model_meta"], str(e)))
            )

        # in order to allow a model instance to be saved multiple times, set attrs back.
        self._optimizer = optimizer
        self._network = network
        self._callback_container = callback_container
        return

    @staticmethod
    def load(path: str) -> "PaddleBaseClassifier":
        """
        Loads a PaddleBaseClassifier from a file.

        Args:
            path(str): A path string containing a model file name.

        Returns:
            PaddleBaseClassifier: the loaded PaddleBaseClassifier instance.
        """
        abs_path = os.path.abspath(path)
        raise_if_not(os.path.exists(abs_path), "model file does not exist: %s" % abs_path)
        raise_if(os.path.isdir(abs_path), "path must be a file path, not a directory: %s" % abs_path)

        # 1.1 model
        with open(abs_path, "rb") as f:
            model = pickle.load(f)
        raise_if_not(
            isinstance(model, PaddleBaseClassifier),
            "loaded model type must be inherited from %s, but actual loaded model type: %s" %
            (PaddleBaseClassifier, model.__class__)
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
