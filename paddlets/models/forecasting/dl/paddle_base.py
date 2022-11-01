# !/usr/bin/env python3
# -*- coding:utf-8 -*-

from paddlets.models.base import BaseModel
from paddlets import TSDataset
from paddlets.logger import raise_if, raise_if_not, raise_log

import os
import pickle
import paddle
import abc
from typing import Optional, List, Union
import json


class PaddleBaseModel(BaseModel, metaclass=abc.ABCMeta):
    """
    Base class for all paddle deep learning models.

    Args:
        in_chunk_len(int): The size of the loopback window, i.e., the number of time steps feed to the model.
        out_chunk_len(int): The size of the forecasting horizon, i.e., the number of time steps output by the model.
        skip_chunk_len(int): Optional, the number of time steps between in_chunk and out_chunk for a single sample.
            The skip chunk is neither used as a feature (i.e. X) nor a label (i.e. Y) for a single sample. By
            default, it will NOT skip any time steps.

    Attributes:
        _network(paddle.nn.Layer): A paddle.nn.Layer instance.
        _optimizer(paddle.optimizer.Optimizer) A paddle.optimizer.Optimizer instance.
        _callback_container(paddlets.models.dl.paddlepaddle.callbacks.CallbackContainer): a container containing one or more
            callback instance(s).
    """
    def __init__(
        self,
        in_chunk_len: int,
        out_chunk_len: int,
        skip_chunk_len: int = 0
    ):
        super(PaddleBaseModel, self).__init__(
            in_chunk_len=in_chunk_len,
            out_chunk_len=out_chunk_len,
            skip_chunk_len=skip_chunk_len
        )
        self._network = None
        self._optimizer = None
        self._callback_container = None

    @abc.abstractmethod
    def fit(
        self,
        train_data: Union[TSDataset, List[TSDataset]],
        valid_data: Optional[Union[TSDataset, List[TSDataset]]] = None
    ):
        """
        Fit a paddle deep learning model instance.

        Any non-abstract classes inherited from this class should implement this method.

        Args:
            train_data(Union[TSDataset, List[TSDataset]]): training dataset.
            valid_data(Optional[Union[TSDataset, List[TSDataset]]]): validation dataset, optional.
        """
        pass

    @abc.abstractmethod
    def predict(self, data: TSDataset) -> TSDataset:
        """
        Make prediction.

        Any non-abstract classes inherited from this class should implement this method.

        Args:
            data(TSDataset): TSDataset to predict.

        Returns:
            TSDataset: Predicted result, in type of TSDataset.
        """
        pass

    @abc.abstractmethod
    def _init_network(self) -> paddle.nn.Layer:
        """
        Internal method, used for initializing a paddle.nn.Layer instance for current model.

        Any non-abstract classes inherited from this class should implement this method.

        Returns:
            paddle.nn.Layer: An initialized paddle.nn.Layer instance.
        """
        pass

    def save(self, path: str) -> None:
        """
        Saves a PaddleBaseModel instance to a disk file.

        1> A PaddleBaseModel (or any child classes inherited from PaddleBaseModel) instance have a set of member
        variables, they can be divided into 3 categories:
        `pickle-serializable members` (e.g. python built-in type such as int, str, dict, etc.),
        `paddle-related pickle-not-serializable members` (e.g. paddle.nn.Layer, paddle.optimizer.Optimizer),
        `paddle-not-related pickle-not-serializable members`.

        2> To call this method, self._network and self._optimizer must not be None.

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
            # paddlets.models.dl.paddlepaddle.xxx
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
    def load(path: str) -> "PaddleBaseModel":
        """
        Loads a PaddleBaseModel from a file.

        As optimizer does not affect the model prediction results, currently optimizer will NOT be loaded.

        Args:
            path(str): A path string containing a model file name.

        Returns:
            PaddleBaseModel: the loaded PaddleBaseModel instance.
        """
        abs_path = os.path.abspath(path)
        raise_if_not(os.path.exists(abs_path), "model file does not exist: %s" % abs_path)
        raise_if(os.path.isdir(abs_path), "path must be a file path, not a directory: %s" % abs_path)

        # 1.1 model
        with open(abs_path, "rb") as f:
            model = pickle.load(f)
        raise_if_not(
            isinstance(model, PaddleBaseModel),
            "loaded model type must be inherited from %s, but actual loaded model type: %s" %
            (PaddleBaseModel, model.__class__)
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
