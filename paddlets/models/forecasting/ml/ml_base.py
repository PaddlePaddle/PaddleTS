# !/usr/bin/env python3
# -*- coding:utf-8 -*-

from paddlets.models.base import BaseModel
from paddlets import TSDataset
from paddlets.logger import Logger, raise_if_not, raise_if, raise_log

import os
import abc
import pickle
from typing import Optional, List, Union
import json

logger = Logger(__file__)


class MLBaseModel(BaseModel, metaclass=abc.ABCMeta):
    """
    Base class for all machine learning models.

    Args:
        in_chunk_len(int): The length of past target time series chunk for a single sample.
        out_chunk_len(int): The length of future target time series chunk for a single sample.
        skip_chunk_len(int): The length of time series chunk between past target and future target for a single sample.
             The skip chunk are neither used as feature (i.e. X) nor label (i.e. Y) for a single sample.
    """
    def __init__(
        self,
        in_chunk_len: int,
        out_chunk_len: int,
        skip_chunk_len: int
    ):
        super(MLBaseModel, self).__init__(
            in_chunk_len=in_chunk_len,
            out_chunk_len=out_chunk_len,
            skip_chunk_len=skip_chunk_len
        )

    @abc.abstractmethod
    def fit(
        self,
        train_data: Union[TSDataset, List[TSDataset]],
        valid_data: Optional[Union[TSDataset, List[TSDataset]]] = None
    ):
        """
        Fit a machine learning model instance.

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
            TSDataset: TSDataset with predictions.
        """
        pass

    def save(self, path: str) -> None:
        """
        Saves a MLBaseModel instance to a disk file.

        Args:
            path(str): a path string containing a model file name.

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

        # path to save other internal files.
        # adding modelname as each internal file name prefix to allow multiple models to be saved at same dir.
        # examples (assume there are 2 models `a` and `b`):
        # a.modelname = "a"
        # a.model_meta_name = "a_model_meta"
        # b.modelname = "b"
        # b.model_meta_name = "b_model_meta"
        # given above example, adding name prefix avoids conflicts between a.internal files and b.internal files.
        modelname = os.path.basename(abs_model_path)
        internal_filename_map = {"model_meta": "%s_%s" % (modelname, "model_meta")}

        # internal files must not conflict with existing files.
        conflict_files = {*internal_filename_map.values()} - set(os.listdir(abs_root_path))
        raise_if(
            len(conflict_files) < len(internal_filename_map),
            "failed to save model internal files, these files cause conflict: %s" % conflict_files
        )

        # start to save.
        # 1 save model
        try:
            with open(abs_model_path, "wb") as f:
                pickle.dump(self, f)
        except Exception as e:
            raise_log(ValueError("error occurred while saving model: %s, error: %s" % (abs_model_path, str(e))))

        # 2 save model meta
        model_meta = {
            # ChildModel,MLBaseModel,BaseModel,Trainable,object
            "ancestor_classname_set": [clazz.__name__ for clazz in self.__class__.mro()],
            # paddlets.models.ml.xxx
            "modulename": self.__module__
        }
        try:
            with open(os.path.join(abs_root_path, internal_filename_map["model_meta"]), "w") as f:
                json.dump(model_meta, f, ensure_ascii=False)
        except Exception as e:
            raise_log(
                ValueError("error occurred while saving %s, err: %s" % (internal_filename_map["model_meta"], str(e)))
            )
        return

    @staticmethod
    def load(path: str) -> "BaseModel":
        """
        Loads a MLBaseModel instance from a file.

        Args:
            path(str): A path string containing a model file name.

        Returns:
            BaseModel: A loaded MLBaseModel instance.
        """
        abs_path = os.path.abspath(path)
        raise_if_not(os.path.exists(abs_path), "model file does not exist: %s" % abs_path)
        raise_if(os.path.isdir(abs_path), "path must be a file path, not a directory: %s" % abs_path)

        with open(path, "rb") as f:
            model = pickle.load(f)
        raise_if_not(
            isinstance(model, MLBaseModel),
            "loaded model type must be inherited from %s, but actual loaded model type: %s" %
            (MLBaseModel, model.__class__)
        )
        return model
