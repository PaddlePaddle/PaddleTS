# !/usr/bin/env python3
# -*- coding:utf-8 -*-

import abc
from typing import Optional
import numpy as np
import pandas as pd
import math

from paddlets import TSDataset, TimeSeries
from paddlets.logger import raise_if, raise_log
# WARN: import paddlets.models.utils here would cause circular reference, below is raised error:
# ImportError: cannot import name 'BaseModel' from 'paddlets.models' (/home/work/paddlets/paddlets/models/__init__.py)


class Trainable(object, metaclass=abc.ABCMeta):
    """
    Base class for all trainable classes.

    Any classes need to be fitted (e.g. :class:`~paddlets.models.base.BaseModel`, :class:`~paddlets.pipeline.Pipeline`, etc.) may
    inherit from this base class and optionally implement :func:`fit` method.
    """
    def __init__(self):
        pass

    @abc.abstractmethod
    def fit(
        self,
        train_data: TSDataset,
        valid_data: Optional[TSDataset] = None
    ):
        """
        Fit a trainable instance.

        Any non-abstract classes inherited from this class should implement this method.

        Args:
            train_data(TSDataset): Training dataset.
            valid_data(TSDataset, optional): Validation dataset, optional.
        """
        pass

    @abc.abstractmethod
    def predict(self, data: TSDataset) -> TSDataset:
        """
        Make prediction.

        Any non-abstract classes inherited from this class should implement this method.

        Args:
            data(TSDataset): A TSDataset for time series forecasting.

        Returns:
            TSDataset: Predicted result, in type of TSDataset.
        """
        pass


class BaseModel(Trainable, metaclass=abc.ABCMeta):
    """
    Base class for all machine learning and deep learning models.

    Args:
        in_chunk_len(int): The size of the loopback window, i.e., the number of time steps feed to the model.
        out_chunk_len(int): The size of the forecasting horizon, i.e., the number of time steps output by the model.
        skip_chunk_len(int): Optional, the number of time steps between in_chunk and out_chunk for a single sample.
            The skip chunk is neither used as a feature (i.e. X) nor a label (i.e. Y) for a single sample. By
            default, it will NOT skip any time steps.

    Attributes:
        _in_chunk_len(int): The size of the loopback window, i.e., the number of time steps feed to the model.
        _out_chunk_len(int): The size of the forecasting horizon, i.e., the number of time steps output by the model.
        _skip_chunk_len(int): The length of time series chunk between past target and future target for a single sample.
             The skip chunk are neither used as feature (i.e. X) nor label (i.e. Y) for a single sample.
    """
    def __init__(
        self,
        in_chunk_len: int,
        out_chunk_len: int,
        skip_chunk_len: int
    ):
        super(BaseModel, self).__init__()

        self._in_chunk_len = in_chunk_len
        self._out_chunk_len = out_chunk_len
        self._skip_chunk_len = skip_chunk_len

    @abc.abstractmethod
    def fit(
        self,
        train_data: TSDataset,
        valid_data: Optional[TSDataset] = None
    ):
        """
        Fit a BaseModel instance.

        Any non-abstract classes inherited from this class should implement this method.

        Args:
            train_data(TSDataset): Training dataset.
            valid_data(TSDataset, optional): Validation dataset, optional.
        """
        pass

    @abc.abstractmethod
    def predict(self, data: TSDataset) -> TSDataset:
        """
        Make prediction.

        Any non-abstract classes inherited from this class should implement this method.

        Args:
            data(TSDataset): A TSDataset for time series forecasting.

        Returns:
            TSDataset: Predicted result, in type of TSDataset.
        """
        pass

    @abc.abstractmethod
    def save(self, path: str) -> None:
        """
        Saves a BaseModel instance to a disk file.

        Any non-abstract classes inherited from this class should implement this method.

        Args:
            path(str): A path string containing a model file name.
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def load(path: str) -> "BaseModel":
        """
        Loads a :class:`~/paddlets.models.base.BaseModel` instance from a file.

        Any non-abstract classes inherited from this class should implement this method.

        Args:
            path(str): A path string containing a model file name.

        Returns:
            BaseModel: A loaded model.
        """
        pass
