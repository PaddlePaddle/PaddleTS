# !/usr/bin/env python3
# -*- coding:utf-8 -*-

import abc
from typing import Optional, List, Union
import numpy as np
import pandas as pd
import math

from paddlets import TSDataset, TimeSeries
from paddlets.logger import raise_if, raise_log, raise_if_not
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

    def _check_multi_tsdataset(self, datasets: List[TSDataset]):
        """
        Check the validity of multi time series combination transform

        Args:
            datasets(List[TSDataset]): Training datasets.
        """
        raise_if(
            len(datasets) == 0,
            "The Length of datasets cannot be 0!"
        )
        columns_set = set(tuple(sorted(dataset.columns.items())) for dataset in datasets)
        raise_if_not(
            len(columns_set) == 1,
            "The schema of datasets is not same! Cannot be combined for conversion!"
        )

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

    def recursive_predict(
            self,
            tsdataset: TSDataset,
            predict_length: int,
    ) -> TSDataset:
        """
        Apply `self.predict` method iteratively for multi-step time series forecasting, the predicted results from the
        current call will be appended to the `TSDataset` object and will appear in the loopback window for next call.
        Note that each call of `self.predict` will return a result of length `out_chunk_len`, so it will be called
        ceiling(`predict_length`/`out_chunk_len`) times to meet the required length.

        Args:
            tsdataset(TSDataset): Data to be predicted.
            predict_length(int): Length of predicted results.

        Returns:
            TSDataset: Predicted results.
        """
        # Not supported when _skip_chunk !=0
        raise_if(self._skip_chunk_len != 0, f"recursive_predict not supported when \
            _skip_chunk_len!=0, got {self._skip_chunk_len}.")
        # raise_if(predict_length < self._out_chunk_len, f"predict_length must be >= \
        #    self._out_chunk_len, got {predict_length}.")
        raise_if(predict_length <= 0, f"predict_length must be > \
            0, got {predict_length}.")
        tsdataset_copy = tsdataset.copy()
        # Preprocess tsdataset
        if isinstance(tsdataset.get_target().data.index, pd.RangeIndex):
            dataset_end_time = max(
                tsdataset_copy.get_target().end_time + \
                math.ceil(predict_length / self._out_chunk_len) * self._out_chunk_len * \
                (tsdataset_copy.get_target().time_index.step),
                tsdataset_copy.get_known_cov().end_time \
                    if tsdataset_copy.get_known_cov() is not None \
                    else tsdataset_copy.get_target().start_time,
                tsdataset_copy.get_observed_cov().end_time \
                    if tsdataset_copy.get_observed_cov() is not None \
                    else tsdataset_copy.get_target().start_time
            )
        elif isinstance(tsdataset.get_target().data.index, pd.DatetimeIndex):
            dataset_end_time = max(
                tsdataset_copy.get_target().end_time + \
                math.ceil(predict_length / self._out_chunk_len) * self._out_chunk_len * \
                (tsdataset_copy.get_target().time_index.freq),
                tsdataset_copy.get_known_cov().end_time \
                    if tsdataset_copy.get_known_cov() is not None \
                    else tsdataset_copy.get_target().start_time,
                tsdataset_copy.get_observed_cov().end_time \
                    if tsdataset_copy.get_observed_cov() is not None \
                    else tsdataset_copy.get_target().start_time
            )
        else:
            raise_log(ValueError(f"time col type not support, \
                index type:{type(tsdataset.get_target().data.index)}"))
        # Reindex data and the default fill value is np.nan
        fill_value = np.nan
        if tsdataset_copy.get_known_cov() is not None:
            if isinstance(tsdataset_copy.get_known_cov().data.index, pd.RangeIndex):
                tsdataset_copy.get_known_cov().reindex(
                    pd.RangeIndex(start=tsdataset_copy.get_known_cov().start_time,
                                  stop=dataset_end_time + 1,
                                  step=tsdataset_copy.get_known_cov().time_index.step),
                    fill_value=fill_value
                )
            else:
                tsdataset_copy.get_known_cov().reindex(
                    pd.date_range(start=tsdataset_copy.get_known_cov().start_time,
                                  end=dataset_end_time,
                                  freq=tsdataset_copy.get_known_cov().time_index.freq),
                    fill_value=fill_value
                )
        if tsdataset_copy.get_observed_cov() is not None:
            if isinstance(tsdataset_copy.get_observed_cov().data.index, pd.RangeIndex):
                tsdataset_copy.get_observed_cov().reindex(
                    pd.RangeIndex(start=tsdataset_copy.get_observed_cov().start_time,
                                  stop=dataset_end_time + 1,
                                  step=tsdataset_copy.get_observed_cov().time_index.step),
                    fill_value=fill_value
                )
            else:
                tsdataset_copy.get_observed_cov().reindex(
                    pd.date_range(start=tsdataset_copy.get_observed_cov().start_time,
                                  end=dataset_end_time,
                                  freq=tsdataset_copy.get_observed_cov().time_index.freq),
                    fill_value=fill_value
                )
        return self._recursive_predict(tsdataset_copy, predict_length)

    def _recursive_predict(
            self,
            tsdataset: TSDataset,
            predict_length: int,
    ) -> np.ndarray:
        """
        Recursive predict core.

        Args:
            tsdataset(TSDataset): Data to be predicted.
            predict_length(int): Length of predicted results.

        Returns:
            TSDataset: Predicted results.
        """
        recursive_rounds = math.ceil(predict_length / self._out_chunk_len)
        results = []
        for _ in range(recursive_rounds):
            # Model predict
            output = self.predict(tsdataset)
            # Update data using predicted value
            tsdataset = TSDataset.concat([tsdataset, output])
            results.append(output)
        # Concat results
        result = TSDataset.concat(results)
        # Resize result
        result.set_target(
            TimeSeries(result.get_target().data[0: predict_length], result.freq)
        )
        return result
