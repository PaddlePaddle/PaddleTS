# !/usr/bin/env python3
# -*- coding:utf-8 -*-


import abc
import copy
from typing import Union
from typing import List 

import pandas as pd
import numpy as np
from sklearn import preprocessing

from paddlets.transform.base import BaseTransform
from paddlets.datasets.tsdataset import TimeSeries
from paddlets.datasets.tsdataset import TSDataset
from paddlets.logger import Logger
from paddlets.logger import raise_if_not
from paddlets.logger import raise_if
from paddlets.logger import raise_log
from paddlets.logger.logger import log_decorator

logger = Logger(__name__)


class MinMaxScaler(BaseTransform):
    """
    Transform a dataset by scaling the values of sepcified column(s) to the expected range: [min, max].
    
    The transformation is done by:
    
    X_std = (X - X.min) / (X.max - X.min)
    
    X_scaled = X_std * (max - min) + min


    Args:
        cols(str|List): Column name(s) to be scaled.
        f_range(tuple): tuple (min, max), default=(0, 1), Desired range of transformed values.
        clip(bool): Set to True to clip transformed values of held-out data to provided feature range.

    Returns:
        None
    """ 
    def __init__(self, cols: Union[str, List[str]]=None, f_range: tuple=(0, 1), clip: bool=False):
        super(MinMaxScaler, self).__init__()
        self._cols = cols
        if isinstance(cols, str):
            self._cols = [cols]
        if self._cols is not None and len(self._cols) < 1:
            raise_log(ValueError("The feature column is not specified!"))
        raise_if_not(isinstance(f_range, tuple), "f_range is not a tuple.")
        raise_if_not(isinstance(clip, bool), "clip is not a boolean.")
        self.f_range = f_range
        self.scaler = preprocessing.__getattribute__('MinMaxScaler')(feature_range=f_range, clip=clip)
        self.cols_scaler_dict = {}

    @log_decorator
    def fit(self, dataset: TSDataset):
        """
        Compute the MIN and MAX parameters needed by the scaler.

        Args:
            dataset(TSDataset): dataset from which to compute the parameters. 

        Returns:
            self
        """
        raise_if_not(dataset is not None, "The dataset is None, please check your data!")
        data_cols = [col for (col, col_type) in dataset.columns.items()]
        if self._cols is None:
            self._cols = data_cols
        for col in self._cols:
            if col in data_cols:
                scaler = copy.deepcopy(self.scaler) 
                scaler.fit(dataset[col].values.reshape(-1, 1))
                self.cols_scaler_dict[col] = scaler
        return self

    @log_decorator
    def transform(self, dataset: TSDataset, inplace: bool=False) -> TSDataset:
        """
        Transform the dataset based on the computed parameters.

        Args:
            dataset(TSDataset): Dataset to be transformed.
            inplace(bool): Set to True to perform inplace row normalization and avoid a copy.

        Returns:
            new_ts(TSDataset): Transformed TSDataset.
        """
        raise_if_not(dataset is not None, "The dataset is None, please check your data!")
        new_ts = dataset
        if not inplace:
            new_ts = dataset.copy()
        _cols = [col for (col, col_type) in dataset.columns.items()]
        # transform 
        for col in _cols:
            if col in self.cols_scaler_dict:
                cur_trans_np = self.cols_scaler_dict[col].transform(new_ts[col].values.reshape(-1, 1))
                cur_trans_np = np.squeeze(cur_trans_np)
                new_ts.set_column(
                    column=col, 
                    value=pd.Series(cur_trans_np, index=new_ts[col].index),
                    type=new_ts.columns[col]
                )

        return new_ts

    def inverse_transform(self, dataset: TSDataset, cols: Union[str, List[str]]=None, 
                          inplace: bool=False) -> TSDataset:
        """
        Inversely transform the scaled dataset.

        Args:
            dataset(TSDataset): dataset to be inversely transformed.
            inplace(bool): Set to True to perform inplace operation and avoid data copy.

        Returns:
            TSDataset: Inversely transformed TSDataset.
        """
        raise_if_not(dataset is not None, "The dataset is None, please check your data!")
        new_ts = dataset
        if not inplace:
            new_ts = dataset.copy() 
        _cols = [col for (col, col_type) in dataset.columns.items()]
        # inverse_transform
        for col in _cols:
            if col in self.cols_scaler_dict:
                cur_trans_np = self.cols_scaler_dict[col].inverse_transform(new_ts[col].values.reshape(-1, 1))
                cur_trans_np = np.squeeze(cur_trans_np)
                new_ts.set_column(
                    column=col, 
                    value=pd.Series(cur_trans_np, index=new_ts[col].index)
                )
        return new_ts

    def fit_transform(self, dataset: TSDataset, inplace: bool=False) -> TSDataset:
        """
        First fit the scaler, and then do transformation.

        Args:
            tsdata(TSDataset): dataset to be processed.
            inplace(bool): Set to True to perform inplace operation and avoid data copy.

        Returns:
            TSDataset: Transformed TSDataset.
        """
        return self.fit(dataset).transform(dataset, inplace)


class StandardScaler(BaseTransform):
    """
    Transform a dataset by scaling the values of sepcified column(s) to zero mean and unit variance.

    The transformation is done by:
    z = (x - u) / s.

    where u is the MEAN or zero if with_mean=False, and s is the standard deviation or one if with_std=False.

    Args:
        cols(str|List):Column name or names to be scaled.
        with_mean(bool): If True, center the data before scaling. 
        with_std(bool):If True, scale the data to unit variance.

    Returns:
        None
    """
    def __init__(self, cols: Union[str, List[str]]=None, with_mean: bool=True, with_std: bool=True): 
        super(StandardScaler, self).__init__()
        self._cols = cols
        if isinstance(cols, str):
            self._cols = [cols]
        if self._cols is not None and len(self._cols) < 1:
            raise_log(ValueError("The feature column is not specified!"))
        raise_if_not(isinstance(with_mean, bool), "with_std is not a boolean.")
        raise_if_not(isinstance(with_std, bool), "with_mean is not a boolean.")
        self.scaler = preprocessing.__getattribute__('StandardScaler')(with_mean=with_mean, with_std=with_std)
        self.cols_scaler_dict = {}

    @log_decorator
    def fit(self, dataset: TSDataset):
        """
        Compute the mean and std parameters needed by the scaler.

        Args:
            dataset(TSDataset): dataset from which to compute parameters.

        Returns:
            self
        """
        raise_if_not(dataset is not None, "The dataset is None, please check your data!")
        data_cols = [col for (col, col_type) in dataset.columns.items()]
        if self._cols is None:
            self._cols = data_cols
        for col in self._cols:
            if col in data_cols:
                scaler = copy.deepcopy(self.scaler) 
                scaler.fit(dataset[col].values.reshape(-1, 1))
                self.cols_scaler_dict[col] = scaler
        return self

    @log_decorator
    def transform(self, dataset: TSDataset, inplace: bool=False) -> TSDataset:
        """
        Transform the dataset based on the computed parameters.

        Args:
            dataset(TSDataset): dataset to be transformed.
            inplace(bool): Set to True to perform inplace operation and avoid data copy.

        Returns:
            new_ts(TSDataset): Transformed TSDataset.
        """
        raise_if_not(dataset is not None, "The dataset is None, please check your data!")
        new_ts = dataset
        if not inplace:
            new_ts = dataset.copy()
        _cols = [col for (col, col_type) in dataset.columns.items()]
        # transform
        for col in _cols:
            if col in self.cols_scaler_dict:
                cur_trans_np = self.cols_scaler_dict[col].transform(new_ts[col].values.reshape(-1, 1))
                cur_trans_np = np.squeeze(cur_trans_np)
                new_ts.set_column(
                    column=col, 
                    value=pd.Series(cur_trans_np, index=new_ts[col].index)
                )
        return new_ts

    def inverse_transform(self, dataset: TSDataset, inplace: bool=False) -> TSDataset:
        """
        Inversely transform the scaled dataset.

        Args:
            dataset(TSDataset): dataset to be inversely transformed.
            inplace(bool): Set to True to perform inplace operation and avoid data copy.

        Returns:
            TSDataset: Inversely transformed TSDataset.
        """
        raise_if_not(dataset is not None, "The dataset is None, please check your data!")
        new_ts = dataset
        if not inplace:
            new_ts = dataset.copy()
        _cols = [col for (col, col_type) in dataset.columns.items()]
        # inverse_transform
        for col in _cols:
            if col in self.cols_scaler_dict:
                cur_trans_np = self.cols_scaler_dict[col].inverse_transform(new_ts[col].values.reshape(-1, 1))
                cur_trans_np = np.squeeze(cur_trans_np)
                new_ts.set_column(
                    column=col, 
                    value=pd.Series(cur_trans_np, index=new_ts[col].index)
                )
        return new_ts

    def fit_transform(self, dataset: TSDataset, inplace: bool=False) -> TSDataset:
        """
        First fit the scaler, and then do transformation.

        Args:
            tsdata(TSDataset): dataset to be processed.
            inplace(bool): Set to True to perform inplace operation and avoid data copy.

        Returns:
            TSDataset: Transformed TSDataset.
        """
        return self.fit(dataset).transform(dataset, inplace)
