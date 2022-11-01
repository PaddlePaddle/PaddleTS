# !/usr/bin/env python3
# -*- coding:utf-8 -*-


import abc
from typing import Union
from typing import List 

import pandas as pd
import numpy as np

from paddlets.transform.base import BaseTransform
from paddlets.datasets.tsdataset import TimeSeries
from paddlets.datasets.tsdataset import TSDataset
from paddlets.logger import Logger
from paddlets.logger import raise_if_not
from paddlets.logger import raise_if
from paddlets.logger import raise_log
from paddlets.logger.logger import log_decorator

logger = Logger(__name__)


class Fill(BaseTransform):
    """
    This class is designed to fill missing values in columns. There are three kinds of ways to fulfill this task, including

    Replace the missing values with a statistic computed from a sliding window, e.g. MAX, MIN, MEAN, or MEDIAN;

    Replace the missing values with adjacent values, which could be values previous or next to the missing values;

    Replace the missing values with the value specified by the user.

    Args:
        cols(str|List): Column name(s) to be processed.
        method(str): Method of filling missing values. Totally 8 methods are supported currently:
            max: Use the max value in the sliding window.
            min: Use the min value in the sliding window.
            mean: Use the mean value in the sliding window.
            median: Use the median value in the sliding window.
            pre: Use the previous value.
            next: Use the next value.
            zero: Use 0s.
            default: Use the value specified by the user.
        value(int||float): Only effective when the method is default, value specified by the user to replace the missing values.
        window_size(int): Size of the sliding window.
        min_num_non_missing_values(int): Minimum number of non-missing values in the sliding window, 
            if less than the min_num_non_missing_values, the statistic will be set to np.nan.

    Returns:
        None
    """
    def __init__(self, cols: Union[str, List[str]], method: str='pre', value: int=0, window_size: int=10, min_num_non_missing_values: int=1):
        super(Fill, self).__init__()
        self._cols = cols
        self.method = method
        self.value = value
        self.window_size = window_size
        self.min_num_non_missing_values = min_num_non_missing_values
        self.methods = ['max', 'min', 'mean', 'median', 'pre', 'next', 'zero', 'default']
        if isinstance(cols, str):self._cols = [cols]
        raise_if_not(self._cols, "No column is specified")
        raise_if(method not in self.methods, "The specified filling method doesn't exist.")
        self._cols_lost_dict = {}

        self.need_previous_data = True
        if self.method in ['pre', 'next', 'zero', 'default']:
            self.n_rows_pre_data_need = -1
        else:
            self.n_rows_pre_data_need = self.window_size

    @log_decorator
    def fit_one(self, dataset: TSDataset):
        """
        Args:
            dataset(TSDataset): dataset to process

        Returns:
            self
        """
        return self

    @log_decorator
    def transform_one(self, dataset: TSDataset, inplace: bool=False) -> TSDataset:
        """
        Fill missing values.

        Args:
            dataset(TSDataset): TSDataset or List[TSDataset]
            inplace(bool): Set to True to perform inplace row normalization and avoid a copy.

        Returns:
            new_ts(TSDataset): Transformed TSDataset.

        """
        raise_if_not(dataset is not None, "The specified dataset is None, please check your data!")
        new_ts = dataset
        if not inplace:
            new_ts = dataset.copy()  
        all_method={'pre':{'method':'ffill', 'value':None}, 'next':{'method':'bfill', 'value':None}, 
                   'zero':{'method':None, 'value':self.value}, 'default':{'method':None, 'value':self.value}}

        for col in self._cols:
            sub_data = dataset[col] #.astype(float)
            lack_index = dataset[col][dataset[col].isnull()].index
            self._cols_lost_dict[col] = lack_index
            if self.method in all_method:
                    new_ts[col].fillna(method=all_method[self.method]['method'], 
                                        value=all_method[self.method]['value'], inplace=True)  
            else:
                roll_window = pd.Series.rolling(new_ts[col], window=self.window_size, \
                                                min_periods=self.min_num_non_missing_values)
                for index in lack_index:
                    new_ts[col].loc[index]  = roll_window.__getattribute__(self.method)()[index]           
        return new_ts
