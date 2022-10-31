# !/usr/bin/env python3
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import abc
from typing import Union, List

from paddlets.transform.base import BaseTransform
from paddlets.datasets.tsdataset import TimeSeries, TSDataset
from paddlets.logger import Logger, raise_if_not, raise_if, raise_log
from paddlets.logger.logger import log_decorator

logger = Logger(__name__)


class KSigma(BaseTransform):
    """
    The ksigma method for outlier detection and replacement. It involves:

    1. Calculate the mean (`mu`) and standard deviation (`std`) of a column.

    2. Determine the interval of normal data according to `mu` and `std`: `[mu - k * std, mu + k * std]` 
       where `k` is a hyper-parameter (3.0 by default). Any value of the interval will be considered as an outlier.

    3. Replace the outliers with `mu`.
    
    Args:
        cols(str|List[str]): Column name or Column names
            (Each column will be handled individually when multiple columns are provided).
        k(float): The hyper-parameter which takes a positive value (3.0 by default).
    
    Returns:
        None
    """
    def __init__(self, cols: Union[str, List[str]], k: float = 3.0):
        super(KSigma, self).__init__()
        self._cols = cols
        self._k = k
        if isinstance(cols, str):
            self._cols = [cols]
        if len(self._cols) < 1:
            raise_log(ValueError("At least one column name should be specified."))
        self._cols_stats_dict = {}
    
    @log_decorator
    def fit_one(self, dataset: TSDataset):
        """
        The process to determine the mean (mu), standard deviation (std), and valid interval ([mu - k * std, mu + k * std])
        
        Args:
            dataset(TSDataset): TSDataset
        
        Returns:
            self
        """
        self._cols_stats_dict = {}
        
        #Compute mu, std, and interval and save the results in _cols_stats dict
        for col in self._cols:
            sub_data = dataset[col]
            #Skip columns that are not numerical
            if not (np.issubdtype(sub_data.dtype, np.integer) or np.issubdtype(sub_data.dtype, np.floating)):
                logger.warning("The values in the column %s should be numerical" % (col))
                continue
            mean = sub_data.mean()
            std = sub_data.std()
            lower = mean - self._k * std
            upper = mean + self._k * std
            self._cols_stats_dict[col] = [lower, upper, mean]

        return self

    @log_decorator
    def transform_one(self, dataset: TSDataset, inplace: bool = False) -> TSDataset:
        """
        Replace the outliers with mu
        
        Args:
            dataset(TSDataset): TSDataset
            inplace(bool): Whether to perform transform inplace, the default is False.
        
        Returns:
            TSDataset
        """
        if self._cols_stats_dict == {}:
            raise_log(ValueError("The fit method must be called prior to calling the transform method."))

        new_ts = dataset
        if not inplace:
            new_ts = dataset.copy()
        
        #Replace outliers withe averages
        for col in self._cols:
            #If a column of data in fit stage is not executed normally, 
            #relevant parameters will not be saved in _cols_stats_dict, throw the corresponding information.
            if col not in self._cols_stats_dict:
                logger.warning("%s is not in anomaly_dict" % (col))
                continue
            lower, upper, mean = self._cols_stats_dict[col]
            for i, value in enumerate(new_ts[col].astype(float)):
                new_ts[col][i] = float(np.where(((value < lower)|(value > upper)), mean, value))                
        return new_ts
