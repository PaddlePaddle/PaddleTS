# !/usr/bin/env python3
# -*- coding:utf-8 -*-

import abc
from typing import Union, List, Optional

import pandas as pd
import numpy as np
import chinese_calendar

from paddlets.transform.base import BaseTransform
from paddlets.datasets.tsdataset import TimeSeries, TSDataset
from paddlets.logger import Logger, raise_if_not, raise_if, raise_log
from paddlets.logger.logger import log_decorator

logger = Logger(__name__)


def _cal_year(x: np.datetime64, ):
    """
    Args:
        x(np.datetime64): time
    
    Returns
        int: year
    """
    return x.year


def _cal_month(x: np.datetime64, ):
    """
    Args:
        x(np.datetime64): time
    
    Returns
        int: month of year
    """
    return x.month


def _cal_day(x: np.datetime64, ):
    """
    Args:
        x(np.datetime64): time
    
    Returns
        int: day of month
    """
    return x.day


def _cal_hour(x: np.datetime64, ):
    """
    Args:
        x(np.datetime64): time
    
    Returns
        int: hour of day
    """
    return x.hour


def _cal_weekday(x: np.datetime64, ):
    """
    Args:
        x(np.datetime64): time
    
    Returns
        int: day of weekday
    """
    return x.dayofweek


def _cal_quarter(x: np.datetime64, ):
    """
    Args:
        x(np.datetime64): time
    
    Returns
        int: quarter of year
    """
    return x.quarter


def _cal_dayofyear(x: np.datetime64, ):
    """
    Args:
        x(np.datetime64): time
    
    Returns
        int: day of year
    """
    return x.dayofyear


def _cal_weekofyear(x: np.datetime64, ):
    """
    Args:
        x(np.datetime64): time
    
    Returns
        int: week of year
    """
    return x.weekofyear


def _cal_holiday(x: np.datetime64, ):
    """
    Args:
        x(np.datetime64): time
    
    Returns
        float: holiday
    """
    return float(chinese_calendar.is_holiday(x))


def _cal_workday(x: np.datetime64, ):
    """
    Args:
        x(np.datetime64): time
    
    Returns
        float: workday
    """
    return float(chinese_calendar.is_workday(x))


#THe method of date transform
CAL_DATE_METHOD = {
    'year': _cal_year,
    'month': _cal_month,
    'day': _cal_day,
    'hour': _cal_hour,
    'weekday': _cal_weekday,
    'quarter': _cal_quarter,
    'dayofyear': _cal_dayofyear,
    'weekofyear': _cal_weekofyear,
    'is_holiday': _cal_holiday,
    'is_workday': _cal_workday
}


class TimeFeatureGenerator(BaseTransform):
    """
    Transform time index into specific time features
    
    Args:
        feature_cols(str): Name of feature columns to transform. Currently supported arg values are: year, month, day, weekday, hour, quarter, dayofyear, weekofyear, is_holiday, and is_workday. These time features will be generated by default
        extend_points(int): Extra time points need to be appended to the tail of the existing target time series. 
            Only used when two scenarios are matched simultaneously: 1.the known covariates is None 2.the :func:`predict` method is called. 
            The reason is that the  :func:`predict` method usually requires the tail index of the future target; this index can be calculated from the known cov time series. 
            If known cov is None, this future target tail index needs to be manually extended in this transform and appended to the target.

    Returns:
        None
    """

    def __init__(
            self,
            feature_cols: Optional[List[str]]=[
                'year', 'month', 'day', 'weekday', 'hour', 'quarter',
                'dayofyear', 'weekofyear', 'is_holiday', 'is_workday'
            ],
            extend_points: int=0, ):
        super(TimeFeatureGenerator, self).__init__()
        self.feature_cols = feature_cols
        self.extend_points = extend_points

    @log_decorator
    def fit_one(self, dataset: TSDataset):
        """
        This transformer does not need to be fitted.

        Args:
            dataset(TSDataset): Dataset to be fitted.
        
        Returns:
            TimeFeatureGenerator
        """
        return self

    @log_decorator
    def transform_one(self, dataset: TSDataset,
                      inplace: bool=False) -> TSDataset:
        """
        Transform time column to time features.
        
        Args:
            dataset(TSDataset): Dataset to be transformed.
            inplace(bool): Whether to perform the transformation inplace. default=False
        
        Returns:
            TSDataset
        """
        #Whether to replace data
        new_ts = dataset
        if not inplace:
            new_ts = dataset.copy()
        #Get known_cov
        kcov = new_ts.get_known_cov()
        if not kcov:
            #When time_index.name == None, if useing pd.DataFrame.to_frame, the column name will be automatically defined as 0
            tf_kcov = new_ts.get_target().time_index.to_frame()
        else:
            tf_kcov = kcov.time_index.to_frame()
        #Get time column name
        time_col = tf_kcov.columns[0]
        #Determine time column format
        if np.issubdtype(tf_kcov[time_col].dtype, np.integer):
            raise_log(
                ValueError(
                    "The time_col can't be the type of numpy.integer, and it must be the type of numpy.datetime64"
                ))
        #If kcov == None, it need expand future time periods
        if not kcov:
            freq = new_ts.get_target().freq
            extend_time = pd.date_range(
                start=tf_kcov[time_col][-1],
                freq=freq,
                periods=self.extend_points + 1,
                closed='right',
                name=time_col).to_frame()
            tf_kcov = pd.concat([tf_kcov, extend_time])
        #Generate time index feature content
        for k in self.feature_cols:
            v = tf_kcov[time_col].apply(lambda x: CAL_DATE_METHOD[k](x))
            v.index = tf_kcov[time_col]
            new_ts.set_column(k, v, 'known_cov')

        return new_ts
