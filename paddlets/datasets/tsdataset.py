# !/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
TSDataset is the fundamental data class in PaddleTS, which is designed as the first-class citizen 
to represent the time series data. It is widely used in PaddleTS. In many cases, a function consumes a TSDataset and produces another TSDataset. 
A TSDataset object is comprised of two kinds of time series data: 

	1. Target:  the key time series data in the time series modeling tasks (e.g. those needs to be forecasted in the time series forecasting tasks).
	2. Covariate: the relevant time series data which are usually helpful for the time series modeling tasks.

Currently, it supports the representation of:

	1. Time series of single target w/wo covariates.
	2. Time series of multiple targets w/wo covariates. 

And the covariates can be categorized into one of the following 3 types:

	1. Observed covariates (`observed_cov`): 
		referring to those variables which can only be observed in the historical data, e.g. measured temperatures

	2. Known covariates (`known_cov`):
		referring to those variables which can be determined at present for future time steps, e.g. weather forecasts

	3. Static covariates (`static_cov`):
		referring to those variables which keep constant over time

A TSDataset object includes one or more TimeSeries objects, representing targets, 
known covariates (known_cov), observed covariates (observed_cov), and static covariates (static_cov), respectively.

"""
from copy import deepcopy
import json
import math
import pickle
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from paddlets.logger import Logger, raise_if_not, raise_if, raise_log

logger = Logger(__name__)


class TimeSeries(object):
    """
    TimeSeries is the atomic data structure for representing target(s), observed covariates (observed_cov), and known covariates (known_cov). 
    Each could be comprised of a single or multiple time series data.

    Args:
        data(DataFrame|Series): A Pandas DataFrame or Series containing the time series data
        freq(str|int):  A string or int representing the Pandas DateTimeIndex's frequency or RangeIndex's step size

    Returns:
        None

    """
    def __init__(
        self,
        data: Union[pd.DataFrame, pd.Series],
        freq: Union[int, str],
    ):
        if isinstance(data, pd.Series):
            data = data.to_frame()
        raise_if_not(
            isinstance(data, pd.DataFrame),
            f"The type of param `data` must be pd.DataFrame or pd.Series, but {type(data)} received"
        )
        self._data = data
        self._freq = freq
        if isinstance(self.freq, str):
            try:
                self._data = self._data.asfreq(self._freq)
                self._freq = self._data.index.freqstr
            except ValueError:
                raise_log(
                    ValueError(f"Invalid freq: {self._freq}")
                )

    @classmethod
    def load_from_dataframe(
        cls, 
        data: Union[pd.DataFrame, pd.Series],
        time_col: Optional[str] = None,
        value_cols: Optional[Union[List[str], str]] = None,
        freq: Optional[Union[str, int]] = None,
        drop_tail_nan: bool = False,
        dtype: Optional[Union[type, Dict[str, type]]] = None
    ) -> "TimeSeries":
        """
        Construct a TimeSeries object from the specified columns of a DataFrame

        Args:
            data(DataFrame|Series): A Pandas DataFrame or Series containing the time series data
            time_col(str|None): The name of time column, a Pandas DatetimeIndex or RangeIndex. 
                If not set, the DataFrame's index will be used.
            value_cols(list|str|None): The name of column or the list of columns from which to extract the time series data.
                If set to `None`, all columns except for the time column will be used as value columns.    
            freq(str|int|None): A string or int representing the Pandas DateTimeIndex's frequency or RangeIndex's step size
            drop_tail_nan(bool): Drop time series tail nan value or not, if True, drop all `Nan` value after the last `non-Nan` element in the current time series.
                eg: [nan, 3, 2, nan, nan] -> [nan, 3, 2], [3, 2, nan, nan] -> [3, 2], [nan, nan, nan] -> []
            dtype(np.dtype|type|dict): Use a numpy.dtype or Python type to cast entire TimeSeries object to the same type. 
                Alternatively, use {col: dtype, …}, where col is a column label and dtype is a numpy.dtype or 
                Python type to cast one or more of the DataFrame’s columns to column-specific types.

        Returns:
            TimeSeries object

        """
        #get data
        series_data = None
        if value_cols is None:
            if isinstance(data, pd.Series):
                series_data = data.copy()
            else:
                series_data = data.loc[:, data.columns != time_col].copy()
        else:
            series_data = data.loc[:, value_cols].copy()

        if isinstance(series_data, pd.DataFrame):
            raise_if_not(
                series_data.columns.is_unique, 
                "duplicated column names in the `data`!"
            )
        #get time_col_vals
        if time_col:
            raise_if_not(
                time_col in data.columns,
                f"The time column: {time_col} doesn't exist in the `data`!"
            )
            time_col_vals = data.loc[:, time_col]
        else:
            time_col_vals = data.index
        #Duplicated values or NaN are not allowed in the time column
        raise_if(
            time_col_vals.duplicated().any(),
            "duplicated values in the time column!"
        )
        #Try to convert to string and generate DatetimeIndex
        if np.issubdtype(time_col_vals.dtype, np.integer) and isinstance(freq, str):
            time_col_vals = time_col_vals.astype(str)
        #get time_index
        if np.issubdtype(time_col_vals.dtype, np.integer):
            if freq: 
                #The type of freq should be int when the type of time_col is RangeIndex, which is set to 1 by default
                raise_if_not(
                    isinstance(freq, int) and freq >= 1,
                    "The type of freq should be int when the type of time_col is RangeIndex")
            else:
                freq = 1
            start_idx, stop_idx = min(time_col_vals), max(time_col_vals) + freq
            # All integers in the range must be present
            raise_if(
                (stop_idx - start_idx)/freq != len(data),
                "The number of rows doesn't match with the RangeIndex!"
            )
            time_index = pd.RangeIndex(
                start=start_idx, stop=stop_idx, step=freq
            )
        elif np.issubdtype(time_col_vals.dtype, np.object_) or \
            np.issubdtype(time_col_vals.dtype, np.datetime64):
            time_col_vals = pd.to_datetime(time_col_vals, infer_datetime_format=True)
            time_index = pd.DatetimeIndex(time_col_vals)
            if freq: 
                #freq type needs to be string when time_col type is DatetimeIndex
                raise_if_not(
                    isinstance(freq, str),
                    "The type of `freq` should be `str` when the type of `time_col` is `DatetimeIndex`."
                )
            else:
                #If freq is not provided and automatic inference fail, throw exception
                freq = pd.infer_freq(time_index)
                raise_if(
                    freq is None,
                    "Failed to infer the `freq`. A valid `freq` is required."
                )
                if freq[0] == '-':
                    freq = freq[1:]
        else:
            raise_log(ValueError("The type of `time_col` is invalid.")) 
        if isinstance(series_data, pd.Series):
            series_data = series_data.to_frame()
        series_data.set_index(time_index, inplace=True)
        series_data.sort_index(inplace=True)
        ts = TimeSeries(series_data, freq)
        if drop_tail_nan:
            ts.drop_tail_nan()
        if dtype:
            ts.astype(dtype)
        return ts 
    
    @property
    def time_index(self):
        """the time index"""
        return self.data.index

    @property
    def columns(self):
        """the data columns"""
        return self.data.columns
    
    @property
    def start_time(self) -> Union[pd.Timestamp, int]:
        """the first value of the time index"""
        return self.time_index[0]
    
    @property
    def end_time(self) -> Union[pd.Timestamp, int]:
        """the last value of the time index"""
        return self.time_index[-1]  

    @property
    def data(self):
        """DataFrame storing the data"""
        return self._data

    @property
    def freq(self):
        """Frequency of TimeSeries"""
        return self._freq
    
    @property
    def dtypes(self) -> pd.Series:
        """dtypes of TimeSeries"""
        return self._data.dtypes

    def __len__(self):
        """Length of TimeSeries"""
        return len(self._data)

    def __str__(self):
        """str"""
        return self._data.__str__()

    def __repr__(self):
        """repr"""
        return self._data.__repr__()
    
    def astype(self, dtype: Union[np.dtype, type, Dict[str, Union[np.dtype, type]]]):
        """
        Cast a TimeSeries object to the specified dtype

        Args:
            dtype(np.dtype|type|dict): Use a numpy.dtype or Python type to cast entire TimeSeries object to the same type. 
                Alternatively, use {col: dtype, …}, where col is a column label and dtype is a numpy.dtype or 
                Python type to cast one or more of the DataFrame’s columns to column-specific types.
            
        Raise:
            TypeError
            KeyError

        """
        self._data = self._data.astype(dtype)
        return self

    def to_dataframe(self, copy: bool=True) -> pd.DataFrame:
        """
        Return a pd.DataFrame representation of the TimeSeries object

        Args:
            copy(bool):  Return a copy or reference

        Returns:
            pd.DataFrame

        """
        if copy:
            return self.data.copy()
        else:
            return self.data
    
    def to_numpy(self, copy: bool=True) -> np.ndarray:
        """
        Return a numpy.ndarray representation of the TimeSeries object

        Args:
            copy(bool): Return a copy or reference.
                Note that copy=False does not ensure that to_numpy() is no-copy. 
                Rather, copy=True ensure that a copy is made, even if not strictly necessary.
                refer：https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_numpy.html

        Returns:
            np.ndarray

        """
        return self.data.to_numpy(copy=copy)

    def get_index_at_point(
        self, 
        point: Union[pd.Timestamp, str, float, int], 
        after=True
    ) -> int:
        """
        Convert a point along the time axis into an integer index.
        
        Args:
            point(pd.Timestamp|float|int): Time point, supports 3 types

                `pd.Timestamp|str`: It only takes effect when the time_index type is pd.DatatimeIndex, the corresponding index is returned, and str will be forcibly converted to pd.DatatimeIndex
                
                `float`: the parameter will be treated as the proportion of the time series that should lie before the point.
                
                `int`: the parameter will returned as such, provided that it is in the series. Otherwise it will raise a ValueError.
            after(bool): If the provided pandas Timestamp is not in the time series index, whether to return the index of the
                next timestamp or the index of the previous one.

        Returns:
            int: index
        
        Raise:
            ValueError
            TypeError

        """
        point_index = -1
        if isinstance(point, str):
            point = pd.Timestamp(point)
        if isinstance(point, float):
            raise_if_not(
                0.0 <= point <= 1.0,
                "`point` (float) should be between 0.0 and 1.0."
            )
            point_index = math.floor((self.data.shape[0] - 1) * point)
        elif isinstance(point, (int, np.int64)):
            raise_if(
                point not in range(self.data.shape[0]),
                "`point` (int) should be a valid index in series."
            )
            point_index = point
        elif isinstance(point, pd.Timestamp):
            raise_if_not(
                isinstance(self.time_index, pd.DatetimeIndex),
                "The provided `point` is of the Timestamp type, but the type of time column is not DatetimeIndex"
            )
            raise_if_not(
                point >= self.start_time and point <= self.end_time,
                "The `point` is out of the valid range."
            )
            if point in self.time_index:
                point_index = self.time_index.get_loc(point)
            else:
                point_index = self.time_index.get_loc(
                    next(filter(lambda t: t >= point, self.time_index))
                    if after
                    else next(filter(lambda t: t <= point, self.time_index[::-1]))
                )
        else:
            raise_log(
                TypeError(
                    "`point` needs to be either `float`, `int` or `pd.Timestamp`"
                )
            )
        return point_index
    
    def split(
        self, 
        split_point: Union[pd.Timestamp, str, float, int], 
        after=True
    ) -> Tuple["TimeSeries", "TimeSeries"]:
        """
        Split the TimeSeries object into two TimeSeries objects according to `split_point`
        
        Args:
            split_point(pd.Timestamp|float|int): Where to split the TSDataset, which could be

                `pd.Timestamp|str`: Only valid when the type of time_index is pd.DatatimeIndex, and str will be forcibly converted to pd.DatatimeIndex

                `float`: The proportion of the length of the first TSDataset object

                `int`: Only valid when the type of time_index is pd.RangeIndex

                If the data of the split_point exists, it will be included in the first TimeSeries object.
            after(bool): If `split_point` (pd.TimeSeries) doesn't exist in the time index, use the next valid index (True) or the previous one (False)
            
        Returns:
            Tuple["TimeSeries", "TimeSeries"]
        
        Raise:
            ValueError
            TypeError

        """
        point = self.get_index_at_point(split_point, after)
        shift = 0 if isinstance(split_point, (int, np.int64)) else 1
        return (
            TimeSeries(self.data.iloc[: point + shift, :], self.freq),
            TimeSeries(self.data.iloc[point + shift :, ], self.freq)
        )
    
    def copy(self) -> "TimeSeries":
        """
        Make a copy of the TimeSeries object
        
        Returns:
            TimeSeries
        """
        return TimeSeries(self.data.copy(), self.freq)

    def __getitem__(
            self,
            key: Union[
                pd.DatetimeIndex,
                pd.RangeIndex,
                slice,
            ],
    ) -> "TimeSeries":
        """
        Indexing operation on the TimeSeries object

        Args:
            key(pd.DatatimeIndex|pd.RangeIndex|slice):

                `pd.DatatimeIndex`: Only valid when the type of time_index is pd.DatatimeIndex, return a sub TimeSeries according to pd.DatetimeIndex

                `pd.RangeIndex`: Only valid when the type of time_index is pd.RangeIndex, return a sub TimeSeries according to pd.RangeIndex

                `slice`: return a sub TimeSeries by the `slice`, e.g. timeseries[10:20] returns a sub TimeSeries of length 10

        Returns:
            TimeSeries

        Raise:
            ValueError
        
        """
        if isinstance(key, pd.DatetimeIndex):
            raise_if_not(isinstance(self._data.index, pd.DatetimeIndex),
                         f"The TimeSeries' index is of the type {type(self._data.index)}, but the key is of the type pd.DatetimeIndex")
            return self.__class__(self._data.loc[key], freq=key.freqstr)
        elif isinstance(key, pd.RangeIndex):
            raise_if_not(isinstance(self._data.index, pd.RangeIndex),
                         f"The TimeSeries' index is of the type {type(self._data.index)}, but the key is of the type pd.RangeIndex")
            return self.__class__(self._data.loc[key], freq=key.step)
        elif isinstance(key, slice):
            return self.__class__(self._data[key], freq=self.freq)

        raise_log(ValueError(f"Invalid type of `key`: {type(key)}, currently only `pd.DatetimeIndex`, `pd.RangeIndex`, and `slice` are supported"))
        
    @classmethod
    def concat(cls, tss: List["TimeSeries"], axis: int = 0, drop_duplicates: bool = True, keep: str = 'first') -> "TimeSeries":
        """
        Concatenate a list of TimeSeries objects along the specified axis

        Args:
            tss(list[TimeSeries]): A list of TimeSeries objects
                All TimeSeries' freqs are required to be consistent. 
                When axis=1, time_col is required to be non-repetitive; 
                when axis=0, all columns are required to be non-repetitive
            axis(int): The axis along which to concatenate the TimeSeries objects
            drop_duplicates(bool): Drop duplicate indices.
            keep(str): keep 'first' or 'last' when drop duplicates.

        Returns:
            TimeSeries
        
        Raise:
            ValueError

        """
        raise_if_not(
            len(set(i.freq for i in tss)) == 1,
            f"Failed to concatenate, the freqs of TimeSeries objects are not consistent ." 
        )
        raise_if_not(
            keep in ["first", "last"], "keep should set to 'first' or 'last'")
        
        if axis == 0:
            data = pd.concat([ts.data for ts in tss], axis=axis)
            if drop_duplicates:
                data = data[~data.index.duplicated(keep=keep)]
                if isinstance(tss[0].data.index, pd.RangeIndex):
                    #Range index concat完会变成Int64index， 需要转换回Range index
                    data = data.set_index(pd.RangeIndex(data.index.values[0], data.index.values[-1] + tss[0].freq, tss[0].freq))
            else:
                raise_if(
                    data.index.duplicated().any(),
                    "Failed to concatenate, duplicated values found in the time column.\
                    You can set drop_duplicates = True to drop Duplicate values.\
                    And you can set keep = 'first' or 'last' to choose which value to preserve."
                )
            return TimeSeries(data, tss[0].freq)
        elif axis == 1:
            data = pd.concat([ts.data for ts in tss], axis=axis)
            if drop_duplicates:
                data = data.loc[:,~data.columns.duplicated(keep=keep)]
            else:
                raise_if(
                    data.columns.duplicated().any(),
                    "Failed to concatenate, duplicated column names found.\
                    You can set drop_duplicates = True to drop Duplicate columns.\
                    And you can set keep = 'first' or 'last' to choose which value to preserve."
                )
            return TimeSeries(data, tss[0].freq)
        else:
            raise_log(
                ValueError(f"Failed to concatenate, invalid axis: {axis}")
            )
    
    def reindex(self, index, fill_value=np.nan, *args, **kwargs) -> "TimeSeries":
        """
        Reindex the TimeSeries object with optional filling logic

        Args:
            index: array-like, new index to conform. 
                   Preferably an Index object to avoid duplicating data.
            fill_value: Value to use for missing values. NaN by default, but can be any “compatible” value.
            args: Optional arguments passed to `DataFrame.reindex`
            kwargs: Optional arguments passed to `DataFrame.reindex`
        
        Returns:
            TimeSeries
        
        Raise:
            ValueError

        """
        self._data = self._data.reindex(index, fill_value=fill_value, *args, **kwargs)
    
    def sort_columns(self, ascending: bool = True):
        """
        Sort the TimeSeries object by the index

        Args:
            ascending(bool): Sort ascending or descending. When the index is a MultiIndex the sort direction can be controlled for each level individually.

        """
        self._data = self._data.sort_index(axis=1, ascending=ascending)
        return self
    
    def _find_end_index(self) -> int:
        """
        Identify end indices in series which is not Nan.

        Return:
            The end index which is not Nan        

        """
        isna_fun = lambda x: pd.isnull(x).all()
        for i in range(len(self.data) - 1, -1, -1):
            if not isna_fun(self.data.iloc[i]):
                break
            if i == 0:
                return -1
        return i
    
    def drop_tail_nan(self):
        """
        Drop trailing consecutive Nan values
        """
        end_index = self._find_end_index()
        self.reindex(self.time_index[:end_index + 1])
    
    def to_json(self) -> str:
        """
        Return a str json representation of the TimeSeries object.

        Returns:
            str
        """
        json_res = {'freq': self._freq}
        if isinstance(self._freq, str):
            data = self._data.copy()
            data.index = data.index.astype(str)
        else:
            data = self._data
        json_res['data'] = data.to_dict()
        return json.dumps(json_res, ensure_ascii=False)
    
    @classmethod
    def load_from_json(cls, json_data: str, **json_load_kwargs) -> "TimeSeries":
        """
        Construct a TimeSeries object from a str json_data
        
        Args:
            json_data(str):  json object from which to load data
            **json_load_kwargs: Optional arguments passed to `json.loads` function
        
        Returns:
            TimeSeries
        """
        res = json.loads(json_data, **json_load_kwargs)
        freq = res['freq']
        data = res['data']
        data = pd.DataFrame(data)
        if isinstance(freq, str):
            data.index = pd.DatetimeIndex(data.index.to_list())
        else:
            start_idx, stop_idx = min(data.index.astype(int)), max(data.index.astype(int)) + freq
            data.index = pd.RangeIndex(
                start=start_idx, stop=stop_idx, step=freq
            )
        return TimeSeries(data, freq)

    def to_categorical(self, col: Optional[Union[str, List[str]]] = None):
        """
        Modify col's type to int as categorical.

        Args:
            col(Optional[Union[str, List[str]]]): col names in ts

        """
        if col:
            if isinstance(col, str):
                col = [col]
            dtype = {col_one: np.int64 for col_one in col}
            self.astype(dtype)
        else:
            self.astype(np.int64)

    def to_numeric(self, col: Optional[Union[str, List[str]]] = None):
        """
        Modify col's type to float as numeric.

        Args:
            col(Optional[Union[str, List[str]]]): col names in ts

        """
        if col:
            if isinstance(col, str):
                col = [col]
            dtype = {col_one: np.float32 for col_one in col}
            self.astype(dtype)
        else:
            self.astype(np.float32)


class TSDataset(object):
    """
    TSDataset is the fundamental data class in PaddleTS, which is designed as the first-class citizen 
    to represent the time series data. It is widely used in PaddleTS. In many cases, a function consumes a TSDataset and produces another TSDataset. 
    A TSDataset object is comprised of two kinds of time series data: 

        1. Target:  the key time series data in the time series modeling tasks (e.g. those needs to be forecasted in the time series forecasting tasks).
        2. Covariate: the relevant time series data which are usually helpful for the time series modeling tasks.

    Currently, it supports the representation of:

        1. Time series of single target w/wo covariates.
        2. Time series of multiple targets w/wo covariates. 

    And the covariates can be categorized into one of the following 3 types:

        1. Observed covariates (`observed_cov`): 
            referring to those variables which can only be observed in the historical data, e.g. measured temperatures

        2. Known covariates (`known_cov`):
            referring to those variables which can be determined at present for future time steps, e.g. weather forecasts

        3. Static covariates (`static_cov`):
            referring to those variables which keep constant over time

    A TSDataset object includes one or more TimeSeries objects, representing targets, 
    known covariates (known_cov), observed covariates (observed_cov), and static covariates (static_cov), respectively.
            
    Args:
        target(TimeSeries|None): Target
        observed_cov(TimeSeries|None): Observed covariates 
        known_cov(TimeSeries|None): Known covariates
        static_cov(dict|None): Static covariates
        fill_missing_dates(bool): Fill missing dates or not
        fillna_method(str): Method of filling missing values. Totally 7 methods are supported currently:
            max: Use the max value in the sliding window
            min: Use the min value in the sliding window
            avg: Use the mean value in the sliding window
            median:  Use the median value in the sliding window
            pre: Use the previous value
            back: Use the next value
            zero:  Use 0s
        fillna_window_size(int): Size of the sliding window

     Returns:
         None

    """
    def __init__(
        self,
        target: Optional[TimeSeries] = None,
        observed_cov: Optional[TimeSeries] = None,
        known_cov: Optional[TimeSeries] = None,
        static_cov: Optional[dict] = None,
        fill_missing_dates: bool = False,
        fillna_method: str = "pre",
        fillna_window_size: int = 10,
    ):
        
        self._target = target
        self._observed_cov = observed_cov
        self._known_cov = known_cov
        self._static_cov = static_cov

        #The type of freq is str when When time_index is DatetimeIndex, int when time_index is RangeIndex
        self._freq : Optional[str, int] = None
        self._check_data()

        #Get built-in analysis operators
        from paddlets.analysis import TSDataset_Inner_Analyzer
        self._inner_analyzer = TSDataset_Inner_Analyzer

        if fill_missing_dates:
            #Fill the missing values
            from paddlets.transform import Fill
            fill_obj = Fill(
                cols=list(self.columns.keys()), 
                method=fillna_method,
                window_size=fillna_window_size
            )
            fill_obj.fit_transform(self, inplace=True)
    
    def __getattr__(self, name: str) -> Callable:
        """
        Dynamically integrate and call built-in operators
        (Only analysis operators are currently integrated, and other types of operators may be integrated in the future)

        Args:
            name(str): operator name, eg: summary、max、min

        Returns:
            Callable: operator funtion

        Raise:
            ValueError

        """
        if (name.startswith('__') and name.endswith('__')) or \
           (name.startswith('_') and name.endswith('_')) :
            return super().__getattr__(name)
        #analysis operators
        if name in self._inner_analyzer:
            #the first parameter of the self._inner_analyze operator needs to be TSDataset
            #functools.partial can only provide the ability of default parameters, which is not flexible enough to use here
            def partial(*arg, **kwargs):
                return self._inner_analyzer[name](self, *arg, **kwargs)
            return partial
        else:
            raise_log(
                ValueError(f"attr: {name} doesn't exist!")
            )

    def _check_data(self):
        freq_list = []
        columns_list = []
        if self._target is not None:
            freq_list.append(self._target.freq)
            columns_list += list(self._target.columns)
        if self._observed_cov is not None:
            freq_list.append(self._observed_cov.freq)
            columns_list += list(self._observed_cov.columns)
        if self._known_cov is not None:
            freq_list.append(self._known_cov.freq)
            columns_list += list(self._known_cov.columns)
        if self._static_cov is not None:
            columns_list += list(self._static_cov.keys())
        #check freq
        raise_if(
            len(set(freq_list)) != 1,
            "The freqs of target, observed_covariate, and known_covariate are not consistent."
        )
        self._freq = freq_list[0]
        #check columns

        raise_if(
            len(set(columns_list)) != len(columns_list),
            "Duplicated column names in target, observed_covariate, and known_covariate."
        )

    @classmethod
    def load_from_csv(
        cls,
        filepath_or_buffer: str,
        group_id: str = None,
        time_col: Optional[str] = None,
        target_cols: Optional[Union[List[str], str]] = None,
        label_col: Optional[Union[List[str], str]] = None,
        observed_cov_cols: Optional[Union[List[str], str]] = None,
        feature_cols: Optional[Union[List[str], str]] = None,
        known_cov_cols: Optional[Union[List[str], str]] = None,
        static_cov_cols: Optional[Union[List[str], str]] = None,
        freq: Optional[Union[str, int]] = None,
        fill_missing_dates: bool = False,
        fillna_method: str = "pre",
        fillna_window_size: int = 10,
        drop_tail_nan: bool = False,
        dtype: Optional[Union[type, Dict[str, type]]] = None,
        **kwargs
    ) -> Union["TSDataset", List["TSDataset"]]:
        """
        Construct a TSDataset object from a csv file

        Args:
            filepath_or_buffer(str): The path to the CSV file, or the file object; 
                consistent with the argument of `pandas.read_csv` function
            group_id(str|None): The column name identifying a time series. 
                This means that the group_id identify a sample together with the `time_index`. 
                If you have only one timeseries dataset, Do not pass this parameter or set this to the name of column that is constant.
                If group_id is provided, the function will return a list of TSDataset which length equal to len(group_id.unique()).
                eg: A sample of equipment load detection guarantees the data of multiple equipment, in which the `ID` column is used to distinguish different equipment. In this case, the group_id='ID'.
            time_col(str|None): The name of time column
            target_cols(list|str|None): The names of columns for target
            label_col(list|str|None): The names of columns for label in anomaly detection
            observed_cov_cols(list|str|None): The names of columns for observed covariates
            feature_cols(list|str|None): The names of columns for feature in anomaly detection
            known_cov_cols(list|str|None): The names of columns for konwn covariates
            static_cov_cols(list|str|None): The names of columns for static covariates
            freq(str|int|None): A str or int representing the DateTimeIndex's frequency or RangeIndex's step size
            fill_missing_dates(bool): Fill missing dates or not
            fillna_method(str): Method of filling missing values. Totally 7 methods are supported currently:
                max: Use the max value in the sliding window
                min: Use the min value in the sliding window
                avg: Use the mean value in the sliding window
                median:  Use the median value in the sliding window
                pre: Use the previous value
                back: Use the next value
                zero:  Use 0 
            fillna_window_size(int): Size of the sliding window
            drop_tail_nan(bool): Drop time series tail nan value or not
            dtype(np.dtype|type|dict): Use a numpy.dtype or Python type to cast entire TimeSeries object to the same type. 
                Alternatively, use {col: dtype, …}, where col is a column label and dtype is a numpy.dtype or 
                Python type to cast one or more of the DataFrame’s columns to column-specific types.
            kwargs: Optional arguments passed to `pandas.read_csv`

        Returns:
            Union[TSDataset, List[TSDataset]]
        """
        df = pd.read_csv(filepath_or_buffer=filepath_or_buffer, **kwargs)
        return cls.load_from_dataframe(
            df=df,
            group_id=group_id,
            time_col=time_col,
            target_cols=target_cols,
            label_col=label_col,
            observed_cov_cols=observed_cov_cols,
            feature_cols=feature_cols,
            known_cov_cols=known_cov_cols,
            static_cov_cols=static_cov_cols,
            freq=freq,
            fill_missing_dates=fill_missing_dates,
            fillna_method=fillna_method,
            fillna_window_size=fillna_window_size,
            drop_tail_nan=drop_tail_nan,
            dtype=dtype,
        )

    @classmethod
    def load_from_dataframe(
        cls,
        df: pd.DataFrame,
        group_id: str = None,
        time_col: Optional[str] = None,
        target_cols: Optional[Union[List[str], str]] = None,
        label_col: Optional[Union[List[str], str]] = None,
        observed_cov_cols: Optional[Union[List[str], str]] = None,
        feature_cols: Optional[Union[List[str], str]] = None,
        known_cov_cols: Optional[Union[List[str], str]] = None,
        static_cov_cols: Optional[Union[List[str], str]] = None,
        freq: Optional[Union[str, int]] = None,
        fill_missing_dates: bool = False,
        fillna_method: str = "pre",
        fillna_window_size: int = 10,
        drop_tail_nan: bool = False,
        dtype: Optional[Union[type, Dict[str, type]]] = None
    ) -> Union["TSDataset", List["TSDataset"]]:
        """
        Construct a TSDataset object from a DataFrame

        Args:
            df(pd.DataFrame): panas.DataFrame object from which to load data
            group_id(str|None): The column name identifying a time series. 
                This means that the group_id identify a sample together with the `time_index`. 
                If you have only one timeseries dataset, Do not pass this parameter or set this to the name of column that is constant.
                If group_id is provided, the function will return a list of TSDataset which length equal to len(group_id.unique()).
                eg: A sample of equipment load detection guarantees the data of multiple equipment, in which the `ID` column is used to distinguish different equipment. In this case, the group_id='ID'.
            time_col(str|None): The name of time column
            target_cols(list|str|None): The names of columns for target
            label_col(list|str|None): The names of columns for label in anomaly detection
            observed_cov_cols(list|str|None): The names of columns for observed covariates
            feature_cols(list|str|None): The names of columns for feature in anomaly detection
            known_cov_cols(list|str|None): The names of columns for konwn covariates
            static_cov_cols(list|str|None): The names of columns for static covariates
            freq(str|int|None): A str or int representing the DateTimeIndex's frequency or RangeIndex's step size
            fill_missing_dates(bool): Fill missing dates or not
            fillna_method(str): Method of filling missing values. Totally 7 methods are supported currently:
                max: Use the max value in the sliding window
                min: Use the min value in the sliding window
                avg: Use the mean value in the sliding window
                median:  Use the median value in the sliding window
                pre: Use the previous value
                back: Use the next value
                zero:  Use 0s
            fillna_window_size(int): Size of the sliding window
            drop_tail_nan(bool): Drop time series tail nan value or not
            dtype(np.dtype|type|dict): Use a numpy.dtype or Python type to cast entire TimeSeries object to the same type. 
                Alternatively, use {col: dtype, …}, where col is a column label and dtype is a numpy.dtype or 
                Python type to cast one or more of the DataFrame’s columns to column-specific types.
            
        Returns:
            Union[TSDataset, List[TSDataset]]
        """
        raise_if_not(
            df.columns.is_unique,
            "The column names of the input DataFrame are not unique."
        )
        dfs = []
        if group_id is not None:
            raise_if_not(
                group_id in df.columns,
                f"group_id: {group_id} not in df!"
            )
            group_unique = df[group_id].unique()
            for column in group_unique:
                dfs.append(df[df[group_id].isin([column])])
        else:
            dfs = [df]
        res = []
        if label_col:
            raise_if(
                target_cols is not None,
                "label_col and target_cols cannot pass at the same time!"
            )
            raise_if(
                isinstance(label_col, list) and len(label_col) > 1,
                "The length of label_col must be 1."
            )
            target_cols = label_col
        if feature_cols:
            raise_if(
                observed_cov_cols is not None or known_cov_cols is not None,
                "feature_cols and cov_cols cannot pass at the same time!"
            )
            observed_cov_cols = feature_cols
        for df in dfs:
            target = None
            observed_cov = None
            known_cov = None
            static_cov = dict()
            if not any([target_cols, observed_cov_cols, known_cov_cols, static_cov_cols]):
                #By default all columns are target columns
                target = TimeSeries.load_from_dataframe(
                    df, 
                    time_col,
                    [a for a in df.columns if a != time_col],
                    freq,
                )
                if drop_tail_nan:
                    target.drop_tail_nan()
            else:
                if target_cols:
                    target = TimeSeries.load_from_dataframe(
                        df, 
                        time_col,
                        target_cols,
                        freq,
                    )
                    if drop_tail_nan:
                        target.drop_tail_nan()
                if observed_cov_cols:
                    observed_cov = TimeSeries.load_from_dataframe(
                        df, 
                        time_col,
                        observed_cov_cols,
                        freq,
                    )
                    if drop_tail_nan:
                        observed_cov.drop_tail_nan()
                if known_cov_cols:            
                    known_cov = TimeSeries.load_from_dataframe(
                        df, 
                        time_col,
                        known_cov_cols,
                        freq,
                    )
                    if drop_tail_nan:
                        known_cov.drop_tail_nan()
                if static_cov_cols:
                    if isinstance(static_cov_cols, str):
                        static_cov_cols = [static_cov_cols]
                    for col in static_cov_cols:
                        raise_if(
                            col not in df.columns or len(np.unique(df[col])) != 1,
                            "static cov cals data is not in columns or schema is not right!"
                        )
                        static_cov[col] = df[col].iloc[0]
            res.append(cls(
                target, 
                observed_cov, 
                known_cov, 
                static_cov,
                fill_missing_dates,
                fillna_method,
                fillna_window_size,
            ))
            if dtype:
                for one in res:
                    one.astype(dtype) 
        return res if len(res) > 1 else res[0]
    
    def to_dataframe(self, copy: bool=True) -> pd.DataFrame:
        """
        Return a pd.DataFrame representation of the TSDataset object

        Args:
            copy(bool):  Return a copy of or a reference to the underlying DataFrame objects

        Returns:
            pd.DataFrame

        """
        pd_list = []
        if self._target is not None:
            pd_list.append(self._target.to_dataframe(copy))
        cov = self.get_all_cov()
        if cov is not None:
            pd_list.append(cov.to_dataframe(copy))
        return pd.concat(pd_list, axis=1)

    def to_numpy(self, copy: bool=True) -> np.ndarray:
        """
        Return a np.ndarray representation of the TSDataset object

        Args:
            copy(bool): Return a copy of or a reference to the underlying DataFrame objects,
                Note that copy=False does not ensure that to_numpy() is no-copy. Rather, 
                copy=True ensures that a copy is made, even if not strictly necessary.
                refer：https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_numpy.html

        Returns:
            np.ndarray

        """
        return self.to_dataframe(copy).to_numpy()

    def get_target(self) -> Optional["TimeSeries"]:
        """
        Returns:
            TimeSeries|None: target

        """
        return self._target
    
    def get_label(self) -> Optional["TimeSeries"]:
        """
        Returns:
            TimeSeries|None: target

        """
        return self.get_target()
    
    def get_observed_cov(self) -> Optional["TimeSeries"]:
        """
        Returns:
            TimeSeries|None: observed_cov

        """
        return self._observed_cov
    
    def get_feature(self) -> Optional["TimeSeries"]:
        """
        Returns:
            TimeSeries|None: observed_cov

        """
        return self.get_observed_cov()
    
    def get_known_cov(self) -> Optional["TimeSeries"]:
        """
        Returns:
            TimeSeries|None: known_cov

        """
        return self._known_cov

    def get_static_cov(self) -> Optional[dict]:
        """    
        Returns:
            dict|None: static_cov

        """
        return self._static_cov

    @property
    def target(self) -> Optional["TimeSeries"]:
        """
        Returns:
            TimeSeries|None: target

        """
        return self._target
    
    @property
    def label(self) -> Optional["TimeSeries"]:
        """
        Returns:
            TimeSeries|None: target

        """
        return self.target
    
    @property
    def observed_cov(self) -> Optional["TimeSeries"]:
        """
        Returns:
            TimeSeries|None: observed_cov

        """
        return self._observed_cov
    
    @property
    def feature(self) -> Optional["TimeSeries"]:
        """
        Returns:
            TimeSeries|None: observed_cov

        """
        return self.observed_cov
    
    @property
    def known_cov(self) -> Optional["TimeSeries"]:
        """
        Returns:
            TimeSeries|None: known_cov

        """
        return self._known_cov

    @property
    def static_cov(self) -> Optional[dict]:
        """    
        Returns:
            dict|None: static_cov 

        """
        return self._static_cov

    def get_all_cov(self) -> Optional[TimeSeries]:
        """    
        Returns:
            pd.DataFrame|None: Merge observed_cov and konw_cov

        """
        if self._known_cov is None:
            return self._observed_cov
        elif self._observed_cov is None:
            return self._known_cov
        else:
            return TimeSeries(pd.concat([self._observed_cov.data, self._known_cov.data], axis=1), self._known_cov.freq)

    def set_target(self, target: "TimeSeries"):
        """
        Args:
            target(TimeSeries): New target

        Returns:
            None

        Raise:
            ValueError

        """
        self._target = target
        self._check_data()
    
    def set_label(self, label: "TimeSeries"):
        """
        Args:
            label(TimeSeries): New label

        Returns:
            None

        Raise:
            ValueError

        """
        return self.set_target(label)
    
    def set_observed_cov(self, observed_cov: "TimeSeries"):
        """
        Args:
            observed_cov(TimeSeries): New observed_cov

        Returns:
            None

        Raise:
            ValueError

        """
        self._observed_cov = observed_cov
        self._check_data()   

    def set_feature(self, feature: "TimeSeries"):
        """
        Args:
            feature(TimeSeries): New feature

        Returns:
            None

        Raise:
            ValueError

        """
        return self.set_observed_cov(feature)  

    def set_known_cov(self, known_cov: "TimeSeries"):
        """
        Args:
            known_cov(TimeSeries): New known_cov

        Returns:
            None

        Raise:
            ValueError

        """
        self._known_cov = known_cov
        self._check_data()

    def set_static_cov(self, static_cov: "dict", append: bool=True):
        """
        Args:
            static_cov(dict): New static_cov
            append(bool): Append to the existing static_cov or replace the existing satic_cov
            
        Returns:
            None

        Raise:
            ValueError

        """
        if append and self._static_cov:
            self._static_cov = {**self._static_cov, **static_cov}
        else:  
            self._static_cov = static_cov
        self._check_data()

    @target.setter
    def target(self, target: "TimeSeries"):
        """
        Args:
            target(TimeSeries): New target

        Returns:
            None

        Raise:
            ValueError

        """
        self._target = target
        self._check_data()

    @label.setter
    def label(self, label: "TimeSeries"):
        """
        Args:
            label(TimeSeries): New target

        Returns:
            None

        Raise:
            ValueError

        """
        return self.set_label(label)
    
    @observed_cov.setter
    def observed_cov(self, observed_cov: "TimeSeries"):
        """
        Args:
            observed_cov(TimeSeries): New observed_cov

        Returns:
            None

        Raise:
            ValueError

        """
        self._observed_cov = observed_cov
        self._check_data()

    @feature.setter
    def feature(self, feature: "TimeSeries"):
        """
        Args:
            feature(TimeSeries): New feature

        Returns:
            None

        Raise:
            ValueError

        """
        return self.set_feature(feature)

    @known_cov.setter
    def known_cov(self, known_cov: "TimeSeries"):
        """
        Args:
            known_cov(TimeSeries): New known_cov

        Returns:
            None

        Raise:
            ValueError

        """
        self._known_cov = known_cov
        self._check_data()

    @static_cov.setter
    def static_cov(self, static_cov: "dict"):
        """
        Args:
            static_cov(dict): New static_cov
          
        Returns:
            None

        Raise:
            ValueError

        """
        self._static_cov = static_cov
        self._check_data()

    def split(
        self, 
        split_point: Union[pd.Timestamp, str, float, int], 
        after=True
    ) -> Tuple["TSDataset", "TSDataset"]:
        """
        Splits the TSDataset object into two TSDataset objects according to `split_point`, only valid when `self._target` is not None
        
        Args:
            split_point(pd.Timestamp|float|int): Where to split the TSDataset, which could be

                `pd.Timestamp|str`: Only valid when the type of time_index is pd.DatatimeIndex, and str will be forcibly converted to pd.DatatimeIndex

                `float`: The proportion of the length of the first TSDataset object

                `int`: Only valid when the type of time_index is pd.RangeIndex

                If the data of the split_point exists, it will be included in the first data
            after(bool): If `split_point` (pd.TimeSeries) doesn't exist in the time column, 
                use the next valid index (True) or the previous one (False)                 

        Returns:
            Tuple["TSDataset", "TSDataset"]
        
        Raise:
            ValueError
            TypeError

        """

        if self.target is not None:
            train_target, test_target = self._target.split(split_point, after)
            time_point_split = train_target.end_time
            if isinstance(time_point_split, int):
                time_point_split = (time_point_split - train_target.start_time) // train_target.freq + 1
        else:
            raise_if(
                self.observed_cov is not None and self.known_cov is not None,
                "Failed to split, the TSDataset's target can not be None when both observed_cov and known_cov are not None."
            )
            train_target, test_target = (None, None)
            time_point_split = split_point
        train_observed_cov, test_observed_cov = self._observed_cov.split(time_point_split, after) \
            if self._observed_cov else (None, None)
        _, test_known_cov = self._known_cov.split(time_point_split, after) \
            if self._known_cov else (None, None)
        return (
            TSDataset(train_target, train_observed_cov, self._known_cov, self._static_cov),
            TSDataset(test_target, test_observed_cov, test_known_cov, self._static_cov)
        )  

    def get_item_from_column(self, column: Union[str, int]) -> Union["TimeSeries", dict]:
        """
        Get the underlying TimeSeries object for targets, observed covariates, and know covariates, or the dict for static_covs according to the column name
        
        Args:   
            column(str): column name

        Returns: 
            Union["TimeSeries", dict]
        
        Raise:
            ValueError

        """
        if self._target and column in self._target.columns:
            return self.get_target()
        elif self._observed_cov and column in self._observed_cov.columns:
            return self.get_observed_cov()
        elif self._known_cov and column in self._known_cov.columns:
            return self.get_known_cov()
        elif self._static_cov and column in self._static_cov:
            return self.get_static_cov()
        else:
            raise ValueError(f"column: {column} not exists!")
    
    def __getitem__(
        self,
        columns: Union[str, int, List[Union[str, int]]]
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Get data from the specified columns
        
        Args:   
            columns(str|int|List): column names

        Returns:
            Union[pd.Series, pd.DataFrame]
        
        Raise:
            ValueError

        """
        if isinstance(columns, str) or isinstance(columns, int):
            columns = [columns]
        raise_if_not(
            len(set(columns)) == len(columns),
            "Duplicated values found in the columns"
        )
        res = None
        if self._target:
            columns_in_target = [v for v in columns if v in self._target.columns]
            if columns_in_target:
                if len(columns_in_target) == 1:
                    columns_in_target = columns_in_target[0]
                res = pd.concat([res, self._target.data[columns_in_target]], axis=1) \
                    if res is not None else self._target.data[columns_in_target]
        if self._observed_cov:
            columns_in_observed_cov = [v for v in columns if v in self._observed_cov.columns]
            if columns_in_observed_cov:
                if len(columns_in_observed_cov) == 1:
                    columns_in_observed_cov = columns_in_observed_cov[0]
                res = pd.concat([res, self._observed_cov.data[columns_in_observed_cov]], axis=1) \
                    if res is not None else self._observed_cov.data[columns_in_observed_cov]
        if self._known_cov:
            columns_in_known_cov = [v for v in columns if v in self._known_cov.columns]
            if columns_in_known_cov:
                if len(columns_in_known_cov) == 1:
                    columns_in_known_cov = columns_in_known_cov[0]
                res = pd.concat([res, self._known_cov.data[columns_in_known_cov]], axis=1) \
                    if res is not None else self._known_cov.data[columns_in_known_cov]
        if self._static_cov:
            columns_in_static_cov = [v for v in columns if v in self._static_cov]
            if columns_in_static_cov:
                len_static = 1 if res is None else res.shape[0]
                index_static = [1] if res is None else res.index
                for tmp in columns_in_static_cov:
                    tmp_df = pd.Series(
                        [self._static_cov[tmp] for i in range(len_static)],
                        index=index_static,
                        name=tmp
                    )
                    res = pd.concat([res, tmp_df], axis=1) \
                    if res is not None else tmp_df
        
        count = 0
        if res is not None:
            count = res.shape[1] if isinstance(res, pd.DataFrame) else 1
        raise_if_not(
            count == len(columns),
            "The specified columns don't exist!"
        )
        if isinstance(res, pd.DataFrame):
            return res[columns]
        else:
            return res
    
    def set_column(
        self,
        column: Union[str, int],
        value: Union[pd.Series, str, int],
        type: str = 'known_cov'
    ):
        """
        Add a new column or update the existing column
        
        Args:   
            column(str|int): column name
            value(pd.Series|str|int): New column values. When value=pd.Series, its index must be same as
                the index of the TSDataset object. When type='static_cov', value can only be int or str.
            type(str): Only effective when adding a new column, where to put the new column. By default, the new column will be added to known_cov.

        Returns:
            None
        
        Raise:
            ValueError

        """
        try:
            #Get the underlying TimeSeries object when the column exists
            attr = self.get_item_from_column(column)
        except ValueError:
            #If the column doesn't exist, then add a new column
            if type == 'target':
                raise_if_not(
                    isinstance(value, pd.Series),
                    "New column added to the target should be pd.Series."
                )
                if self._target is not None:
                    self._target.data[column] = value.reindex(self._target.time_index)
                else:
                    self._target = TimeSeries.load_from_dataframe(pd.DataFrame(
                        value.rename(column), 
                        index=value.index
                    ))
            elif type == 'known_cov':
                raise_if_not(
                    isinstance(value, pd.Series),
                    "New column added to the target should be pd.Series."
                )
                if self._known_cov is not None:
                    self._known_cov.data[column] = value.reindex(self._known_cov.time_index)
                else:
                    self._known_cov = TimeSeries.load_from_dataframe(pd.DataFrame(
                        value.rename(column), 
                        index=value.index
                    ))
            elif type == 'observed_cov':
                raise_if_not(
                    isinstance(value, pd.Series),
                    "New column added to the observed_cov should be pd.Series."
                )
                if self._observed_cov is not None:
                    self._observed_cov.data[column] = value.reindex(self._observed_cov.time_index)
                else:
                    self._observed_cov = TimeSeries.load_from_dataframe(pd.DataFrame(
                        value.rename(column), 
                        index=value.index
                    ))
            elif type == 'static_cov':
                raise_if_not(
                    isinstance(value, int) or isinstance(value, str),
                    "New column added to the static_cov should be int or str"
                )
                if self._static_cov is not None:
                    self._static_cov[column] = value
                else:
                    self._static_cov = {column: value}
            else:
                raise_log(
                    ValueError(f"Illegal type: {type}")
                )
            self._check_data()
            return
        #modify
        if attr == self._static_cov:
            raise_if_not(
                isinstance(value, str) or isinstance(value, int),
                "value is illegal!"
            )
            attr[column] = value 
        else:
            raise_if_not(
                isinstance(value, pd.Series),
                "value is illegal!"
            )
            attr.data[column] = value.reindex(attr.time_index)
    
    def __setitem__(
        self,
        column: Union[str, int],
        value: Union[pd.Series, str, int]
    ):
        """
        Update an existing column or add a new column to known_cov.
        For update, the column can be from the target, known_cov, observed_cov, or static_cov. 
        For addition, new columns will be added to known_cov, and see set_column for other operations.
        
        Args:   
            column(str|int): column name
            value(pd.Series|str|int): columns object, Its index must be the same as the index of the target property,
                the value can only be int or str when updating a column in static_cov

        Returns:
            None
        
        Raise:
            ValueError

        """
        # tsdataset['a'] = b only works for adding or updating columns in know_cov and turn to set_column for other cases
        type = "known_cov"
        self.set_column(column, value, type)

    def __str__(self):
        """str"""
        return self.to_dataframe().__str__()

    def __repr__(self):
        """repr"""
        return self.to_dataframe().__repr__()
    
    def drop(
        self,
        columns: Union[str, int, List[Union[str, int]]]
    ):
        """
        Drop column or columns
        
        Args:   
            columns(str|int|List): Column name or column names

        Returns:
            None
        
        Raise:
            ValueError

        """
        if isinstance(columns, str) or isinstance(columns, int):
            columns = [columns]
        raise_if_not(
            len(set(columns)) == len(columns),
            "Duplicated column names found"
        )
        if self._target is not None:
            columns_in_target = [v for v in columns if v in self._target.columns]
            if columns_in_target:
                self._target.data.drop(columns_in_target, axis=1, inplace=True)
                if self._target.data.shape[1] == 0:
                    self._target = None
        if self._observed_cov is not None:
            columns_in_observed_cov = [v for v in columns if v in self._observed_cov.columns]
            if columns_in_observed_cov:
                self._observed_cov.data.drop(columns_in_observed_cov, axis=1, inplace=True)
                if self._observed_cov.data.shape[1] == 0:
                    self._observed_cov = None
        if self._known_cov is not None:
            columns_in_known_cov = [v for v in columns if v in self._known_cov.columns]
            if columns_in_known_cov:
                self._known_cov.data.drop(columns_in_known_cov, axis=1, inplace=True)
                if self._known_cov.data.shape[1] == 0:
                    self._known_cov = None
        if self._static_cov is not None:
            columns_in_static_cov = [v for v in columns if v in self._static_cov]
            if columns_in_static_cov:
                for tmp in columns_in_static_cov:
                    del self._static_cov[tmp]
                if len(self._static_cov) == 0:
                    self._static_cov = None

    def plot(self, 
             columns:Union[List[str], str] = None, 
             add_data:Union[List["TSDataset"], "TSDataset"] = None,
             labels:Union[List[str], str] = None,
             low_quantile:float = 0.05,
             high_quantile:float = 0.95,
             central_quantile:float = 0.5,
             **kwargs) -> "pyplot":
        """
        plot function, a wrapper for Dataframe.plot()
        
        Args:   
            columns(str|List): The names of columns to be plot. 
                When columns is None, the targets will be plot by default.
            add_data(List|TSDataset): Add data for joint plotprinting, the default is None
            labels(str|List): Custom labels, length should be equal to nums of added datasets.
            central_quantile(float):The quantile (between 0 and 1) to plot as a "central" value, For instance, setting `central_quantile=0.5` will plot the
                median of each component.  (only used when dataset is probability forecasting output )
            low_quantile(float): The quantile to use for the lower bound of the plotted confidence interval. Similar to `central_quantile`,
                this is applied to each component separately (i.e., displaying marginal distributions). No confidence
                interval is shown if `confidence_low_quantile` is None (default 0.05). (only used when dataset is probability forecasting output )
            high_quantile(float):The quantile to use for the upper bound of the plotted confidence interval. Similar to `central_quantile`,
                this is applied to each component separately (i.e., displaying marginal distributions). No confidence
                interval is shown if `high_quantile` is None (default 0.95). (only used when dataset is probability forecasting output )
            **kwargs: Optional arguments passed to `Dataframe.plot` function

        Returns:
            matplotlib.pyplot object
        
        Raise:
            ValueError

        """
        quantile_cols = self._get_quantile_cols_origin_names()
        if not columns:
            if len(quantile_cols) == 0:
                columns = self._target.columns.values.tolist()
            else:
                columns = quantile_cols
        if isinstance(columns, str):
            columns = [columns]

        if len(columns) > 10:
            logger.info(f"To many columns to print ({len(columns)}), Plotting only the first 10 columns.")
            columns = columns[:10]

        #The type of plot, the default is line chart
        kind = "line"
        if "kind" not in kwargs:
            kwargs["kind"] = kind

        #Whether background grid is required, default is required
        grid = True
        if "grid" not in kwargs:
            kwargs["grid"] = grid

        #plot size
        figsize = (10,3)
        if "figsize" not in kwargs:
            kwargs["figsize"] = figsize

        if len(quantile_cols) == 0:
            all_cols = self.columns.keys()
        else:
            all_cols = quantile_cols
            if self.known_cov:
                all_cols = all_cols + self.known_cov.columns.values.tolist()
            if self.observed_cov:
                all_cols = all_cols + self.observed_cov.columns.values.tolist()
        # plot self data
        raise_if_not(set(columns) <= set(all_cols),
                     f"Columns {set(columns) - set(all_cols)} do not exist in origin datasets!")

        label = []
        for column in columns:
            # quantile plot
            if column in quantile_cols:
                central_quantile_str = "@quantile" + str(float(central_quantile * 100))
                df = self.__getitem__(column + central_quantile_str)
                plot = df.plot(**kwargs, label = column)
                self._fill_between_quantiles(column, low_quantile, high_quantile, **kwargs)
            # normal plot
            else:
                df = self.__getitem__(column)
                plot = df.plot(**kwargs, label = column)
        plot.legend()
        #plot added data

        if add_data:
            if isinstance(add_data, TSDataset):
                add_data = [add_data]
            col_len = len(columns)
            for ts in add_data:
                ts_quantile_cols = ts._get_quantile_cols_origin_names()

                if len(ts_quantile_cols) == 0:
                    all_cols = ts.columns.keys()
                else:
                    all_cols = ts_quantile_cols
                    if ts.known_cov:
                        all_cols = all_cols + ts.known_cov.columns.values.tolist()
                    if ts.observed_cov:
                        all_cols = all_cols + ts.observed_cov.columns.values.tolist()

                raise_if_not(set(columns) <= set(all_cols),
                             f"Columns {set(columns) - set(all_cols)} do not exist in added datasets!")

                if ts.freq != self.freq:
                    logger.warning("Add datas have different frequency with origin data!")

                for column in columns:
                    if column in ts_quantile_cols:
                        central_quantile_str = "@quantile" + str(float(central_quantile * 100))
                        df = ts.__getitem__(column + central_quantile_str)
                        plot = df.plot(**kwargs)
                        ts._fill_between_quantiles(column, low_quantile, high_quantile, **kwargs)
                    else:
                        df = ts.__getitem__(column)
                        plot = df.plot(**kwargs)

            # change labels
            _, origin_labels = plot.get_legend_handles_labels()
            origin_labels = [origin_labels[i].split("@quantile50")[0] for i in range(len(origin_labels))]
            if labels:
                if isinstance(labels, str):
                    labels = [labels]
                custome_labels = labels
                labels = origin_labels
                raise_if(len(custome_labels) != len(add_data), f"Custom labels does not match added datasets num:{len(add_data)}")
                count = 1
                while count <= len(add_data):
                    for i in range(col_len * count, col_len * (count + 1)):
                        labels[i] = custome_labels[count - 1] + "-" + labels[i]
                    count = count + 1  
            else:
                labels = origin_labels        
                count = 1
                while count <= len(add_data):
                    for i in range(col_len * count, col_len * (count + 1)):
                        labels[i] = "Add" + str(count) + "-" + labels[i]
                    count = count + 1       
            plot.legend(labels)
            
        return plot

    def _fill_between_quantiles(self,
                                column: str = None,
                                low_quantile: float = 0.05,
                                high_quantile: float = 0.95,
                                **kwargs):
        """
        Fill color between quantiles

        Args:   
            columns(str|List): The names of columns to be plot. 
                When columns is None, the targets will be plot by default.
            low_quantile(float): The quantile to use for the lower bound of the plotted confidence interval. No confidence
                interval is shown if `confidence_low_quantile` is None (default 0.05). (only used when dataset is probability forecasting output )
            high_quantile(float):The quantile to use for the upper bound of the plotted confidence interval. No confidence
                interval is shown if `high_quantile` is None (default 0.95). (only used when dataset is probability forecasting output )

        Return:
            None      

        """
        alpha = 0.25
        raise_if_not("@quantile" in self._target.columns[0], "This dataset do not have quantile info!")
        raise_if(low_quantile < 0 or low_quantile > 1, "Low quantile value should between 0 and 1!")
        raise_if(high_quantile < 0 or high_quantile > 1, "High quantile value should between 0 and 1!")
        raise_if(high_quantile < low_quantile, "Low quantile value should smaller than high quantile!")
        low_quantile_str = "@quantile" + str(float(low_quantile * 100))
        high_quantile_str = "@quantile" + str(float(high_quantile * 100))

        plt.fill_between(self.target.data.index,
                         self[column + low_quantile_str].values,
                         self[column + high_quantile_str].values,
                         alpha=(
                             alpha
                             if "alpha" not in kwargs
                             else kwargs["alpha"]
                         ),
                         label =str(int(low_quantile * 100)) + "%-" + str(int(high_quantile * 100)) + "% probability interval"
                         )

    def _get_quantile_cols_origin_names(self) -> List[str]:
        """
        Get quantile cols origin names

        Return:
            List[str]

        """
        origin_columns = []
        for name in self.target.columns:
            tmp = name.split("@quantile")
            if tmp[0] not in origin_columns and len(tmp) > 1:
                origin_columns.append(tmp[0])

        return origin_columns
        

    def copy(self) -> "TSDataset":
        """
        Make a copy of the TSDataset object
        
        Returns:
            TSDataset
        
        """
        target = self._target.copy() if self._target else None
        observed_cov = self._observed_cov.copy() if self._observed_cov else None
        known_cov = self._known_cov.copy() if self._known_cov else None
        static_cov = deepcopy(self._static_cov) if self._static_cov else None
        return TSDataset(target, observed_cov, known_cov, static_cov)

    def save(self, file: str):
        """
        Save TSDataset object to a file

        Args:   
            file(str): file path
        
        """
        with open(file, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file: str) -> "TSDataset":
        """
        Load TSDataset from the saved file
        
        Args:   
            file(str): file path

        Returns:
            TSDataset
        
        """
        with open(file, 'rb') as f:
            return pickle.load(f)
    
    def to_json(self) -> str:
        """
        Return a str json representation of the TSDataset object.

        Returns:
            str
        """
        attrs = ['target', 'observed_cov', 'known_cov']
        res = {}
        for attr in attrs:
            attr_ts = getattr(self, attr) 
            if attr_ts is not None:
                res[attr] = attr_ts.to_json()
            else:
                res[attr] = None
        if self.static_cov is not None:
            res['static_cov'] = json.dumps(self.static_cov, ensure_ascii=False)
        else:
            res['static_cov'] = None
        return json.dumps(res)

    @classmethod
    def load_from_json(cls, json_data: str, **json_load_kwargs) -> "TSDataset":
        """
        Construct a TSDataset object from a str json_data
        
        Args:
            json_data(str):  json object from which to load data
            **json_load_kwargs: Optional arguments passed to `json.loads` function
        
        Returns:
            TSDataset
        """
        res = json.loads(json_data, **json_load_kwargs)
        attrs = ['target', 'observed_cov', 'known_cov']
        params = {}
        for attr in attrs:
            if res[attr] is not None:
                params[attr] = TimeSeries.load_from_json(res[attr], **json_load_kwargs)
        if res['static_cov'] is not None:
            params['static_cov'] = json.loads(res['static_cov'], **json_load_kwargs)
        return TSDataset(**params)
    
    @property
    def columns(self) -> dict:
        """return all columns(except static columns)
        
        Returns:
            dict: The key is the column name, and the value is the type, including target, known_cov, and observed_cov
        """
        res = {}
        if self._target is not None:
            for column in self._target.columns:
                res[column] = 'target'
        if self._known_cov is not None:
            for column in self._known_cov.columns:
                res[column] = 'known_cov'
        if self._observed_cov is not None:
            for column in self._observed_cov.columns:
                res[column] = 'observed_cov'
        return res
    
    @property
    def freq(self):
        """Frequency of TSDataset"""
        return self._freq

    @classmethod
    def concat(cls, tss: List["TSDataset"], axis: int = 0, drop_duplicates = True, keep = 'first') -> "TSDataset":
        """
        Concatenate a list of TSDataset objects along the specified axis

        Args:
            tss(list[TimeSeries]): A list of TSDataset objects.
                All TSDatasets' freqs are required to be consistent. 
                When axis=1, time_col is required to be non-repetitive; 
                when axis=0, all columns are required to be non-repetitive
            axis(int): The axis along which to concatenate the TimeSeries objects
            drop_duplicates(bool): Drop duplicate indices.
            keep(str): keep 'first' or 'last' when drop duplicates.

        Returns:
            TSDataset
        
        Raise:
            ValueError

        """
        targets = [ts.get_target() for ts in tss if ts.get_target() is not None]
        target = TimeSeries.concat(targets, axis, drop_duplicates=drop_duplicates, keep=keep) if len(targets) != 0 else None
        known_covs = [ts.get_known_cov() for ts in tss if ts.get_known_cov() is not None]
        known_cov = TimeSeries.concat(known_covs, axis, drop_duplicates=drop_duplicates, keep=keep) if len(known_covs) != 0 else None
        observed_covs = [ts.get_observed_cov() for ts in tss if ts.get_observed_cov() is not None]
        observed_cov = TimeSeries.concat(observed_covs, axis, drop_duplicates=drop_duplicates, keep=keep) if len(observed_covs) != 0 else None
        static_cov = {}
        for ts in tss:
            if ts.get_static_cov() is not None:
                for key, value in ts.get_static_cov().items():
                    if key in static_cov:
                        raise_if_not(
                            static_cov[key] == value,
                            f"static cov key: {key} have diffent value! concat failed!"
                        )
                    else:
                        static_cov[key] = value
        return TSDataset(target, observed_cov, known_cov, static_cov)

    def astype(self, dtype: Union[np.dtype, type, Dict[str, Union[np.dtype, type]]]):
        """
        Cast a TSDataset object to the specified dtype

        Args:
            dtype(Union[np.dtype, type, Dict[str, Union[np.dtype, type]]]): Use a numpy.dtype or Python type to cast entire TimeSeries object to the same type. 
                Alternatively, use {col: dtype, …}, where col is a column label and dtype is a numpy.dtype or 
                Python type to cast one or more of the DataFrame’s columns to column-specific types.

        Raise:
            TypeError
            KeyError

        """
        target_type = {}
        known_cov_type = {}
        observed_cov_type = {}
        static_cov_type = {}
        if isinstance(dtype, dict):
            for key, value in dtype.items():
                raise_if_not(
                    key in self.columns or \
                    (self._static_cov and key in self._static_cov),
                    f"Invaild key: {key}"
                )
                if self._static_cov and key in self._static_cov:
                    static_cov_type[key] = value
                elif self.columns[key] == 'target':
                    target_type[key] = value
                elif self.columns[key] == 'known_cov':
                    known_cov_type[key] = value
                elif self.columns[key] == 'observed_cov':
                    observed_cov_type[key] = value
        else:
            target_type = known_cov_type = observed_cov_type = static_cov_type = dtype   
        if self._target is not None and target_type:
            self._target.astype(target_type)
        if self._known_cov is not None and known_cov_type:
            self._known_cov.astype(known_cov_type)
        if self._observed_cov is not None and observed_cov_type:
            self._observed_cov.astype(observed_cov_type)
        if self._static_cov and static_cov_type:
            if isinstance(static_cov_type, dict):
                for key, value in static_cov_type.items():
                    self._static_cov[key] = np.array([self._static_cov[key]]).astype(value)[0]
            else:
                for key, value in self._static_cov.items():
                    self._static_cov[key] = np.array([value]).astype(static_cov_type)[0]
        return self
    
    @property
    def dtypes(self) -> pd.Series:
        """
        Get dtypes of target, known_covs, observed_covs

        Returns:
            pd.Series: <column name, dtype>
        """
        type_list = []
        if self._target is not None:
            type_list.append(self._target.dtypes)
        if self._known_cov is not None:
            type_list.append(self._known_cov.dtypes)
        if self._observed_cov is not None:
            type_list.append(self._observed_cov.dtypes)
        if self._static_cov:
            type_list.append(pd.DataFrame(self._static_cov, index=[0]).dtypes)
        return pd.concat(type_list)

    def sort_columns(self, ascending: bool = True):
        """
        Sort the TSDataset object by the index
        
        Args:
            ascending(bool): Ascending or descending. When the index is a 
                MultiIndex the sort direction can be controlled for each level individually.

        """
        if self._target is not None:
            self._target.sort_columns(ascending)
        if self._known_cov is not None:
            self._known_cov.sort_columns(ascending)
        if self._observed_cov is not None:
            self._observed_cov.sort_columns(ascending)
        return self

    def to_categorical(self, col: Optional[Union[str, List[str]]] = None):
        """
        Modify col's type to int as categorical.

        Args:
            col(Optional[Union[str, List[str]]]): col names in ts

        """
        if col:
            if isinstance(col, str):
                col = [col]
            dtype = {col_one: np.int64 for col_one in col}
            self.astype(dtype)
        else:
            self.astype(np.int64)

    def to_numeric(self, col: Optional[Union[str, List[str]]] = None):
        """
        Modify col's type to float as numeric.

        Args:
            col(Optional[Union[str, List[str]]]): col names in ts

        """
        if col:
            if isinstance(col, str):
                col = [col]
            dtype = {col_one: np.float32 for col_one in col}
            self.astype(dtype)
        else:
            self.astype(np.float32)
