# !/usr/bin/env python3
# -*- coding:utf-8 -*-

from typing import Any, Callable, List, Optional, Sequence, Tuple, Union, Dict

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
from scipy.signal import argrelmax
from scipy.stats import norm
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf

from paddlets import TimeSeries, TSDataset
from paddlets.logger import Logger, raise_if_not, raise_if, raise_log
from paddlets.analysis.base import Analyzer

logger = Logger(__name__)


class Seasonality(Analyzer):
    """
    Compute the seasonality period of given columns.

    Args:
        period(int): The period of the data. If None(by default), we will calculate a unique seasonality period.
        nlags(int): Number of lags to return autocorrelation for, default=300.
        alpha(float): The confidence intervals for the seasonality. default=0.05. 
        mode(str):  Type of seasonal component. Abbreviations are accepted. Optional("additive", "multiplicative").
        order(int): How many points on each side to use for the comparisont o consider ``comparator(n, n+x)`` to be True.
        kwargs: Other parameters.

    """
    def __init__(self, 
                 period: Optional[int] = None,
                 nlags: int = 300,
                 alpha: float = 0.05,
                 mode: str = 'additive',
                 order: int = 1,
                 **kwargs):
        super().__init__(**kwargs)
        if period:
            raise_if(period < 2, 'period must be more than 1!')
        self.period = period
        self.nlags = nlags
        self.alpha = alpha
        self.mode = mode
        self.order = order
        self.period_dict = {}
        self.seasonality_dict = {}
        
    def analyze(
        self, 
        X: Union[pd.Series, pd.DataFrame]
    ) -> Union[Any, pd.Series]:
        """
        Compute the seasonality period of given columns

        Args:
            X(pd.Series|pd.DataFrame): columns to be analyzed

        Returns:
            (dict, dict): The seasonality period and seasonality values

        Raise:
            ValueError

        """
        raise_if(
            self.nlags is None or not (1 <= self.nlags < len(X)),
            "nlags must be greater than or equal to 1 and less than len(X).",
        )
        if isinstance(X, pd.Series):
            X = X.to_frame()
        #calculate period
        self.period_dict = self._period(X)
        #extrack seasonality values
        self.seasonality_dict = self._seasonality(X, self.period_dict)
        return (self.period_dict, self.seasonality_dict)
    
    def _seasonality(
        self, 
        X: pd.DataFrame,
        period_dict: dict = {},
    ) -> Union[Any, pd.Series]:
        """
        extrack the seasonality values of given columns

        Args:
            X(pd.DataFrame): columns to be analyzed

        Returns:
            dict: The seasonality values by column names

        Raise:
            ValueError

        """
        season_dict = {}
        for col in period_dict:
            if period_dict[col] is not None and period_dict[col] * 2 < len(X[col]):
                ret = sm.tsa.seasonal_decompose(X[col].dropna().values, freq=period_dict[col], model=self.mode, extrapolate_trend="freq")
                season_dict[col] = ret.seasonal[: period_dict[col]]
            else:
                season_dict[col] = None
        return season_dict
    
    def _period(
        self, 
        X: pd.DataFrame
    ) -> Union[Any, pd.Series]:
        """
        Compute the seasonality period of given columns

        Args:
            X(pd.DataFrame): columns to be analyzed

        Returns:
            dict: The seasonality period by column names

        Raise:
            ValueError

        """
        period_dict = {}
        for col in X.columns:
            if self.period:
                period_dict[col] = self.period
            else:
                series = X[col]
                period_value = self._cal_period(series, col)
                period_dict[col] = period_value
        return period_dict
    
    def _cal_period(self, X: pd.Series, col: str):
        """
        Compute the period of given pd.Series

        Args:
            X(pd.Series): column to be analyzed

        Returns:
            int: period values

        Raise:
            ValueError
        """
        #calculate acf values
        acf_values, confident = acf(X.values, 
                                    nlags=self.nlags, 
                                    missing='drop', 
                                    qstat=False, 
                                    fft=False, 
                                    alpha=self.alpha,
                                   )
        #adjust whether were empty
        if len(acf_values) == 0:
            logger.warning('No seasonality in %s' % col)
            return None
        #select period
        periods = argrelmax(acf_values, order=self.order)[0]
        #adjust confidence
        interval = [confident[lag][1] - acf_values[lag] for lag in range(1, self.nlags + 1)]
        period_first = None
        for period in periods:
            if interval[period] >= acf_values[period]:
                continue
            period_first = period
            return period_first
        return period_first
    
    def plot(self) -> "pyplot":
        """
        display seasonality result.

        Args:
            None

        Returns:
            plt(matplotlib.pyplot object): The seasonality figure

        Raise:
            None
        """
        columns = [x for x in self.period_dict.keys() if self.period_dict[x]]
        columns_num = len(columns)
        if not columns:
            return
        fig, ax = plt.subplots(columns_num, 1, squeeze=False, sharex=False, figsize=(10,columns_num * 5))
        for i in range(0, columns_num):
            col_name = columns[i]
            if self.seasonality_dict[col_name] is None:
                continue
            x = range(self.period_dict[col_name])
            y = self.seasonality_dict[col_name]
            ax[i, 0].plot(x, y)
            ax[i, 0].set_title(col_name)
            ax[i, 0].set_xlabel('period points')
            ax[i, 0].set_ylabel('value')
            ax[i, 0].grid()
        plt.tight_layout()
        
        
        return plt
    
    @classmethod
    def get_properties(cls) -> Dict:
        """
        Get the properties of the analyzer.

        Returns:
            Dict
        """

        return {
            "name": "seasonality",
            "report_heading": "SEASONALITY",
            "report_description": "Seasonality of given columns"
        }


class Acf(Analyzer):
    """
    Compute the acf values of given columns.

    Args:
        nlags(int): Number of lags to return autocorrelation for, default=300.
        alpha(float): The confidence intervals for the acf. default=0.05. 
        kwargs: Other parameters.

    """
    def __init__(self, 
                 nlags: int = 300,
                 alpha: float = 0.05,
                 **kwargs):
        super().__init__(**kwargs)
        raise_if(
            alpha is None or not (0 < alpha < 1),
            "alpha must be greater than 0 and less than 1.",
        )
        self.nlags = nlags
        self.alpha = alpha
        self.acf_dict = {}
    
    def analyze(
        self, 
        X: Union[pd.Series, pd.DataFrame]
    ) -> Union[Any, pd.Series]:
        """
        Compute the acf values of given columns

        Args:
            X(pd.Series|pd.DataFrame): columns to be analyzed

        Returns:
            dict: The acf values and confident values

        Raise:
            ValueError

        """
        raise_if(
            self.nlags is None or not (1 <= self.nlags < len(X)),
            "nlags must be greater than or equal to 1 and less than len(X).",
        )
        if isinstance(X, pd.Series):
            X = X.to_frame()
        #calculate acf
        for col in X.columns:
            series = X[col]
            self.acf_dict[col] = self._acf(series, col)
        return self.acf_dict
    
    def _acf(self, X: pd.Series, col: str):
        """
        Compute the acf of given pd.Series

        Args:
            X(pd.Series): column to be analyzed

        Returns:
            (np.array, list): acf values and confident interval

        Raise:
            ValueError
        """
        #calculate acf values
        acf_values, confident = acf(
                X.values,
                nlags=self.nlags,
                fft=False,
                alpha=self.alpha,
                qstat=False,
                missing='drop'
        )
        if len(acf_values) == 0:
            return None
        interval = [confident[lag][1] - acf_values[lag] for lag in range(1, self.nlags + 1)]
        return (acf_values, interval)
    
    def plot(self) -> "pyplot":
        """
        display acf result.

        Args:
            None

        Returns:
            plt(matplotlib.pyplot object): The acf figure

        Raise:
            None
        """
        columns = [x for x in self.acf_dict.keys() if self.acf_dict[x]]
        if not columns:
            return
        columns_num = len(columns)
        fig, ax = plt.subplots(columns_num, 1, squeeze=False, sharex=False, figsize=(10,columns_num * 5))
        for i in range(0, columns_num):
            col_name = columns[i]
            if self.acf_dict[col_name] is None:
                continue
            acf_values = self.acf_dict[col_name][0]
            confident = self.acf_dict[col_name][1]
            ax[i, 0].plot(acf_values)
            ax[i, 0].fill_between(
                np.arange(1, self.nlags + 1), confident, [-x for x in confident], alpha=0.25
            )
            ax[i, 0].set_title(col_name)
            ax[i, 0].set_xlabel('nlags')
            ax[i, 0].set_ylabel('acf values')
            ax[i, 0].grid()
        plt.tight_layout()
        
        return plt
    
    @classmethod
    def get_properties(cls) -> Dict:
        """
        Get the properties of the analyzer.

        Returns:
            Dict
        """

        return {
            "name": "acf",
            "report_heading": "ACF",
            "report_description": "Acf of given columns"
        }
    
class Correlation(Analyzer):
    """
    Compute the correlation values of given columns.

    Args:
        method(str) : {'pearson', 'kendall', 'spearman'} or callable
        lag(int): lag time points.
        lag_cols(List[str], str): columns that need lag.
        kwargs: Other parameters.

    """
    def __init__(self, 
                 method: str = 'pearson',
                 lag: int = 0,
                 lag_cols: Optional[Union[str, List[str]]] = [],
                 **kwargs):
        super().__init__(**kwargs)
        self.method = method
        self.lag = lag
        self.corrs = None
        self.lag_cols = lag_cols
    
    def analyze(
        self, 
        X: pd.DataFrame
    ) -> Union[Any, pd.Series]:
        """
        Compute the correlation values of given columns.

        Args:
            X(pd.DataFrame): columns to be analyzed

        Returns:
            dict: The acf values and confident values

        Raise:
            ValueError

        """
        raise_if(len(X.columns) < 2, 'column number must be more than 1!')
        raise_if(len(X) < self.lag, 'lag must be less than len(X)!')
        #columns that need lag while lag > 0
        if self.lag > 0:
            for col in self.lag_cols:
                X[col] = X[col].shift(self.lag)
        #calculate correlation values
        self.corrs = self._correlation(X)
        return self.corrs
    
    def _correlation(self, X: pd.DataFrame):
        """
        Compute the correlation values of given columns.

        Args:
            X(pd.DataFrame): column to be analyzed

        Returns:
            list: correlation values

        Raise:
            ValueError
        """
        #calculate correlation values
        corr = X.corr(method=self.method)
        return corr
    
    def plot(self) -> "pyplot":
        """
        display correlation result.

        Args:
            None

        Returns:
            plt(matplotlib.pyplot object): The correlation figure

        Raise:
            None
        """
        columns = list(self.corrs.keys())
        columns_num = len(columns)
        plt.figure(figsize=(20, 16))
        mask = np.zeros_like(self.corrs, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        g = sns.heatmap(self.corrs, mask=mask, square=True, annot=True, fmt='0.2f', linewidths=1)
        plt.show()
        return plt
    
    @classmethod
    def get_properties(cls) -> Dict:
        """
        Get the properties of the analyzer.

        Returns:
            Dict
        """

        return {
            "name": "correlation",
            "report_heading": "CORRELATION",
            "report_description": "Correlation of given columns"
        }
