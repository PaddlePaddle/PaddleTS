# !/usr/bin/env python3
# -*- coding:utf-8 -*-

from typing import Any, Callable, List, Optional, Sequence, Tuple, Union, Dict

import numpy as np
import pandas as pd

from paddlets import TimeSeries, TSDataset
from paddlets.logger import Logger, raise_if_not, raise_if, raise_log
from paddlets.analysis.base import Analyzer

logger = Logger(__name__)


class Summary(Analyzer):
    """
    Statistical indicators, currently support: numbers, mean, variance, minimum, 25% median, 50% median, 75% median, maximum value, missing percentage, stationarity p value

    Args:
        kwargs: Argument positions left for sub-classes.

    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def analyze(
        self, 
        X: Union[pd.Series, pd.DataFrame]
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Calculate Statistical indicators.

        Args:
            X(pd.Series|pd.DataFrame): Pd.Series or pd.DataFrame to be analyzed.

        Returns:
            pd.Series|pd.DataFrame: Analysis results.

        Raise:
            ValueError

        """
        des = X.describe()
        #TODO Add more Statistical indicators.
        # Add missing percentage indicator.
        if isinstance(X, pd.DataFrame):
            missing = (X.isna().sum()/X.shape[0]).rename('missing').to_frame().T
            return pd.concat([missing, des])
        else:
            des['missing'] = X.isna().sum()/X.shape[0]
            return des

    @classmethod
    def get_properties(cls) -> Dict:
        """
        Get the properties of the analyzer.

        Returns:
            Dict
        """

        return {
            "name": "summary",
            "report_heading": "SUMMARY",
            "report_description": "Specified statistical indicators, currently support: numbers, mean, \
                variance, minimum, 25% median, 50% median, 75% median, maximum value, missing percentage, stationarity p value"
        }


# Default instance for Summary
summary = Summary()


class Max(Analyzer):
    """
    Compute maximum values of given columns

    Args:
        kwargs: Argument positions left for sub-classes.

    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def analyze(
        self, 
        X: Union[pd.Series, pd.DataFrame]
    ) -> Union[Any, pd.Series]:
        """
        Compute the maximum values of given columns

        Args:
            X(pd.Series|pd.DataFrame): columns to be analyzed

        Returns:
            Any|pd.Series: The maximum value or the maximum values indexed by column names

        Raise:
            ValueError

        """
        res = X.max(axis=0, skipna=True)
        return res

    
    @classmethod
    def get_properties(cls) -> Dict:
        """
        Get the properties of the analyzer.

        Returns:
            Dict
        """

        return {
            "name": "max",
            "report_heading": "MAX",
            "report_description": "Maximum values of given columns"
        }


# Default instance for Max
max = Max()
