# !/usr/bin/env python3
# -*- coding:utf-8 -*-

from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot 

from paddlets import TimeSeries, TSDataset
from paddlets.logger import Logger, raise_if_not, raise_if, raise_log

logger = Logger(__name__)


class Analyzer(ABC):
    """
    ``Analyzer`` is the base class for all analyzers.
    Analyzer module is designed to perform specific mathematical analysis on time series data.
    All analyzer need to override analyze method.   

    Args:
        kwargs: Argument positions left for sub-classes.

    """
    def __init__(self, **kwargs):
        self._res = None
        self._kwargs = kwargs

    def _build_analysis_data(
        self,
        tsdataset: "TSDataset",
        columns: Optional[Union[str, List[str]]] = None,
        **kwargs
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Build a pd.Series or pd.DataFrame From a TSDataset or specific columns in a TSDataset.
        
        Args:
            tsdataset(TSDataset): TSDataset to be analyzed.
            columns(str|list[str]): Specific columns used to build pd.Series or pd.DataFrame.
            kwargs: argument positions Left for overide methods.

        Returns:
            pd.Series|pd.DataFrame: pd.Series or pd.DataFrame that convert from the TSDataset.

        Raise:
            ValueError

        """
        if columns is None:
            if tsdataset.get_target() is None and tsdataset.get_all_cov() is None:
                raise_log(
                    ValueError('tsdataset is empty!')
                )
            elif tsdataset.get_all_cov() is None:
                return tsdataset.get_target().data
            elif tsdataset.get_target() is None:
                return tsdataset.get_all_cov().data
            else:
                return pd.concat([tsdataset.get_target().data, tsdataset.get_all_cov().data], axis=1)
        else:
            return tsdataset[columns]

    @abstractmethod
    def analyze(
        self, 
        X: Union[pd.Series, pd.DataFrame],
        **kwargs
    ) -> Any:
        """
        Analyze data, need to be implemented by sub-classes. 

        Args:
            X(pd.Series|pd.DataFrame): Pd.Series or pd.DataFrame to be analyzed.
            kwargs: Argument positions left for override methods.

        Returns:
            Any: Analysis results.

        Raise:
            ValueError

        """
        pass

    @abstractmethod
    def get_properties(self) -> Dict:
        """
        Get the properties of the analyzer.
        All sub-classses should implements.

        Args:
            None

        Returns:
            Dict

        Raise:
            None

        """
        pass

    def plot(self) -> "pyplot":
        """
        The plot method of the Analyzer to show figures of Analysis results, optionally override this method.
        If analyzers need to  displays figures in the analysis report, this method needs to be overrided. 

        Args:
            None

        Returns:
            matplotlib.pyplot object

        Raise:
            None

        Examples:
            .. code-block:: python

                # Analysis results is DataFrame
                def plot(self):
                    fig = self.res.plot().get_figure()
                return fig

                # Other
                #Implement the plot method by yourself, return need to be matplotlib.pyplot object.

        """
        return None
    
    
    def __call__(
        self,
        tsdataset: "TSDataset",
        columns: Optional[Union[str, List[str]]] = None
    )-> Any:
        """
        Compute analysis's value from TSDataset
        
        Args:
            tsdataset(TSDataset): TSDataset to be analyzed.
            columns(str|List[str]): Specific columns to be analyzed. Analyze complete TSDataset by default.

        Returns:
            Any: Analysis results.

        Raise:
            ValueError

        """
        array = self._build_analysis_data(tsdataset, columns, **self._kwargs)
        self._res = self.analyze(array, **self._kwargs)
        return self._res
