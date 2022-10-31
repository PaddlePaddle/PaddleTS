# !/usr/bin/env python3
# -*- coding:utf-8 -*-

from typing import Union, List

import pandas as pd
import numpy as np

from paddlets.transform.base import BaseTransform
from paddlets.datasets.tsdataset import TimeSeries, TSDataset
from paddlets.logger import Logger, raise_if_not, raise_if, raise_log
from paddlets.logger.logger import log_decorator

logger = Logger(__name__)

STATISTICS = ['median', 'mean', 'max', 'min', 'std']

class StatsTransform(BaseTransform):
    """
    Statistical features: 'median', 'mean', 'max', 'min', 'std'
    
    Args:
        cols(str|List): Name of columns to transform. 
        start(int): Start coordinates.
        end(int): End coordinates.
        statistics(str|List): Indicators that need to be counted, default=['median', 'mean', 'max', 'min', 'std'].

    Returns:
        None

    Examples:
        .. code-block:: python

            Given X:
                X
                1
                2
                3
                4
        
            statistics = ['mean'], start = 0, end = 2
        
            after transform:
                X X_mean
                1 nan
                2 1.5
                3 2.5
                4 3.5
            Remark: since the first element(1) start index has no value, the result of calculating the mean is nan

            statistics = ['mean'], start = 1, end = 3
        
            after transform:
                X X_mean
                1 nan
                2 nan
                3 1.5
                4 2.5
    """

    def __init__(self, cols: Union[str, List],
                 start: int = 0,
                 end: int = 1,
                 statistics: List = STATISTICS):
        super(StatsTransform, self).__init__()
        self._cols = cols
        if isinstance(cols, str):
            self._cols=[cols]
        if len(self._cols) < 1:
            raise_log(ValueError("The feature column is not specified!"))

        self._statistics = statistics
        if len(self._statistics) < 1:
            raise_log(ValueError("The statistics are not specified!"))
        if not set(self._statistics) <= set(STATISTICS):
            raise_log(ValueError("%s not in %s" % (self._statistics, STATISTICS)))
        
        if start < 0 or end < 0:
            raise_log(ValueError("Start or end index less than 0"))
        if end <= start:
            raise_log(ValueError("Start index greater than end"))
        
        self._start = start
        self._end = end
        self._map = {}
        for e in STATISTICS:
            self._map[e] = []
        
        self.need_previous_data = True
        self.n_rows_pre_data_need = self._end

    @log_decorator
    def fit_one(self, tsdata: TSDataset):
        """
        Fit the StatsTransform to dataset.
        
        Args:
            tsdata(TSDataset): Dataset to be fitted.
        
        Returns:
            StatsTransform
        """
        return self

    @log_decorator
    def transform_one(self, tsdata: TSDataset, inplace: bool = False) -> TSDataset:
        """
        Transform dataset to statstransform codes.
        
        Args:
            tsdata(TSDataset): Dataset to be transformed.
            inplace(bool): Whether to perform the transformation inplace. default=False
        
        Returns:
            TSDataset
        """
        new_ts = tsdata
        if not inplace:
            new_ts = tsdata.copy()
        
        statics_df = tsdata[self._cols]
        if isinstance(statics_df, pd.core.series.Series):
            statics_df = statics_df.to_frame()
        for col in self._cols:
            data_item = new_ts.get_item_from_column(col).data
            try:
                cur_series = statics_df[col].astype('float')
            except ValueError:
                logger.warning("Values in the column %s should be numerical" % (col))
            
            start_old = self._start
            end_old = self._end
            for i in range(len(cur_series)):
                end = i - start_old + 1
                start = i - end_old + 1
                for e in self._statistics:
                    self._map[e].append(cur_series[start: end].__getattr__(e)())

            for e in self._statistics:
                new_name = '%s_%s' % (col, e)
                if new_ts.columns[col] == 'target':
                    new_value = pd.Series(self._map[e])
                    new_value.index = new_ts[col].index
                    new_ts.set_column(new_name, new_value, 'observed_cov')
                else:
                    data_item[new_name] = self._map[e]
                self._map[e].clear()

        return new_ts
