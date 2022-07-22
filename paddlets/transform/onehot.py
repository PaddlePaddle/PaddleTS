# !/usr/bin/env python3
# -*- coding:utf-8 -*-

import abc
from typing import Union, List

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

from paddlets.transform import base
from paddlets.datasets.tsdataset import TimeSeries, TSDataset
from paddlets.logger import Logger, raise_if_not, raise_if, raise_log
from paddlets.logger.logger import log_decorator

logger = Logger(__name__)


class OneHot(base.BaseTransform):
    """
    Transform categorical columns with OneHot encoder.
    
    Args:
        cols(str|List): Column(s) to be encoded.
        handle_unknown(str): {'error', 'ignore'}, default='error'
        drop(bool): Whether to delete the original column, default=False
        dtype(object): Data type, default=float
        categorie(str|List): 'auto' or a list of array-like, default='auto', if categorie is 'auto', it determine categories automatically from the dataset.
        
    Returns:
        None
    """
    def __init__(self, cols: Union[str, List], 
                dtype: object = np.float64, 
                handle_unknown: str = "error", 
                categories: Union[str, List] = 'auto', 
                drop: bool = False):
        super(OneHot, self).__init__()
        self._cols = cols
        self._drop = drop
        if isinstance(cols, str):
            self._cols=[cols]
        if len(self._cols) < 1:
            raise_log(ValueError("The column is not specified."))
        self._dtype = dtype
        self._handle_unknown = handle_unknown
        self._categories = categories
        self._preprocessor = None
        self._fitted = False
        try:
            self._preprocessor = OneHotEncoder(handle_unknown=self._handle_unknown,
                                           categories=self._categories,
                                           dtype=self._dtype)
        except Exception as e:
            raise_log(ValueError("init error: %s" % (str(e))))    
        
    @log_decorator
    def fit(self, dataset: TSDataset):
        """
        Fit the ecnoder with the dataset.
        
        Args:
            dataset(TSDataset): dataset from which to fit the encoder
        
        Returns:
            self
        """
        encoder_df = dataset[self._cols]
        if isinstance(encoder_df, pd.core.series.Series):
            encoder_df = encoder_df.to_frame()
        self._preprocessor.fit(encoder_df)
        self._fitted = True

        return self

    @log_decorator
    def transform(self, dataset: TSDataset, inplace: bool = False) -> TSDataset:
        """
        Transform the dataset with the fitted encoder.
        
        Args:
            dataset(TSDataset): dataset to be transformed.
            inplace(bool): whether to replace the original data. default=False
        
        Returns:
            TSDataset
        """
        if not self._fitted:
            raise_log(ValueError("This encoder is not fitted yet. Call 'fit' before applying the transform method." ))

        new_ts = dataset
        if not inplace:
            new_ts = dataset.copy()

        transform_df = new_ts[self._cols]
        if isinstance(transform_df, pd.core.series.Series):
            transform_df = transform_df.to_frame()
        result = self._preprocessor.transform(transform_df).toarray()
        result = pd.DataFrame(result).set_index(transform_df.index)

        pre_len = 0
        for i in range(len(self._cols)):
            data_item = new_ts.get_item_from_column(self._cols[i])
            series_len = len(set(new_ts[self._cols[i]]))
            for j in range(series_len):
                new_name = self._cols[i] + "_" + str(j)
                data_item.data[new_name] = result[pre_len + j]
            pre_len += series_len
            if self._drop:
                new_ts.drop(self._cols[i])
                
        return new_ts

    def fit_transform(self, dataset: TSDataset, inplace: bool = False) -> TSDataset:
        """
        First fit the encoder, and then transform the dataset.
        
        Args:
            dataset(TSDataset): dataset to be processed.
            inplace(bool): whether to replace the original data. default=False
        
        Returns:
            TSDataset
        """
        return self.fit(dataset).transform(dataset, inplace)
