# !/usr/bin/env python3
# -*- coding:utf-8 -*-

from typing import Union, List

import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

from paddlets.transform import base
from paddlets.datasets.tsdataset import TimeSeries, TSDataset
from paddlets.logger import Logger, raise_if_not, raise_if, raise_log
from paddlets.logger.logger import log_decorator

logger = Logger(__name__)


class Ordinal(base.BaseTransform):
    """
    Encode categorical features as an integer array.
    
    Args:
        cols(str|List): Name of columns to Encode 
        handle_unknown(str): {‘error’, ‘use_encoded_value’}, default=’error’
        drop(bool): Whether to delete the original column, default=False.
        dtype(object): Number type, default=float.
        unknown_value(str): int or np.nan, default=None.
        categorie(str|List): 'auto' or a list of array-like, default='auto',if categorie is 'auto', it determine categories automatically from the training data. if categorie is list, categories[i] holds the categories expected in the ith column. The passed categories should not mix strings and numeric values, and should be sorted in case of numeric values.
    
    Returns:
        None
    """
    def __init__(self, 
                 cols: Union[str, List], 
                 dtype: np.dtype = np.dtype("float64"), 
                 categories: Union[str, List] = 'auto', 
                 unknown_value: Union[int, None] = None,
                 handle_unknown: str = 'error',
                 drop: bool = False):
        super(Ordinal, self).__init__()
        self._cols = cols
        self._drop = drop
        if isinstance(cols, str):
            self._cols=[cols]
        if len(self._cols) < 1:
            raise_log(ValueError("The feature column that needs to be encoded is not specified!"))
        self._dtype = dtype
        self._categories = categories
        self._handle_unknown = handle_unknown
        self._unknown_value = unknown_value
        self._preprocessor = None
        self._fitted = False
        try:
            self._preprocessor = OrdinalEncoder(categories=self._categories, 
                                                dtype=self._dtype, 
                                                handle_unknown=self._handle_unknown,
                                                unknown_value=self._unknown_value
                                                )
        except Exception as e:
            raise_log(ValueError("init error: %s" % (str(e))))  
        
    @log_decorator
    def fit(self, dataset: TSDataset):
        """
        Fit the OrdinalEncoder to dataset.
        
        Args:
            dataset(TSDataset): Dataset to be fitted.
        
        Returns:
            Ordinal
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
        Transform dataset to ordinal codes
        
        Args:
            dataset(TSDataset):  Dataset to be transformed.
            inplace(bool): Whether to perform the transformation inplace. default=False
        
        Returns:
            TSDataset
        """
        if not self._fitted:
            raise_log(ValueError("This OrdinalEncoder instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator." ))

        new_ts = dataset
        if not inplace:
            new_ts = dataset.copy()

        transform_df = new_ts[self._cols]
        if isinstance(transform_df, pd.core.series.Series):
            transform_df = transform_df.to_frame()
        result = self._preprocessor.transform(transform_df)
        result = pd.DataFrame(result).set_index(transform_df.index)
        
        for i in range(len(self._cols)):
            data_item = new_ts.get_item_from_column(self._cols[i])
            new_name = self._cols[i] + "_" + "encoder"
            data_item.data[new_name] = result[i]
            if self._drop:
                new_ts.drop(self._cols[i])            
        return new_ts

    def fit_transform(self, dataset: TSDataset, inplace: bool = False) -> TSDataset:
        """
        Fit OrdinalEncoder to dataset, then transform dataset.
        
        Args:
            dataset(TSDataset): Dataset to be fitted and transformed.
            inplace(bool): Whether to perform the transformation inplace.default=False
        
        Returns:
            TSDataset
        """
        return self.fit(dataset).transform(dataset, inplace)
