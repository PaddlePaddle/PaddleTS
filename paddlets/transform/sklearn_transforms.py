# !/usr/bin/env python3
# -*- coding:utf-8 -*-

import abc
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union, Dict

import pandas as pd
import numpy as np
import scipy

from sklearn import preprocessing

from paddlets.transform.sklearn_transforms_base import SklearnTransformWrapper

class OneHot(SklearnTransformWrapper):
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
    def __init__(
        self, 
        cols: Union[str, List[str]], 
        dtype: object = np.float64, 
        handle_unknown: str = "error", 
        categories: Union[str, List[str]] = 'auto', 
        drop: bool = False
    ):
        super().__init__(
            preprocessing.__getattribute__('OneHotEncoder'),
            in_col_names=cols,
            per_col_transform=True,
            drop_origin_columns=drop,
            dtype=dtype,
            handle_unknown=handle_unknown,
            categories=categories,
        )

class Ordinal(SklearnTransformWrapper):
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
    def __init__(
        self, 
        cols: Union[str, List[str]], 
        dtype: np.dtype = np.dtype("float64"), 
        categories: Union[str, List[str]] = 'auto', 
        unknown_value: Union[int, None] = None,
        handle_unknown: str = 'error',
        drop: bool = False
    ):
        super().__init__(
            preprocessing.__getattribute__('OrdinalEncoder'),
            in_col_names=cols,
            per_col_transform=False,
            drop_origin_columns=drop,
            dtype=dtype,
            handle_unknown=handle_unknown,
            categories=categories,
            unknown_value=unknown_value,
        )

class MinMaxScaler(SklearnTransformWrapper):
    """
    Transform a dataset by scaling the values of sepcified column(s) to the expected range: [min, max].
    
    The transformation is done by:
    
    X_std = (X - X.min) / (X.max - X.min)
    
    X_scaled = X_std * (max - min) + min


    Args:
        cols(str|List): Column name(s) to be scaled.
        f_range(tuple): tuple (min, max), default=(0, 1), Desired range of transformed values.
        clip(bool): Set to True to clip transformed values of held-out data to provided feature range.

    Returns:
        None
    """ 
    def __init__(
        self, 
        cols: Union[str, List[str]]=None, 
        f_range: tuple=(0, 1), 
        clip: bool=False
    ):
        super().__init__(
            preprocessing.__getattribute__('MinMaxScaler'),
            in_col_names=cols,
            drop_origin_columns=True,
            per_col_transform=True,
            feature_range=f_range,
            clip=clip
        )

class StandardScaler(SklearnTransformWrapper):
    """
    Transform a dataset by scaling the values of sepcified column(s) to zero mean and unit variance.

    The transformation is done by:
    z = (x - u) / s.

    where u is the MEAN or zero if with_mean=False, and s is the standard deviation or one if with_std=False.

    Args:
        cols(str|List): Column name or names to be scaled.
        with_mean(bool): If True, center the data before scaling. 
        with_std(bool):If True, scale the data to unit variance.

    Returns:
        None
    """
    def __init__(
        self, 
        cols: Union[str, List[str]]=None, 
        with_mean: bool=True, 
        with_std: bool=True
    ):
        super().__init__(
            preprocessing.__getattribute__('StandardScaler'),
            in_col_names=cols,
            drop_origin_columns=True,
            per_col_transform=True,
            with_mean=with_mean,
            with_std=with_std
        )
