# !/usr/bin/env python3
# -*- coding:utf-8 -*-

import abc
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union, Dict

import pandas as pd
import numpy as np
import scipy
import copy

from sklearn import preprocessing

from paddlets.transform.sklearn_transforms_base import SklearnTransformWrapper
from paddlets.logger import Logger, raise_if_not

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

    def _check_multi_tsdataset_static_cov(self, datasets: List["TSDataset"]):
        """
        Check the validity of static cov for multi time series combination transform

        Args:
            datasets(List[TSDataset]): datasets from which to fit or transform the transformer.
        """
        columns_set = set(tuple(sorted(dataset.static_cov.keys())) for dataset in datasets)
        raise_if_not(
            len(columns_set) == 1,
            f"Invalid tsdatasets. The given tsdataset statoc column schema ({[ts.static_cov for ts in datasets]}) must be same."
        )

    def fit(self, dataset: Union["TSDataset", List["TSDataset"]]):
        """
        Learn the parameters from the dataset needed by the transformer.

        Any non-abstract class inherited from this class should implement this method.

        The parameters fitted by this method is transformer-specific. For example, the `MinMaxScaler` needs to 
        compute the MIN and MAX, and the `StandardScaler` needs to compute the MEAN and STD (standard deviation)
        from the dataset. 

        Args:
            dataset(Union[TSDataset, List[TSDataset]]): dataset from which to fit the transformer.
        """

        if isinstance(dataset, list) and all(ds.static_cov is not None for ds in dataset):
            static_col = [col for col in self._cols if col in dataset[0].static_cov]
            if static_col:
                self._check_multi_tsdataset_static_cov(dataset)
                self._static_col = static_col
                self._cols = [col for col in self._cols if col not in self._static_col]
                self._static_ud_transformer = copy.deepcopy(self._ud_transformer)                
                static_np = np.array([[ds.static_cov[col] for col in static_col] for ds in dataset])
                self._static_ud_transformer.fit(static_np)
        
        return super().fit(dataset)


    def transform(
        self,
        dataset: Union["TSDataset", List["TSDataset"]],
        inplace: bool = False
    ) -> Union["TSDataset", List["TSDataset"]]:
        """
        Apply the fitted transformer on the dataset

        Any non-abstract class inherited from this class should implement this method.

        Args:
            dataset(Union[TSDataset, List[TSDataset]): dataset to be transformed.
            inplace(bool, optional): Set to True to perform inplace transformation. Default is False.
            
        Returns:
            Union[TSDataset, List[TSDataset]]: transformed dataset.
        """

        dataset = super().transform(dataset, inplace)

        if hasattr(self, "_static_ud_transformer") \
           and isinstance(dataset, list) \
           and all(ds.static_cov is not None for ds in dataset):
            self._check_multi_tsdataset_static_cov(dataset)  
            static_np = np.array([[ds.static_cov[col] for col in self._static_col] for ds in dataset])
            static_res = self._static_ud_transformer.transform(static_np)
            for col_id, col in enumerate(self._static_col):
                for ds_id, ds in enumerate(dataset):
                    ds.static_cov[col] = static_res[ds_id][col_id]
        
        return dataset


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
