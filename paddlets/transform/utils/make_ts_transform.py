# !/usr/bin/env python3
# -*- coding:utf-8 -*-

import abc
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union, Dict

import pandas as pd
import numpy as np

from paddlets.logger import Logger, raise_if_not, raise_if, raise_log
from paddlets.transform.base import UdBaseTransform
from paddlets.transform.sklearn_transforms_base import SklearnTransformWrapper

def _check_is_sklearn_transform(transform_class) -> bool:
    """
    Check whether it is a sklearn style transform class.

    Args:
        User define or third-party transformer class.
    
    Returns:
        bool
    """
    if hasattr(transform_class, "fit") and \
        hasattr(transform_class, "transform"):
        return True
    else:
        return False

def make_ts_transform(
    ud_transform_class,
    in_col_names: Optional[Union[str, List]]=None,
    per_col_transform: bool=False,
    drop_origin_columns: bool=False,
    out_col_types: Optional[Union[str, List[str]]]=None,
    out_col_names: Optional[List[str]]=None,
    **sklearn_transform_params,
)-> UdBaseTransform:
    """
    Wrap the third-party or user-define transform into time series transform.

    Args:
        ud_transform_class: The transformer class from sklearn data transformation.
        in_col_names(Optional[Union[str, List]]): Column name or names to be transformed.
        per_col_transform(bool): Whether each column of data is transformed independently, default False.
        drop_origin_columns(bool): Whether to delete the original column, default=False.
        out_col_types(Optional[Union[str, List[str]]]): The type of output columns, None values represent automatic inference based on input.
        out_col_names(Optional[List[str]]): The name of output columns, None values represent automatic inference based on input.
        sklearn_transform_params: Optional arguments passed to sklearn_transform_class.
    
    Returns:
        UdBaseTransform.
    """
    if _check_is_sklearn_transform(ud_transform_class):
        return SklearnTransformWrapper(
            ud_transform_class,
            in_col_names,
            per_col_transform,
            drop_origin_columns,
            out_col_types,
            out_col_names,
            **sklearn_transform_params,
        )
    else:
        #TODO Support more types of third-party transform.
        raise_log(
            TypeError(f"Invaild ud transform class type: {type(ud_transform_class)}")
        )
