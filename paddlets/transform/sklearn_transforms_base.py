# !/usr/bin/env python3
# -*- coding:utf-8 -*-

import abc
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union, Dict

import pandas as pd
import numpy as np
import scipy

from paddlets.transform.base import UdBaseTransform
from paddlets.datasets.tsdataset import TimeSeries, TSDataset
from paddlets.logger import Logger, raise_if_not, raise_if, raise_log
from paddlets.logger.logger import log_decorator

logger = Logger(__name__)


class SklearnTransformWrapper(UdBaseTransform):
    """
    Wrapper for data transformation classes provided by skearn data transformation.

    Args:
        sklearn_transform_class: The transformer class from sklearn data transformation.
        in_col_names(Optional[Union[str, List[str]]]): Column name or names to be transformed.
        per_col_transform(bool): Whether each column of data is transformed independently, default False.
        drop_origin_columns(bool): Whether to delete the original column, default=False.
        out_col_types(Optional[Union[str, List[str]]]): The type of output columns, None represents automatic inference based on input.
        out_col_names(Optional[List[str]]): The name of output columns, None represents automatic inference based on input.
        sklearn_transform_params: Optional arguments passed to sklearn_transform_class.

    """
    def __init__(
        self,
        sklearn_transform_class,
        in_col_names: Optional[Union[str, List[str]]]=None,
        per_col_transform: bool=False,
        drop_origin_columns: bool=False,
        out_col_types: Optional[Union[str, List[str]]]=None,
        out_col_names: Optional[List[str]]=None,
        **sklearn_transform_params,
    ):
        ud_sklearn_transformer = sklearn_transform_class(**sklearn_transform_params)
        super().__init__(
            ud_sklearn_transformer,
            in_col_names,
            per_col_transform,
            drop_origin_columns,
            out_col_types,
            out_col_names
        )
    
    def _gen_output(
        self,
        raw_output
    )->np.ndarray:
        """
        Generate the np.ndarray output base on the raw_output from ud transform.

        Args:
            raw_output(TSDataset): raw_output from ud transform.

        Returns:
            output(np.ndarray)
        """
        if isinstance(raw_output, scipy.sparse.csr.csr_matrix):
            return raw_output.toarray()
        else:
            return super()._gen_output(raw_output) 

    def _fit(self, input: pd.DataFrame):
        """
        Learn the parameters from the dataset needed by the transformer.
        
        Args:
            input(pd.DataFrame): The input to transformer.
        
        Returns:
            None
        """
        self._ud_transformer.fit(input)

    def _transform(
        self, 
        input: np.ndarray
    ):
        """
        Transform the dataset with the fitted transformer.
        
        Args:
            input(pd.DataFrame): The input to transformer.
         
        """
        if not self._fitted:
            raise_log(ValueError("This encoder is not fitted yet. Call 'fit' before applying the transform method." ))
        return self._ud_transformer.transform(input)

    def _inverse_transform(
            self, 
            input: np.ndarray
        ):
        """
        Inversely transform the dataset output by the `transform` method.

        Args:
            input(pd.DataFrame): The input to transformer.
        
        """
        if not self._fitted:
            raise_log(ValueError("This encoder is not fitted yet. Call 'fit' before applying the transform method." ))
        if hasattr(self._ud_transformer, 'inverse_transform'):
            return self._ud_transformer.inverse_transform(input)
        else:
            raise_log(
                NotImplementedError(f"inverse_transform not implemented")
            )
