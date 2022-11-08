#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from typing import Callable
import functools

import pandas as pd
import numpy as np

from paddlets.logger import raise_if_not, raise_if, raise_log, Logger
from paddlets.datasets import TSDataset
from paddlets.models import BaseModel

logger = Logger(__name__)

SAMPLE_ATTR_NAME = "_num_samples"
QUANTILE_OUTPUT_MODE = "quantiles"
QUANTILE_NUM = 101


def get_target_from_tsdataset(tsdataset: TSDataset):
    """
    Just reserve target in tsdataset.
   
    Args:
        tsdataset(TSDataset): Data to be converted.
    """
    if tsdataset.known_cov is not None or tsdataset.observed_cov is not None or tsdataset.static_cov is not None:
        logger.warning('covariant exists and will be filtered.')
        tsdataset = TSDataset(tsdataset.target)
    return tsdataset


def check_tsdataset(tsdataset: TSDataset):
    """Ensure the robustness of input data (consistent feature order), at the same time, 
    check whether the data types are compatible. If not, the processing logic is as follows.  

        1> Integer: Convert to np.int64. 

        2> Floating: Convert to np.float32. 

        3> Missing value: Warning. 

        4> Other: Illegal.

    Args:
        tsdataset(TSDataset): Data to be checked.
    """
    new_dtypes = {}
    for column, dtype in tsdataset.dtypes.items():
        if np.issubdtype(dtype, np.floating):
            new_dtypes.update({column: "float32"})
        elif np.issubdtype(dtype, np.integer): 
            new_dtypes.update({column: "int64"})
        else:
            msg = f"{dtype} data type not supported, the illegal columns contains: " \
                + f"{tsdataset.dtypes.index[tsdataset.dtypes==dtype].tolist()}"
            raise_log(TypeError(msg))

        # Check whether the data contains NaN.
        if np.isnan(tsdataset[column]).any() or np.isinf(tsdataset[column]).any():
            msg = f"np.inf or np.NaN, which may lead to unexpected results from the model"
            msg = f"Input `{column}` contains {msg}."
            logger.warning(msg)

    if new_dtypes:
        tsdataset.astype(new_dtypes)
    tsdataset.sort_columns() # Ensure the robustness of input data (consistent feature order).
            

def to_tsdataset(
    scenario: str = "forecasting"
) -> Callable[..., Callable[..., TSDataset]]:
    """A decorator, used for converting ndarray to tsdataset 
    (compatible with both DL and ML, compatible with both forecasting and anomaly).

    Args:
        scenario(str): The task type. ["forecasting", "anomaly_label", "anomaly_score"] is optional.

    Returns:
        Callable[..., Callable[..., TSDataset]]: Wrapped core function.
    """
    def decorate(func) -> Callable[..., TSDataset]:
        @functools.wraps(func)
        def wrapper(
            obj: BaseModel,
            tsdataset: TSDataset
        ) -> TSDataset:
            """Core processing logic.

            Args:
                obj(BaseModel): BaseModel instance.
                tsdataset(TSDataset): tsdataset.

            Returns:
                TSDataset: tsdataset.
            """
            raise_if_not(
                scenario in ("forecasting", "anomaly_label", "anomaly_score"),
                f"{scenario} not supported, ['forecasting', 'anomaly_label', 'anomaly_score'] is optional."
            )
            
            results = func(obj, tsdataset)
            if scenario == "anomaly_label" or scenario == "anomaly_score":
                # Generate target cols
                target_cols = tsdataset.get_target()
                if target_cols is None:
                    target_cols = [scenario]
                else:
                    target_cols = target_cols.data.columns
                    if scenario == "anomaly_score":
                        target_cols = target_cols + '_score'
                # Generate target index freq
                target_index = tsdataset.get_observed_cov().data.index
                if isinstance(target_index, pd.RangeIndex):
                    freq = target_index.step
                else:
                    freq = target_index.freqstr
                results_size = results.size
                raise_if(
                    results_size == 0,
                    f"There is something wrong, anomaly predict size is 0, you'd better check the tsdataset or the predict logic."
                )
                target_index = target_index[-results_size:]
                anomaly_target = pd.DataFrame(results, index=target_index, columns=target_cols)
                return TSDataset.load_from_dataframe(anomaly_target, freq=freq)
                
            past_target_index = tsdataset.get_target().data.index
            if isinstance(past_target_index, pd.RangeIndex):
                freq = past_target_index.step
                future_target_index = pd.RangeIndex(
                    past_target_index[-1] + (1 + obj._skip_chunk_len) * freq,
                    past_target_index[-1] + (1 + obj._skip_chunk_len + obj._out_chunk_len) * freq,
                    step=freq
                )
            else:
                freq = past_target_index.freqstr
                future_target_index = pd.date_range(
                    past_target_index[-1] + (1 + obj._skip_chunk_len) * past_target_index.freq,
                    periods=obj._out_chunk_len,
                    freq=freq
                )
            target_cols = tsdataset.get_target().data.columns
            # for probability forecasting and quantile output
            if hasattr(obj, SAMPLE_ATTR_NAME) and obj._output_mode == QUANTILE_OUTPUT_MODE: 
                target_cols = [x + "@" + "quantile" + str(y) for x in target_cols for y in range(QUANTILE_NUM)]
            future_target = pd.DataFrame(
                np.reshape(results, newshape=[obj._out_chunk_len, -1]),
                index=future_target_index,
                columns=target_cols
            )
            return TSDataset.load_from_dataframe(future_target, freq=freq)
        return wrapper
    return decorate

def to_tsdataset2(
    scenario: str = "anomaly_label"
    ) -> Callable[..., Callable[..., TSDataset]]:
    """A decorator, used for converting ndarray to tsdataset 
    (compatible with both DL and ML, compatible with both forecasting and anomaly).

    Args:
        scenario(str): The task type. ["anomaly_label", "anomaly_score"] is optional.
    Returns:
        Callable[..., Callable[..., TSDataset]]: Wrapped core function.
    """
    def decorate(func) -> Callable[..., TSDataset]:
        @functools.wraps(func)
        def wrapper(
            obj: BaseModel,
            tsdataset: TSDataset,
            traindataset: TSDataset,
            res_adjust: bool,
        ) -> TSDataset:
            """Core processing logic.

            Args:
                obj(BaseModel): BaseModel instance.
                tsdataset(TSDataset): tsdataset.
                traindataset(TSDataset): tsdataset.
                res_adjust(bool): Wether or not use result adjust.
                
            Returns:
                TSDataset: tsdataset.
            """
            raise_if_not(
                scenario in ("anomaly_label", "anomaly_score"),
                f"{scenario} not supported, ['anomaly_label', 'anomaly_score'] is optional."
            )
            
            results = func(obj, tsdataset, traindataset, res_adjust)
            if scenario == "anomaly_label" or scenario == "anomaly_score":
                # Generate target cols
                target_cols = tsdataset.get_target()
                if target_cols is None:
                    target_cols = [scenario]
                else:
                    target_cols = target_cols.data.columns
                    if scenario == "anomaly_score":
                        target_cols = target_cols + '_score'
                # Generate target index freq
                target_index = tsdataset.get_observed_cov().data.index
                if isinstance(target_index, pd.RangeIndex):
                    freq = target_index.step
                else:
                    freq = target_index.freqstr
                results_size = results.size
                raise_if(
                    results_size == 0,
                    f"There is something wrong, anomaly predict size is 0, you'd better check the tsdataset or the predict logic."
                )
                target_index = target_index[:results_size]# [-results_size:]
                anomaly_target = pd.DataFrame(results, index=target_index, columns=target_cols)
                return TSDataset.load_from_dataframe(anomaly_target, freq=freq)
        return wrapper
    return decorate

