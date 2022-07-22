#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from typing import Callable
import functools

import pandas as pd
import numpy as np

from paddlets.logger import raise_if, raise_log, Logger
from paddlets.datasets import TSDataset
from paddlets.models import BaseModel

logger = Logger(__name__)


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
            

def to_tsdataset(func) -> Callable[..., TSDataset]:
    """A decorator, used for converting ndarray to tsdataset (compatible with both DL and ML).

    Args:
        func(Callable[..., np.ndarray]): Core function.

    Returns:
        Callable[..., TSDataset]: Wrapped core function.
    """
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
        results = func(obj, tsdataset)
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
        future_target = pd.DataFrame(
            np.reshape(results, newshape=[obj._out_chunk_len, -1]),
            index=future_target_index,
            columns=tsdataset.get_target().data.columns
        )
        return TSDataset.load_from_dataframe(future_target, freq=freq)
    return wrapper

