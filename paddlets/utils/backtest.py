#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from typing import List, Dict, Any, Callable, Optional, Tuple, Union
import math

import pandas as pd
import numpy as np

from paddlets.metrics import Metric, MSE
from paddlets.models.base import Trainable
from paddlets.datasets import TSDataset, TimeSeries
from paddlets.pipeline import Pipeline
from paddlets.logger import Logger, raise_if
from paddlets.utils.utils import check_model_fitted

logger = Logger(__name__)


def backtest(
        data: TSDataset,
        model: Trainable,
        start: Union[pd.Timestamp, int, str ,float] = None,
        predict_window: int = 1,
        stride: int = 1,
        metric: Optional[Metric] = None,
        return_score: bool = True,
        reduction: Union[Callable[[np.ndarray], float], None] = np.mean
) -> Union[TSDataset, List[TSDataset], float]:
    """
    Backtest
    A repeated forecasting and validating process. It first use data with the length of predict_window, 
    and then moves the end of the training set forward by `stride` time steps. 
    By default, Backtest will generate a TSdataset with length (data_length - model.skip_chunk_len) as output.
    If set predict_window != stride Backtest will generate a List of TSdataset as output
    If set return_score=True, A metric (given by the `metric` function) is then evaluated 
    on the forecast and the actual values. It will returns a mean-of all these metric scores by default.

    Args:
        data(TSDataset): The  TSdataset to use to successively evaluate the historical forecasts
        model(Trainable): The  fitted model to use to successively evaluate the historical forecasts
            start(pd.Timestamp|int|None): The first prediction time, at which a prediction is computed for a future time.
        predict_window(int): The predict window for the  prediction.
        stride(int): The number of time steps between two consecutive predict window.
        metric(Metric): A function that takes two ``TSdataset`` instances as inputs and returns an error value.
        return_score(bool):  If set return_score=True, A metric (given by the `metric` function) is then evaluated 
            on the forecast and the actual values. It will returns a mean-of all these metric scores by default.
        reduction(Callable[[np.ndarray]|None):A function used to combine the individual error scores obtained when predict_window ！= stride.
            If explicitely set to `None`, the method will return a list of the individual error scores instead.
            Set to ``np.mean`` by default.

    Returns:
        Union[TSDataset,List[TSdataset],float]
         
    Raise:
        ValueError

    """

    if isinstance(model, Pipeline):
        #如果是None, 后续的check会报错，这里就不做重复处理了
        if model._model_exist:
            model_in_chunk_len = model._model._in_chunk_len
            model_skip_chunk_len = model._model._skip_chunk_len
            model_out_chunk_len = model._model._out_chunk_len
    else:
        model_in_chunk_len = model._in_chunk_len
        model_skip_chunk_len = model._skip_chunk_len
        model_out_chunk_len = model._out_chunk_len

    def _check():
        # Check whether model fitted or not.
        check_model_fitted(model)
        raise_if(start < model_in_chunk_len, f"Parameter 'start' value should >= in_chunk_len {model_in_chunk_len}")
        raise_if(start > target_length, f"Parameter 'start' value should not exceed data target_len {target_length}")
        raise_if(predict_window <= 0, "Parameter 'predict_window' should be positive integer")
        raise_if(stride <= 0, "Parameter 'stride' should be positive integer")
        # When the prediction window is larger than output_chunk_len, recursive prediction is required.
        # Recursive prediction does not support skip_chunk_len !=0.
        raise_if(model_skip_chunk_len != 0 and predict_window > model._out_chunk_len, "Backtest can not work when _skip_chunk_len!=0 and \
                window > _out_chunk_len at same time.")
        # If skip_chunk_len !=0, prediction will start from start + skip_chunk_len.
        if model_skip_chunk_len != 0 and predict_window > model_out_chunk_len:
            logger.info(f"model.skip_chunk_len is {model_skip_chunk_len}, backtest will start at \
                index {start + model_skip_chunk_len} (start + skip_chunk_len)")

    data = data.copy()
    all_target = data.get_target()
    target_length = len(all_target)

    # If start is not set, set to model._in_chunk_len by default
    if start is None:
        start = model_in_chunk_len
        logger.info(f"Parameter 'start' not set, default set to model.in_chunk_len {model_in_chunk_len}")
    start = all_target.get_index_at_point(start)
    # Parameter check
    _check()
    # When predict_window == stride, the prediction will form a complete continuous time series, which will be automatically merged by default and return the complete TSdataset
    # If predict_window! = stride, the forecast will form a discontinuous time series, do not processed by default and returns List[TSdataset]
    return_tsdataset = True if predict_window == stride else False

    length = target_length - start
    predict_rounds = math.ceil(length / stride)
    results = []
    scores = []
    index = start

    for _ in range(predict_rounds):
        data._target, rest = all_target.split(index) 
        rest_len = len(rest)
        if rest_len <= predict_window:
            if data.known_cov is not None:
                target_end_time = data._target.end_time
                known_index = data.known_cov.get_index_at_point(target_end_time)
                if len(data.known_cov) - known_index - 1 < predict_window:
                    break
            predict_window = rest_len
            predict_window = predict_window - model_skip_chunk_len

        # Use recursive prediction if the prediction window is larger than the model output_chunk_len
        if predict_window > model_out_chunk_len:

            output = model.recursive_predict(data, predict_length=predict_window)
        else:
            output = model.predict(data)
            output.set_target(
                TimeSeries(output.get_target().data[0: predict_window], output.freq)
            )

        results.append(output)
        real_values = TimeSeries(all_target.data[index: index + predict_window], output.freq).to_numpy()
        predict_values = output._target.to_numpy()

        if return_score:
            if metric is None:
                metric = MSE()
            score = metric.metric_fn(real_values, predict_values)
            scores.append(score)
        index = index + stride

    if return_tsdataset:
        results = TSDataset.concat(results)

    if reduction:
        return reduction(scores) if return_score else results
    else:
        return scores if return_score else results

