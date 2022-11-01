#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from typing import List, Dict, Any, Callable, Optional, Tuple, Union
import math
from collections import defaultdict, Iterable

import pandas as pd
import numpy as np
from tqdm import tqdm

from paddlets.metrics import Metric, MSE
from paddlets.models.base import Trainable
from paddlets.datasets import TSDataset, TimeSeries
from paddlets.logger import Logger, raise_if
from paddlets.utils.utils import check_model_fitted

logger = Logger(__name__)


def backtest(
        data: TSDataset,
        model: Trainable,
        start: Union[pd.Timestamp, int, str ,float] = None,
        predict_window: Optional[int] = None,
        stride: Optional[int] = None,
        metric: Optional[Metric] = None,
        return_predicts: bool = False,
        reduction: Union[Callable[[np.ndarray], float], None] = np.mean,
        verbose: bool = True
) -> Union[float, Tuple[float, Union[TSDataset, List[TSDataset]]]]:
    """
    Backtest
    A repeated forecasting and validating process. It first use data with the length of predict_window, 
    and then moves the end of the training set forward by `stride` time steps. 
    By default, Backtest will generate a TSdataset with length (data_length - model.skip_chunk_len) as output.
    If set predict_window != stride Backtest will generate a List of TSdataset as output

    Args:
        data(TSDataset): The  TSdataset to use to successively evaluate the historical forecasts
        model(Trainable): The  fitted model to use to successively evaluate the historical forecasts
        start(pd.Timestamp|int|None): The first prediction time, at which a prediction is computed for a future time.
        predict_window(int|None): The predict window for the  prediction.
        stride(int|None): The number of time steps between two consecutive predict window.
        metric(Metric): A function that takes two ``TSdataset`` instances as inputs and returns an error value.
        return_predicts(bool):  If set return_predicts=True, the predict results will return additionaly.
        reduction(Callable[[np.ndarray]|None):A function used to combine the individual error scores obtained when predict_window ！= stride.
            If explicitely set to `None`, the method will return a list of the individual error scores instead.
            Set to ``np.mean`` by default.
        verbose(bool): Turn on Verbose mode,set to true by default.

    Returns:
        float|(float, TSDataset|List[TSDataset]): Return score by default, If set return_predicts=True, the predict results will return additionaly.
         
    Raise:
        ValueError

    """
    from paddlets.pipeline import Pipeline
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
        raise_if(start < model_in_chunk_len + model_skip_chunk_len, f"Parameter 'start' value should >= in_chunk_len {model_in_chunk_len} + skip_chunk_len {model_skip_chunk_len}")
        raise_if(start > target_length, f"Parameter 'start' value should not exceed data target_len {target_length}")
        raise_if(predict_window <= 0, "Parameter 'predict_window' should be positive integer")
        raise_if(stride <= 0, "Parameter 'stride' should be positive integer")
        # When the prediction window is larger than output_chunk_len, recursive prediction is required.
        # Recursive prediction does not support skip_chunk_len !=0.
        raise_if(model_skip_chunk_len != 0 and predict_window > model_out_chunk_len, "Backtest can not work when _skip_chunk_len!=0 and \
                window > _out_chunk_len at same time.")
        # If skip_chunk_len !=0, prediction will start from start + skip_chunk_len.
        if model_skip_chunk_len != 0:
            if verbose:
                logger.info(f"model.skip_chunk_len is {model_skip_chunk_len}, backtest will start at \
                    index {start + model_skip_chunk_len} (start + skip_chunk_len)")

    data = data.copy()
    all_target = data.get_target()
    all_observe = data.get_observed_cov() if data.get_observed_cov() else None
    target_length = len(all_target)
    if predict_window is None:
        predict_window = model_out_chunk_len
        if verbose:
            logger.info(f"Parameter 'predict_window' not set, default set to model.out_chunk_len {model_out_chunk_len}")
    if stride is None:
        stride = predict_window
        if verbose:
            logger.info(f"Parameter 'stride' not set, default set to predict_window {predict_window}")
    # If start is not set, set to model._in_chunk_len by default
    if start is None:
        start = model_in_chunk_len + model_skip_chunk_len
        if verbose:
            logger.info(f"Parameter 'start' not set, default set to model_in_chunk_len {model_in_chunk_len} + skip_chunk_len {model_skip_chunk_len}")
    start = all_target.get_index_at_point(start)
    # Parameter check
    _check()
    start = start - model_skip_chunk_len
    # When predict_window == stride, the prediction will form a complete continuous time series, which will be automatically merged by default and return the complete TSdataset
    # If predict_window! = stride, the forecast will form a discontinuous time series, do not processed by default and returns List[TSdataset]
    return_tsdataset = True if predict_window == stride else False

    length = target_length - start - model_skip_chunk_len
    predict_rounds = math.ceil(length / stride)
    predicts = []
    scores = []
    index = start

    TQDM_PREFIX = "Backtest Progress"
    for _ in tqdm(range(predict_rounds), desc=TQDM_PREFIX, disable=not verbose):
        data._target, rest = all_target.split(index) 
        data._observed_cov, _ = all_observe.split(index) if all_observe else (None, None)
        rest_len = len(rest)

        if rest_len < model_out_chunk_len + model_skip_chunk_len:
            if data.known_cov is not None:
                target_end_time = data._target.end_time
                known_index = data.known_cov.get_index_at_point(target_end_time)
                if len(data.known_cov) - known_index - 1 < model_out_chunk_len + model_skip_chunk_len:
                    break

        if rest_len < predict_window + model_skip_chunk_len:
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
        predicts.append(output)
        real = TSDataset(target=TimeSeries(all_target.data[index + model_skip_chunk_len: index + predict_window + model_skip_chunk_len], output.freq))
        predict = output

        if metric is None:
            metric = MSE()
        score_dict = metric(real, predict)
        scores.append(score_dict)
        index = index + stride

    if reduction:
        if metric._TYPE == "quantile" and isinstance(list(scores[0].values())[0], dict):
            target_cols = [x for x in scores[0].keys()]

            tmp = {}
            for cols in target_cols:
                tmp[cols] = defaultdict(list)
                for dct in [x[cols] for x in scores]:
                    for k, v in dct.items():
                        if isinstance(v, Iterable):
                            tmp[cols][k].extend(v)
                        else:
                            tmp[cols][k].append(v)
                tmp[cols] = {k: reduction(v) for k, v in tmp[cols].items()}

            scores = tmp

        else:
            tmp = defaultdict(list)
            for dct in [x for x in scores]:
                for k, v in dct.items():
                    if isinstance(v, Iterable):
                        tmp[k].extend(v)
                    else:
                        tmp[k].append(v)

            tmp = {k: reduction(v) for k, v in tmp.items()}
            scores = tmp


    if return_predicts:
        if return_tsdataset:
            predicts = TSDataset.concat(predicts)
        return scores, predicts
    else:
        return scores
