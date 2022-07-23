#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from typing import List, Dict

import sklearn.metrics as metrics
import numpy as np

from paddlets.metrics.base import Metric
from paddlets.metrics.utils import ensure_2d


class MSE(Metric):
    """Mean Squared Error.

    Args:
        mode(str): Supported metric modes, only normal and prob are valid values.
            Set to normal for non-probability use cases, set to prob for probability use cases.
            Note that mode = prob is currently not supported.

    Attributes:
        _NAME(str): Metric name.
        _MAXIMIZE(bool): Identify optimization direction.
    """
    _NAME = "mse"
    _MAXIMIZE = False

    def __init__(
        self, 
        mode: str = "normal"
    ):
        super(MSE, self).__init__(mode)
   
    @ensure_2d
    def metric_fn(
        self, 
        y_true: np.ndarray, 
        y_score: np.ndarray
    ) -> float:
        """Mean squared error regression loss.

        Args:
            y_true(np.ndarray): Ground truth (correct) target values.
            y_score(np.ndarray): Estimated target values.

        Returns:
            float: Mean squared error regression loss. A non-negative floating point value (the best value is 0.0).
        """
        return metrics.mean_squared_error(y_true, y_score)


class MAE(Metric):
    """Mean Absolute Error.

    Args:
        mode(str): Supported metric modes, only normal and prob are valid values.
            Set to normal for non-probability use cases, set to prob for probability use cases.
            Note that mode = prob is currently not supported.
 
    Attributes:
        _NAME(str): Metric name.
        _MAXIMIZE(bool): Identify optimization direction.
    """
    _NAME = "mae"
    _MAXIMIZE = False

    def __init__(
        self, 
        mode: str = "normal"
    ):
        super(MAE, self).__init__(mode)
    
    @ensure_2d
    def metric_fn(
        self, 
        y_true: np.ndarray, 
        y_score: np.ndarray
    ) -> float:
        """Mean absolute error regression loss.

        Args:
            y_true(np.ndarray): Ground truth (correct) target values.
            y_score(np.ndarray): Estimated target values.

        Returns:
            float: Mean absolute error regression loss. A non-negative floating point value (the best value is 0.0).
        """
        return metrics.mean_absolute_error(y_true, y_score)
        

class LogLoss(Metric):
    """Log loss or cross-entropy loss.

    Args:
        mode(str): Supported metric modes, only normal and prob are valid values.
            Set to normal for non-probability use cases, set to prob for probability use cases.
            Note that mode = prob is currently not supported.

    Attributes:
        _NAME(str): Metric name.
        _MAXIMIZE(bool): Identify optimization direction.
    """
    _NAME = "logloss"
    _MAXIMIZE = False

    def __init__(
        self,
        mode: str = "normal"
    ):
        super(LogLoss, self).__init__(mode)
    
    def metric_fn(
        self, 
        y_true: np.ndarray, 
        y_score: np.ndarray
    ) -> float:
        """Log loss or cross-entropy loss.

        Args:
            y_true(np.ndarray): Ground truth (correct) labels.
            y_score(np.ndarray): Predicted probabilities.

        Returns:
            float: Log loss or cross-entropy loss.
        """
        return metrics.log_loss(y_true, y_score)


class MetricContainer(object):
    """Container holding a list of metrics.

    Args:
        metric_names(List[str]): List of metric names.
        prefix(str): Prefix of metric names.

    Attributes:
        _prefix(str): Prefix of metric names.
        _metrics(List[Metric]): List of metric instance.
        _names(List[str]): List of metric names associated with eval_name.
    """
    def __init__(
        self,
        metric_names: List[str],
        prefix: str = ""
    ):
        self._prefix = prefix
        self._metrics = Metric.get_metrics_by_names(metric_names)
        self._names = [prefix + name for name in metric_names]

    def __call__(
        self, 
        y_true: np.ndarray, 
        y_score: np.ndarray
    ) -> Dict[str, float]:
        """Compute all metrics and store into a dict.

        Args:
            y_true(np.ndarray): Ground truth (correct) target values.
            y_score(np.ndarray): Estimated target values.

        Returns:
            Dict[str, float]: Dict of metrics.
        """
        logs = {}
        for metric in self._metrics:
            res = metric.metric_fn(y_true, y_score)
            logs[self._prefix + metric._NAME] = res
        return logs

