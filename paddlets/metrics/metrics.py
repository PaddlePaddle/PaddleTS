#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from typing import List, Dict, Union

import sklearn.metrics as metrics
import numpy as np
import paddle
from paddle import distribution

from paddlets.metrics.base import Metric
from paddlets.metrics.utils import ensure_2d
from paddlets.logger import Logger, raise_if_not, raise_if, raise_log


logger = Logger(__name__)


class MSE(Metric):
    """Mean Squared Error.

    Args:
        mode(str): Supported metric modes, only normal and prob are valid values.
            Set to normal for non-probability use cases, set to prob for probability use cases.

    Attributes:
        _NAME(str): Metric name.
        _MAXIMIZE(bool): Identify optimization direction.
    """
    _NAME = "mse"
    _TYPE = "point"
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
 
    Attributes:
        _NAME(str): Metric name.
        _MAXIMIZE(bool): Identify optimization direction.
    """
    _NAME = "mae"
    _TYPE = "point"
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

    Attributes:
        _NAME(str): Metric name.
        _MAXIMIZE(bool): Identify optimization direction.
    """
    _NAME = "logloss"
    _TYPE = "point"
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


class QuantileLoss(Metric):
    """
    Quantile loss, following the article: `Bayesian Intermittent Demand Forecasting for Large Inventories <https://papers.nips.cc/paper/2016/file/03255088ed63354a54e0e5ed957e9008-Paper.pdf>`_ .
        A quantile of ``q=0.5`` will give half of the mean absolute error as it is calcualted as ``max(q * (y-y_pred), (1-q) * (y_pred-y))``.

    Args:
        q_points(List[float]): The quantile points of interest, the default value is None. 
            In the evaluation of the prediction, while q_points is specified, 
            output a dict which contains each quantile result respect to quantile points.
        mode(str): Supported metric modes, only normal and prob are valid values. 
            Set to normal for non-probability use cases, set to prob for probability use cases.
    """
    _NAME = "quantile_loss"
    _TYPE = "quantile"
    _MAXIMIZE = False

    def __init__(
        self,
        q_points: List[float]=[0.1, 0.5, 0.9],
        quantile_level: Union[np.ndarray, List[float], None] = None,
        mode: str="prob",
    ):
        self._train = True
        self._q_points = q_points
        raise_if_not(mode=="prob", "QuantileLoss metric only support `prob` mode")
        if quantile_level is not None:
            logger.warning(f"The parameter `quantile_level` has been deprecated and will be removed in future update.")
        super(QuantileLoss, self).__init__(mode=mode, q_points=q_points)
        
    def __call__(
        self,
        tsdataset_true: "TSDataset",
        tsdataset_pred: "TSDataset",
    )-> Dict[str, float]:
        """
        Compute metric's value from TSDataset, overwrite `__call__` in base class.
        
        Args:
            tsdataset_true(TSDataset): TSDataset containing ground truth (correct) target values.
            tsdataset_pred(TSDataset): TSDataset containing estimated target values.

        Returns:
            Dict[str, float]: Dict of metrics. key is the name of target, and value is specific metric value. 
        """
        self._train = False
        return super(QuantileLoss, self).__call__(tsdataset_true, tsdataset_pred)
        
    def metric_fn(
        self,
        y_true: np.ndarray,
        y_pred_sample: np.ndarray, # sampling result
    ) -> Union[float, dict]:
        """
        Quantile loss.

        Args:
            y_true(np.ndarray): Ground truth (correct) labels.
            y_score(np.ndarray): Predicted quantiles.
        
        Returns:
            float: Quantile loss.
        """
        q_points = self._kwargs["q_points"]
        quantiles = np.quantile(y_pred_sample, q_points, axis=-1, interpolation="nearest")
        errors = [y_true - quantiles[i] for i in range(len(q_points))]
        losses = [np.max(np.stack([(q_points[i] - 1) * errors[i], q_points[i] * errors[i]], axis=-1), axis=-1)
                  for i in range(len(q_points))]
        losses_array = np.stack(losses, axis=-1)
        if self._train:
            # sum losses over quantiles and average across time and observations, for training scenario
            return np.sum(losses_array, axis=-1).mean(axis=-1).mean()  # a scalar (shapeless)
        else:
            # compute q_risk for each quantile, for eval scenario
            q_risk = 2 * [losses_array[..., i].sum() for i in range(losses_array.shape[-1])] / np.abs(y_true).sum()
            return dict(zip(q_points, q_risk))
        

class ACC(Metric):
    """Accuracy_score.

    Args:
        mode(str): Supported metric modes, only anomaly is valid value.

    Attributes:
        _NAME(str): Metric name.
    """
    _NAME = "acc"

    def __init__(
        self, 
        mode: str = "anomaly"
    ):
        super(ACC, self).__init__(mode)
   
    @ensure_2d
    def metric_fn(
        self, 
        y_true: np.ndarray, 
        y_score: np.ndarray
    ) -> float:
        """Accuracy_score.

        Args:
            y_true(np.ndarray): Ground truth (correct) target values.
            y_score(np.ndarray): Estimated target values.

        Returns:
            float: accuracy_score. A non-negative floating point value (the best value is 1.0).
        """
        return metrics.accuracy_score(y_true, y_score)


class Precision(Metric):
    """Precision_score.

    Args:
        mode(str): Supported metric modes, only anomaly is valid value.

    Attributes:
        _NAME(str): Metric name.
    """
    _NAME = "precision"

    def __init__(
        self, 
        mode: str = "anomaly"
    ):
        super(Precision, self).__init__(mode)
   
    @ensure_2d
    def metric_fn(
        self, 
        y_true: np.ndarray, 
        y_score: np.ndarray,
        **kwargs
    ) -> float:
        """Precision_score.

        Args:
            y_true(np.ndarray): Ground truth (correct) target values.
            y_score(np.ndarray): Estimated target values.

        Returns:
            float: precision_score. A non-negative floating point value (the best value is 1.0).
        """
        return metrics.precision_score(y_true, y_score, **kwargs)

    
class Recall(Metric):
    """Recall_score.

    Args:
        mode(str): Supported metric modes, only anomaly is valid value.

    Attributes:
        _NAME(str): Metric name.
    """
    _NAME = "recall"

    def __init__(
        self, 
        mode: str = "anomaly"
    ):
        super(Recall, self).__init__(mode)
   
    @ensure_2d
    def metric_fn(
        self, 
        y_true: np.ndarray, 
        y_score: np.ndarray,
        **kwargs
    ) -> float:
        """Recall_score.

        Args:
            y_true(np.ndarray): Ground truth (correct) target values.
            y_score(np.ndarray): Estimated target values.

        Returns:
            float: recall_score. A non-negative floating point value (the best value is 1.0).
        """
        return metrics.recall_score(y_true, y_score, **kwargs)

    
class F1(Metric):
    """F1_score.

    Args:
        mode(str): Supported metric modes, only anomaly is valid value.

    Attributes:
        _NAME(str): Metric name.
    """
    _NAME = "f1"

    def __init__(
        self, 
        mode: str = "anomaly"
    ):
        super(F1, self).__init__(mode)
   
    @ensure_2d
    def metric_fn(
        self, 
        y_true: np.ndarray, 
        y_score: np.ndarray,
        **kwargs
    ) -> float:
        """F1_score.

        Args:
            y_true(np.ndarray): Ground truth (correct) target values.
            y_score(np.ndarray): Estimated target values.

        Returns:
            float: f1_score. A non-negative floating point value (the best value is 1.0).
        """
        return metrics.f1_score(y_true, y_score, **kwargs)
    

class MetricContainer(object):
    """Container holding a list of metrics.

    Args:
        metrics(List[str]|List[Metric]): List of metric or metric names.
        prefix(str): Prefix of metric names.

    Attributes:
        _prefix(str): Prefix of metric names.
        _metrics(List[Metric]): List of metric instance.
        _names(List[str]): List of metric names associated with eval_name.
    """
    def __init__(
        self,
        metrics: Union[List[str],List[Metric]],
        prefix: str = ""
    ):
        self._prefix = prefix
        self._metrics = (
            metrics if (metrics and isinstance(metrics[-1], Metric))
            else Metric.get_metrics_by_names(metrics)
        )
        self._names = [prefix + metric._NAME for metric in self._metrics]
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

