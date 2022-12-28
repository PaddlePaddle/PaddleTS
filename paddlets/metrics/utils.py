#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from typing import Callable
import functools

import numpy as np

from paddlets.metrics.base import Metric


def ensure_2d(func) -> Callable[..., float]:
    """A decorator, used for ensuring that the parameter of the wrapped function 
    is a 2-dimentional tensor so that it fits sklearn.metrics.

    Args:
        func(Callable[..., float]): Core function.

    Returns:
        Callable[..., float]: Wrapped core function.
    """
    @functools.wraps(func)
    def wrapper(
        obj: Metric,
        y_true: np.ndarray, 
        y_score: np.ndarray,
        **kwargs
    ) -> float:
        """Core processing logic.

        Args:
            obj(Metric): Metirc instance.
            y_true(np.ndarray): Ground truth (correct) labels.
            y_score(np.ndarray): Estimated target values.

        Returns:
            float: metric.
        """
        batch_nd_true, batch_nd_score = y_true.shape[0], y_score.shape[0]
        y_true = np.reshape(y_true, (batch_nd_true, -1))
        y_score = np.reshape(y_score, (batch_nd_score, -1))
        return func(obj, y_true, y_score, **kwargs)
    return wrapper

