#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from typing import Any, List, Tuple, Dict
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from paddlets import TimeSeries, TSDataset
from paddlets.logger import Logger, raise_if_not, raise_if, raise_log

logger = Logger(__name__)


class Metric(ABC):
    """Abstract base class used to build new Metric.
    
    Args:
        mode(str): Supported metric modes, only normal and prob are valid values. 
            Set to normal for non-probability use cases, set to prob for probability use cases.
        kwargs: Keyword parameters of specific metric functions.
    """
    def __init__(self, mode: str="normal", **kwargs):
        self._kwargs = kwargs
        raise_if_not(mode in {"normal", "prob"}, 
                     f"Metric mode should be one of {{`normal`, `prob`}}, got `{mode}`.")
        self._mode = mode

    def _build_metrics_data(
        self,
        tsdataset_true: "TSDataset",
        tsdataset_pred: "TSDataset",
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Convert TSDataset of normal mode to ndarray. 
        
        Args:
            tsdataset_true(TSDataset): TSDataset containing Ground truth (correct) target values.
            tsdataset_pred(TSDataset): TSDataset containing Estimated target values.

        Returns:
            Dict[str, Tuple[np.ndarray, np.ndarray]]: Dict of tuple, 
                key is the name of target, and value is tuple type (y_true, y_score).

        Raises:
            ValueError.
        """
        target_true = tsdataset_true.get_target().sort_columns()
        target_pred = tsdataset_pred.get_target().sort_columns()
        raise_if(
            target_true is None or target_pred is None,
            "TSDataset target is None!"
        )
        raise_if_not(
            len(target_true.columns) == len(target_pred.columns),
            "In `normal` mode, only point forecasting data is supported!"
        )
        raise_if_not(
            (target_true.columns == target_pred.columns).all(),
            "tsdataset true's and pred's columns are not the same!"
        )
        target_pred = TimeSeries(
            target_pred.data.reindex(target_true.time_index), 
            target_true.freq
        )
        for column in target_pred.columns:
            raise_if(
                target_pred.data[column].isna().all(),
                "tsdataset true's and pred's time_index do not match!"
            )
        res = {}
        for target in target_true.columns:
            res[target] = (target_true.data[target].to_numpy(), target_pred.data[target].to_numpy())
        return res
    
    def _build_prob_metrics_data(
        self,
        tsdataset_true: "TSDataset",
        tsdataset_pred: "TSDataset",
        data_type: str,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Convert TSDataset of prob mode to ndarray.
        
        Args:
            tsdataset_true(TSDataset): TSDataset containing ground truth (correct) target values.
            tsdataset_pred(TSDataset): TSDataset containing estimated target values.

        Returns:
            Dict[str, Tuple[np.ndarray, np.ndarray]]: Dict of tuple, 
                key is the name of target, and value is tuple type.

        Raises:
            ValueError.
        """
        target_true = tsdataset_true.get_target().sort_columns()
        target_pred = tsdataset_pred.get_target().sort_columns()
        # check validation
        raise_if(
            target_true is None or target_pred is None,
            "TSDataset target is None!")
        # check columns
        target_set = set(target_true.columns)
        pred_target_set = set([col.rsplit("@", 1)[0] for col in target_pred.columns])
        raise_if_not(target_set == pred_target_set, 
                     "Prediction is not coherent with ground truth.")
        target_pred.reindex(target_true.time_index)
        for column in target_pred.columns:
            raise_if(
                target_pred.data[column].isna().all(),
                "tsdataset true's and pred's time_index do not match!")

        target_true = target_true.to_dataframe()
        target_pred = target_pred.to_dataframe()
        res = {}
        target_pred_names = target_pred.columns

        for target_name in target_true.columns:
            cur_pred_target_names = [x for x in target_pred_names if x.rsplit("@", 1)[0] == target_name]
            target_pred_cur = target_pred[cur_pred_target_names]            
            if data_type == "quantile":
                res[target_name] = (target_true[target_name].to_numpy(), target_pred_cur.to_numpy())
            else: # data_type: "point"
                target_pred_cur_median = np.median(target_pred_cur.to_numpy(), axis = -1)
                res[target_name] = (target_true[target_name].to_numpy(), target_pred_cur_median)
        return res

    @abstractmethod
    def metric_fn(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
    ) -> float:
        """
        Compute metric's value from ndarray.

        Args:
            y_true(np.ndarray): Ground truth (correct) target values.
            y_pred(np,ndarray): Estimated target values.

        Returns:
            float: Computed metric value.

        Raises:
            ValueError.
        """
        pass
    
    def __call__(
        self,
        tsdataset_true: "TSDataset",
        tsdataset_pred: "TSDataset",
    )-> Dict[str, float]:
        """
        Compute metric's value from TSDataset.
        
        Args:
            tsdataset_true(TSDataset): TSDataset containing ground truth (correct) target values.
            tsdataset_pred(TSDataset): TSDataset containing estimated target values.

        Returns:
            Dict[str, float]: Dict of metrics. key is the name of target, and value is specific metric value. 

        Raises:
            ValueError.
        """
        if self._mode == "normal":
            res_array = self._build_metrics_data(tsdataset_true, tsdataset_pred)
        else: # "prob"
             res_array = self._build_prob_metrics_data(tsdataset_true, tsdataset_pred, self._TYPE)
        res = {}
        for target, value in res_array.items():
            res[target] = self.metric_fn(value[0], value[1])
        return res

    @classmethod
    def get_metrics_by_names(cls, names: List[str]) -> List["Metric"]:
        """Get list of metric classes.

        Args:
            names(List[str]): List of metric names.

        Returns:
            List[Metric]: List of metric classes.
        """
        available_metrics = cls.__subclasses__()
        available_names = [metric._NAME for metric in available_metrics]
        metrics = []
        for name in names:
            assert (name in available_names
            ), f"{name} is not available, choose in {available_names}"
            idx = available_names.index(name)
            metric = available_metrics[idx]()
            metrics.append(metric)
        return metrics

