#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from typing import Any, List, Tuple, Dict
from abc import ABC, abstractmethod
import operator

import numpy as np
import pandas as pd
from paddlets import TimeSeries, TSDataset
from paddlets.logger import Logger, raise_if_not, raise_if, raise_log

logger = Logger(__name__)


class Metric(ABC):
    """Abstract base class used to build new Metric.
    
    Args:
        mode(str): Supported metric modes, only normal, prob and anomaly are valid values. 
            Set to normal for non-probability use cases, set to prob for probability use cases, set to anomaly for anomaly dection.
        kwargs: Keyword parameters of specific metric functions.
    """
    def __init__(self, mode: str="normal", **kwargs):
        self._kwargs = kwargs
        raise_if_not(mode in {"normal", "prob", "anomaly"}, 
                     f"Metric mode should be one of {{`normal`, `prob`, `anomaly`}}, got `{mode}`.")
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
        # Reindex the true and pred data based on the time_index intersection.
        target_true, target_pred = self._reindex_data(target_true, target_pred)
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
         # Reindex the true and pred data based on the time_index intersection.
        target_true, target_pred = self._reindex_data(target_true, target_pred)
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
    
    def _build_anomaly_metrics_data(
        self,
        tsdataset_true: "TSDataset",
        tsdataset_pred: "TSDataset",
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Convert TSDataset of anomaly mode to ndarray. 
        
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
            "TSDataset label is None!"
        )
        raise_if(
            len(target_true) == 0 or len(target_pred) == 0,
            "In `anomaly` mode, the length of the true and pred can't be 0!"
        )
        raise_if_not(
            len(target_true) >= len(target_pred),
            "In `anomaly` mode, the length of the true must be greater than or equal to the length of the pred!"
        )
        raise_if_not(
            (target_true.columns == target_pred.columns).all(),
            "tsdataset true's and pred's columns are not the same!"
        )
        # ensure the values in true and pred are 0 or 1
        for column in target_true.columns:
            true_value = list(set(target_true.data[column].to_numpy()))
            pred_value = list(set(target_pred.data[column].to_numpy()))
            raise_if_not(
                operator.eq(true_value, [0, 1]) or operator.eq(true_value, [0]) or operator.eq(true_value, [1]),
                'In `anomaly` mode, the value in true label must be 0 or 1, please check your data!'
            )
            raise_if_not(
                operator.eq(pred_value, [0, 1]) or operator.eq(pred_value, [0]) or operator.eq(pred_value, [1]),
                'In `anomaly` mode, the value in pred label must be 0 or 1, please check your data!'
            )   
        # Reindex the true and pred data based on the time_index intersection.
        target_true, target_pred = self._reindex_data(target_true, target_pred)
        res = {}
        for target in target_true.columns:
            res[target] = (target_true.data[target].to_numpy(), target_pred.data[target].to_numpy())
        return res
    
    def _reindex_data(
        self,
        target_true: TimeSeries,
        target_pred: TimeSeries
    )-> List[TimeSeries]:
        """
        Reindex the true and pred data based on the time_index intersection.
        
        Args:
            target_true(TimeSeries): TimeSeries containing Ground truth (correct) target values.
            target_pred(TimeSeries): TimeSeries containing Estimated target values.

        Returns:
            List[TimeSeries]: The target_true and target_pred after reindex.

        Raises:
            ValueError.
        """
        raise_if(
            type(target_true.time_index) != type(target_pred.time_index),
            "The time_index type of true and pred are inconsistent, please check you data!"
        )
        merge_index = pd.merge(target_true.time_index.to_frame(index=False), target_pred.time_index.to_frame(index=False))
        raise_if(
            len(merge_index) == 0,
            "The time_index intersection of true and pred is empty, please check you data!"
        )
        if len(merge_index) != len(target_true):
            logger.warning("Tsdataset true's and pred's time_index do not match, the result will be calculated according to the intersection!")
        index_name = merge_index.columns[0]
        if isinstance(target_true.time_index, pd.RangeIndex):
            merge_index = pd.RangeIndex(merge_index[index_name].iloc[0], merge_index[index_name].iloc[-1], target_true.freq)
        else:
            merge_index = pd.DatetimeIndex(merge_index[index_name], freq=target_true.freq)
            
        target_true = TimeSeries(
            target_true.data.reindex(merge_index),
            target_true.freq
        )
        target_pred = TimeSeries(
            target_pred.data.reindex(merge_index),
            target_pred.freq
        )
        return [target_true, target_pred]

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
        elif self._mode == "prob": # "prob"
            res_array = self._build_prob_metrics_data(tsdataset_true, tsdataset_pred, self._TYPE)
        elif self._mode == "anomaly": # "anomaly"
            res_array = self._build_anomaly_metrics_data(tsdataset_true, tsdataset_pred)
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

