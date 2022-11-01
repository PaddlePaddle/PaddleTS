# !/usr/bin/env python3
# -*- coding:utf-8 -*-

import abc
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from paddlets.datasets.tsdataset import TSDataset, TimeSeries
from paddlets.logger import raise_if_not, Logger
from paddlets.ensemble.ensemble_forecaster_base import EnsembleForecasterBase

logger = Logger(__name__)
SUPPORT_MODES = ["mean", "min", "max", "median"]


class WeightingEnsembleForecaster(EnsembleForecasterBase, metaclass=abc.ABCMeta):
    """
    The WeightingEnsembleBase Class.

    Args:

        in_chunk_len(int): The size of the loopback window, i.e., the number of time steps feed to the model.
        out_chunk_len(int): The size of the forecasting horizon, i.e., the number of time steps output by the model.
        skip_chunk_len(int): Optional, the number of time steps between in_chunk and out_chunk for a single sample.
            The skip chunk is neither used as a feature (i.e. X) nor a label (i.e. Y) for a single sample. By
            default, it will NOT skip any time steps.
        estimators(List[Tuple[object, dict]] ): A list of tuple (class,params) consisting of several paddlets models 

        model: weighting mode, support ["mean","min","max","median"] for now, set to "mean" by default.
        verbose(bool): Turn on Verbose mode,set to true by default.

    """
    def __init__(self,
                 in_chunk_len: int,
                 out_chunk_len: int,
                 skip_chunk_len: int,
                 estimators: List[Tuple[object, dict]],
                 mode="mean",
                 verbose: bool = False
                 ) -> None:

        super().__init__(in_chunk_len, out_chunk_len, skip_chunk_len, estimators, verbose)
        raise_if_not(isinstance(mode, str), "Mode should in type of string")
        raise_if_not(mode in SUPPORT_MODES,
                     "Unsupported ensemble mode, please use WeightingEnsemble.get_support_modes() to check supported modes")
        self._mode = mode

    def _fit(self,
            train_tsdataset: TSDataset,
            valid_tsdataset: Optional[TSDataset] = None) -> None:
        """
        fit 

        Args:
            train_tsdataset(TSDataset): Train dataset.
            valid_tsdataset(TSDataset, optional): Valid dataset.
        """
        self._fit_estimators(train_tsdataset, valid_tsdataset)

    def predict(self, tsdataset: TSDataset) -> TSDataset:
        """
        predict 

        Args:
            tsdataset(TSDataset): Dataset to predict.

        """
        predictions = self._predict_estimators(tsdataset)

        target_names = predictions[0].target.data.columns.values.tolist()
        target_df = pd.DataFrame(columns=target_names)

        for name in target_names:
            meta = np.concatenate(
                [np.array(prediction[name]).reshape(-1, 1) for prediction in predictions], axis=1)

            y = self._weighting(meta)
            target_df[name] = y

        target_df.index = predictions[0].target.data.index
        return TSDataset(target=TimeSeries(target_df, predictions[0].target.freq))

    def _weighting(self, meta) -> TSDataset:
        """
        Get weighting results

        Args:
            meta(np.array): meta data to weight.  
        """

        if self._mode == "mean":
            y = np.mean(meta, axis=1)
        elif self._mode == "min":
            y = np.min(meta, axis=1)
        elif self._mode == "max":
            y = np.max(meta, axis=1)
        elif self._mode == "median":
            y = np.median(meta, axis=1)

        return y

    @classmethod
    def get_support_modes(cls) -> None:
        """
        Get support modes
        """
        logger.info(f"Supported ensemble modes:{SUPPORT_MODES}")
