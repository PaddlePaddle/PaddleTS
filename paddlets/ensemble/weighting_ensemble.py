# !/usr/bin/env python3
# -*- coding:utf-8 -*-

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pyod.utils.utility import standardizer

from paddlets.ensemble.base import EnsembleBase
from paddlets.models import BaseModel
from paddlets.datasets.tsdataset import TSDataset
from paddlets.logger import raise_if_not, Logger, raise_if
from paddlets.pipeline import Pipeline
from paddlets.models.utils import to_tsdataset
from paddlets.models.anomaly.dl.anomaly_base import AnomalyBaseModel
from paddlets.models.ml_model_wrapper import PyodModelWrapper

logger = Logger(__name__)
FORECASTER_SUPPORT_MODES = ["mean", "min", "max", "median"]
ANOMALY_SUPPORT_MODES = ["mean", "min", "max", "median", "voting"]

class WeightingEnsembleBase(EnsembleBase):
    """
    The WeightingEnsembleBase Class.

    Args:

        estimators(List[Tuple[object, dict]] ): A list of tuple (class,params) consisting of several paddlets models 
        model: weighting mode, support ["mean","min","max","median"] for now, set to "mean" by default.
        verbose(bool): Turn on Verbose mode,set to true by default.

    """
    def __init__(self,
                 estimators: List[Tuple[object, dict]],
                 mode="mean",
                 verbose: bool = False
                 ) -> None:

        raise_if_not(isinstance(mode, str), "Mode should in type of string")
        self._mode = mode
        super().__init__(estimators, verbose)

    def fit(self,
            train_tsdataset: TSDataset,
            valid_tsdataset: Optional[TSDataset] = None) -> None:
        """
        Fit 

        Args:
            train_tsdataset(TSDataset): Train dataset.
            valid_tsdataset(TSDataset, optional): Valid dataset.
        """
        self._fit_estimators(train_tsdataset, valid_tsdataset)
        self._fitted = True

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

        return target_df.to_numpy()

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



class WeightingEnsembleForecaster(WeightingEnsembleBase, BaseModel):
    """
    The WeightingEnsembleForecaster Class.

    Args:

        in_chunk_len(int): The size of the loopback window, i.e., the number of time steps feed to the model.
        out_chunk_len(int): The size of the forecasting horizon, i.e., the number of time steps output by the model.
        skip_chunk_len(int): Optional, the number of time steps between in_chunk and out_chunk for a single sample.
            The skip chunk is neither used as a feature (i.e. X) nor a label (i.e. Y) for a single sample. Bydefault, it will NOT skip any time steps.
        estimators(List[Tuple[object, dict]] ): A list of tuple (class,params) consisting of several paddlets models 
        mode: weighting mode, support ["mean","min","max","median"] for now, set to "mean" by default.
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

        raise_if_not(mode in FORECASTER_SUPPORT_MODES,
                     f"Unsupported ensemble mode,supported ensemble modes: {FORECASTER_SUPPORT_MODES}")
        BaseModel.__init__(self, in_chunk_len, out_chunk_len, skip_chunk_len)
        WeightingEnsembleBase.__init__(self, estimators, mode, verbose)

    def _set_params(self, estimators) -> List:
        """
        Set estimators params

        Set params and initial estimators.

        Args:

            estimators(List[Tuple[object, dict]] ): A list of tuple (class,params) consisting of several paddlets models.

        """
        for index in range(len(estimators)):
            e = estimators[index]
            if issubclass(e[0], BaseModel):
                e[-1]["in_chunk_len"] = self._in_chunk_len
                e[-1]["out_chunk_len"] = self._out_chunk_len
                e[-1]["skip_chunk_len"] = self._skip_chunk_len
            elif issubclass(e[0], Pipeline):
                e[1]["steps"][-1][1]["in_chunk_len"] = self._in_chunk_len
                e[1]["steps"][-1][1]["out_chunk_len"] = self._out_chunk_len
                e[1]["steps"][-1][1]["skip_chunk_len"] = self._skip_chunk_len
        return super()._set_params(estimators)

    def _check_estimators(self, estimators) -> None:
        """
        Check estimators

        Check and valid estimators

        Args:

            estimators(List[Tuple[object, dict]] ): A list of tuple (class,params) consisting of several paddlets models.

        """
        super()._check_estimators(estimators)
        if all([issubclass(e[0], BaseModel) or issubclass(e[0], Pipeline) for e in estimators]):
            pass
        
        else:
            raise ValueError("Estimators have unsupported or uncompatible models")

    def fit(self,
            train_tsdataset: TSDataset,
            valid_tsdataset: Optional[TSDataset] = None) -> None:
        """
        fit 

        Args:
            train_tsdataset(TSDataset): Train dataset.
            valid_tsdataset(TSDataset, optional): Valid dataset.
        """
        super().fit(train_tsdataset, valid_tsdataset)
        
    @to_tsdataset(scenario="forecasting")
    def predict(self, tsdataset: TSDataset) -> TSDataset:
        """
        predict 

        Args:
            tsdataset(TSDataset): Dataset to predict.

        """
        return super().predict(tsdataset)

    def save(self, path: str, ensemble_file_name: str = "paddlets-weighting-ensemble-forecaster-partial.pkl") -> None:
        """
        Save the ensemble model to a directory.

        Args:
            path(str): Output directory path.
            ensemble_file_name(str): Name of ensemble object. This file contains meta information of ensemble model.
        """
        return super().save(path, ensemble_file_name)

    @staticmethod
    def load(path: str,
             ensemble_file_name: str = "paddlets-weighting-ensemble-forecaster-partial.pkl") -> "WeightingEnsembleForecaster":
        """
        Load the ensemble model from a directory.

        Args:
            path(str): Input directory path.
            ensemble_file_name(str): Name of ensemble object. This file contains meta information of ensemble.

        Returns:
            The loaded ensemble model.
        """
        return EnsembleBase.load(path, ensemble_file_name)

class WeightingEnsembleAnomaly(WeightingEnsembleBase):
    """
    The WeightingEnsembleAnomaly Class.

    Args:

        in_chunk_len(int): The size of the loopback window, i.e., the number of time steps feed to the model.
        estimators(List[Tuple[object, dict]] ): A list of tuple (class,params) consisting of several paddlets Anomly models or Pyod models
        model: weighting mode, support ["mean","min","max","median"] for now, set to "mean" by default.
        contamination(int):Anomaly rate, should in [0,0.5). 
                            For example, when anomaly rate=0.1, the top 10% values in trian scores will set to threshold.
                            Set to 0 by default, use the max score on train as threshold.
        standardization : bool, optional (default=True)
            If True, perform standardization first to convert
            prediction score to zero mean and unit variance.
        verbose(bool): Turn on Verbose mode,set to true by default.

    """
    def __init__(self,
                 in_chunk_len,
                 estimators: List[Tuple[object, dict]],
                 mode="mean",
                 contamination: int = 0,
                 standardization: bool = True, 
                 verbose: bool = False
                 ) -> None:
        raise_if_not(contamination < 0.5 and contamination >= 0, "anomly_rate should in[0,0.5)")
        self._contamination = contamination

        raise_if_not(isinstance(in_chunk_len, int) and in_chunk_len > 0,"in_chunk_len should be a positive integer")
        self._in_chunk_len = in_chunk_len

        raise_if_not(mode in ANOMALY_SUPPORT_MODES,
                     f"Unsupported ensemble mode,supported ensemble modes: {ANOMALY_SUPPORT_MODES}")

        self._standardization = standardization
        super().__init__(estimators, mode, verbose)

    def _check_estimators(self, estimators) -> None:
        """
        Check estimators

        Check and valid estimators

        Args:

            estimators(List[Tuple[object, dict]] ): A list of tuple (class,params) consisting of several paddlets models.

        """
        super()._check_estimators(estimators)
        if all([issubclass(e[0], PyodModelWrapper) or issubclass(e[0], AnomalyBaseModel or issubclass(e[0], Pipeline)) for e in estimators]):
            pass
        else:
            raise ValueError("Estimators have unsupported or uncompatible models")

    def _set_params(self, estimators) -> List:
        """
        Set estimators params

        Set params and initial estimators.

        Args:

            estimators(List[Tuple[object, dict]] ): A list of tuple (class,params) consisting of several paddlets models.

        """
        for index in range(len(estimators)):
            e = estimators[index]
            if issubclass(e[0], PyodModelWrapper) or issubclass(e[0], AnomalyBaseModel):
                e[-1]["in_chunk_len"] = self._in_chunk_len
            elif issubclass(e[0], Pipeline) and (issubclass(e[0]._steps[-1][0], PyodModelWrapper) or issubclass(e[0]._steps[-1][0], AnomalyBaseModel)) :
                e[1]["steps"][-1][1]["in_chunk_len"] = self._in_chunk_len
        return super()._set_params(estimators)

    def fit(self,
            train_tsdataset: TSDataset,
            valid_tsdataset: Optional[TSDataset] = None) -> None:
        """
        Fit 

        Args:
            train_tsdataset(TSDataset): Train dataset.
            valid_tsdataset(TSDataset, optional): Valid dataset.
        """
        super().fit(train_tsdataset, valid_tsdataset)

        #compute threshold
        if self._mode != "voting":
            scores = self.predict_score(train_tsdataset).target.to_dataframe().to_numpy()
            max_score_on_train = np.max(scores)
            if self._contamination == 0:
                self._threshold = max_score_on_train
            else:
                self._threshold = np.percentile(scores,
                                                100 * (1 - self._contamination))

    @to_tsdataset(scenario="anomaly_label")
    def predict(self, tsdataset: TSDataset) -> TSDataset:
        """
        Predict 

        Args:
            tsdataset(TSDataset): Dataset to predict.

        """
        anomaly_label = []
        if self._mode == "voting":
            predictions = self._predict_estimators(tsdataset)

            l = []
            for predict in predictions:
                l.append(predict.target.data.to_numpy())
            scores = np.stack(l, axis=1)

            for i in range(len(scores)):
                anomaly_label.append(np.argmax(np.bincount(scores[i].ravel())))
        else:
            anomaly_score = self.predict_score(tsdataset)

            for score in anomaly_score.target.to_numpy():
                label = 0 if score < self._threshold else 1
                anomaly_label.append(label)
        return np.array(anomaly_label)

    @to_tsdataset(scenario="anomaly_score")
    def predict_score(
            self,
            tsdataset: TSDataset
    ) -> TSDataset:
        """
        Get anomaly score on a batch. the result are output as tsdataset.

        Args:
            tsdataset(TSDataset): Data to be predicted.

        Returns:
            TSDataset.
        """
        raise_if(self._mode == "voting", 
                    "predict_score() not work when mode == voting")

        predictions = []
        for estimator in self._estimators:
            predictions.append(estimator.predict_score(tsdataset))

        target_names = predictions[0].target.data.columns.values.tolist()
        target_df = pd.DataFrame(columns=target_names)

        for name in target_names:
            meta = np.concatenate(
                [np.array(prediction[name]).reshape(-1, 1) for prediction in predictions], axis=1)

            if self._standardization:
                meta = standardizer(meta)
            y = self._weighting(meta)
            target_df[name] = y

        return target_df.to_numpy()

    def save(self, path: str, ensemble_file_name: str = "paddlets-weighting-ensemble-anomly-partial.pkl") -> None:
        """
        Save the ensemble model to a directory.

        Args:
            path(str): Output directory path.
            ensemble_file_name(str): Name of ensemble object. This file contains meta information of ensemble model.
        """
        return super().save(path, ensemble_file_name)

    @staticmethod
    def load(path: str,
             ensemble_file_name: str = "paddlets-weighting-ensemble-anomly-partial.pkl") -> "WeightingEnsembleAnomaly":
        """
        Load the ensemble model from a directory.

        Args:
            path(str): Input directory path.
            ensemble_file_name(str): Name of ensemble object. This file contains meta information of ensemble.

        Returns:
            The loaded ensemble model.
        """
        return EnsembleBase.load(path, ensemble_file_name)
