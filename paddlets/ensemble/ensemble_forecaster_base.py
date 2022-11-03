# !/usr/bin/env python3
# -*- coding:utf-8 -*-

import abc
import os
import pickle

from typing import List, Optional, Tuple

from paddlets.models.base import BaseModel
from paddlets.datasets.tsdataset import TSDataset
from paddlets.logger import raise_log
from paddlets.models.model_loader import load as paddlets_model_load
from paddlets.pipeline.pipeline import Pipeline


class EnsembleForecasterBase(BaseModel, metaclass=abc.ABCMeta):
    """
    The EnsembleBase Class.

    Args:

        in_chunk_len(int): The size of the loopback window, i.e., the number of time steps feed to the model.
        out_chunk_len(int): The size of the forecasting horizon, i.e., the number of time steps output by the model.
        skip_chunk_len(int): Optional, the number of time steps between in_chunk and out_chunk for a single sample.
            The skip chunk is neither used as a feature (i.e. X) nor a label (i.e. Y) for a single sample. By
            default, it will NOT skip any time steps.
        estimators(List[Tuple[object, dict]] ): A list of tuple (class,params) consisting of several paddlets forecasters.
        verbose(bool): Turn on Verbose mode,set to true by default.

    """
    def __init__(self,
                 in_chunk_len: int,
                 out_chunk_len: int,
                 skip_chunk_len: int = 0,
                 estimators: List[Tuple[object, dict]] = None,
                 verbose=False
                 ) -> None:

        def _check_estimators():
            # when estimator is type of int, skip check, use for model save and load
            if isinstance(estimators, int):
                return
            if (
                    estimators is None
                    or len(estimators) == 0
                    or not isinstance(estimators, list)
                    or not all([len(estimator) == 2 for estimator in estimators])
            ):
                raise ValueError(
                    "Invalid 'estimators' attribute, 'estimators' should be a list"
                    " of (class,params) tuples."
                )

            from paddlets.models.representation.dl.repr_base import ReprBaseModel
            if all([issubclass(e[0], BaseModel) or issubclass(e[0], Pipeline) for e in estimators]):
                pass
            elif all([issubclass(e[0], ReprBaseModel) for e in estimators]):
                pass
            else:
                raise ValueError("Estimators have unsupported or uncompatible models")

        _check_estimators()
        self._in_chunk_len = in_chunk_len
        self._out_chunk_len = out_chunk_len
        self._skip_chunk_len = skip_chunk_len
        self._verbose = verbose
        self.fitted = False

        self._estimators = []
        for index in range(len(estimators)):
            e = estimators[index]
            model_params = e[-1]
            if issubclass(e[0], BaseModel):
                model_params["in_chunk_len"] = self._in_chunk_len
                model_params["out_chunk_len"] = self._out_chunk_len
                model_params["skip_chunk_len"] = self._skip_chunk_len
            elif issubclass(e[0], Pipeline):
                e[1]["steps"][-1][1]["in_chunk_len"] = self._in_chunk_len
                e[1]["steps"][-1][1]["out_chunk_len"] = self._out_chunk_len
                e[1]["steps"][-1][1]["skip_chunk_len"] = self._skip_chunk_len
            try:
                estimator = e[0](**model_params)
            except Exception as e:
                raise_log(ValueError("init error: %s" % (str(e))))
            self._estimators.append(estimator)

    def fit(self,
            train_tsdataset: TSDataset,
            valid_tsdataset: Optional[TSDataset] = None) -> None:
        """
        Fit 

        Args:
            train_tsdataset(TSDataset): Train dataset.
            valid_tsdataset(TSDataset, optional): Valid dataset.
        """
        if valid_tsdataset:
            self._fit(train_tsdataset, valid_tsdataset)
        else:
            self._fit(train_tsdataset)
        self.fitted = True

    @abc.abstractmethod
    def _fit(self,
            train_tsdataset: TSDataset,
            valid_tsdataset: Optional[TSDataset] = None) -> None:
        """
        Fit 

        Args:
            train_tsdataset(TSDataset): Train dataset.
            valid_tsdataset(TSDataset, optional): Valid dataset.
        """
        pass

    def _fit_estimators(self,
                        train_tsdataset: TSDataset,
                        valid_tsdataset: Optional[TSDataset] = None) -> None:
        """
        Fit estimators

        Args:
            train_tsdataset(TSDataset): Train dataset.
            valid_tsdataset(TSDataset, optional): Valid dataset.
        """
        for estimator in self._estimators:
            estimator.fit(train_tsdataset, valid_tsdataset)

    @abc.abstractmethod
    def predict(self, tsdataset: TSDataset) -> None:
        """
        Predict 

        Args:
            tsdataset(TSDataset): Dataset to predict.

        """
        pass
    
    def _predict_estimators(self,
                            tsdataset: TSDataset) -> List[TSDataset]:
        """
        Predict estimators

        Args:
            tsdataset(TSDataset): Dataset to predict.

        """
        predictions = []
        for estimator in self._estimators:
            predictions.append(estimator.predict(tsdataset))
        return predictions

    def save(self, path: str, ensemble_file_name: str = "paddlets-ensemble-partial.pkl") -> None:
        """
        Save the ensemble model to a directory.

        Args:
            path(str): Output directory path.
            ensemble_file_name(str): Name of ensemble object. This file contains meta information of ensemble model.
        """
        if not os.path.exists(path):
            # Check path
            os.makedirs(path)
        elif not os.path.isdir(path):
            raise_log(ValueError(f"path is not a directory, path : {path}"))
        # Check file not exist
        ensemble_file_path = os.path.join(path, ensemble_file_name)
        if os.path.exists(ensemble_file_path):
            raise_log(FileExistsError(f"paddlets-ensemble-partial file already exist, path : {ensemble_file_path}"))
        # 1.Save model
        for i in range(len(self._estimators)):
            model = self._estimators[i]
            model.save(os.path.join(path, "paddlets-ensemble-model" + str(i)))
        # 2.Save ensemble(without final model)
        model_tmp = self._estimators
        self._estimators = len(self._estimators)
        try:
            with open(ensemble_file_path, "wb") as f:
                pickle.dump(self, f)
        except Exception as e:
            raise_log(ValueError("error occurred while saving ensemble, file path: %s, err: %s" \
                                 % (ensemble_file_path, str(e))))
        # Reset model
        self._estimators = model_tmp

    @staticmethod
    def load(path: str, ensemble_file_name: str = "paddlets-ensemble-partial.pkl") -> "EnsembleForecasterBase":
        """
        Load the ensemble model from a directory.

        Args:
            path(str): Input directory path.
            ensemble_file_name(str): Name of loaded ensemble object. This file contains meta information of ensemble model.

        Returns:
            The loaded ensemble model.
        """
        if not os.path.exists(path):
            raise_log(FileNotFoundError(f"path not exist, path : {path}"))
        if not os.path.isdir(path):
            raise_log(ValueError(f"path is not a directory, path : {path}"))
        # 1.Load ensemble
        # Check file exist
        ensemble_file_path = os.path.join(path, ensemble_file_name)
        if not os.path.exists(ensemble_file_path):
            raise_log(FileExistsError(f"paddlets-ensemble-partial file not exist, path : {ensemble_file_path}"))
        try:
            with open(ensemble_file_path, "rb") as f:
                ensemble = pickle.load(f)
        except Exception as e:
            raise_log(RuntimeError(
                "error occurred while loading ensemble, path: %s, error: %s" % (ensemble_file_path, str(e))))
        # 2.Load model
        model_number = ensemble._estimators
        estimators = []
        for i in range(model_number):
            model = paddlets_model_load(os.path.join(path, "paddlets-ensemble-model" + str(i)))
            estimators.append(model)
        # Add model to ensemble
        ensemble._estimators = estimators
        return ensemble
