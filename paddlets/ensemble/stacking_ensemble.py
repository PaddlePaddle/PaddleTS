# !/usr/bin/env python3
# -*- coding:utf-8 -*-
import abc
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.base import is_regressor, clone

from paddlets.models import BaseModel
from paddlets.pipeline import Pipeline
from paddlets.datasets.tsdataset import TSDataset
from paddlets.datasets.splitter import ExpandingWindowSplitter, HoldoutSplitter
from paddlets.utils.validation import cross_validate, fit_and_score
from paddlets.logger import raise_if
from paddlets.ensemble.base import EnsembleBase
from paddlets.models.utils import to_tsdataset

class StackingEnsembleBase(EnsembleBase):
    """
    The StackingEnsembleBase Class.

    Args:

        estimators(List[Tuple[object, dict]] ): A list of tuple (class,params) consisting of several paddlets models 
        final_learner(Callable): The final learner on stack level 2, should be a sklearn-like regressor, set to GradientBoostingRegressor(max_depth=5) by default.
        verbose(bool): Turn on Verbose mode,set to true by default.

    """
    def __init__(self,
                 estimators: List[Tuple[object, dict]],
                 final_learner: Callable = None,
                 verbose: bool = False
                 ) -> None:

        super().__init__(estimators, verbose)
        self._final_learner = self._check_final_learner(final_learner)


    def fit(self,
            train_tsdataset: TSDataset,
            valid_tsdataset: Optional[TSDataset] = None) -> None:
        """
        fit 

        Args:
            train_tsdataset(TSDataset): Train dataset.
            valid_tsdataset(TSDataset, optional): Valid dataset.
        """
        X_meta, y_meta = self._generate_fit_meta_data(train_tsdataset, valid_tsdataset)
        self._final_learner.fit(X_meta, y_meta)
        self._fitted = True

    @abc.abstractmethod
    def _generate_fit_meta_data(self, train_tsdataset: TSDataset,
                            valid_tsdataset: Optional[TSDataset] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        generate fit meta data
        generate fit meta data needed for level 2

        Args:
            train_tsdataset(TSDataset): Train dataset.
            valid_tsdataset(TSDataset, optional): Valid dataset.
        """

        pass

    @abc.abstractmethod
    def _check_final_learner(self, final_learner) -> None:
        """Check if a final learner is given and if it is valid, otherwise set default regressor.

        Args:
            final_learner(Callable):A sklearn-like regressor
        Returns:

            regressor
        Raises:

            ValueError
                Raise error if given regressor is not a valid sklearn-like regressor.

        """

        pass

    def predict(self, tsdataset: TSDataset) -> TSDataset:
        """
        predict 

        Args:
            tsdataset(TSDataset): Dataset to predict.

        """
        X_meta = self._generate_predict_meta_data(tsdataset)
        y_pred = self._final_learner.predict(X_meta)

        return y_pred

    @abc.abstractmethod
    def _generate_predict_meta_data(self, train_tsdataset: TSDataset,
                            valid_tsdataset: Optional[TSDataset] = None) -> np.ndarray:
        """
        generate predict meta data
        generate predict meta data needed for level 2

        Args:
            train_tsdataset(TSDataset): Train dataset.
            valid_tsdataset(TSDataset, optional): Valid dataset.
        """

        pass


class StackingEnsembleForecaster(StackingEnsembleBase, BaseModel):
    """
    The StackingEnsembleForecaster Class.

    Args:

        in_chunk_len(int): The size of the loopback window, i.e., the number of time steps feed to the model.
        out_chunk_len(int): The size of the forecasting horizon, i.e., the number of time steps output by the model.
        skip_chunk_len(int): Optional, the number of time steps between in_chunk and out_chunk for a single sample.
            The skip chunk is neither used as a feature (i.e. X) nor a label (i.e. Y) for a single sample. By
            default, it will NOT skip any time steps.
        estimators(List[Tuple[object, dict]] ): A list of tuple (class,params) consisting of several paddlets Forecasting models.
        final_learner(Callable): The final learner on stack level 2, should be a sklearn-like regressor, set to GradientBoostingRegressor(max_depth=5) by default.
        resampling_strategy(str): A string of resampling strategies.Supported resampling strategy are "cv", "holdout".
        split_ratio(Union[str, float]): The proportion of the dataset included in the validation split for holdout.The split_ratio should be in the range of (0, 1). 
        k_fold(Union[str, int]): Number of folds for cv.The k_fold should be in the range of (0, 10], defaults to 3.
        use_backtest(bool): If use backtest on predictions.
        verbose(bool): Turn on Verbose mode,set to true by default.

    """
    def __init__(self,
                 in_chunk_len: int,
                 out_chunk_len: int,
                 skip_chunk_len: int,
                 estimators: List[Tuple[object, dict]],
                 final_learner: Callable = None,
                 use_backtest: bool = True,
                 resampling_strategy: str = 'cv',
                 split_ratio: Union[str, float] = 0.1,
                 k_fold: Union[str, int] = 3,
                 verbose: bool = False
                 ) -> None:

        self._use_backtest = use_backtest
        self._resampling_strategy = resampling_strategy
        self._split_ratio = split_ratio
        self._k_fold = k_fold
        BaseModel.__init__(self, in_chunk_len, out_chunk_len, skip_chunk_len)
        StackingEnsembleBase.__init__(self, estimators,final_learner,verbose)

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

    def _generate_fit_meta_data(self, train_tsdataset: TSDataset,
                            valid_tsdataset: Optional[TSDataset] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate fit meta data needed for level 2

        Args:
            train_tsdataset(TSDataset): Train dataset.
            valid_tsdataset(TSDataset, optional): Valid dataset.
        """
        predictions = []
        X_meta = None
        y_meta = None

        if valid_tsdataset:
            for estimator in self._estimators:
                prediction = fit_and_score(train_data=train_tsdataset, valid_data=valid_tsdataset, estimator=estimator,
                                           use_backtest=self._use_backtest)["predicts"]
                predictions.append(prediction)
            y_meta = np.array(valid_tsdataset.target.data.loc[predictions[0].target.data.index])
        else:
            if self._resampling_strategy == 'holdout':
                raise_if(self._split_ratio > 1 or self._split_ratio < 0, "split_ratio out of range (0, 1)")
                splitter = HoldoutSplitter(test_size=1 - self._split_ratio)
            elif self._resampling_strategy == 'cv':
                if self._k_fold > 10 or self._k_fold <= 0:
                    raise ValueError("k_fold out of range (0,10]")
                splitter = ExpandingWindowSplitter(n_splits=self._k_fold)
            else:
                raise NotImplementedError("Unknown resampling_strategy")

            for estimator in self._estimators:
                cv_predictions = [res["predicts"] for res in
                                  cross_validate(data=train_tsdataset, estimator=estimator, splitter=splitter,
                                                 return_score=False, use_backtest=self._use_backtest,
                                                 verbose=self._verbose)]
                prediction = TSDataset.concat(cv_predictions, axis=0)
                predictions.append(prediction)

            y_meta = np.array(train_tsdataset.target.data.loc[predictions[0].target.data.index])
        X_meta = np.concatenate([np.array(prediction.target.data) for prediction in predictions], axis=1)

        return X_meta, y_meta


    def _check_final_learner(self, final_learner) -> "MultiOutputRegressor":
        """
        Check if a final learner is given and if it is valid, otherwise set default regressor.

        Args:
            final_learner(Callable):A sklearn-like regressor
        Returns:

            MultiOutputRegressor
        Raises:

            ValueError
                Raise error if given regressor is not a valid sklearn-like regressor.

        """
        if final_learner is None:
            final_learner = GradientBoostingRegressor(max_depth=5)
        else:
            if not is_regressor(final_learner):
                raise ValueError(
                    f"`final learner` should be a sklearn-like regressor, "
                    f"but found: {final_learner}"
                )
            final_learner = clone(final_learner)
        return MultiOutputRegressor(final_learner)

    @to_tsdataset(scenario="forecasting")
    def predict(self, tsdataset: TSDataset) -> TSDataset:
        """
        Predict 

        Args:
            tsdataset(TSDataset): Dataset to predict.

        """
        return super().predict(tsdataset)

    def _generate_predict_meta_data(self, tsdataset: TSDataset) -> np.ndarray:
        """
        Generate predict meta data needed for level 2

        Args:
            tsdataset(TSDataset): Predcit dataset.
        """
        predictions = self._predict_estimators(tsdataset)
        X_meta = np.concatenate([np.array(prediction.target.data) for prediction in predictions], axis=1)

        return X_meta

    def save(self, path: str, ensemble_file_name: str = "paddlets-stacking-forecaster-partial.pkl") -> None:
        """
        Save the ensemble model to a directory.

        Args:
            path(str): Output directory path.
            ensemble_file_name(str): Name of ensemble object. This file contains meta information of ensemble model.
        """
        return super().save(path, ensemble_file_name)

    @staticmethod
    def load(path: str, ensemble_file_name: str = "paddlets-stacking-forecaster-partial.pkl") -> "StackingEnsembleForecaster":
        """
        Load the ensemble model from a directory.

        Args:
            path(str): Input directory path.
            ensemble_file_name(str): Name of ensemble object. This file contains meta information of ensemble.

        Returns:
            The loaded ensemble model.
        """
        return EnsembleBase.load(path, ensemble_file_name)


