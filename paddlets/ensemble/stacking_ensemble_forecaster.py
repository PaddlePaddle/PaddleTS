# !/usr/bin/env python3
# -*- coding:utf-8 -*-
import abc
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.base import is_regressor, clone

from paddlets.ensemble.ensemble_forecaster_base import EnsembleForecasterBase
from paddlets.datasets.tsdataset import TSDataset, TimeSeries
from paddlets.datasets.splitter import ExpandingWindowSplitter, HoldoutSplitter
from paddlets.utils.validation import cross_validate, fit_and_score
from paddlets.logger import raise_if

class StackingEnsembleForecaster(EnsembleForecasterBase, metaclass=abc.ABCMeta):
    """
    The StackingEnsemble Class.

    Args:

        in_chunk_len(int): The size of the loopback window, i.e., the number of time steps feed to the model.
        out_chunk_len(int): The size of the forecasting horizon, i.e., the number of time steps output by the model.
        skip_chunk_len(int): Optional, the number of time steps between in_chunk and out_chunk for a single sample.
            The skip chunk is neither used as a feature (i.e. X) nor a label (i.e. Y) for a single sample. By
            default, it will NOT skip any time steps.
        estimators(List[Tuple[object, dict]] ): A list of tuple (class,params) consisting of several paddlets models 
        verbose(bool): Turn on Verbose mode,set to true by default.
        final_learner(Callable): The final learner on stack level 2, should be a sklearn-like regressor, set to GradientBoostingRegressor(max_depth=5) by default.
        resampling_strategy(str): A string of resampling strategies.Supported resampling strategy are "cv", "holdout".
        split_ratio(Union[str, float]): The proportion of the dataset included in the validation split for holdout.The split_ratio should be in the range of (0, 1). 
        k_fold(Union[str, int]): Number of folds for cv.The k_fold should be in the range of (0, 10].
        use_backtest(bool): If use backtest on predictions.

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
        super().__init__(in_chunk_len, out_chunk_len, skip_chunk_len, estimators, verbose)

        final_learner = self._check_final_learner(final_learner)
        self._final_learner = MultiOutputRegressor(final_learner)
        self._use_backtest = use_backtest
        self._resampling_strategy = resampling_strategy
        self._split_ratio = split_ratio
        self._k_fold = k_fold

    def _fit(self,
            train_tsdataset: TSDataset,
            valid_tsdataset: Optional[TSDataset] = None) -> None:
        """
        fit 

        Args:
            train_tsdataset(TSDataset): Train dataset.
            valid_tsdataset(TSDataset, optional): Valid dataset.
        """
        X_meta, y_meta = self._generate_meta_data(train_tsdataset, valid_tsdataset)
        self._final_learner.fit(X_meta, y_meta)

    def _generate_meta_data(self, train_tsdataset: TSDataset,
                            valid_tsdataset: Optional[TSDataset] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        generate meta data
        generate meta data needed for level 2

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

        if final_learner is None:
            final_learner = GradientBoostingRegressor(max_depth=5)
        else:
            if not is_regressor(final_learner):
                raise ValueError(
                    f"`final learner` should be a sklearn-like regressor, "
                    f"but found: {final_learner}"
                )
            final_learner = clone(final_learner)
        return final_learner

    def predict(self, tsdataset: TSDataset) -> TSDataset:
        """
        predict 

        Args:
            tsdataset(TSDataset): Dataset to predict.

        """
        predictions = self._predict_estimators(tsdataset)
        X_meta = np.concatenate([np.array(prediction.target.data) for prediction in predictions], axis=1)

        y_pred = self._final_learner.predict(X_meta)
        target_names = predictions[0].target.data.columns.values.tolist()
        target_df = pd.DataFrame(y_pred, columns=target_names)
        target_df.index = predictions[0].target.data.index

        return TSDataset(target=TimeSeries(target_df, predictions[0].target.freq))
