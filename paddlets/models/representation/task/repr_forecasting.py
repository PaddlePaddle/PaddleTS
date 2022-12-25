# !/usr/bin/env python3
# -*- coding:utf-8 -*-
from typing import Callable, Tuple

import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from sklearn.base import is_regressor, clone

from paddlets.datasets.tsdataset import TSDataset
from paddlets.models.representation.dl.repr_base import ReprBaseModel
from paddlets.models.representation import TS2Vec, CoST
from paddlets.ensemble.base import EnsembleBase
from paddlets.ensemble.stacking_ensemble import StackingEnsembleBase
from paddlets.logger.logger import raise_if, Logger, raise_log
from paddlets.models.data_adapter import DataAdapter
from paddlets.models.utils import to_tsdataset
from paddlets.models import BaseModel
from paddlets.utils.utils import repr_results_to_tsdataset

logger = Logger(__name__)


class ReprForecasting(StackingEnsembleBase, BaseModel):
    """
    The ReprForecasting Class.

    Args:

        in_chunk_len(int): The size of previous time point window  to use for representation results
        out_chunk_len(int): The size of the forecasting horizon, i.e., the number of time steps output by the model.
        repr_model(ReprBasemodel): Representation model to use for forcast.
        skip_chunk_len(int): Optional, the number of time steps between in_chunk and out_chunk for a single sample.
            The skip chunk is neither used as a feature (i.e. X) nor a label (i.e. Y) for a single sample. By
            default, it will NOT skip any time steps.
        sampling_stride: Sampling intervals between two adjacent samples.
        repr_model_params(dict):params for reprmodel init.
        encode_params(dict):params for reprmodel encode, "slide_len" will set to in_chunk_len by force.
        downstream_learner(Callable): The downstream learner, should be a sklearn-like regressor, set to Ridge(alpha=0.5) by default.
        verbose(bool): Turn on Verbose mode,set to true by default.

    """

    def __init__(self,
                 in_chunk_len: int,
                 out_chunk_len: int,
                 repr_model: ReprBaseModel,
                 skip_chunk_len: int = 0,
                 sampling_stride: int = 1,
                 repr_model_params: dict = None,
                 encode_params: dict = None,
                 downstream_learner: Callable = None,
                 verbose: bool = False
                 ) -> None:

        self._sampling_stride = sampling_stride
        if repr_model_params is None:
            repr_model_params = {}
        if encode_params is None:
            encode_params = {}
        raise_if(not isinstance(repr_model_params, dict), "model_params should be a params dict")
        raise_if(not isinstance(encode_params, dict), "encode_params should be a params dict")
        self._repr_model = repr_model
        self._repr_model_params = repr_model_params
        self._encode_params = encode_params
        BaseModel.__init__(self, in_chunk_len, out_chunk_len, skip_chunk_len)
        super().__init__([(repr_model, repr_model_params)], downstream_learner, verbose)

    def _set_params(self, estimators):
        """
        Set estimators params

        Set params and initial estimators.

        Args:

            estimators(List[Tuple[object, dict]] ): A list of tuple (class,params) consisting of several paddlets models.

        """
        if estimators[0][0] is TS2Vec:
            if "sliding_len" in self._encode_params:
                raise_if(self._encode_params["sliding_len"] != self._in_chunk_len,
                         f" Sliding_len ({self._encode_params['sliding_len']})should \
                 equal to in_chunk_len{self._in_chunk_len} when use ts2vec")
            else:
                # sliding_len decide how many previous time point to use for repr res in Ts2vec
                self._encode_params["sliding_len"] = self._in_chunk_len - 1
        elif estimators[0][0] is CoST:
            if "segment_size" in estimators[0][1]:
                raise_if(estimators[0][1]["segment_size"] != self._in_chunk_len,
                         f"Segment_size ({self._repr_model_params['segment_size']})should \
                 equal to in_chunk_len{self._in_chunk_len}.(except ts2vec model)")
            else:
                estimators[0][1]["segment_size"] = self._in_chunk_len
        else:
            raise_log(f"Unsupported model type {type(self._repr_model)}")
        if "verbose" not in self._encode_params:
            self._encode_params["verbose"] = False

        return super()._set_params(estimators)

    def _check_estimators(self, estimators):
        """
        Check estimators

        Check and valid estimators

        Args:

            estimators(List[Tuple[object, dict]] ): A list of tuple (class,params) consisting of several paddlets models.

        """
        super()._check_estimators(estimators)

        if all([issubclass(e[0], ReprBaseModel) for e in estimators]):
            pass
        else:
            raise ValueError("Estimators have unsupported or uncompatible models")

    def fit(self,
            tsdataset: TSDataset) -> None:
        """
        fit 

        Args:
            train_tsdataset(TSDataset): Train dataset.
        """

        target_len = len(tsdataset.target)
        if tsdataset._observed_cov and tsdataset.observed_cov.end_time > tsdataset.target.end_time:
            tsdataset._observed_cov, _ = tsdataset._observed_cov.split(target_len)

        # TODO: future know_cov support 
        if tsdataset.known_cov and tsdataset.known_cov.end_time > tsdataset.target.end_time:
            tsdataset._known_cov, _ = tsdataset._known_cov.split(target_len)

        super().fit(tsdataset)

    def _generate_fit_meta_data(self, tsdataset: TSDataset, valid_data: TSDataset = None) -> Tuple[
        np.ndarray, np.ndarray]:
        """
        Generate meta data needed for level 2

        Args:
            train_tsdataset(TSDataset): Train dataset.

        Return:
            Tuple[np.ndarray, np.ndarray]
        """
        # fit repr
        logger.info("Repr model fit start")
        self._estimators[0].fit(tsdataset)
        logger.info("Repr model fit end")
        # encode
        encode = self._estimators[0].encode(tsdataset, **self._encode_params)
        # repr res to tsdataset
        encode_tsdataset = repr_results_to_tsdataset(encode, tsdataset)
        # get samples
        samples = DataAdapter().to_sample_dataset(encode_tsdataset,
                                                  in_chunk_len=1,
                                                  out_chunk_len=self._out_chunk_len,
                                                  skip_chunk_len=self._skip_chunk_len,
                                                  sampling_stride=self._sampling_stride).samples
        X_meta = []
        y_meta = []
        # assemble meta
        # currently not support known_cov
        for sample in samples:
            target = sample["future_target"]
            target = np.array(target).flatten()
            y_meta.append(target)
            feature = sample["observed_cov_numeric"]
            feature = np.array(feature).flatten()
            X_meta.append(feature)

        X_meta = np.array(X_meta)
        y_meta = np.array(y_meta)
        return X_meta, y_meta

    def _check_final_learner(self, final_learner) -> None:
        """
        Check if a final learner is given and if it is valid, otherwise set default regressor.

        Args:
            final_learner(Callable):A sklearn-like regressor
        Returns:

            regressor
        Raises:

            ValueError
                Raise error if given regressor is not a valid sklearn-like regressor.

        """

        if final_learner is None:
            final_learner = Ridge(alpha=0.5)
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
        tsdataset = tsdataset.copy()
        target_len = len(tsdataset.target)

        raise_if(target_len < self._in_chunk_len,
                 f"tsdataset.target length ({target_len}) must < model._in_chunk_len ({self._in_chunk_len})")

        if tsdataset._observed_cov and tsdataset.observed_cov.end_time > tsdataset.target.end_time:
            tsdataset._observed_cov, _ = tsdataset._observed_cov.split(target_len)

        # TODO: future know_cov support 
        if tsdataset.known_cov and tsdataset.known_cov.end_time > tsdataset.target.end_time:
            tsdataset._known_cov, _ = tsdataset._known_cov.split(target_len)

        if target_len > self._in_chunk_len:
            _, tsdataset = tsdataset.split(target_len - self._in_chunk_len)

        # target_num = len(tsdataset.target.columns)
        # y_pred = y_pred.reshape(self._out_chunk_len, target_num)

        return super().predict(tsdataset)

    def _generate_predict_meta_data(self, tsdataset: TSDataset) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predict meta data needed for level 2

        Args:
            tsdataset(TSDataset): Predcit dataset.
        """
        encode = self._estimators[0].encode(tsdataset, **self._encode_params)
        encode = encode[0]
        X_meta = encode[-1].reshape(1, -1)

        return X_meta

    def save(self, path: str, repr_forecaster_file_name: str = "repr-forecaster-partial.pkl") -> None:
        """
        Save the repr-forecaster model to a directory.

        Args:
            path(str): Output directory path.
            ensemble_file_name(str): Name of repr-forecaster model object. This file contains meta information of repr-forecaster model.
        """
        return super().save(path, repr_forecaster_file_name)

    @staticmethod
    def load(path: str, repr_forecaster_file_name: str = "repr-forecaster-partial.pkl") -> "StackingEnsembleForecaster":
        """
        Load the repr-forecaster model from a directory.

        Args:
            path(str): Input directory path.
            ensemble_file_name(str): Name of repr-forecaster model object. This file contains meta information of repr-forecaster model.

        Returns:
            The loaded ensemble model.
        """
        return EnsembleBase.load(path, repr_forecaster_file_name)
