# !/usr/bin/env python3
# -*- coding:utf-8 -*-
import abc
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from sklearn.base import is_regressor, clone

from paddlets.datasets.tsdataset import TSDataset
from paddlets.models.representation.dl.repr_base import ReprBaseModel
from paddlets.ensemble.ensemble_forecaster_base import EnsembleForecasterBase
from paddlets.logger.logger import raise_if,Logger
from paddlets.models.forecasting.dl.adapter import DataAdapter
from paddlets.models.utils import to_tsdataset
from paddlets.utils.utils import repr_results_to_tsdataset

logger = Logger(__name__)
class ReprForecasting(EnsembleForecasterBase, metaclass=abc.ABCMeta):
    """
    The ReprForecasting Class.

    Args:

        in_chunk_len(int): The size of previous time point window  to use for representation results
        out_chunk_len(int): The size of the forecasting horizon, i.e., the number of time steps output by the model.
        repr_model(ReprBasemodel): Representation model to use for forecasting.
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

        raise_if(not isinstance(repr_model_params, dict), "model_params should be a params dict")
        estimator = [(repr_model, repr_model_params)]

        super().__init__(in_chunk_len, out_chunk_len, skip_chunk_len, estimator, verbose)

        if encode_params is None:
            encode_params = {}
        raise_if(not isinstance(encode_params, dict), "encode_params should be a params dict")

        # sliding_len decide how many previous time point to use for repr res
        encode_params["sliding_len"] = self._in_chunk_len
        if "verbose" not in encode_params:
            encode_params["verbose"] = False
        self._encode_params = encode_params

        self._sampling_stride = sampling_stride
        downstream_learner = self._check_final_learner(downstream_learner)
        self._final_learner = MultiOutputRegressor(downstream_learner)

    def _fit(self,
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
        X_meta, y_meta = self._generate_meta_data(tsdataset)
        logger.info("Downstream model fit start")
        self._final_learner.fit(X_meta, y_meta)
        logger.info("Downstream model fit end")

    def _generate_meta_data(self, tsdataset: TSDataset) -> Tuple[np.ndarray, np.ndarray]:
        """
        generate meta data
        generate meta data needed for level 2

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
        samples = DataAdapter().to_paddle_dataset(encode_tsdataset,
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
            final_learner = Ridge(alpha=0.5)
        else:
            if not is_regressor(final_learner):
                raise ValueError(
                    f"`final learner` should be a sklearn-like regressor, "
                    f"but found: {final_learner}"
                )
            final_learner = clone(final_learner)
        return final_learner

    @to_tsdataset(scenario="forecasting")
    def predict(self, tsdataset: TSDataset) -> TSDataset:
        """
        predict 

        Args:
            tsdataset(TSDataset): Dataset to predict.

        """
        tsdataset = tsdataset.copy()
        target_len = len(tsdataset.target)

        raise_if(target_len < self._in_chunk_len, f"tsdataset.target length ({target_len}) must < model._in_chunk_len ({self._in_chunk_len})")

        if tsdataset._observed_cov and tsdataset.observed_cov.end_time > tsdataset.target.end_time:
            tsdataset._observed_cov, _ = tsdataset._observed_cov.split(target_len)

        # TODO: future know_cov support 
        if tsdataset.known_cov and tsdataset.known_cov.end_time > tsdataset.target.end_time:
            tsdataset._known_cov, _ = tsdataset._known_cov.split(target_len)

        if target_len > self._in_chunk_len:
            _, tsdataset = tsdataset.split(target_len - self._in_chunk_len)
    
        encode = self._estimators[0].encode(tsdataset, **self._encode_params)
        encode = encode[0]
        X_meta = encode[-1].reshape(1, -1)
        y_pred = self._final_learner.predict(X_meta)
        target_num = len(tsdataset.target.columns)
        y_pred = y_pred.reshape(self._out_chunk_len, target_num)

        return y_pred
