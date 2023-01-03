# !/usr/bin/env python3
# -*- coding:utf-8 -*-

from typing import List, Optional, Type, Union

import numpy as np

from paddlets.logger import raise_if, Logger
from paddlets.logger.logger import log_decorator
from paddlets.datasets.tsdataset import TSDataset
from paddlets.models.base import BaseModel
from paddlets.transform.base import BaseTransform
from paddlets.metrics.metrics import MAE, MSE, LogLoss
from paddlets.automl.searcher import Searcher
from paddlets.automl.search_space_configer import SearchSpaceConfiger
from paddlets.automl.optimize_runner import OptimizeRunner
from paddlets.utils import check_train_valid_continuity
from paddlets.models.forecasting.dl.paddle_base import PaddleBaseModel
from paddlets.models.forecasting.ml.ml_base import MLBaseModel


logger = Logger(__name__)

METRICS = {
    "mae": MAE,
    "mse": MSE,
    "logloss": LogLoss
}
DEFAULT_SPLIT_RATIO = 0.1
DEFAULT_K_FOLD = 3
DEFAULT_DL_REFIT_TRAIN_PROPORTION = 0.9
NP_RANDOM_SEED = 2022


class AutoTS(BaseModel):
    """
    The AutoTS Class.
    AutoTS is an automated machine learning tool for PaddleTS.
    It frees the user from selecting hyperparameters for PaddleTS models or PaddleTS pipelines.

    Args:
        estimator(Union[str, Type[BaseModel], List[Union[str, Type[BaseTransform], Type[BaseModel]]]]): A class of
            a paddlets model or a list of classes consisting of several paddlets transformers and a paddlets model
        in_chunk_len(int): The size of the loopback window, i.e., the number of time steps feed to the model.
        out_chunk_len(int): The size of the forecasting horizon, i.e., the number of time steps output by the model.
        skip_chunk_len(int): Optional, the number of time steps between in_chunk and out_chunk for a single sample.
            The skip chunk is neither used as a feature (i.e. X) nor a label (i.e. Y) for a single sample. By
            default, it will NOT skip any time steps.
        sampling_stride(int): Sampling intervals between two adjacent samples.
        search_space(Union[str, dict]): The domain of the automl to be optimized.
            If search_space is 'auto', the default search space will be used.
        search_alg(str): The algorithm for optimization.
            Supported algorithms are "auto", "Random", "CMAES", "TPE", "CFO", "BlendSearch", "Bayes". When the algorithm
            is "auto", search_alg is set to "TPE" based on experimental experiences.
        resampling_strategy(str): A string of resampling strategies.
            Supported resampling strategy are "auto", "cv", "holdout".When the strategy is "auto", resampling_strategy
            is set to "holdout" and split_ratio is set to DEFAULT_SPLIT_RATIO by default.
        split_ratio(Union[str, float]): The proportion of the dataset included in the validation split for holdout.
            The split_ratio should be in the range of (0, 1). When the split_ratio is "auto", split_ratio is set to
            DEFAULT_SPLIT_RATIO by default.
            Note that the split_ratio will be ignored if valid_tsdataset is provided in the `AutoTS.fit()`.
        k_fold(Union[str, int]): Number of folds for cv.
            The k_fold should be in the range of (0, 10].When the k_fold is "auto", k_fold is set to DEFAULT_K_FOLD by default.
            Note that the k_fold will be ignored if valid_tsdataset is provided in the `AutoTS.fit()`.
        metric(str): A string of the metric name. The specified metric will be used to calculate validation loss reported
            to the search_algo.
            Supported metric are "mae", "mse", "logloss". When the metric is "auto", metric is set to "mae" by
            default.
        mode(str): According to the mode, the metric is maximized or minimized.
            Supported mode are "min", "max". When the mode is "auto", metric is set to "min" by default.
        refit(bool): Whether to refit the model with the best parameter on full training data.If refit is True, the
            AutoTS object can be used to predict. If refit is False, the AutoTS
            object can be used to get the best parameter, but can not make predictions.
        local_dir(str): Local dir to save training results and log to. Defaults to `./`.
        ensemble(bool): Not supported yet. This feature will be comming in future.
        n_jobs(int): Not supported yet. This feature will be comming in future.
        verbose(int): Not supported yet. This feature will be comming in future.

    Examples:

        >>> from paddlets.automl.autots import AutoTS
        >>> from paddlets.models.forecasting import MLPRegressor
        >>> from paddlets.datasets.repository import get_dataset
        >>> tsdataset = get_dataset("UNI_WTH")
        >>> autots_model = AutoTS(MLPRegressor, 96, 2)
        >>> autots_model.fit(tsdataset)
        >>> predicted_tsdataset = autots_model.predict(tsdataset)
        >>> best_param = autots_model.best_param
    """

    def __init__(
            self,
            estimator: Union[str, Type[BaseModel], List[Union[str, Type[BaseTransform], Type[BaseModel]]]],
            in_chunk_len: int,
            out_chunk_len: int,
            skip_chunk_len: int = 0,
            sampling_stride: int = 1,
            search_space: Union[str, dict] = 'auto',
            search_alg: str = 'auto',
            resampling_strategy: str = 'auto',
            split_ratio: Union[str, float] = 'auto',
            k_fold: Union[str, int] = 'auto',
            metric: str = 'auto',
            mode: str = 'auto',
            refit: bool = True,
            ensemble: bool = False,
            local_dir: Optional[str] = None,
            n_jobs: int = -1,
            verbose: int = 4
    ):
        np.random.seed(NP_RANDOM_SEED)
        super(AutoTS, self).__init__(
            in_chunk_len=in_chunk_len,
            out_chunk_len=out_chunk_len,
            skip_chunk_len=skip_chunk_len
        )
        self._sampling_stride = sampling_stride
        self._paddlets_configer = SearchSpaceConfiger()
        self._best_param = None
        self._best_estimator = None
        self._check_estimator_valid(estimator)
        self._estimator = estimator
        self._is_pipeline = False
        self._refitted = False
        if isinstance(self._estimator, list):
            self._is_pipeline = True
        if search_space == 'auto':
            if self._is_pipeline:
                # search space cannot be 'auto' when estimator is a pipeline
                raise NotImplementedError("\nSearch space cannot be 'auto' when estimator is a pipeline.\n"
                                          + self._paddlets_configer.recommend(estimator, verbose=False))
            self._search_space = self._paddlets_configer.get_default_search_space(self._estimator)
        else:
            self._search_space = search_space

        if search_alg == 'auto':
            self._search_alg = 'TPE'
        elif search_alg in Searcher.get_supported_algs():
            self._search_alg = search_alg
        else:
            raise NotImplementedError("Unknown search_alg")

        self._k_fold = DEFAULT_K_FOLD
        self._split_ratio = DEFAULT_SPLIT_RATIO
        if resampling_strategy == 'auto':
            self._resampling_strategy = 'holdout'
            self._split_ratio = DEFAULT_SPLIT_RATIO if split_ratio == 'auto' else split_ratio
            raise_if(self._split_ratio > 1 or self._split_ratio < 0, "split_ratio out of range (0, 1)")
        elif resampling_strategy in ['cv', 'holdout']:
            self._resampling_strategy = resampling_strategy
            if self._resampling_strategy == 'cv':
                if k_fold > 10 or k_fold <= 0:
                    raise ValueError("k_fold out of range (0,10]")
                self._k_fold = DEFAULT_K_FOLD if k_fold == 'auto' else k_fold
        else:
            raise NotImplementedError("Unknown resampling_strategy")

        if metric == 'auto':
            self._metric = MAE
        elif metric in METRICS.keys():
            self._metric = METRICS[metric]
        else:
            raise NotImplementedError("Unknown metric")

        if mode == 'auto':
            self._mode = "min"
        elif mode in ["min", "max"]:
            self._mode = mode
        else:
            raise NotImplementedError("Unknown mode, supported: [min,max]")

        self._ensemble = ensemble
        self._refit = refit
        self._n_jobs = n_jobs
        self._verbose = verbose
        self._optimize_runner = OptimizeRunner(search_alg=self._search_alg)
        if local_dir is None:
            self._local_dir = "./"
        else:
            self._local_dir = local_dir

    @log_decorator
    def fit(
            self,
            train_tsdataset: Union[TSDataset, List[TSDataset]],
            valid_tsdataset: Union[TSDataset, List[TSDataset]] = None,
            n_trials: int = 20,
            cpu_resource: float = 1.0,
            gpu_resource: float = 0,
            max_concurrent_trials: int = 1,
    ):
        """
        Fit the estimator with the given tsdataset.
        The way fit is done is that the search algorithm will suggest configurations from the hyperparameter search
        space, then choose the best parameter from all configurations.
        If refit is True, the fit() will refit the model with the best parameters on full training data.

        Args:
            train_tsdataset(Union[TSDataset, List[TSDataset]]): Train dataset.
            valid_tsdataset(Union[TSDataset, List[TSDataset]], optional): Valid dataset.
            n_trials(int): The number of configurations suggested by the search algorithm.
            cpu_resource(float): CPU resources to allocate per trial.
            gpu_resource(float): GPU resources to allocate per trial. Note that GPUs will not be assigned if you do
                not specify them here.
            max_concurrent_trials(int): The maximum number of trials running concurrently.


        Returns:
            Optional(BaseModel, Pipeline): Refitted estimator.
        """
        if cpu_resource < 0 or gpu_resource < 0 or max_concurrent_trials <= 0:
            raise NotImplementedError("invalid cpu_resource || gpu_resource || max_concurrent_trials")
        if isinstance(train_tsdataset, list):
            # check valid tsdataset exist
            if valid_tsdataset is None:
                raise NotImplementedError("When the train_tsdataset is a list, valid_tsdataset is required!")
        analysis = self._optimize_runner.optimize(self._estimator,
                                                  self._in_chunk_len,
                                                  self._out_chunk_len,
                                                  train_tsdataset,
                                                  valid_tsdataset=valid_tsdataset,
                                                  sampling_stride=self._sampling_stride,
                                                  skip_chunk_len=self._skip_chunk_len,
                                                  metric=self._metric,
                                                  search_space=self._search_space,
                                                  mode=self._mode,
                                                  resampling_strategy=self._resampling_strategy,
                                                  split_ratio=self._split_ratio,
                                                  k_fold=self._k_fold,  # cv的fold切分数, 默认DEFAULT_K_FOLD折切分
                                                  n_trials=n_trials,
                                                  cpu_resource=cpu_resource,
                                                  gpu_resource=gpu_resource,
                                                  local_dir=self._local_dir,
                                                  max_concurrent_trials=max_concurrent_trials,
                                                  )
        self._best_param = analysis.best_config
        if self._refit:
            logger.info("AutoTS: start refit")
            self._best_estimator = self._optimize_runner.setup_estimator(config=self._best_param,
                                                   paddlets_estimator=self._estimator,
                                                   in_chunk_len=self._in_chunk_len,
                                                   out_chunk_len=self._out_chunk_len,
                                                   skip_chunk_len=self._skip_chunk_len,
                                                   sampling_stride=self._sampling_stride)

            estimator_model = self._estimator[-1] if self._is_pipeline else self._estimator
            if hasattr(estimator_model, "__mro__") and PaddleBaseModel in estimator_model.__mro__:
                if valid_tsdataset is None:
                    if self._resampling_strategy == "holdout":
                        train_tsdataset, valid_tsdataset = train_tsdataset.split(1 - self._split_ratio)
                        self._best_estimator.fit(train_tsdataset, valid_tsdataset)
                    elif len(train_tsdataset.get_target()) * (1 - DEFAULT_DL_REFIT_TRAIN_PROPORTION) \
                            > self._in_chunk_len + self._skip_chunk_len:
                        train_tsdataset, valid_tsdataset = train_tsdataset.split(DEFAULT_DL_REFIT_TRAIN_PROPORTION)
                        self._best_estimator.fit(train_tsdataset, valid_tsdataset)
                    else:
                        self._best_estimator.fit(train_tsdataset)
                else:
                    self._best_estimator.fit(train_tsdataset, valid_tsdataset)
            elif hasattr(estimator_model, "__mro__") and MLBaseModel in estimator_model.__mro__:
                # if is ml model && data is continuity, concat
                if valid_tsdataset is not None \
                        and not isinstance(train_tsdataset, list)\
                        and not isinstance(valid_tsdataset, list)\
                        and check_train_valid_continuity(train_tsdataset, valid_tsdataset):
                    train_tsdataset = TSDataset.concat([train_tsdataset, valid_tsdataset])
                self._best_estimator.fit(train_tsdataset)
            self._refitted = True
            logger.info("AutoTS: refitted")
            return self._best_estimator

    def predict(self, tsdataset: TSDataset) -> TSDataset:
        """
        Make prediction.

        Args:
            tsdataset: Data to be predicted.

        Returns:
            TSDataset: Predicted results of calling `self.predict` on the refitted estimator.

        """
        if not self._refit:
            raise NotImplementedError("The best_estimator is not refitted.")
        return self._best_estimator.predict(tsdataset)

    @property
    def best_param(self):
        """
        Return the best parameters in optimization.

        Returns:
            Dict: The dict of the best parameters.
        """
        return self._best_param

    def best_estimator(self):
        """
        Return the best_estimator in optimization.

        Returns:
            estimator: The best_estimator in optimization.
        """
        if not self._refit:
            raise NotImplementedError("The best_estimator is not refitted.")
        return self._best_estimator

    def search_space(self):
        """
        Return the search space.
        If search_space is 'auto', it will return the default search space.

        Returns:
            Dict: The dict of search space.
        """
        return self._search_space

    def save(self, path: str) -> None:
        """
        AutoTS doesn't support save() yet.
        """
        raise NotImplementedError("Not supported yet")

    @classmethod
    def load(path: str):
        """
        AutoTS doesn't support save() yet.
        """
        raise NotImplementedError("Not supported yet")

    def _check_estimator_valid(self, estimator):
        """
        _check_estimator_valid
        """
        if hasattr(estimator, "__mro__"):
            if not (BaseTransform in estimator.__mro__ or BaseModel in estimator.__mro__):
                raise NotImplementedError("Unknown estimator")
        elif isinstance(estimator, str):
            # str form is not supported yet
            raise NotImplementedError("Estimator in str form is not supported yet")
        elif isinstance(estimator, list):
            # todo: 必须要有模型
            if len(estimator) == 0:
                raise NotImplementedError("Estimator list must not be empty")
            for e in estimator:
                if isinstance(estimator, str):
                    # str form is not supported yet
                    raise NotImplementedError("Estimator in str form is not supported yet")
                if not (BaseTransform in e.__mro__ or BaseModel in e.__mro__):
                    raise NotImplementedError("Unknown estimator")
            # The last estimator must be model
            if not (BaseModel in estimator[-1].__mro__):
                raise NotImplementedError("The last estimator must be model")
        else:
            # estimator is unknown type
            raise NotImplementedError("Unkonwn estimator")

    def is_refitted(self):
        """

        Returns:
            Bool: Whether the autots model has been refitted

        """
        return self._refitted
