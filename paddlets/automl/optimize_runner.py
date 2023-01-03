# !/usr/bin/env python3
# -*- coding:utf-8 -*-

from typing import List, Optional, Type, Union
import os
import copy
import inspect

from abc import ABCMeta
import ray
from ray import tune
from ray.tune.sample import Categorical
from ray.tune import choice

from paddlets.logger import Logger
from paddlets.datasets.tsdataset import TSDataset
from paddlets.models.base import BaseModel
from paddlets.transform.base import BaseTransform
from paddlets.pipeline.pipeline import Pipeline
from paddlets.metrics import MAE
from paddlets.utils import cross_validate, fit_and_score
from paddlets.datasets.splitter import HoldoutSplitter, ExpandingWindowSplitter
from paddlets.automl.search_space_configer import SearchSpaceConfiger
from paddlets.automl.searcher import Searcher

logger = Logger(__name__)
MODEL_SETUP_SEED = 2022
NEED_METRIC_PROB_ESTIMATORS = ["DeepARModel", "TFTModel"]

class OptimizeRunner:
    """
    Optimize runner is for experiment execution and hyperparameter tuning.

    Args:
        search_alg(str): The algorithm for optimization.Supported algorithms are "auto", "Random", "CMAES", "TPE",
            "CFO", "BlendSearch", "Bayes".

    """

    def __init__(self, search_alg: str = "Random"):
        self.search_alg = search_alg
        self.paddlets_configer = SearchSpaceConfiger()
        self.report_metric = "loss"
        self._track_choice_mapping = {}

    def setup_estimator(self,
                        config: dict,
                        paddlets_estimator: Union[Type[BaseModel], List[Union[Type[BaseTransform], Type[BaseModel]]]],
                        in_chunk_len: int,
                        out_chunk_len: int,
                        skip_chunk_len: int,
                        sampling_stride: int
                        ):
        """

        Build a paddlets estimator with config.

        Args:
            config(dict): Algorithm configuration for estimator.
            paddlets_estimator(Union[Type[BaseModel], List[Union[Type[BaseTransform], Type[BaseModel]]]]): A class of a paddlets model
                or a list of classes consisting of several paddlets transformers and a paddlets model.
            in_chunk_len(int): The size of the loopback window, i.e., the number of time steps feed to the model.
            out_chunk_len(int): The size of the forecasting horizon, i.e., the number of time steps output by the model.
            skip_chunk_len: The number of time steps between in_chunk and out_chunk for a single sample.
            sampling_stride(int): Sampling intervals between two adjacent samples.

        Returns:
            BaseModel: paddlets estimator.

        """
        running_config = copy.deepcopy(config)
        self._preprocess_config(running_config, self._track_choice_mapping)
        # model param
        if isinstance(paddlets_estimator, list):
            model_init_signature = inspect.signature(paddlets_estimator[-1].__init__)
        else:
            model_init_signature = inspect.signature(paddlets_estimator.__init__)
        model_init_param = list(model_init_signature.parameters)
        model_base_param = {
            "in_chunk_len": in_chunk_len,
            "out_chunk_len": out_chunk_len,
            "skip_chunk_len": skip_chunk_len,
        }
        if "sampling_stride" in model_init_param:
            model_base_param.update({"sampling_stride": sampling_stride})
        if "seed" in model_init_param:
            model_base_param.update({"seed": MODEL_SETUP_SEED})

        if isinstance(paddlets_estimator, list):
            # init pipeline
            pipeline_param = []

            # Requires python version >= 3.6 to ensure the order-preserving of dict
            # Refer to https://mail.python.org/pipermail/python-dev/2017-December/151283.html
            for idx, estimator_name in enumerate(running_config):
                pipeline_param.append((paddlets_estimator[idx], running_config[estimator_name]))

            pipeline_param[-1][1].update(model_base_param)
            logger.info(f"setup_estimator: init pipeline. Params: {pipeline_param}")
            estimator = Pipeline(pipeline_param)
        else:
            model_param = running_config
            model_param.update(model_base_param)
            logger.info(f"setup_estimator: init model. Params: {model_param}")
            estimator = paddlets_estimator(**model_param)

        return estimator

    def optimize(self,
                 paddlets_estimator: Union[Type[BaseModel], List[Union[Type[BaseTransform], Type[BaseModel]]]],
                 in_chunk_len: int,
                 out_chunk_len: int,
                 train_tsdataset: Union[TSDataset, List[TSDataset]],
                 valid_tsdataset: Union[TSDataset, List[TSDataset]] = None,
                 sampling_stride: int = 1,
                 skip_chunk_len: int = 0,
                 search_space: Optional[dict] = None,
                 metric: ABCMeta = MAE,
                 mode: str = "min",
                 resampling_strategy: str = "holdout",
                 split_ratio: float = 0.1,
                 k_fold: int = 3,  # cv的fold切分数, 默认5折切分
                 n_trials: int = 5,
                 cpu_resource: float = 1.0,
                 gpu_resource: float = 0,
                 max_concurrent_trials: int = 1,
                 local_dir: Optional[str] = None,
                 ):
        """
        Execute optimization.

        Args:
            paddlets_estimator(Union[Type[BaseModel], List[Union[Type[BaseTransform], Type[BaseModel]]]]):
                A class of a paddlets model or a list of classes consisting of several paddlets transformers and a paddlets model.
            in_chunk_len(int): The size of the loopback window, i.e., the number of time steps feed to the model.
            out_chunk_len(int): The size of the forecasting horizon, i.e., the number of time steps output by the model.
            train_tsdataset(Union[TSDataset, List[TSDataset]]): Train dataset.
            valid_tsdataset(Union[TSDataset, List[TSDataset]], optional): Valid dataset.
            sampling_stride(int): Sampling intervals between two adjacent samples.
            skip_chunk_len(int): Optional, the number of time steps between in_chunk and out_chunk for a single sample.
            search_space(Optional[dict]): The domain of the automl to be optimized.
            metric(ABCMeta): A class of a metric, e.g. MAE, MSE.
            mode(str): According to the mode, the metric is maximized or minimized. Supported mode are "min", "max"
            resampling_strategy(str): A string of resampling strategies. Supported resampling strategy are "cv",
                "holdout".
            split_ratio(float): The proportion of the dataset included in the validation split for holdout.
            k_fold(int): Number of folds for cv.
            n_trials(int): The number of configurations suggested by the search algorithm.
            cpu_resource(float): CPU resources to allocate per trial.
            gpu_resource(float): GPU resources to allocate per trial. Note that GPUs will not be assigned if you do
                not specify them here.
            max_concurrent_trials(int): The maximum number of trials running concurrently.
            local_dir(str): Local dir to save training results to. Defaults to `./`.

        Returns:
            ExperimentAnalysis: Object for experiment analysis.

        Raises:
            TuneError: Any trials failed.

        """

        def run_trial(config):

            logger.info(f"trial config: {config}")

            # setup estimator
            estimator = self.setup_estimator(config=config,
                                             paddlets_estimator=paddlets_estimator,
                                             in_chunk_len=in_chunk_len,
                                             out_chunk_len=out_chunk_len,
                                             skip_chunk_len=skip_chunk_len,
                                             sampling_stride=sampling_stride)
            score = None
            metric_instance = None
            if estimator.__class__.__name__ in NEED_METRIC_PROB_ESTIMATORS:
                metric_instance = metric(mode="prob")
            else:
                metric_instance = metric()
            if valid_tsdataset:
                logger.info("Got user-defined valid_tsdataset.If trian and valid dataset are not continious.\n"
                            "The first input_chunk_len target value of valid_tsdataset will be used to build the feature, "
                            "but not be used to calculate the optimization target.")
                score = fit_and_score(train_data=train_tsdataset,
                                      valid_data=valid_tsdataset,
                                      estimator=estimator,
                                      metric=metric_instance)["score"]
            else:
                if resampling_strategy == "holdout":
                    splitter = HoldoutSplitter(test_size=split_ratio)
                if resampling_strategy == "cv":
                    splitter = ExpandingWindowSplitter(n_splits=k_fold)
                score = cross_validate(data=train_tsdataset,
                                       splitter=splitter,
                                       estimator=estimator,
                                       metric=metric_instance)

            tune.report(**{self.report_metric: score})

        if local_dir is None:
            local_dir = "./"

        if isinstance(train_tsdataset, list):
            # check valid tsdataset exist
            if valid_tsdataset is None:
                raise NotImplementedError("When the train_tsdataset is a list, valid_tsdataset is required!")

        # 若sp不存在，则获取默认search space
        # if sp is None, get default search space using paddlets_configer
        if search_space is None:
            search_space = self.paddlets_configer.get_default_search_space(paddlets_estimator)

        searcher = Searcher.get_searcher(self.search_alg, max_concurrent=max_concurrent_trials)

        running_search_space = copy.deepcopy(search_space)
        self._track_choice_mapping = self._preprocess_search_space(running_search_space)

        # 将ray临时文件夹修改为用户自定义文件夹
        # 临时文件夹的设置会导致ray启动错误 https://github.com/ray-project/ray/issues/30650
        # set ray temp dir to local_dir
        # The setting of _temp_dir will cause ray startup error, issue: https://github.com/ray-project/ray/issues/30650
        # os.environ['RAY_TMPDIR'] = local_dir + "/ray_log/"

        return tune.run(run_trial, num_samples=n_trials, config=running_search_space,
                        metric=self.report_metric,
                        mode=mode,
                        search_alg=searcher,
                        fail_fast=True,
                        resources_per_trial={
                            "cpu": cpu_resource,
                            "gpu": gpu_resource
                        },
                        local_dir=local_dir+"/ray_results",
                        )

    def _backtrack_traverse_search_space(self, sp, track, track_choice_mapping):
        """
        _backtrack_traverse_search_space
        """
        if isinstance(sp, Categorical):
            choice_list = sp.categories
            for e in choice_list:
                if isinstance(e, list):
                    # build mapping
                    str_choice = ["Choice_%d: %s" % (idx, str(e)) for idx, e in enumerate(choice_list)]
                    str_choice_mapping = {}
                    for idx, value in enumerate(str_choice):
                        str_choice_mapping[value] = choice_list[idx]
                    track_choice_mapping[tuple(track)] = str_choice_mapping
                    return str_choice
            return
        for k, v in sp.items():
            if not isinstance(v, dict) and not isinstance(v, Categorical):
                continue
            track.append(k)
            traverse_res = self._backtrack_traverse_search_space(sp[k], track, track_choice_mapping)
            if isinstance(traverse_res, list):
                # Traverse to leaf nodes
                sp[k] = choice(traverse_res)
            track.pop()

    def _preprocess_search_space(self, sp):
        """
        _preprocess_search_space
        """
        track = []
        track_choice_mapping = {}
        self._backtrack_traverse_search_space(sp, track, track_choice_mapping)
        return track_choice_mapping

    def _preprocess_config(self, config, track_choice_mapping):
        """
        _preprocess_config
        """
        for k, v in track_choice_mapping.items():
            key_list = list(k)
            mapping_key = self._config_get_value(config, key_list)
            key_list = list(k)
            self._config_set_value(config, key_list, v[mapping_key])

    def _config_set_value(self, config, key_list, value):
        """
        _config_set_value
        """
        if len(key_list) == 1:
            config[key_list[0]] = value
            return
        k = key_list.pop(0)
        self._config_set_value(config[k], key_list, value)

    def _config_get_value(self, config, key_list):
        """
        _config_get_value
        """
        if len(key_list) == 0:
            return
        k = key_list.pop(0)
        if not (k in config):
            return
        if len(key_list) == 0:
            return config[k]
        return self._config_get_value(config[k], key_list)
