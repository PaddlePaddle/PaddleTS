# !/usr/bin/env python3
# -*- coding:utf-8 -*-

import copy
import json

from ray.tune import uniform, quniform, loguniform, randn, randint, qrandint, lograndint, choice
from ray.tune.sample import Float, Integer, Categorical
from ray.tune.sample import Quantized, Normal

from paddlets.logger import Logger
from paddlets.pipeline import Pipeline
from paddlets.models.forecasting.dl.paddle_base import PaddleBaseModel
from paddlets.models.forecasting.ml.ml_base import MLBaseModel
from paddlets.transform.base import BaseTransform

logger = Logger(__name__)

USER_DEFINED_SEARCH_SPACE = "USER_DEFINED_SEARCH_SPACE"
RAY_SAMPLE = {
    Float: {
        Float._Uniform: uniform.__name__,
        Float._Normal: randn.__name__,
        Float._LogUniform: loguniform.__name__,
    },
    Integer: {
        Integer._Uniform: randint.__name__,
        Integer._LogUniform: lograndint.__name__,
    },
    Categorical: {
        Categorical._Uniform: choice.__name__
    }
}


class SearchSpaceConfiger:
    """
    SearchSpaceConfiger is for getting the default search space for the paddlets transformer, paddlets model, or paddlets pipeline used
    by automl.

    """

    def get_default_search_space(self, paddlets_estimator):
        """

        Args:
            paddlets_estimator: A class(or str) of a paddlets model or a list of classes(or str) consisting of several paddlets
                transformers and a paddlets model.

        Returns:
            dict: The domain of the automl to be optimized.

        """
        if hasattr(paddlets_estimator, "__mro__") and BaseTransform in paddlets_estimator.__mro__:
            for class_name, config in self.paddlets_default_search_space["transform"].items():
                if paddlets_estimator.__name__ == class_name:
                    return config
        elif hasattr(paddlets_estimator, "__mro__") and PaddleBaseModel in paddlets_estimator.__mro__:
            for class_name, config in self.paddlets_default_search_space["models"]["dl"]["paddlepaddle"].items():
                if paddlets_estimator.__name__ == class_name:
                    return config
        elif hasattr(paddlets_estimator, "__mro__") and MLBaseModel in paddlets_estimator.__mro__:
            for class_name, config in self.paddlets_default_search_space["models"]["ml"].items():
                if paddlets_estimator.__name__ == class_name:
                    return config
        elif isinstance(paddlets_estimator, Pipeline):
            config_dict = {}
            for step in paddlets_estimator.steps:
                e = step[0]
                founded = False
                estimator_index = 0
                for class_name, config in {**self.paddlets_default_search_space["transform"],
                                           **self.paddlets_default_search_space["models"]["dl"]["paddlepaddle"],
                                           **self.paddlets_default_search_space["models"]["ml"]}.items():
                    if e.__name__ == class_name:
                        config_dict[e.__name__ + "-" + str(estimator_index)] = config
                        founded = True
                        break
                if founded is False:
                    config_dict[e.__name__ + "-" + str(estimator_index)] = {}
                estimator_index = estimator_index + 1
            if self._sp_empty(config_dict):
                raise NotImplementedError(f"search space is empty, sp: {self.search_space_to_str(config_dict)}")
            return config_dict
        elif isinstance(paddlets_estimator, str):
            for class_name, config in {**self.paddlets_default_search_space["transform"],
                                       **self.paddlets_default_search_space["models"]["dl"]["paddlepaddle"],
                                       **self.paddlets_default_search_space["models"]["ml"]}.items():
                if paddlets_estimator == class_name:
                    return config
        elif isinstance(paddlets_estimator, list):
            config_dict = {}
            estimator_index = 0
            for e in paddlets_estimator:
                founded = False
                for class_name, config in {**self.paddlets_default_search_space["transform"],
                                           **self.paddlets_default_search_space["models"]["dl"]["paddlepaddle"],
                                           **self.paddlets_default_search_space["models"]["ml"]}.items():
                    if isinstance(e, str) and e == class_name:
                        config_dict[e + "-" + str(estimator_index)] = config
                        founded = True
                        break
                    elif not isinstance(e, str) and e.__name__ == class_name:
                        config_dict[e.__name__ + "-" + str(estimator_index)] = config
                        founded = True
                        break
                if founded is False:
                    if isinstance(e, str):
                        config_dict[e + "-" + str(estimator_index)] = {}
                    else:
                        config_dict[e.__name__ + "-" + str(estimator_index)] = {}
                estimator_index = estimator_index + 1
            if self._sp_empty(config_dict):
                raise NotImplementedError(f"search space is empty, sp: {self.search_space_to_str(config_dict)}")
            return config_dict
        # paddlets_estimator is unknown type
        raise NotImplementedError("Unknown estimator")

    def recommend(self, estimator, verbose=True):
        """
        Recommend a search space for the paddlets estimator.

        Args:
            estimator: A class(or str) of a paddlets model or a list of classes(or str) consisting of several paddlets
                transformers and a paddlets model.

        Returns:
            str: Search space in form of str

        """
        recommended_sp = self.get_default_search_space(estimator)
        recommended_sp = self.search_space_to_str(recommended_sp)
        res = "The recommended search space are as follows: \n" \
              "=======================================================\n" \
              "from ray.tune import uniform, quniform, loguniform, qloguniform, " \
              "randn, qrandn, randint, qrandint, lograndint, qlograndint, choice\n" \
              "recommended_sp = " \
              + recommended_sp + \
              "\n=====================================================\n" \
              "Please note that the **USER_DEFINED_SEARCH_SPACE** " \
              "parameters need to be set by the user\n"
        if verbose:
            logger.info(res)
        return res

    def search_space_to_str(self, search_space):
        """
        Convert search space to string

        Args:
            search_space: A class(or str) of a paddlets model or a list of classes(or str) consisting of several paddlets
                transformers and a paddlets model.

        Returns:
            str: Search space in form of str

        """
        res = copy.deepcopy(search_space)
        self._dfs_search_space_to_str(res)
        return self._to_pretty_str(res)

    def _dfs_search_space_to_str(self, search_space):
        """

        _dfs_search_space_to_str

        """
        for e, sp in search_space.items():
            if isinstance(sp, dict):
                search_space[e] = self._dfs_search_space_to_str(search_space[e])
            else:
                search_space[e] = self._param_search_space_to_str(sp)
        return search_space

    def _param_search_space_to_str(self, sp):
        """

        _param_search_space_to_str

        """
        if not hasattr(sp, "sampler"):
            if isinstance(sp, str) and sp == USER_DEFINED_SEARCH_SPACE:
                return "**" + USER_DEFINED_SEARCH_SPACE + "**"
            else:
                # may throw an exception
                return f"{sp}"
        elif isinstance(sp.sampler, Quantized):
            return "q" + RAY_SAMPLE[sp.__class__][sp.sampler.sampler.__class__] \
                   + "(" + str(sp.lower) + ", " + str(sp.upper) + ", q=" + str(sp.sampler.q) + ")"
        else:
            if isinstance(sp, Categorical):
                return RAY_SAMPLE[sp.__class__][sp.sampler.__class__] + "(" + sp.domain_str + ")"
            else:
                if isinstance(sp.sampler, Normal):
                    return RAY_SAMPLE[sp.__class__][sp.sampler.__class__] + "(" + str(sp.sampler.mean) + ", " + str(
                        sp.sampler.sd) + ")"
                else:
                    return RAY_SAMPLE[sp.__class__][sp.sampler.__class__] + "(" + str(sp.lower) + ", " + str(
                        sp.upper) + ")"

    def _to_pretty_str(self, sp_dict):
        """

        _to_pretty_str

        """

        return json.dumps(sp_dict, indent="\t").replace('",', ",").replace(': "', ": ").replace('"\n', '\n')

    @property
    def paddlets_default_search_space(self):
        """

        Default search space for paddlets

        """
        return {
            "transform": {
                "Fill": {
                    "cols": USER_DEFINED_SEARCH_SPACE,
                    "method": choice(['max', 'min', 'mean', 'median', 'pre', 'next', 'zero']),
                    "window_size": lograndint(10, 30),
                    "min_num_non_missing_values": lograndint(1, 10),
                },
                "OneHot": {
                    "cols": USER_DEFINED_SEARCH_SPACE
                },
                "Ordinal": {
                    "cols": USER_DEFINED_SEARCH_SPACE
                },
                "StatsTransform": {
                    "cols": USER_DEFINED_SEARCH_SPACE,
                    "start": 0,
                    "end": 5,
                },
                "MinMaxScaler": {
                    "col": USER_DEFINED_SEARCH_SPACE,
                    "clip": choice([True, False]),
                },
                "StandardScaler": {
                    "col": USER_DEFINED_SEARCH_SPACE,
                },
                "KSigma": {
                    "cols": USER_DEFINED_SEARCH_SPACE,
                    "k": quniform(0.5, 10, q=0.5)
                },
                "TimeFeatureGenerator": {}
            },
            "models": {
                "dl": {
                    "paddlepaddle": {
                        "MLPRegressor": {
                            "hidden_config": choice([[64], [64] * 2, [64] * 3, [128], [128] * 2, [128] * 3]),
                            "use_bn": choice([True, False]),
                            "batch_size": qrandint(8, 128, q=8),
                            "max_epochs": qrandint(30, 600, q=30),
                            "optimizer_params": {
                                "learning_rate": uniform(1e-4, 1e-2)
                            },
                            "patience": qrandint(5, 50, q=5)
                        },
                        "RNNBlockRegressor": {
                            "rnn_type_or_module": choice(["SimpleRNN", "LSTM", "GRU"]),
                            "fcn_out_config": choice([[16], [32], [64], [128], [256],
                                                      [16] * 2, [32] * 2, [64] * 2, [128] * 2, [256] * 2,
                                                      [16] * 3, [32] * 3, [64] * 3, [128] * 3,
                                                      [256] * 3]),
                            "hidden_size": qrandint(32, 512, q=32),
                            "num_layers_recurrent": randint(1, 4),
                            "dropout": quniform(0, 0.5, q=0.05),
                            "optimizer_params": {
                                "learning_rate": uniform(1e-4, 1e-2)
                            },
                            "batch_size": qrandint(8, 128, q=8),
                            "max_epochs": qrandint(30, 600, q=30),
                            "patience": qrandint(5, 50, q=5)
                        },
                        "NBEATSModel": {
                            "generic_architecture": choice([True, False]),
                            "num_stacks": randint(2, 6),
                            "num_blocks": randint(2, 6),
                            "num_layers": randint(1, 6),
                            "layer_widths": qrandint(32, 512, q=32),
                            "expansion_coefficient_dim": qrandint(32, 512, q=32),
                            "trend_polynomial_degree": randint(2, 6),
                            "optimizer_params":{
                                "learning_rate": uniform(1e-4, 1e-2)
                            },
                            "batch_size": qrandint(8, 128, q=8),
                            "max_epochs": qrandint(300, 1000, q=100),
                            "patience": qrandint(50, 100, q=5)
                        },
                        "NHiTSModel": {
                            "num_stacks": randint(2, 6),
                            "num_blocks": randint(2, 6),
                            "num_layers": randint(2, 6),
                            "layer_widths": 512,
                            "batch_norm": choice([True, False]),
                            "dropout": quniform(0, 0.5, 0.05),
                            "activation": choice(
                                ["ReLU", "PReLU", "ELU", "Softplus", "Tanh", "SELU", "LeakyReLU", "Sigmoid",
                                 "GELU"]),
                            "MaxPool1d": choice([True, False]),
                            "optimizer_params": {
                                "learning_rate": uniform(1e-4, 1e-2)
                            },
                            "batch_size": qrandint(8, 128, q=8),
                            "max_epochs": qrandint(30, 600, q=30),
                            "verbose": 1,
                            "patience": qrandint(5, 50, q=5)
                        },
                        "LSTNetRegressor": {
                            "skip_size": 1,
                            "channels": choice([1, 2, 4, 8, 16, 32, 64]),
                            "kernel_size": choice([1, 3, 7]),
                            "rnn_cell_type": choice(["GRU", "LSTM"]),
                            "rnn_num_cells": choice([1, 2, 4, 8, 16, 32, 64]),
                            "skip_rnn_cell_type": choice(["GRU", "LSTM"]),
                            "skip_rnn_num_cells": choice([1, 2, 4, 8, 16, 32, 64]),
                            "dropout_rate": quniform(0, 0.5, 0.05),
                            "output_activation": None,
                            "batch_size": qrandint(8, 128, q=8),
                            "max_epochs": qrandint(30, 600, q=30),
                            "optimizer_params": {
                                "learning_rate": uniform(1e-4, 1e-2)
                            },
                            "patience": qrandint(5, 50, q=5)
                        },
                        "TransformerModel": {
                            "nhead": choice([1, 2, 4, 8]),
                            "num_encoder_layers": randint(1, 11),
                            "num_decoder_layers": randint(1, 11),
                            "dim_feedforward": qrandint(32, 512, q=32),
                            "activation": choice(["relu", "gelu"]),
                            "dropout_rate": quniform(0, 0.5, q=0.05),
                            "d_model": qrandint(32, 512, q=32),
                            "batch_size": qrandint(8, 128, q=8),
                            "max_epochs": qrandint(30, 600, q=30),
                            "optimizer_params": {
                                "learning_rate": uniform(1e-4, 1e-2)
                            },
                            "patience": qrandint(5, 50, q=5)
                        },
                        "TCNRegressor": {
                            "hidden_config": choice([[64], [64] * 2, [64] * 3, [128] * 2, [128] * 3,
                                                     [8] * 3, [8] * 5, [8] * 7,
                                                     [16] * 3, [16] * 5, [16] * 7]),
                            "kernel_size": choice([3, 5, 7]),
                            "dropout_rate": quniform(0, 0.5, 0.05),
                            "batch_size": qrandint(8, 128, q=8),
                            "max_epochs": qrandint(300, 1500, q=100),
                            "optimizer_params": {
                                "learning_rate": uniform(1e-4, 1e-2)
                            },
                            "patience": qrandint(50, 150, q=10)
                        },
                        "InformerModel": {
                            "nhead": choice([1, 2, 4, 8]),
                            "num_encoder_layers": randint(1, 11),
                            "num_decoder_layers": randint(1, 11),

                            "activation": choice(["relu", "gelu"]),
                            "dropout_rate": quniform(0, 0.5, q=0.05),
                            "d_model": qrandint(32, 512, q=32),
                            "batch_size": qrandint(8, 128, q=8),
                            "max_epochs": qrandint(30, 600, q=30),
                            "optimizer_params": {
                                "learning_rate": uniform(1e-4, 1e-2)
                            },
                            "patience": qrandint(5, 50, q=5)
                        },
                        "DeepARModel": {
                            "rnn_type_or_module": choice(["LSTM", "GRU"]),
                            "fcn_out_config": choice([[16], [32], [64], [128], [256],
                                                      [16] * 2, [32] * 2, [64] * 2, [128] * 2, [256] * 2,
                                                      [16] * 3, [32] * 3, [64] * 3, [128] * 3,
                                                      [256] * 3]),
                            "hidden_size": qrandint(32, 512, q=32),
                            "num_layers_recurrent": randint(1, 4),
                            "dropout": quniform(0, 0.5, q=0.05),
                            "optimizer_params": {"learning_rate": uniform(1e-4, 1e-2)},
                            "batch_size": qrandint(8, 128, q=8),
                            "max_epochs": qrandint(30, 600, q=30),
                            "patience": qrandint(5, 50, q=5)
                        }
                    },
                    "pytorch": {}
                },
                "ml": {
                    "LGBM": {
                        "num_boost_round": randint(10, 2000),
                        "early_stopping_rounds": 0,
                        "params": {
                            "boosting": choice(["gbdt", "rf", "dart"]),
                            "objective": "regression",
                            "metric": choice(["mse", "mae"]),
                            "learning_rate": loguniform(1e-4, 0.1),
                            "lambda_l1": quniform(0, 0.3, 0.05),
                            "lambda_l2": quniform(0, 0.3, 0.05),
                            "num_leaves": randint(15, 255),
                            "max_depth": -1,
                            "bagging_freq": randint(1, 6),
                            "bagging_fraction": quniform(0.3, 0.95, 0.05),
                            "feature_fraction": quniform(0.3, 1, 0.1),
                            "min_data_in_leaf": randint(1, 32),
                            "verbose": -1,
                            "num_threads": 1,
                            "seed": 28,
                        },
                    },
                    "ArimaModel": {
                        "p": 0,
                        "d": 0,
                        "q": 1,
                        "trend": choice(["c", "nc"]),
                    }
                }
            }
        }

    def _sp_empty(self, sp_dict):
        """

        Check if the search space for a pipeline is empty

        """
        for k, v in sp_dict.items():
            if bool(v):
                return False
        return True
