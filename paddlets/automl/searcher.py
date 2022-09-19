# !/usr/bin/env python3
# -*- coding:utf-8 -*-

from ray.tune.suggest import BasicVariantGenerator
from ray.tune.suggest.optuna import OptunaSearch
from ray.tune.suggest.flaml import CFO
from ray.tune.suggest.flaml import BlendSearch
from ray.tune.suggest.bohb import TuneBOHB
from optuna.samplers import CmaEsSampler
from optuna.samplers import TPESampler
from ray.tune.suggest import ConcurrencyLimiter

MAX_CONCURRENT = 4

class Searcher:
    """

    Searcher is for getting suggesting algorithms for automl.

    """

    @classmethod
    def get_supported_algs(cls):
        """

        Returns:
            list: A list of supported algorithms.

        """
        return ["Random", "CMAES", "TPE", "CFO", "BlendSearch", "Bayes"]

    @classmethod
    def get_searcher(cls, search_algo):
        """
        Get a searcher by string

        Args:
            search_algo: The algorithm for optimization.
                Supported algorithms are "Random", "CMAES", "TPE", "CFO", "BlendSearch", "Bayes".

        Returns:
            Searcher: An object of the suggesting algorithm.

        """
        if search_algo == "Random":
            algo = BasicVariantGenerator(max_concurrent=MAX_CONCURRENT)
        else:
            if search_algo == "CMAES":
                algo = OptunaSearch(sampler=CmaEsSampler())
            elif search_algo == "TPE":
                algo = OptunaSearch(sampler=TPESampler())
            elif search_algo == "CFO":
                algo = CFO()
            elif search_algo == "BlendSearch":
                algo = BlendSearch()
            elif search_algo == "Bayes":
                algo = TuneBOHB()
            else:
                raise NotImplementedError("Unknown searcher")
            algo = ConcurrencyLimiter(algo, max_concurrent=MAX_CONCURRENT)

        return algo
