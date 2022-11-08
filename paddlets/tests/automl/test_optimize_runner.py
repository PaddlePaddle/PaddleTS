# !/usr/bin/env python3
# -*- coding:utf-8 -*-

import unittest
from unittest import TestCase

import pandas as pd
import numpy as np

from paddlets.datasets.tsdataset import TimeSeries, TSDataset
from paddlets.automl.optimize_runner import OptimizeRunner
from paddlets.transform.fill import Fill
from paddlets.models.forecasting import MLPRegressor


class TestOptimizeRunner(TestCase):
    def setUp(self):
        """
        unittest function
        """
        super().setUp()

    def test_optimize(self):
        """
        unittest function
        """

        np.random.seed(2022)
        target1 = pd.Series(
            np.random.randn(200).astype(np.float32),
            index=pd.date_range("2022-01-01", periods=200, freq="15T"),
            name="target_test")
        observed_cov = pd.DataFrame(
            np.random.randn(200, 2).astype(np.float32),
            index=pd.date_range("2022-01-01", periods=200, freq="15T"),
            columns=["b", "c"])
        known_cov = pd.DataFrame(
            np.random.randn(200, 2).astype(np.float32),
            index=pd.date_range("2022-01-01", periods=200, freq="15T"),
            columns=["b1", "c1"])
        static_cov = {"f": 1., "g": 2.}

        # index为DatetimeIndex类型
        tsdataset = TSDataset(
            TimeSeries.load_from_dataframe(target1),
            TimeSeries.load_from_dataframe(observed_cov),
            TimeSeries.load_from_dataframe(known_cov),
            static_cov)

        optimize_runner = OptimizeRunner(search_alg="Random")
        analysis = optimize_runner.optimize(MLPRegressor,
                                            10,
                                            4,
                                            tsdataset,
                                            n_trials=1
                                            )
        best_trial = analysis.best_trial
        dfs = analysis.trial_dataframes

        # index为DatetimeIndex类型
        tsdataset = TSDataset(
            TimeSeries.load_from_dataframe(target1),
            TimeSeries.load_from_dataframe(observed_cov),
            TimeSeries.load_from_dataframe(known_cov),
            static_cov)

        optimize_runner = OptimizeRunner(search_alg="CFO")
        analysis = optimize_runner.optimize(MLPRegressor,
                                            10,
                                            4,
                                            tsdataset,
                                            n_trials=1
        )
        best_trial = analysis.best_trial
        dfs = analysis.trial_dataframes

        # index为DatetimeIndex类型
        tsdataset = TSDataset(
            TimeSeries.load_from_dataframe(target1),
            TimeSeries.load_from_dataframe(observed_cov),
            TimeSeries.load_from_dataframe(known_cov),
            static_cov)

        optimize_runner = OptimizeRunner(search_alg="BlendSearch")
        analysis = optimize_runner.optimize(MLPRegressor,
                                            10,
                                            4,
                                            tsdataset,
                                            n_trials=1
        )
        best_trial = analysis.best_trial
        dfs = analysis.trial_dataframes

        #cv
        tsdataset = TSDataset(
            TimeSeries.load_from_dataframe(target1),
            TimeSeries.load_from_dataframe(observed_cov),
            TimeSeries.load_from_dataframe(known_cov),
            static_cov)

        optimize_runner = OptimizeRunner(search_alg="TPE")
        analysis = optimize_runner.optimize(MLPRegressor,
                                            10,
                                            4,
                                            tsdataset,
                                            resampling_strategy="cv",
                                            n_trials=1
        )
        best_trial = analysis.best_trial
        dfs = analysis.trial_dataframes

        #specify valid
        tsdataset = TSDataset(
            TimeSeries.load_from_dataframe(target1),
            TimeSeries.load_from_dataframe(observed_cov),
            TimeSeries.load_from_dataframe(known_cov),
            static_cov)
        ts1, ts2 = tsdataset.split(0.5)
        optimize_runner = OptimizeRunner(search_alg="CMAES")
        analysis = optimize_runner.optimize(MLPRegressor,
                                            10,
                                            4,
                                            ts1,
                                            valid_tsdataset= ts2,
                                            n_trials=1
        )
        best_trial = analysis.best_trial
        dfs = analysis.trial_dataframes

        # Pipeline
        optimize_runner = OptimizeRunner(search_alg="Bayes")
        from ray.tune import uniform, qrandint, choice
        sp = {
            "Fill": {
                "cols": ['b', 'b1'],
                "method": choice(['max', 'min', 'mean', 'median', 'pre', 'next', 'zero']),
                "value": uniform(0.1, 0.9),
                "window_size": qrandint(20, 50, q=1)
            },
            "MLPRegressor": {
                "batch_size": qrandint(16, 64, q=16),
                "use_bn": choice([True, False]),
                "max_epochs": qrandint(10, 50, q=10)
            }
        }
        analysis = optimize_runner.optimize([Fill, MLPRegressor],
                                            10,
                                            4,
                                            tsdataset,
                                            search_space=sp,
                                            n_trials=1
                                            )
        best_trial = analysis.best_trial
        dfs = analysis.trial_dataframes

if __name__ == "__main__":
    unittest.main()
