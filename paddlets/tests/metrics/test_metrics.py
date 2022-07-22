# !/usr/bin/env python3
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np

import unittest
from unittest import TestCase

from paddlets import TimeSeries, TSDataset
from paddlets.metrics import (
    MetricContainer,
    LogLoss,
    Metric,
    MSE,
    MAE,
)


class TestMetrics(TestCase):
    def setUp(self):
        """unittest function
        """
        super().setUp()

    def test_MSE(self):
        """unittest function
        """
        # case1
        mse = MSE()
        y_true = np.random.randn(5)
        y_score = y_true.copy()
        res = mse.metric_fn(y_true, y_score)
        self.assertEqual(res, 0.)

        # case2
        periods = 100
        df = pd.DataFrame(
            [1 for i in range(periods)],
            index = pd.date_range("2022-01-01", periods=periods, freq="1D"),
            columns=["target"]
        )
        ts = TSDataset.load_from_dataframe(df, target_cols="target")
        ts2 = ts.copy()
        ts2["target"] = ts["target"] + 1
        self.assertEqual(mse(ts, ts2), {"target": 1.0})

        # case3
        ts.set_column("target2", ts["target"] + 1, "target")
        ts2 = ts.copy()
        ts2["target"] = ts["target"] + 1
        ts2["target2"] = ts["target"] + 1
        self.assertEqual(mse(ts, ts2), {"target": 1., "target2": 0.})

    def test_MAE(self):
        """unittest function
        """
        # case1
        mae = MAE()
        y_true = np.random.randn(5)
        y_score = y_true.copy()
        res = mae.metric_fn(y_true, y_score)
        self.assertEqual(res, 0.)

        # case2
        periods = 100
        df = pd.DataFrame(
            [1 for i in range(periods)],
            index = pd.date_range("2022-01-01", periods=periods, freq="1D"),
            columns=["target"]
        )
        ts = TSDataset.load_from_dataframe(df, target_cols="target")
        ts2 = ts.copy()
        ts2["target"] = ts["target"] + 1
        self.assertEqual(mae(ts, ts2), {"target": 1.0})

        # case3
        ts.set_column("target2", ts["target"] + 1, "target")
        ts2 = ts.copy()
        ts2["target"] = ts["target"] + 1
        ts2["target2"] = ts["target"] + 1
        self.assertEqual(mae(ts, ts2), {"target": 1., "target2": 0.})

    def test_LogLoss(self):
        """unittest function
        """
        # case1
        logloss = LogLoss()
        y_true = [1, 0, 1, 1]
        y_score = [0.9, 0.1, 0.8, 0.8]
        res = logloss.metric_fn(y_true, y_score)
        self.assertAlmostEqual(res, 0.16425, delta=1e-5)

        # # case2
        df = pd.DataFrame(
            [1, 0, 1, 1],
            index = pd.date_range("2022-01-01", periods=4, freq="1D"),
            columns=["target"]
        )
        df2 = pd.DataFrame(
            [0.9, 0.1, 0.8, 0.8],
            index = pd.date_range("2022-01-01", periods=4, freq="1D"),
            columns=["target"]
        )
        ts = TSDataset.load_from_dataframe(df, target_cols="target")
        ts2 = TSDataset.load_from_dataframe(df2, target_cols="target")
        expect_output = {"target": 0.16425}
        ret = logloss(ts, ts2)
        for schema in ret:
            self.assertAlmostEqual(ret[schema], expect_output[schema], delta=1e-5)

        # case3
        ts.set_column("target2", 1 - ts["target"], "target")
        ts2 = ts2.copy()
        ts2.set_column("target2", ts2["target"] + 0.1, "target")
        expect_output = {"target": 0.16425, "target2": 10.18854}
        ret = logloss(ts, ts2)
        for schema in ret:
            self.assertAlmostEqual(ret[schema], expect_output[schema], delta=1e-5)

    def test_get_metrics_by_names(self):
        """unittest function
        """
        # case1
        fake_input = ["mse", "mae"]
        expect_output = [MSE(), MAE()]
        expect_output = [obj._NAME for obj in expect_output]
        ret = Metric.get_metrics_by_names(fake_input)
        ret = [obj._NAME for obj in ret]
        self.assertEqual(ret, expect_output)
        
        # case2
        fake_input = ["mse", "mape"]
        with self.assertRaises(AssertionError):
            Metric.get_metrics_by_names(fake_input)


class TestMetricContainer(TestCase):
    def setUp(self):
        """unittest function
        """
        super().setUp()

    def test_call(self):
        """unittest function
        """
        # case1
        contrainer = MetricContainer(
            metric_names=["mse", "mae"],
            prefix="val_"
        )
        fake_y_true = np.random.randn(5)
        fake_y_score = fake_y_true.copy()
        ret = contrainer(fake_y_true, fake_y_score)
        expect_output = {"val_mse": 0., "val_mae": 0.}
        for schema in ret:
            self.assertAlmostEqual(ret[schema], expect_output[schema], delta=1e-5)

        # case2
        fake_y_score = fake_y_true + 1.
        ret = contrainer(fake_y_true, fake_y_score)
        expect_output = {"val_mse": 1., "val_mae": 1.}
        for schema in ret:
            self.assertAlmostEqual(ret[schema], expect_output[schema], delta=1e-5)

'''
class TestCheckMetrics(TestCase):
    def setUp(self):
        """unittest function
        """
        super().setUp()

    def test_check_metrics(self):
        """unittest function
        """
        # case1
        fake_input = ["mse", "mae"]
        expect_output = ["mse", "mae"]
        ret = check_metrics(fake_input)
        self.assertEqual(expect_output, ret)

        # case2
        fake_input = ["mse", MAE()]
        expect_output = ["mse", "mae"]
        ret = check_metrics(fake_input)
        self.assertEqual(expect_output, ret)

        # case3
        fake_input = ["mse", 5]
        with self.assertRaises(TypeError):
            check_metrics(fake_input)
'''


if __name__ == "__main__":
    unittest.main()

