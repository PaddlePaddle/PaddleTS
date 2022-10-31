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
    QuantileLoss,
    ACC,
    Precision,
    Recall,
    F1
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

    def test_prob(self):
        periods = 100
        mse_prob = MSE("prob")
        df1 = pd.DataFrame(
            np.ones([periods, 2]),
            index = pd.date_range("2022-01-01", periods=periods, freq="1D"),
            columns=["target", "target_2" ]
        )

        ts1 = TSDataset.load_from_dataframe(df1)
        df2 = pd.DataFrame(
            np.zeros([periods, 4]),
            index = pd.date_range("2022-01-01", periods=periods, freq="1D"),
            columns=["target@1", "target@2", "target_2@1", "target_2@2"]
        )
        ts2 = TSDataset.load_from_dataframe(df2)
        mse_prob = MSE("prob")
        self.assertEqual(mse_prob(ts1, ts2), {"target": 1., "target_2": 1.})
        mae_prob = MAE("prob")
        self.assertEqual(mae_prob(ts1, ts2), {"target": 1., "target_2": 1.})
        q_loss = QuantileLoss()
        self.assertEqual(q_loss(ts1, ts2), {"target": 1., "target_2": 1.})
        
    def test_ACC(self):
        """unittest function
        """
        # case1
        acc = ACC()
        y_true = np.random.randint(0,2,5)
        y_score = y_true.copy()
        res = acc.metric_fn(y_true, y_score)
        self.assertEqual(res, 1.)
        
        #case2 
        y_true = np.array([1, 1, 0, 0, 0])
        y_score = np.array([1, 0, 1, 0, 0])
        res = acc.metric_fn(y_true, y_score)
        self.assertEqual(res, 0.6)

        # case3
        periods = 5
        df = pd.DataFrame(
            y_true, index = pd.date_range("2022-01-01", periods=periods, freq="1D"),
            columns=["label"]
        )
        ts = TSDataset.load_from_dataframe(df, label_col="label")
        ts2 = ts.copy()
        self.assertEqual(acc(ts, ts2), {"label": 1.0})
        
        # case4
        periods = 4
        df = pd.DataFrame(
            y_true[1:5], index = pd.date_range("2022-01-02", periods=periods, freq="1D"),
            columns=["label"]
        )
        ts2 = TSDataset.load_from_dataframe(df, label_col="label")
        self.assertEqual(acc(ts, ts2), {"label": 1.0})
        
        # case5
        periods = 10
        y_true = np.random.randint(0,2,10)
        df = pd.DataFrame(
            y_true, index = pd.date_range("2022-01-02", periods=periods, freq="1D"),
            columns=["label"]
        )
        message = ''
        succeed = True
        ts2 = TSDataset.load_from_dataframe(df, label_col="label")
        try:
            self.assertEqual(acc(ts, ts2), {"label": 1.0})
        except Exception as e:
            succeed = False
            message = e
        self.assertFalse(succeed)
        self.assertEqual(
            str(message),
            "In `anomaly` mode, the length of the true must be greater than or equal to the length of the pred!"
        )
        
        # case6
        periods = 5
        y_true = np.random.randint(2,4,5)
        df = pd.DataFrame(
            y_true, index = pd.date_range("2022-01-01", periods=periods, freq="1D"),
            columns=["label"]
        )
        message = ''
        succeed = True
        ts = TSDataset.load_from_dataframe(df, label_col="label")
        ts2 = ts.copy()
        try:
            self.assertEqual(acc(ts, ts2), {"label": 1.0})
        except Exception as e:
            succeed = False
            message = e
        self.assertFalse(succeed)
        self.assertEqual(
           str(message),
           "In `anomaly` mode, the value in true label must be 0 or 1, please check your data!"
        )
        
        # case7
        y_true = np.random.randint(0,2,5)
        index = pd.RangeIndex(0, 5, 1)
        periods = 5
        df = pd.DataFrame(
            y_true, index = pd.date_range("2022-01-01", periods=periods, freq="1D"),
            columns=["label"]
        )
        df = df.reset_index(drop=True).reindex(index)
        ts = TSDataset.load_from_dataframe(df, label_col="label")
        ts2 = ts.copy()
        self.assertEqual(acc(ts, ts2), {"label": 1.0})
        
    def test_Precision(self):
        """unittest function
        """
        # case1
        precision = Precision()
        y_true = np.random.randint(0,2,5)
        y_score = y_true.copy()
        res = precision.metric_fn(y_true, y_score)
        self.assertEqual(res, 1.)
        
        #case2 
        y_true = np.array([1, 1, 0, 0, 0])
        y_score = np.array([1, 0, 1, 0, 0])
        res = precision.metric_fn(y_true, y_score)
        self.assertEqual(res, 0.5)

        # case3
        periods = 5
        df = pd.DataFrame(
            y_true, index = pd.date_range("2022-01-01", periods=periods, freq="1D"),
            columns=["label"]
        )
        ts = TSDataset.load_from_dataframe(df, label_col="label")
        ts2 = ts.copy()
        self.assertEqual(precision(ts, ts2), {"label": 1.0})
        
        # case4
        periods = 4
        df = pd.DataFrame(
            y_true[1:5], index = pd.date_range("2022-01-02", periods=periods, freq="1D"),
            columns=["label"]
        )
        ts2 = TSDataset.load_from_dataframe(df, label_col="label")
        self.assertEqual(precision(ts, ts2), {"label": 1.0})
        
        # case5
        periods = 10
        y_true = np.random.randint(0,2,10)
        df = pd.DataFrame(
            y_true, index = pd.date_range("2022-01-02", periods=periods, freq="1D"),
            columns=["label"]
        )
        message = ''
        succeed = True
        ts2 = TSDataset.load_from_dataframe(df, label_col="label")
        try:
            self.assertEqual(precision(ts, ts2), {"label": 1.0})
        except Exception as e:
            succeed = False
            message = e
        self.assertFalse(succeed)
        self.assertEqual(
            str(message),
            "In `anomaly` mode, the length of the true must be greater than or equal to the length of the pred!"
        )
        
        # case6
        periods = 5
        y_true = np.random.randint(2,4,5)
        df = pd.DataFrame(
            y_true, index = pd.date_range("2022-01-01", periods=periods, freq="1D"),
            columns=["label"]
        )
        message = ''
        succeed = True
        ts = TSDataset.load_from_dataframe(df, label_col="label")
        ts2 = ts.copy()
        try:
            self.assertEqual(precision(ts, ts2), {"label": 1.0})
        except Exception as e:
            succeed = False
            message = e
        self.assertFalse(succeed)
        self.assertEqual(
           str(message),
           "In `anomaly` mode, the value in true label must be 0 or 1, please check your data!"
        )
        
        # case7
        y_true = np.random.randint(0,2,5)
        index = pd.RangeIndex(0, 5, 1)
        periods = 5
        df = pd.DataFrame(
            y_true, index = pd.date_range("2022-01-01", periods=periods, freq="1D"),
            columns=["label"]
        )
        df = df.reset_index(drop=True).reindex(index)
        ts = TSDataset.load_from_dataframe(df, label_col="label")
        ts2 = ts.copy()
        self.assertEqual(precision(ts, ts2), {"label": 1.0})
        
    def test_Recall(self):
        """unittest function
        """
        # case1
        recall = Recall()
        y_true = np.random.randint(0,2,5)
        y_score = y_true.copy()
        res = recall.metric_fn(y_true, y_score)
        self.assertEqual(res, 1.)
        
        #case2 
        y_true = np.array([1, 1, 0, 0, 0])
        y_score = np.array([1, 0, 1, 0, 0])
        res = recall.metric_fn(y_true, y_score)
        self.assertEqual(res, 0.5)

        # case3
        periods = 5
        df = pd.DataFrame(
            y_true, index = pd.date_range("2022-01-01", periods=periods, freq="1D"),
            columns=["label"]
        )
        ts = TSDataset.load_from_dataframe(df, label_col="label")
        ts2 = ts.copy()
        self.assertEqual(recall(ts, ts2), {"label": 1.0})
        
        # case4
        periods = 4
        df = pd.DataFrame(
            y_true[1:5], index = pd.date_range("2022-01-02", periods=periods, freq="1D"),
            columns=["label"]
        )
        ts2 = TSDataset.load_from_dataframe(df, label_col="label")
        self.assertEqual(recall(ts, ts2), {"label": 1.0})
        
        # case5
        periods = 10
        y_true = np.random.randint(0,2,10)
        df = pd.DataFrame(
            y_true, index = pd.date_range("2022-01-02", periods=periods, freq="1D"),
            columns=["label"]
        )
        message = ''
        succeed = True
        ts2 = TSDataset.load_from_dataframe(df, label_col="label")
        try:
            self.assertEqual(recall(ts, ts2), {"label": 1.0})
        except Exception as e:
            succeed = False
            message = e
        self.assertFalse(succeed)
        self.assertEqual(
            str(message),
            "In `anomaly` mode, the length of the true must be greater than or equal to the length of the pred!"
        )
        
        # case6
        periods = 5
        y_true = np.random.randint(2,4,5)
        df = pd.DataFrame(
            y_true, index = pd.date_range("2022-01-01", periods=periods, freq="1D"),
            columns=["label"]
        )
        message = ''
        succeed = True
        ts = TSDataset.load_from_dataframe(df, label_col="label")
        ts2 = ts.copy()
        try:
            self.assertEqual(recall(ts, ts2), {"label": 1.0})
        except Exception as e:
            succeed = False
            message = e
        self.assertFalse(succeed)
        self.assertEqual(
           str(message),
           "In `anomaly` mode, the value in true label must be 0 or 1, please check your data!"
        )
        
        # case7
        y_true = np.random.randint(0,2,5)
        index = pd.RangeIndex(0, 5, 1)
        periods = 5
        df = pd.DataFrame(
            y_true, index = pd.date_range("2022-01-01", periods=periods, freq="1D"),
            columns=["label"]
        )
        df = df.reset_index(drop=True).reindex(index)
        ts = TSDataset.load_from_dataframe(df, label_col="label")
        ts2 = ts.copy()
        self.assertEqual(recall(ts, ts2), {"label": 1.0})
        
    def test_F1(self):
        """unittest function
        """
        # case1
        f1 = F1()
        y_true = np.random.randint(0,2,5)
        y_score = y_true.copy()
        res = f1.metric_fn(y_true, y_score)
        self.assertEqual(res, 1.)
        
        #case2 
        y_true = np.array([1, 1, 0, 0, 0])
        y_score = np.array([1, 0, 1, 0, 0])
        res = f1.metric_fn(y_true, y_score)
        self.assertEqual(res, 0.5)

        # case3
        periods = 5
        df = pd.DataFrame(
            y_true, index = pd.date_range("2022-01-01", periods=periods, freq="1D"),
            columns=["label"]
        )
        ts = TSDataset.load_from_dataframe(df, label_col="label")
        ts2 = ts.copy()
        self.assertEqual(f1(ts, ts2), {"label": 1.0})
  
        # case4
        periods = 4
        df = pd.DataFrame(
            y_true[1:5], index = pd.date_range("2022-01-02", periods=periods, freq="1D"),
            columns=["label"]
        )
        ts3 = TSDataset.load_from_dataframe(df, label_col="label")
        self.assertEqual(f1(ts, ts2), {"label": 1.0})
        
        # case5
        periods = 10
        y_true = np.random.randint(0,2,10)
        df = pd.DataFrame(
            y_true, index = pd.date_range("2022-01-02", periods=periods, freq="1D"),
            columns=["label"]
        )
        message = ''
        succeed = True
        ts2 = TSDataset.load_from_dataframe(df, label_col="label")
        try:
            self.assertEqual(f1(ts, ts2), {"label": 1.0})
        except Exception as e:
            succeed = False
            message = e
        self.assertFalse(succeed)
        self.assertEqual(
            str(message),
            "In `anomaly` mode, the length of the true must be greater than or equal to the length of the pred!"
        )
        
        # case6
        periods = 5
        y_true = np.random.randint(2,4,5)
        df = pd.DataFrame(
            y_true, index = pd.date_range("2022-01-01", periods=periods, freq="1D"),
            columns=["label"]
        )
        message = ''
        succeed = True
        ts = TSDataset.load_from_dataframe(df, label_col="label")
        ts2 = ts.copy()
        try:
            self.assertEqual(f1(ts, ts2), {"label": 1.0})
        except Exception as e:
            succeed = False
            message = e
        self.assertFalse(succeed)
        self.assertEqual(
           str(message),
           "In `anomaly` mode, the value in true label must be 0 or 1, please check your data!"
        )

        # case7
        y_true = np.random.randint(0,2,5)
        index = pd.RangeIndex(0, 5, 1)
        periods = 5
        df = pd.DataFrame(
            y_true, index = pd.date_range("2022-01-01", periods=periods, freq="1D"),
            columns=["label"]
        )
        df = df.reset_index(drop=True).reindex(index)
        ts = TSDataset.load_from_dataframe(df, label_col="label")
        ts2 = ts.copy()
        self.assertEqual(f1(ts, ts2), {"label": 1.0})

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
