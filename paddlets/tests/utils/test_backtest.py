# !/usr/bin/env python3
# -*- coding:utf-8 -*-
import sys

sys.path.append(".")
from typing import List
from unittest import TestCase
import unittest
import random

import pandas as pd
import numpy as np

from paddlets.models.forecasting import LSTNetRegressor
from paddlets.datasets import TimeSeries, TSDataset
from paddlets.utils.backtest import backtest
from paddlets.metrics import Metric, MAE


class TestBacktest(TestCase):
    def setUp(self):
        """unittest function
        """
        np.random.seed(2022)
        target1 = TimeSeries.load_from_dataframe(
            pd.Series(
                np.random.randn(2000).astype(np.float32),
                index=pd.date_range("2022-01-01", periods=2000, freq="15T"),
                name="a"
            ))
        target2 = TimeSeries.load_from_dataframe(
            pd.DataFrame(
                np.random.randn(2000, 2).astype(np.float32),
                index=pd.date_range("2022-01-01", periods=2000, freq="15T"),
                columns=["a1", "a2"]
            ))
        observed_cov = TimeSeries.load_from_dataframe(
            pd.DataFrame(
                np.random.randn(2000, 2).astype(np.float32),
                index=pd.date_range("2022-01-01", periods=2000, freq="15T"),
                columns=["b", "c"]
            ))
        known_cov = TimeSeries.load_from_dataframe(
            pd.DataFrame(
                np.random.randn(2500, 2).astype(np.float32),
                index=pd.date_range("2022-01-01", periods=2500, freq="15T"),
                columns=["b1", "c1"]
            ))
        static_cov = {"f": 1.0, "g": 2.0}
        self.tsdataset1 = TSDataset(target1, observed_cov, known_cov, static_cov)
        self.tsdataset2 = TSDataset(target2, observed_cov, known_cov, static_cov)
        super().setUp()

    def test_backtest(self):
        """unittest function
        """
        # case1 default
        lstnet = LSTNetRegressor(
            in_chunk_len=1 * 96 + 20 * 4,
            out_chunk_len=96,
            max_epochs=1
        )
        lstnet.fit(self.tsdataset1, self.tsdataset1)
        res = backtest(self.tsdataset1, lstnet)
        assert res != 0

        # case2 add window,stride, window = stride
        lstnet = LSTNetRegressor(
            in_chunk_len=1 * 96 + 20 * 4,
            out_chunk_len=96,
            max_epochs=1
        )
        lstnet.fit(self.tsdataset1, self.tsdataset1)
        score, predicts = backtest(self.tsdataset1, lstnet, start=200, predict_window=50, stride=50, return_predicts=True)

        # case2 add window,stride, window = stride
        lstnet = LSTNetRegressor(
            in_chunk_len=1 * 96 + 20 * 4,
            out_chunk_len=96,
            max_epochs=1
        )
        lstnet.fit(self.tsdataset1, self.tsdataset1)
        score, predicts = backtest(self.tsdataset1, lstnet, start=pd.Timestamp('2022-01-07T12'), predict_window=50, stride=50,
                       return_predicts=True)

        start = 624
        data_len = len(self.tsdataset1.get_target())
        assert len(predicts.get_target()) == data_len - start

        # case3 add window,stride, window != stride
        lstnet = LSTNetRegressor(
            in_chunk_len=1 * 96 + 20 * 4,
            out_chunk_len=96,
            max_epochs=1
        )
        lstnet.fit(self.tsdataset1, self.tsdataset1)
        score, predicts = backtest(self.tsdataset1, lstnet, start=200, predict_window=50, stride=60, return_predicts=True)
        assert score != 0 

        # case4 add skip_chunk_len
        lstnet = LSTNetRegressor(
            in_chunk_len=1 * 96 + 20 * 4,
            out_chunk_len=96,
            skip_chunk_len=4 * 4,
            max_epochs=1
        )
        lstnet.fit(self.tsdataset1, self.tsdataset1)
        score, predicts = backtest(self.tsdataset1, lstnet, start=200, predict_window=50, stride=50, return_predicts=True)

        start = 200
        data_len = len(self.tsdataset1.get_target())
        assert len(predicts.get_target()) == data_len - start


        # case5 add return score
        lstnet = LSTNetRegressor(
            in_chunk_len=1 * 96 + 20 * 4,
            out_chunk_len=96,
            skip_chunk_len=4 * 4,
            max_epochs=1
        )
        lstnet.fit(self.tsdataset1, self.tsdataset1)
        res = backtest(self.tsdataset1, lstnet, start=192, predict_window=50, stride=50)
        assert res != 0

        # case6 add metric,  return score
        lstnet = LSTNetRegressor(
            in_chunk_len=1 * 96 + 20 * 4,
            out_chunk_len=96,
            skip_chunk_len=4 * 4,
            max_epochs=1
        )
        lstnet.fit(self.tsdataset1, self.tsdataset1)
        res = backtest(self.tsdataset1, lstnet, metric=MAE(), start=192, predict_window=50,
                       stride=50)
        assert res != 0

        # case7 add metric, add reduction,  return score
        lstnet = LSTNetRegressor(
            in_chunk_len=1 * 96 + 20 * 4,
            out_chunk_len=96,
            skip_chunk_len=4 * 4,
            max_epochs=1
        )
        lstnet.fit(self.tsdataset1, self.tsdataset1)
        res = backtest(self.tsdataset1, lstnet, metric=MAE(), start=192, predict_window=50, reduction=np.median,
                       stride=50)
        assert res != 0

        # case8 badcase start < model._in_chunk_len
        lstnet = LSTNetRegressor(
            in_chunk_len=1 * 96 + 20 * 4,
            out_chunk_len=96,
            skip_chunk_len=4 * 4,
            max_epochs=1
        )
        with self.assertRaises(ValueError):
            lstnet.fit(self.tsdataset1, self.tsdataset1)
            res = backtest(self.tsdataset1, lstnet, metric=MAE(), start=20, predict_window=50, stride=50,
                           return_predicts=True)

        # case9 badcase model._skip_chunk_len != 0 and window > model._out_chunk_len
        lstnet = LSTNetRegressor(
            in_chunk_len=1 * 96 + 20 * 4,
            out_chunk_len=96,
            skip_chunk_len=4 * 4,
            max_epochs=1
        )
        with self.assertRaises(ValueError):
            lstnet.fit(self.tsdataset1, self.tsdataset1)
            res = backtest(self.tsdataset1, lstnet, metric=MAE(), start=176, predict_window=150, stride=50,
                           return_predicts=True)

        # case10 badcase window<0
        lstnet = LSTNetRegressor(
            in_chunk_len=1 * 96 + 20 * 4,
            out_chunk_len=96,
            skip_chunk_len=4 * 4,
            max_epochs=1
        )
        with self.assertRaises(ValueError):
            lstnet.fit(self.tsdataset1, self.tsdataset1)
            res = backtest(self.tsdataset1, lstnet, predict_window=-1, return_predicts=True)

        # case11 badcase stride<0
        lstnet = LSTNetRegressor(
            in_chunk_len=1 * 96 + 20 * 4,
            out_chunk_len=96,
            skip_chunk_len=4 * 4,
            max_epochs=1
        )
        with self.assertRaises(ValueError):
            lstnet.fit(self.tsdataset1, self.tsdataset1)
            res = backtest(self.tsdataset1, lstnet, stride=-1, return_predicts=True)

        # case12 start > target_len
        lstnet = LSTNetRegressor(
            in_chunk_len=1 * 96 + 20 * 4,
            out_chunk_len=96,
            skip_chunk_len=4 * 4,
            max_epochs=1
        )
        with self.assertRaises(ValueError):
            lstnet.fit(self.tsdataset1, self.tsdataset1)
            res = backtest(self.tsdataset1, lstnet, start=5000, return_predicts=True)

        # case13 default(multi-target)
        lstnet = LSTNetRegressor(
            in_chunk_len=1 * 96 + 20 * 4,
            out_chunk_len=96,
            max_epochs=1
        )
        lstnet.fit(self.tsdataset2, self.tsdataset2)
        res = backtest(self.tsdataset2, lstnet)
        assert res != 0

        # case14 add metric, add reduction,  return score (multi-target)
        lstnet = LSTNetRegressor(
            in_chunk_len=1 * 96 + 20 * 4,
            out_chunk_len=96,
            skip_chunk_len=4 * 4,
            max_epochs=1
        )
        lstnet.fit(self.tsdataset2, self.tsdataset2)
        res = backtest(self.tsdataset2, lstnet, metric=MAE(), start=192, predict_window=50, reduction=np.median,
                       stride=50)
        assert res != 0

        #quantile metric

        target = TimeSeries.load_from_dataframe(
            pd.DataFrame(np.random.randn(400,2).astype(np.float32),
                    index=pd.date_range("2022-01-01", periods=400, freq="15T"),
                        columns=["a1", "a2"]
                    ))

        observed_cov = TimeSeries.load_from_dataframe(
            pd.DataFrame(
                np.random.randn(400, 2).astype(np.float32),
                index=pd.date_range("2022-01-01", periods=400, freq="15T"),
                columns=["index", "c"]
            ))
        known_cov = TimeSeries.load_from_dataframe(
            pd.DataFrame(
                np.random.randn(500, 2).astype(np.float32),
                index=pd.date_range("2022-01-01", periods=500, freq="15T"),
                columns=["b1", "c1"]
            ))
        static_cov = {"f": 1.0, "g": 2.0}
        dataset = TSDataset(target, observed_cov, known_cov, static_cov)
        from paddlets.models.forecasting import DeepARModel
        from paddlets.metrics import MSE,QuantileLoss
        reg = DeepARModel(
            in_chunk_len=10,
            out_chunk_len=5,
            skip_chunk_len=4 * 4,
            eval_metrics=["mse", "mae"],
            batch_size=512,
            num_samples = 101,
            regression_mode="sampling",
            output_mode="quantiles",
            max_epochs=5
        )

        reg.fit(dataset, dataset)
        score = backtest(dataset, reg,metric=MSE("prob"),verbose=False)
        assert isinstance(score,dict)

        score = backtest(dataset, reg,metric=QuantileLoss(q_points=[0.1,0.9]),verbose=False)
        assert isinstance(score["a1"],dict)

if __name__ == "__main__":
    unittest.main()
