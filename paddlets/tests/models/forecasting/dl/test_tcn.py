# !/usr/bin/env python3
# -*- coding:utf-8 -*-

from unittest import TestCase
import unittest
import random

import pandas as pd
import numpy as np

from paddlets.models.forecasting import TCNRegressor
from paddlets.datasets import TimeSeries, TSDataset


class TestTCNRegressor(TestCase):
    def setUp(self):
        """unittest function
        """
        np.random.seed(2022)
        target1 = pd.Series(
                np.random.randn(2000).astype(np.float32),
                index=pd.date_range("2022-01-01", periods=2000, freq="15T"),
                name="a") 
        target2 = pd.DataFrame(
                np.random.randn(2000, 2).astype(np.float32),
                index=pd.date_range("2022-01-01", periods=2000, freq="15T"),
                columns=["a1", "a2"])
        observed_cov = pd.DataFrame(
                np.random.randn(2000, 2).astype(np.float32),
                index=pd.date_range("2022-01-01", periods=2000, freq="15T"),
                columns=["b", "c"])
        known_cov = pd.DataFrame(
                np.random.randn(2500, 2).astype(np.float32),
                index=pd.date_range("2022-01-01", periods=2500, freq="15T"),
                columns=["b1", "c1"])
        static_cov = {"f": 1.0, "g": 2.0}

        # index为DatetimeIndex类型
        self.tsdataset1 = TSDataset(
            TimeSeries.load_from_dataframe(target1), 
            TimeSeries.load_from_dataframe(observed_cov), 
            TimeSeries.load_from_dataframe(known_cov), 
            static_cov)
        self.tsdataset2 = TSDataset(
            TimeSeries.load_from_dataframe(target2), 
            TimeSeries.load_from_dataframe(observed_cov), 
            TimeSeries.load_from_dataframe(known_cov), 
            static_cov)

        # index为RangeIndex类型
        index = pd.RangeIndex(0, 2000, 2)
        index2 = pd.RangeIndex(0, 2500, 2)
        target2 = target2.reset_index(drop=True).reindex(index)
        observed_cov = observed_cov.reset_index(drop=True).reindex(index)
        known_cov = known_cov.reset_index(drop=True).reindex(index2)
        self.tsdataset3 = TSDataset(
            TimeSeries.load_from_dataframe(target2, freq=index.step),
            TimeSeries.load_from_dataframe(observed_cov, freq=index.step),
            TimeSeries.load_from_dataframe(known_cov, freq=index2.step),
            static_cov) 
        super().setUp()

    def test_init(self):
        """unittest function
        """
        # case1 (参数全部合法)
        param1 = {
            "hidden_config": [10],
            "kernel_size": 3,

        }
        tcn = TCNRegressor(
            in_chunk_len=1 * 96 + 20 * 4,
            out_chunk_len=96,
            skip_chunk_len=4 * 4,
            max_epochs=1,
            **param1
        )
        tcn.fit(self.tsdataset1)
        
        # case2 (hidden_config 不合法)
        param2 = {
            "hidden_config": [-10],
            "kernel_size": 3,

        }
        with self.assertRaises(ValueError):
            tcn = TCNRegressor(
                in_chunk_len=1 * 96 + 20 * 4,
                out_chunk_len=96,
                skip_chunk_len=4 * 4,
                max_epochs=1,
                **param2
            )
            tcn.fit(self.tsdataset1)

        # case3 (kernel_size不合法)
        param3 = {
            "hidden_config": [10],
            "kernel_size": -3,

        }
        with self.assertRaises(ValueError):
            tcn = TCNRegressor(
                in_chunk_len=1 * 96 + 20 * 4,
                out_chunk_len=96,
                skip_chunk_len=4 * 4,
                max_epochs=1,
                **param3
            )
            tcn.fit(self.tsdataset1)

        # case4 (hidden_config 为None)
        param4 = {
            "hidden_config": None,
            "kernel_size": 3,

        }
        tcn = TCNRegressor(
            in_chunk_len=1 * 96 + 20 * 4,
            out_chunk_len=96,
            skip_chunk_len=4 * 4,
            max_epochs=1,
            **param4
        )
        tcn.fit(self.tsdataset1)
        self.assertEqual(len(tcn._network._temporal_layers), 6)

        # case5 (hidden_config 为不为None, 但是导致感受野超过in_chunk_len)
        param5 = {
            "kernel_size": 3,
            "hidden_config": [1, 1, 1, 1, 1, 1]
        }
        with self.assertLogs("paddlets", level="WARNING") as captured:
            tcn = TCNRegressor(
                in_chunk_len=1 * 96 + 20 * 4,
                out_chunk_len=96,
                skip_chunk_len=4 * 4,
                max_epochs=1,
                **param5
            )
            tcn.fit(self.tsdataset1)
            # self.assertEqual(len(captured.records), 1) # check that there is only one log message
            self.assertEqual(
                captured.records[1].getMessage(),
                "The receptive field of TCN exceeds the in_chunk_len."
            )

        # case6 (out_chunk_len/in_chunk_len 不合法)
        param6 = {
            "hidden_config": [10],
            "kernel_size": 3,
        }
        with self.assertRaises(ValueError):
            tcn = TCNRegressor(
                in_chunk_len=20 * 4,
                out_chunk_len=96,
                skip_chunk_len=4 * 4,
                max_epochs=1,
                **param6
            )
            tcn.fit(self.tsdataset1)

        # case5 (训练集的target为非float类型. 视具体模型而定, 目前tcn模型不支持target为除float之外的类型)
        tsdataset = self.tsdataset1.copy()
        tsdataset.astype({"a": "int32"})
        with self.assertRaises(ValueError):
            tcn.fit(tsdataset, tsdataset)

    def test_fit(self):
        """unittest function
        """
        # case1 (用户只传入训练集, log不显示评估指标, 达到最大epochs训练结束)
        tcn = TCNRegressor(
            in_chunk_len=1 * 96 + 20 * 4,
            out_chunk_len=96,
            skip_chunk_len=4 * 4,
            optimizer_params=dict(learning_rate=1e-1),
            batch_size=512,
            max_epochs=10,
            patience=1,
            hidden_config=[10]
        )
        tcn.fit(self.tsdataset1)

        # case2 (用户同时传入训练/评估集和, log显示评估指标, 同时early_stopping生效)
        tcn = TCNRegressor(
            in_chunk_len=1 * 96 + 20 * 4,
            out_chunk_len=96,
            skip_chunk_len=4 * 4,
            optimizer_params=dict(learning_rate=1e-1),
            batch_size=512,
            max_epochs=10,
            patience=1,
            hidden_config=[10]
        )
        tcn.fit(self.tsdataset1, self.tsdataset1)

    def test_predict(self):
        """unittest function
        """
        # case1 (单变量预测)
        tcn = TCNRegressor(
            in_chunk_len=1 * 96 + 20 * 4,
            out_chunk_len=96,
            skip_chunk_len=4 * 4,
            max_epochs=1
        )
        tcn.fit(self.tsdataset1, self.tsdataset1)
        res = tcn.predict(self.tsdataset1)

        self.assertIsInstance(res, TSDataset)
        self.assertEqual(res.get_target().data.shape, (96, 1))

        # case2 (多变量预测)
        tcn.fit(self.tsdataset2, self.tsdataset2)
        res = tcn.predict(self.tsdataset2)
        self.assertIsInstance(res, TSDataset)
        self.assertEqual(res.get_target().data.shape, (96, 2))

    def test_recursive_predict(self):
        """unittest function
        """
        # case1 (单变量预测)
        tcn = TCNRegressor(
            in_chunk_len=1 * 96 + 20 * 4,
            out_chunk_len=96,
            skip_chunk_len=16,
            eval_metrics=["mse", "mae"],
            max_epochs=1
        )
        tcn.fit(self.tsdataset1, self.tsdataset1)
        #not supported when skip_chunk_len > 0
        with self.assertRaises(ValueError):
            res = tcn.recursive_predict(self.tsdataset1, 96)

        tcn = TCNRegressor(
            in_chunk_len=7 * 96 + 20 * 4,
            out_chunk_len=96,
            skip_chunk_len=0,
            eval_metrics=["mse", "mae"],
            max_epochs=1
        )
        tcn.fit(self.tsdataset1, self.tsdataset1)
        #not supported when predict_lenth < 0
        with self.assertRaises(ValueError):
            tcn = tcn.recursive_predict(self.tsdataset1, 0)
        res = tcn.recursive_predict(self.tsdataset1, 1100)
        self.assertIsInstance(res, TSDataset)
        self.assertEqual(res.get_target().data.shape, (1100, 1))

        res = tcn.recursive_predict(self.tsdataset1, 10)
        self.assertIsInstance(res, TSDataset)
        self.assertEqual(res.get_target().data.shape, (10, 1))

        # case2 (多变量预测)
        tcn.fit(self.tsdataset2, self.tsdataset2)
        res = tcn.recursive_predict(self.tsdataset2, 200)
        self.assertIsInstance(res, TSDataset)
        self.assertEqual(res.get_target().data.shape, (200, 2))

        # case3 (range Index)
        tcn.fit(self.tsdataset3, self.tsdataset3)
        res = tcn.recursive_predict(self.tsdataset3, 2500)
        self.assertIsInstance(res, TSDataset)
        self.assertEqual(res.get_target().data.shape, (2500, 2))
 

if __name__ == "__main__":
    unittest.main()
