# !/usr/bin/env python3
# -*- coding:utf-8 -*-

from unittest import TestCase
import unittest
import random

import pandas as pd
import numpy as np

from paddlets.models.forecasting import LSTNetRegressor
from paddlets.datasets import TimeSeries, TSDataset


class TestLSTNetRegressor(TestCase):
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
        super().setUp()

    def test_init(self):
        """unittest function
        """
        # case1 (参数全部合法)
        param1 = {
            "skip_size": 96,
            "channels": 1,
            "kernel_size": 3,
            "rnn_cell_type": "GRU",
            "skip_rnn_cell_type": "GRU",
            "output_activation": None

        }
        lstnet = LSTNetRegressor(
            in_chunk_len=1 * 96 + 20 * 4,
            out_chunk_len=96,
            skip_chunk_len=4 * 4,
            max_epochs=1,
            **param1
        )
        lstnet.fit(self.tsdataset1)
        
        # case2 (channels 不合法)
        param2 = {
            "skip_size": 96,
            "channels": -1,
            "kernel_size": 3,
            "rnn_cell_type": "GRU",
            "skip_rnn_cell_type": "GRU",
            "output_activation": None

        }
        with self.assertRaises(ValueError):
            lstnet = LSTNetRegressor(
                in_chunk_len=1 * 96 + 20 * 4,
                out_chunk_len=96,
                skip_chunk_len=4 * 4,
                max_epochs=1,
                **param2
            )
            lstnet.fit(self.tsdataset1)
        
        # case3 (rnn_cell_type不合法)
        param3 = {
            "skip_size": 96,
            "channels": 1,
            "kernel_size": 3,
            "rnn_cell_type": "RNN",
            "skip_rnn_cell_type": "GRU",
            "output_activation": None

        }
        with self.assertRaises(ValueError):
            lstnet = LSTNetRegressor(
                in_chunk_len=1 * 96 + 20 * 4,
                out_chunk_len=96,
                skip_chunk_len=4 * 4,
                max_epochs=1,
                **param3
            )
            lstnet.fit(self.tsdataset1)

        # case4 (skip_rnn_cell_type不合法)
        param4 = {
            "skip_size": 96,
            "channels": 1,
            "kernel_size": 3,
            "rnn_cell_type": "GRU",
            "skip_rnn_cell_type": "RNN",
            "output_activation": None

        }
        with self.assertRaises(ValueError):
            lstnet = LSTNetRegressor(
                in_chunk_len=1 * 96 + 20 * 4,
                out_chunk_len=96,
                skip_chunk_len=4 * 4,
                max_epochs=1,
                **param4
            )
            lstnet.fit(self.tsdataset1)
            
        # case5 (output_activation不合法)
        param5 = {
            "skip_size": 96,
            "channels": 1,
            "kernel_size": 3,
            "rnn_cell_type": "GRU",
            "skip_rnn_cell_type": "GRU",
            "output_activation": "relu"

        }
        with self.assertRaises(ValueError):
            lstnet = LSTNetRegressor(
                in_chunk_len=1 * 96 + 20 * 4,
                out_chunk_len=96,
                skip_chunk_len=4 * 4,
                max_epochs=1,
                **param5
            )
            lstnet.fit(self.tsdataset1)

        # case6 (kernel_size不合法)
        param6 = {
            "skip_size": 96,
            "channels": 1,
            "kernel_size": 97,
            "rnn_cell_type": "GRU",
            "skip_rnn_cell_type": "GRU",
            "output_activation": None

        }
        with self.assertRaises(ValueError):
            lstnet = LSTNetRegressor(
                in_chunk_len=1 * 96 + 20 * 4,
                out_chunk_len=96,
                skip_chunk_len=4 * 4,
                max_epochs=1,
                **param6
            )
            lstnet.fit(self.tsdataset1)

        # case5 (训练集的target为非float类型. 视具体模型而定, 目前lstnet模型不支持target为除float之外的类型)
        tsdataset = self.tsdataset1.copy()
        tsdataset.astype({"a": "int32"})
        with self.assertRaises(ValueError):
            lstnet.fit(tsdataset, tsdataset)

    def test_fit(self):
        """unittest function
        """
        # case1 (用户只传入训练集, log不显示评估指标, 达到最大epochs训练结束)
        lstnet = LSTNetRegressor(
            in_chunk_len=1 * 96 + 20 * 4,
            out_chunk_len=96,
            skip_chunk_len=4 * 4,
            optimizer_params=dict(learning_rate=1e-1),
            batch_size=512,
            max_epochs=10,
            patience=1
        )
        lstnet.fit(self.tsdataset1)

        # case2 (用户同时传入训练/评估集和, log显示评估指标, 同时early_stopping生效)
        lstnet = LSTNetRegressor(
            in_chunk_len=1 * 96 + 20 * 4,
            out_chunk_len=96,
            skip_chunk_len=4 * 4,
            optimizer_params=dict(learning_rate=1e-1),
            batch_size=512,
            max_epochs=10,
            patience=1
        )
        lstnet.fit(self.tsdataset1, self.tsdataset1)

        lstnet = LSTNetRegressor(
            in_chunk_len=1 * 96 + 20 * 4,
            out_chunk_len=96,
            skip_chunk_len=4 * 4,
            optimizer_params=dict(learning_rate=1e-1),
            batch_size=512,
            max_epochs=10,
            patience=1
        )
        lstnet.fit([self.tsdataset1, self.tsdataset1], self.tsdataset1)

    def test_predict(self):
        """unittest function
        """
        # case1 (单变量预测)
        lstnet = LSTNetRegressor(
            in_chunk_len=1 * 96 + 20 * 4,
            out_chunk_len=96,
            skip_chunk_len=4 * 4,
            max_epochs=1
        )
        lstnet.fit(self.tsdataset1, self.tsdataset1)
        res = lstnet.predict(self.tsdataset1)

        self.assertIsInstance(res, TSDataset)
        self.assertEqual(res.get_target().data.shape, (96, 1))

        # case2 (多变量预测)
        lstnet.fit(self.tsdataset2, self.tsdataset2)
        res = lstnet.predict(self.tsdataset2)
        self.assertIsInstance(res, TSDataset)
        self.assertEqual(res.get_target().data.shape, (96, 2))

 
if __name__ == "__main__":
    unittest.main()
