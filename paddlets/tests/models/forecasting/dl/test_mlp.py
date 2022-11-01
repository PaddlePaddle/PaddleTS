# !/usr/bin/env python3
# -*- coding:utf-8 -*-

from unittest import TestCase, mock
import unittest
import random

import pandas as pd
import numpy as np
import paddle

from paddlets.models.common.callbacks import Callback
from paddlets.models.forecasting import MLPRegressor
from paddlets.datasets import TimeSeries, TSDataset


class TestMLPRegressor(TestCase):
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
            "batch_size": 1,
            "max_epochs": 1,
            "verbose": 1,
            "patience": 1,
        }
        mlp = MLPRegressor(
            in_chunk_len=7 * 96 + 20 * 4,
            out_chunk_len=96,
            skip_chunk_len=4 * 4,
            **param1
        )

        # case2 (batch_size 不合法)
        param2 = {
            "batch_size": 0,
            "max_epochs": 1,
            "verbose": 1,
            "patience": 1,
        }
        with self.assertRaises(ValueError):
            mlp = MLPRegressor(
                in_chunk_len=7 * 96 + 20 * 4,
                out_chunk_len=96,
                skip_chunk_len=4 * 4,
                **param2
            )

        # case3 (max_epochs 不合法)
        param3 = {
            "batch_size": 1,
            "max_epochs": 0,
            "verbose": 1,
            "patience": 1,
        }
        with self.assertRaises(ValueError):
            mlp = MLPRegressor(
                in_chunk_len=7 * 96 + 20 * 4,
                out_chunk_len=96,
                skip_chunk_len=4 * 4,
                **param3
            )

        # case4 (verbose 不合法)
        param4 = {
            "batch_size": 1,
            "max_epochs": 1,
            "verbose": 0,
            "patience": 1,
        }
        with self.assertRaises(ValueError):
            mlp = MLPRegressor(
                in_chunk_len=7 * 96 + 20 * 4,
                out_chunk_len=96,
                skip_chunk_len=4 * 4,
                **param4
            )

        # case5 (patience 不合法)
        param5 = {
            "batch_size": 1,
            "max_epochs": 1,
            "verbose": 1,
            "patience": -1,
        }
        with self.assertRaises(ValueError):
            mlp = MLPRegressor(
                in_chunk_len=7 * 96 + 20 * 4,
                out_chunk_len=96,
                skip_chunk_len=4 * 4,
                **param5
            )

        # case6 hidden_config 不合法
        param6 = {
            "batch_size": 128,
            "max_epochs": 1,
            "verbose": 1,
            "patience": 1,
            "hidden_config": [100, -100]
        }
        with self.assertRaises(ValueError):
            mlp = MLPRegressor(
                in_chunk_len=7 * 96 + 20 * 4,
                out_chunk_len=96,
                skip_chunk_len=4 * 4,
                **param6
            )
            mlp.fit(self.tsdataset1)

        # case7 (use_bn = True)
        param7 = {
            "batch_size": 128,
            "max_epochs": 1,
            "verbose": 1,
            "patience": 1,
            "use_bn": True
        }
        mlp = MLPRegressor(
            in_chunk_len=7 * 96 + 20 * 4,
            out_chunk_len=96,
            skip_chunk_len=4 * 4,
            **param7
        )
        mlp.fit(self.tsdataset1)
        self.assertIsInstance(mlp._network._nn[1], paddle.nn.BatchNorm1D)

    def test_init_dataloader(self):
        """unittest function
        """
        mlp = MLPRegressor(
            in_chunk_len=7 * 96 + 20 * 4,
            out_chunk_len=96,
            skip_chunk_len=4 * 4,
            max_epochs=1
        )
        # case1 (评估集未传入函数)
        _, valid_dataloaders = mlp._init_fit_dataloaders(self.tsdataset1)
        self.assertEqual(len(valid_dataloaders), 0)

        # calse2 (评估集传入函数)
        _, valid_dataloaders = mlp._init_fit_dataloaders(self.tsdataset1, self.tsdataset1)
        self.assertNotEqual(len(valid_dataloaders), 0)

        # case3 (训练集协变量包含非float32)
        tsdataset = self.tsdataset1.copy()
        tsdataset.astype({"b": "float64"})
        mlp.fit(tsdataset, tsdataset)
        dtypes = tsdataset.dtypes.to_dict()
        self.assertEqual(dtypes["b"], np.float32)

        tsdataset = self.tsdataset1.copy()
        tsdataset.astype({"c": "int32"})
        with self.assertRaises(ValueError):
            mlp.fit(tsdataset, tsdataset)

        # case4 (训练集包含非法数据类型如object字符串类型)
        tsdataset = self.tsdataset1.copy()
        tsdataset.astype({"b": "O"})
        with self.assertRaises(ValueError):
            mlp.fit(tsdataset, tsdataset)

        # case5 (训练集的target为非float类型. 视具体模型而定, 目前mlp模型不支持target为除float之外的类型)
        tsdataset = self.tsdataset1.copy()
        tsdataset.astype({"a": "int32"})
        with self.assertRaises(ValueError):
            mlp.fit(tsdataset, tsdataset)

        # case6 (训练集包含NaN)
        tsdataset = self.tsdataset1.copy()
        tsdataset["a"][0] = np.NaN
        with self.assertLogs("paddlets", level="WARNING") as captured:
            mlp.fit(tsdataset, tsdataset)
            self.assertEqual(
                captured.records[0].getMessage(), 
                "Input `a` contains np.inf or np.NaN, which may lead to unexpected results from the model."
            )
            
    def test_init_metrics(self):
        """unittest function
        """
        # case1 (以用户传入的metric为第一优先)
        mlp = MLPRegressor(
            in_chunk_len=7 * 96 + 20 * 4,
            out_chunk_len=96,
            skip_chunk_len=4 * 4,
            eval_metrics = ["mse"]
        )
        _, metrics_names, _ = mlp._init_metrics(["val"])
        self.assertEqual(metrics_names[-1], "val_mse")

        # case2 (用户未传入的metric, 取默认metric)
        mlp = MLPRegressor(
            in_chunk_len=7 * 96 + 20 * 4,
            out_chunk_len=96,
            skip_chunk_len=4 * 4,
            patience=1
        )
        _, metrics_names, _ = mlp._init_metrics(["val"])
        self.assertEqual(metrics_names[-1], "val_mae")

    def test_init_callbacks(self):
        """unittest function
        """
        # case1 (patience = 0)
        mlp = MLPRegressor(
            in_chunk_len=7 * 96 + 20 * 4,
            out_chunk_len=96,
            skip_chunk_len=4 * 4,
            patience=0
        )
        mlp._metrics, mlp._metrics_names, _ = mlp._init_metrics(["val"])
        with self.assertLogs("paddlets", level="WARNING") as captured:
            mlp._init_callbacks()
            self.assertEqual(len(captured.records), 1) # check that there is only one log message
            self.assertEqual(
                captured.records[0].getMessage(), 
                "No early stopping will be performed, last training weights will be used."
            )

        # case2 (patience > 0)
        mlp = MLPRegressor(
            in_chunk_len=7 * 96 + 20 * 4,
            out_chunk_len=96,
            skip_chunk_len=4 * 4,
            patience=1
        )
        mlp._metrics, mlp._metrics_names, _ = mlp._init_metrics(["val"])
        _, callback_container = mlp._init_callbacks()

        # case3 (用户未传入callbacks)
        self.assertEqual(len(callback_container._callbacks), 2)

        # case4 (用户传入callbacks)
        callback = Callback()
        mlp = MLPRegressor(
            in_chunk_len=7 * 96 + 20 * 4,
            out_chunk_len=96,
            skip_chunk_len=4 * 4,
            callbacks=[callback]
        )
        mlp._metrics, mlp._metrics_names, _ = mlp._init_metrics(["val"])
        _, callback_container = mlp._init_callbacks()
        self.assertEqual(len(callback_container._callbacks), 3)

    def test_fit(self):
        """unittest function
        """
        # case1 (用户只传入训练集, log不显示评估指标, 达到最大epochs训练结束)
        mlp = MLPRegressor(
            in_chunk_len=7 * 96 + 20 * 4,
            out_chunk_len=96,
            skip_chunk_len=4 * 4,
            optimizer_params=dict(learning_rate=1e-1),
            eval_metrics=["mse", "mae"],
            batch_size=512,
            max_epochs=10,
            patience=1
        )
        mlp.fit(self.tsdataset1)

        # case2 (用户同时传入训练/评估集和, log显示评估指标, 同时early_stopping生效)
        mlp = MLPRegressor(
            in_chunk_len=7 * 96 + 20 * 4,
            out_chunk_len=96,
            skip_chunk_len=4 * 4,
            optimizer_params=dict(learning_rate=1e-1),
            eval_metrics=["mse", "mae"],
            batch_size=512,
            max_epochs=10,
            patience=1
        )
        mlp.fit(self.tsdataset1, self.tsdataset1)
        self.assertEqual(mlp._stop_training, True)

        # case3 (用户传入多组时序数据用于组合训练)
        mlp = MLPRegressor(
            in_chunk_len=7 * 96 + 20 * 4,
            out_chunk_len=96,
            skip_chunk_len=4 * 4,
            optimizer_params=dict(learning_rate=5e-1),
            eval_metrics=["mse", "mae"],
            batch_size=512,
            max_epochs=10,
            patience=1
        )
        mlp.fit([self.tsdataset1, self.tsdataset1], [self.tsdataset1, self.tsdataset1])
        self.assertEqual(mlp._stop_training, True)

    def test_predict(self):
        """unittest function
        """
        # case1 (index为DatetimeIndex的单变量预测)
        mlp = MLPRegressor(
            in_chunk_len=7 * 96 + 20 * 4,
            out_chunk_len=96,
            skip_chunk_len=4 * 4,
            eval_metrics=["mse", "mae"],
            max_epochs=1,
            patience=1
        )
        mlp.fit(self.tsdataset1, self.tsdataset1)
        res = mlp.predict(self.tsdataset1)
        self.assertIsInstance(res, TSDataset)
        self.assertIsInstance(res.get_target().data.index, pd.DatetimeIndex)
        self.assertEqual(res.get_target().data.shape, (96, 1))

        # case2 (index为DatetimeIndex的多变量预测)
        mlp.fit(self.tsdataset2, self.tsdataset2)
        res = mlp.predict(self.tsdataset2)
        self.assertIsInstance(res, TSDataset)
        self.assertIsInstance(res.get_target().data.index, pd.DatetimeIndex)
        self.assertEqual(res.get_target().data.shape, (96, 2))

        # case3 (index为RangeIndex的多变量预测)
        mlp.fit(self.tsdataset3, self.tsdataset3)
        res = mlp.predict(self.tsdataset3)
        self.assertIsInstance(res.get_target().data.index, pd.RangeIndex)
        self.assertIsInstance(res, TSDataset)
        self.assertEqual(res.get_target().data.shape, (96, 2))

    def test_recursive_predict(self):
        """unittest function
        """
        # case1 (单变量预测)
        reg = MLPRegressor(
            in_chunk_len=7 * 96 + 20 * 4,
            out_chunk_len=96,
            skip_chunk_len=16,
            eval_metrics=["mse", "mae"],
            max_epochs=1
        )
        reg.fit(self.tsdataset1, self.tsdataset1)
        #not supported when skip_chunk_len > 0
        with self.assertRaises(ValueError):
            res = reg.recursive_predict(self.tsdataset1, 96)
        
        reg = MLPRegressor(
            in_chunk_len=7 * 96 + 20 * 4,
            out_chunk_len=96,
            skip_chunk_len=0,
            eval_metrics=["mse", "mae"],
            max_epochs=1
        )
        reg.fit(self.tsdataset1, self.tsdataset1)
        #not supported when predict_lenth < 0
        with self.assertRaises(ValueError):
            res = reg.recursive_predict(self.tsdataset1, 0)
        res = reg.recursive_predict(self.tsdataset1, 1100)
        self.assertIsInstance(res, TSDataset)
        self.assertEqual(res.get_target().data.shape, (1100, 1))

        res = reg.recursive_predict(self.tsdataset1, 10)
        self.assertIsInstance(res, TSDataset)
        self.assertEqual(res.get_target().data.shape, (10, 1))

        # case2 (多变量预测)
        reg.fit(self.tsdataset2, self.tsdataset2)
        res = reg.recursive_predict(self.tsdataset2, 200)
        self.assertIsInstance(res, TSDataset)
        self.assertEqual(res.get_target().data.shape, (200, 2)) 

        # case3 (range Index)
        reg.fit(self.tsdataset3, self.tsdataset3)
        res = reg.recursive_predict(self.tsdataset3, 2500)
        self.assertIsInstance(res, TSDataset)
        self.assertEqual(res.get_target().data.shape, (2500, 2)) 


if __name__ == "__main__":
    unittest.main()
