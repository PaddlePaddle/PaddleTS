# !/usr/bin/env python3
# -*- coding:utf-8 -*-

from unittest import TestCase
import unittest

import pandas as pd
import numpy as np

from paddlets.models.representation.dl.cost import CoST
from paddlets.models.common.callbacks.callbacks import Callback
from paddlets.datasets import TimeSeries, TSDataset


class TestCoST(TestCase):
    def setUp(self):
        """unittest function
        """
        np.random.seed(2022)
        target1 = pd.Series(
                np.random.randn(800).astype(np.float32),
                index=pd.date_range("2022-01-01", periods=800, freq="15T"),
                name="a")
        target2 = pd.DataFrame(
                np.random.randn(800, 2).astype(np.float32),
                index=pd.date_range("2022-01-01", periods=800, freq="15T"),
                columns=["a1", "a2"])
        observed_cov = pd.DataFrame(
                np.random.randn(800, 2).astype(np.float32),
                index=pd.date_range("2022-01-01", periods=800, freq="15T"),
                columns=["b", "c"])
        known_cov = pd.DataFrame(
                np.random.randn(800, 2).astype(np.float32),
                index=pd.date_range("2022-01-01", periods=800, freq="15T"),
                columns=["b1", "c1"])
        
        # index为DatetimeIndex类型
        self.tsdataset1 = TSDataset(
            TimeSeries.load_from_dataframe(target1),
            TimeSeries.load_from_dataframe(observed_cov),
            TimeSeries.load_from_dataframe(known_cov),
        )
        self.tsdataset2 = TSDataset(
            TimeSeries.load_from_dataframe(target2),
            TimeSeries.load_from_dataframe(observed_cov),
            TimeSeries.load_from_dataframe(known_cov),
        )
        super().setUp()

    def test_init(self):
        """unittest function
        """
        # case1 (参数合法)
        param1 = {
            "batch_size": 2,
            "queue_size": 32
        }
        cost = CoST(
            segment_size=300,
            sampling_stride=300,
            **param1
        )

        # case1 (参数非法: queue_size不能被batch_size整除)
        param1 = {
            "batch_size": 3,
            "queue_size": 32
        }
        with self.assertRaises(ValueError):
            cost = CoST(
                segment_size=300,
                sampling_stride=300,
                **param1
            )

    def test_init_fit_dataloader(self):
        """unittest function
        """
        cost = CoST(
            segment_size=300,
            sampling_stride=300,
            num_layers=1,
            batch_size=1,
            max_epochs=1
        )

        # case1 (训练集包含非int64 和 非float32)
        tsdataset = self.tsdataset1.copy()
        tsdataset.astype({"b": "float64", "c": "float64"})
        cost.fit(tsdataset)
        dtypes = tsdataset.dtypes.to_dict()
        self.assertEqual(dtypes["b"], np.float32)
        self.assertEqual(dtypes["c"], np.float32)

        # case2 (训练集包含非法数据类型如object字符串类型)
        tsdataset = self.tsdataset1.copy()
        tsdataset.astype({"b": "O"})
        with self.assertRaises(ValueError):
            cost.fit(tsdataset)

        # case3 (训练集的target为非float类型. 视具体模型而定, 目前cost模型不支持target为除float之外的类型)
        tsdataset = self.tsdataset1.copy()
        tsdataset.astype({"a": "int32"})
        with self.assertRaises(ValueError):
            cost.fit(tsdataset)

        # case6 (训练集包含NaN)
        tsdataset = self.tsdataset1.copy()
        tsdataset["a"][0] = np.NaN
        with self.assertLogs("paddlets", level="WARNING") as captured:
            cost.fit(tsdataset)
            self.assertEqual(
                captured.records[0].getMessage(), 
                "Input `a` contains np.inf or np.NaN, which may lead to unexpected results from the model."
            )
            
    def test_init_callbacks(self):
        """unittest function
        """
        # case1 (用户未传入callbacks)
        cost = CoST(
            segment_size=300,
            sampling_stride=300,
            num_layers=1,
            batch_size=1,
            max_epochs=1
        )
        _, callback_container = cost._init_callbacks()
        self.assertEqual(len(callback_container._callbacks), 1)

        # case4 (用户传入callbacks)
        callback = Callback()
        cost = CoST(
            segment_size=300,
            sampling_stride=300,
            callbacks=[callback],
            num_layers=1,
            batch_size=1,
            max_epochs=1
        )
        _, callback_container = cost._init_callbacks()
        self.assertEqual(len(callback_container._callbacks), 2)

    def test_fit(self):
        """unittest function
        """
        # case1 (用户只传入训练集, log不显示评估指标, 达到最大epochs训练结束)
        cost = CoST(
            segment_size=300,
            sampling_stride=300,
            num_layers=1,
            batch_size=1,
            max_epochs=1,
        )
        cost.fit(self.tsdataset1)

        # case2 (用户传入多组时序数据用于多实例训练)
        cost.fit([self.tsdataset1, self.tsdataset1])

    def test_encode(self):
        """unittest function
        """
        # case1 (index为DatetimeIndex的单变量)
        cost = CoST(
            segment_size=300,
            sampling_stride=300,
            num_layers=1,
            batch_size=1,
            max_epochs=1,
        )
        cost.fit(self.tsdataset1)
        res = cost.encode(self.tsdataset1, batch_size=128)
        self.assertIsInstance(res, np.ndarray)
        self.assertEqual(res.shape, (1, 800, 320))

        # case2 (index为DatetimeIndex的多变量)
        cost.fit(self.tsdataset2)
        res = cost.encode(self.tsdataset2, batch_size=128)
        self.assertIsInstance(res, np.ndarray)
        self.assertEqual(res.shape, (1, 800, 320))


if __name__ == "__main__":
    unittest.main()
