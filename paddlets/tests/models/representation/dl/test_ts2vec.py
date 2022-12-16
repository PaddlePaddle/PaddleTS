# !/usr/bin/env python3
# -*- coding:utf-8 -*-

from unittest import TestCase
import unittest

import pandas as pd
import numpy as np

from paddlets.models.representation.dl.ts2vec import TS2Vec
from paddlets.models.common.callbacks.callbacks import Callback
from paddlets.datasets import TimeSeries, TSDataset


class TestTS2Vec(TestCase):
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
        # case1 (参数全部合法)
        param1 = {
            "batch_size": 1,
            "max_epochs": 1,
            "verbose": 1,
        }
        ts2vec = TS2Vec(
            segment_size=300,
            sampling_stride=300,
            **param1
        )

        # case2 (batch_size 不合法)
        param2 = {
            "batch_size": 0,
            "max_epochs": 1,
            "verbose": 1,
        }
        with self.assertRaises(ValueError):
            ts2vec = TS2Vec(
                segment_size=300,
                sampling_stride=300,
                **param2
            )

        # case3 (max_epochs 不合法)
        param3 = {
            "batch_size": 1,
            "max_epochs": 0,
            "verbose": 1,
        }
        with self.assertRaises(ValueError):
            ts2vec = TS2Vec(
                segment_size=300,
                sampling_stride=300,
                **param3
            )

        # case4 (verbose 不合法)
        param4 = {
            "batch_size": 1,
            "max_epochs": 1,
            "verbose": 0,
        }
        with self.assertRaises(ValueError):
            ts2vec = TS2Vec(
                segment_size=300,
                sampling_stride=300,
                **param4
            )

    def test_init_fit_dataloader(self):
        """unittest function
        """
        ts2vec = TS2Vec(
            segment_size=300,
            sampling_stride=300,
            max_epochs=1
        )

        # case1 (训练集包含非int64 和 非float32)
        tsdataset = self.tsdataset1.copy()
        tsdataset.astype({"b": "float64", "c": "float64"})
        ts2vec.fit(tsdataset)
        dtypes = tsdataset.dtypes.to_dict()
        self.assertEqual(dtypes["b"], np.float32)
        self.assertEqual(dtypes["c"], np.float32)

        # case2 (训练集包含非法数据类型如object字符串类型)
        tsdataset = self.tsdataset1.copy()
        tsdataset.astype({"b": "O"})
        with self.assertRaises(ValueError):
            ts2vec.fit(tsdataset)

        # case3 (训练集的target为非float类型. 视具体模型而定, 目前ts2vec模型不支持target为除float之外的类型)
        tsdataset = self.tsdataset1.copy()
        tsdataset.astype({"a": "int32"})
        with self.assertRaises(ValueError):
            ts2vec.fit(tsdataset)

        # case6 (训练集包含NaN)
        tsdataset = self.tsdataset1.copy()
        tsdataset["a"][0] = np.NaN
        with self.assertLogs("paddlets", level="WARNING") as captured:
            ts2vec.fit(tsdataset)
            self.assertEqual(
                captured.records[0].getMessage(), 
                "Input `a` contains np.inf or np.NaN, which may lead to unexpected results from the model."
            )
            
    def test_init_callbacks(self):
        """unittest function
        """
        # case1 (用户未传入callbacks)
        ts2vec = TS2Vec(
            segment_size=300,
            sampling_stride=300,
            max_epochs=1
        )
        _, callback_container = ts2vec._init_callbacks()
        self.assertEqual(len(callback_container._callbacks), 1)

        # case4 (用户传入callbacks)
        callback = Callback()
        ts2vec = TS2Vec(
            segment_size=300,
            sampling_stride=300,
            callbacks=[callback]
        )
        _, callback_container = ts2vec._init_callbacks()
        self.assertEqual(len(callback_container._callbacks), 2)

    def test_fit(self):
        """unittest function
        """
        # case1 (用户只传入训练集, log不显示评估指标, 达到最大epochs训练结束)
        ts2vec = TS2Vec(
            segment_size=300,
            sampling_stride=300,
            max_epochs=1,
        )
        ts2vec.fit(self.tsdataset1)
        
        # case2 (用户传入多组时序数据用于多实例训练)
        ts2vec.fit([self.tsdataset1, self.tsdataset1])

    def test_encode(self):
        """unittest function
        """
        # case1 (index为DatetimeIndex的单变量/非sliding+all_true)
        ts2vec = TS2Vec(
            segment_size=300,
            sampling_stride=300,
            max_epochs=1,
        )
        ts2vec.fit(self.tsdataset1)
        res = ts2vec.encode(self.tsdataset1)
        self.assertIsInstance(res, np.ndarray)
        self.assertEqual(res.shape, (1, 800, 320))

        # case2 (index为DatetimeIndex的多变量/非sliding+all_true)
        ts2vec.fit(self.tsdataset2)
        res = ts2vec.encode(self.tsdataset2)
        self.assertIsInstance(res, np.ndarray)
        self.assertEqual(res.shape, (1, 800, 320))

        # case3 (index为DatetimeIndex的多变量/非sliding+mask_last)
        res = ts2vec.encode(self.tsdataset2, mask="mask_last")
        self.assertIsInstance(res, np.ndarray)
        self.assertEqual(res.shape, (1, 800, 320))

        # case4 (index为DatetimeIndex的多变量/instance level encoding)
        res = ts2vec.encode(self.tsdataset2, encoding_type="full_series")
        self.assertIsInstance(res, np.ndarray)
        self.assertEqual(res.shape, (1, 320))

        # case5 (index为DatetimeIndex的多变量/sliding+full_series)
        res = ts2vec.encode(self.tsdataset2, sliding_len=50, encoding_type="full_series")
        self.assertIsInstance(res, np.ndarray)
        self.assertEqual(res.shape, (1, 320))

        # case6 (index为DatetimeIndex的多变量/sliding+multiscale)
        res = ts2vec.encode(self.tsdataset2, sliding_len=50, encoding_type="multiscale")
        self.assertIsInstance(res, np.ndarray)
        self.assertEqual(res.shape, (1, 800, 1920))

        # case7 (mask 不合法)
        with self.assertRaises(ValueError):
            res = ts2vec.encode(self.tsdataset2, mask="all_false")

        # case8 (encoding_type 不合法)
        with self.assertRaises(ValueError):
            res = ts2vec.encode(self.tsdataset2, encoding_type="multi_scale")


if __name__ == "__main__":
    unittest.main()
