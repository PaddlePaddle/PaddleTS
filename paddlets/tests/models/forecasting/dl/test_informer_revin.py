# !/usr/bin/env python3
# -*- coding:utf-8 -*-

from unittest import TestCase
import unittest
import random

import pandas as pd
import numpy as np

from paddlets.models.forecasting import InformerModel
from paddlets.datasets import TimeSeries, TSDataset


class TestInformer(TestCase):
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
        # case1 (训练集的target为非float类型. 视具体模型而定, 目前transformer模型不支持target为除float之外的类型)
        informer = InformerModel(
            in_chunk_len=96,
            out_chunk_len=96,
            d_model=8,
            nhead=1,
            num_encoder_layers=1,
            num_decoder_layers=1,
            ffn_channels=64,
            batch_size=512,
            max_epochs=1
        )
        tsdataset = self.tsdataset1.copy()
        tsdataset.astype({"a": "int32"})
        with self.assertRaises(ValueError):
            informer.fit(tsdataset, tsdataset)

        # case2 (in_chunk_len 小于 start_token_len)
        informer = InformerModel(
            in_chunk_len=96,
            out_chunk_len=96,
            start_token_len=100,
            d_model=8,
            nhead=1,
            num_encoder_layers=1,
            num_decoder_layers=1,
            ffn_channels=64,
            batch_size=512,
            max_epochs=1
        )
        with self.assertRaises(ValueError):
            informer.fit(self.tsdataset1, self.tsdataset1)

    def test_fit(self):
        """unittest function
        """
        # case1 (用户只传入训练集, log不显示评估指标, 达到最大epochs训练结束)
        informer = InformerModel(
            in_chunk_len=96,
            out_chunk_len=96,
            d_model=8,
            nhead=1,
            num_encoder_layers=1,
            num_decoder_layers=1,
            ffn_channels=64,
            optimizer_params=dict(learning_rate=1e-1),
            use_revin=True,
            batch_size=512,
            max_epochs=5,
            patience=1,
        )
        informer.fit(self.tsdataset1)

        # case2 (用户同时传入训练/评估集和, log显示评估指标, 同时early_stopping生效)
        informer = InformerModel(
            in_chunk_len=96,
            out_chunk_len=96,
            d_model=8,
            nhead=1,
            num_encoder_layers=1,
            num_decoder_layers=1,
            ffn_channels=64,
            optimizer_params=dict(learning_rate=1e-1),
            use_revin=True,
            batch_size=512,
            max_epochs=5,
            patience=1,
        )
        informer.fit(self.tsdataset1, self.tsdataset1)

        informer = InformerModel(
            in_chunk_len=96,
            out_chunk_len=96,
            d_model=8,
            nhead=1,
            num_encoder_layers=1,
            num_decoder_layers=1,
            ffn_channels=64,
            optimizer_params=dict(learning_rate=1e-1),
            use_revin=True,
            batch_size=512,
            max_epochs=5,
            patience=1,
        )
        informer.fit([self.tsdataset1, self.tsdataset1], self.tsdataset1)

    def test_predict(self):
        """unittest function
        """
        # case1 (index为DatetimeIndex的单变量预测)
        informer = InformerModel(
            in_chunk_len=96,
            out_chunk_len=96,
            d_model=8,
            nhead=1,
            num_encoder_layers=1,
            num_decoder_layers=1,
            ffn_channels=64,
            use_revin=True,
            batch_size=512,
            max_epochs=1,
        )
        informer.fit(self.tsdataset1, self.tsdataset1)
        res = informer.predict(self.tsdataset1)

        self.assertIsInstance(res, TSDataset)
        self.assertEqual(res.get_target().data.shape, (96, 1))

        # case2 (index为DatetimeIndex的多变量预测)
        informer.fit(self.tsdataset2, self.tsdataset2)
        res = informer.predict(self.tsdataset2)
        self.assertIsInstance(res, TSDataset)
        self.assertEqual(res.get_target().data.shape, (96, 2))

        # case3 (index为RangeIndex的多变量预测)
        informer.fit(self.tsdataset3, self.tsdataset3)
        res = informer.predict(self.tsdataset3)
        self.assertIsInstance(res.get_target().data.index, pd.RangeIndex)
        self.assertIsInstance(res, TSDataset)
        self.assertEqual(res.get_target().data.shape, (96, 2))

    def test_recursive_predict(self):
        """unittest function
        """
        # case1 (单变量预测)
        informer = InformerModel(
            in_chunk_len=96,
            out_chunk_len=96,
            skip_chunk_len=16,
            d_model=8,
            nhead=1,
            num_encoder_layers=1,
            num_decoder_layers=1,
            ffn_channels=64,
            eval_metrics=["mse", "mae"],
            use_revin=True,
            batch_size=512,
            max_epochs=1
        )
        informer.fit(self.tsdataset1, self.tsdataset1)
        # not supported when skip_chunk_len > 0
        with self.assertRaises(ValueError):
            res = informer.recursive_predict(self.tsdataset1, 96)

        informer = InformerModel(
            in_chunk_len=96,
            out_chunk_len=96,
            skip_chunk_len=0,
            d_model=8,
            nhead=1,
            num_encoder_layers=1,
            num_decoder_layers=1,
            ffn_channels=64,
            use_revin=True,
            eval_metrics=["mse", "mae"],
            batch_size=512,
            max_epochs=1
        )
        informer.fit(self.tsdataset1, self.tsdataset1)
        # not supported when predict_lenth < 0
        with self.assertRaises(ValueError):
            res = informer.recursive_predict(self.tsdataset1, 0)
        res = informer.recursive_predict(self.tsdataset1, 1100)
        self.assertIsInstance(res, TSDataset)
        self.assertEqual(res.get_target().data.shape, (1100, 1))

        res = informer.recursive_predict(self.tsdataset1, 10)
        self.assertIsInstance(res, TSDataset)
        self.assertEqual(res.get_target().data.shape, (10, 1))

        # case2 (多变量预测)
        informer.fit(self.tsdataset2, self.tsdataset2)
        res = informer.recursive_predict(self.tsdataset2, 200)
        self.assertIsInstance(res, TSDataset)
        self.assertEqual(res.get_target().data.shape, (200, 2))

        # case3 (range Index)
        informer.fit(self.tsdataset3, self.tsdataset3)
        res = informer.recursive_predict(self.tsdataset3, 2500)
        self.assertIsInstance(res, TSDataset)
        self.assertEqual(res.get_target().data.shape, (2500, 2))
 

if __name__ == "__main__":
    unittest.main()

