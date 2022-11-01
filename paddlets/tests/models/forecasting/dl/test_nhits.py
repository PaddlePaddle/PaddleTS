# !/usr/bin/env python3
# -*- coding:utf-8 -*-

from unittest import TestCase, mock
import unittest
from enum import Enum
from typing import List, NewType, Tuple, Union

import paddle
import pandas as pd
import numpy as np

from paddlets.datasets import TimeSeries, TSDataset
from paddlets.models.common.callbacks import Callback
from paddlets.models.forecasting import NHiTSModel
from paddlets.datasets.repository import get_dataset, dataset_list
from paddlets.transform import StandardScaler


np.random.seed(2023)
paddle.seed(0)

    
class TestNHitsModel(TestCase):
    def setUp(self):
        # mock data
        target_single = TimeSeries.load_from_dataframe(
            pd.Series(
                np.random.randn(2000).astype(np.float32),
                index=pd.date_range("2022-01-01", periods=2000, freq="15T"),
                name="a"
            ))
        target_multi = TimeSeries.load_from_dataframe(
            pd.DataFrame(
                np.random.randn(2000, 2).astype(np.float32),
                index=pd.date_range("2022-01-01", periods=2000, freq="15T"),
                columns=["a1", "a2"]
            ))
        
        observed_cov = TimeSeries.load_from_dataframe(
            pd.DataFrame(
                np.random.randn(2000, 3).astype(np.float32),
                index=pd.date_range("2022-01-01", periods=2000, freq="15T"),
                columns=["b1", "b2", "b3"]
            ))
        known_cov = TimeSeries.load_from_dataframe(
            pd.DataFrame(
                np.random.randn(2500, 2).astype(np.float32),
                index=pd.date_range("2022-01-01", periods=2500, freq="15T"),
                columns=["c1", "c2"]
            ))
        static_cov = {"f": 1.0, "g": 2.0}

        int_target = TimeSeries.load_from_dataframe(
            pd.Series(
                np.random.randint(0, 10, 2000).astype(np.int32),
                index=pd.date_range("2022-01-01", periods=2000, freq="15T"),
                name="a"
            ))
        category_known_cov = TimeSeries.load_from_dataframe(
            pd.DataFrame(
                np.random.choice(["a", "b", "c"], [2500, 2]),
                index=pd.date_range("2022-01-01", periods=2500, freq="15T"),
                columns=["c1", "c2"]
            ))

        self.tsdataset1 = TSDataset(target_single, observed_cov, known_cov, static_cov)
        self.tsdataset2 = TSDataset(target_multi, observed_cov, known_cov, static_cov)
        self.tsdataset3 = TSDataset(target_single, None, None, None)
        self.tsdataset5 = TSDataset(int_target, None, category_known_cov, None)
        # real world
        ts4 = get_dataset("UNI_WTH")
        param = {"cols": ["WetBulbCelsius"], "with_mean": True, "with_std": True}
        scaler = eval("StandardScaler")(**param)
        scaler.fit(ts4)
        self.tsdataset4 = scaler.transform(ts4)
        super().setUp()

    def test_fit(self):
        # case1: single-targets, real world data
        reg = NHiTSModel(
            in_chunk_len = 7 * 96 + 9 * 4,
            out_chunk_len = 96,
            skip_chunk_len = 15 * 4,
            sampling_stride = 96,
            eval_metrics = ["mse", "mae"],
            layer_widths = [64, 64, 64],
            batch_size = 256,
        )
        #reg.fit(self.tsdataset1)
        #reg.fit(self.tsdataset1, self.tsdataset1)
        reg.fit(self.tsdataset4, self.tsdataset4)

        # case2: multi-targets, mock data
        reg = NHiTSModel(
                in_chunk_len = 7 * 96 + 9 * 4,
                out_chunk_len = 96,
                skip_chunk_len = 15 * 4,
                eval_metrics = ["mse", "mae"],
                layer_widths = [128, 128, 128]
                )
        reg.fit(self.tsdataset2, self.tsdataset2)

        # case3: no covariates
        reg = NHiTSModel(
                in_chunk_len = 7 * 96 + 9 * 4,
                out_chunk_len = 96,
                skip_chunk_len = 15 * 4,
                eval_metrics = ["mse", "mae"],
                )
        reg.fit(self.tsdataset3, self.tsdataset3)

        # case4: bad case, layer_widths invalid
        with self.assertRaises(ValueError):
            reg = NHiTSModel(
                    in_chunk_len = 7 * 96 + 9 * 4,
                    out_chunk_len = 96,
                    skip_chunk_len = 15 * 4,
                    eval_metrics = ["mse", "mae"],
                    layer_widths = [64, 64]
                    )
            reg.fit(self.tsdataset3, self.tsdataset3)
        
        # case5: bad case, activation invalid
        with self.assertRaises(ValueError):
            reg = NHiTSModel(
                    in_chunk_len = 7 * 96 + 9 * 4,
                    out_chunk_len = 96,
                    skip_chunk_len = 15 * 4,
                    eval_metrics = ["mse", "mae"],
                    layer_widths = [64, 64, 64],
                    activation = "relu"
                    )
            reg.fit(self.tsdataset1, self.tsdataset1)
        
        # case6: bad case, invalid dtypes
        with self.assertRaises(ValueError):
            reg = NHiTSModel(
                    in_chunk_len = 7 * 96 + 9 * 4,
                    out_chunk_len = 96,
                    skip_chunk_len = 15 * 4,
                    eval_metrics = ["mse", "mae"],
                    layer_widths = [64, 64, 64],
                    activation = "relu"
                    )
            reg.fit(self.tsdataset5, self.tsdataset5)

    def test_predict(self):
        reg = NHiTSModel(
            in_chunk_len = 7 * 96 + 9 * 4,
            out_chunk_len = 96,
            skip_chunk_len = 15 * 4,
            sampling_stride = 96,
            eval_metrics = ["mse", "mae"]
        )
        # case1: single target, real world data
        reg.fit(self.tsdataset4, self.tsdataset4)
        res = reg.predict(self.tsdataset4)
        self.assertIsInstance(res, TSDataset)
        self.assertEqual(res.get_target().data.shape, (96, 1))

        reg = NHiTSModel(
            in_chunk_len = 7 * 96 + 9 * 4,
            out_chunk_len = 96,
            skip_chunk_len = 15 * 4,
            eval_metrics = ["mse", "mae"]
        )

        # case2: multi-target
        reg.fit(self.tsdataset2, self.tsdataset2)
        res = reg.predict(self.tsdataset2)
        self.assertIsInstance(res, TSDataset)
        self.assertEqual(res.get_target().data.shape, (96, 2))
if __name__ == "__main__":
    unittest.main()
