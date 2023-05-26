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
from paddlets.models.forecasting import NLinearModel
from paddlets.datasets.repository import get_dataset, dataset_list
from paddlets.transform import StandardScaler

np.random.seed(2023)
paddle.seed(0)


class TestNLinearModel(TestCase):
    def setUp(self):
        # mock data
        target_single = TimeSeries.load_from_dataframe(
            pd.Series(
                np.random.randn(2000).astype(np.float32),
                index=pd.date_range(
                    "2022-01-01", periods=2000, freq="15T"),
                name="a", ))
        target_multi = TimeSeries.load_from_dataframe(
            pd.DataFrame(
                np.random.randn(2000, 2).astype(np.float32),
                index=pd.date_range(
                    "2022-01-01", periods=2000, freq="15T"),
                columns=["a1", "a2"], ))

        observed_cov = TimeSeries.load_from_dataframe(
            pd.DataFrame(
                np.random.randn(2000, 3).astype(np.float32),
                index=pd.date_range(
                    "2022-01-01", periods=2000, freq="15T"),
                columns=["b1", "b2", "b3"], ))
        known_cov = TimeSeries.load_from_dataframe(
            pd.DataFrame(
                np.random.randn(2500, 2).astype(np.float32),
                index=pd.date_range(
                    "2022-01-01", periods=2500, freq="15T"),
                columns=["c1", "c2"], ))
        static_cov = {"f": 1.0, "g": 2.0}

        int_target = TimeSeries.load_from_dataframe(
            pd.Series(
                np.random.randint(0, 10, 2000).astype(np.int32),
                index=pd.date_range(
                    "2022-01-01", periods=2000, freq="15T"),
                name="a", ))
        category_known_cov = TimeSeries.load_from_dataframe(
            pd.DataFrame(
                np.random.choice(["a", "b", "c"], [2500, 2]),
                index=pd.date_range(
                    "2022-01-01", periods=2500, freq="15T"),
                columns=["c1", "c2"], ))

        self.data_single_target = TSDataset(target_single, observed_cov,
                                            known_cov, static_cov)
        self.data_multi_target = TSDataset(target_multi, observed_cov,
                                           known_cov, static_cov)
        self.data_no_cov = TSDataset(target_single, None, None, None)
        self.data_int_type = TSDataset(int_target, None, category_known_cov,
                                       None)

        # real world data
        data_wth = get_dataset("UNI_WTH")
        param = {
            "cols": ["WetBulbCelsius"],
            "with_mean": True,
            "with_std": True
        }
        scaler = eval("StandardScaler")(**param)
        scaler.fit(data_wth)
        self.data_real_world = scaler.transform(data_wth)

        super().setUp()

    def test_fit(self):
        # case1: single-target, real world data
        reg = NLinearModel(
            in_chunk_len=7 * 96 + 9 * 4,
            out_chunk_len=96,
            skip_chunk_len=15 * 4,
            sampling_stride=96,
            eval_metrics=["mse", "mae"],
            batch_size=128,
            max_epochs=5, )
        reg.fit(self.data_real_world, self.data_real_world)

        # case2: multi-targets, mock data
        reg = NLinearModel(
            in_chunk_len=7 * 96 + 9 * 4,
            out_chunk_len=96,
            skip_chunk_len=15 * 4,
            eval_metrics=["mse", "mae"],
            batch_size=128,
            max_epochs=5, )
        reg.fit(self.data_multi_target, self.data_multi_target)

        # case3: no covariates
        reg = NLinearModel(
            in_chunk_len=7 * 96 + 9 * 4,
            out_chunk_len=96,
            skip_chunk_len=15 * 4,
            eval_metrics=["mse", "mae"],
            batch_size=128,
            max_epochs=5, )
        reg.fit(self.data_no_cov, self.data_no_cov)

        # case4: add hidden layer
        reg = NLinearModel(
            in_chunk_len=7 * 96 + 9 * 4,
            out_chunk_len=96,
            skip_chunk_len=15 * 4,
            sampling_stride=96,
            eval_metrics=["mse", "mae"],
            batch_size=128,
            max_epochs=5,
            hidden_config=[100], )
        reg.fit(self.data_real_world, self.data_real_world)

        # case5: use batch normalization
        reg = NLinearModel(
            in_chunk_len=7 * 96 + 9 * 4,
            out_chunk_len=96,
            skip_chunk_len=15 * 4,
            sampling_stride=96,
            eval_metrics=["mse", "mae"],
            batch_size=128,
            max_epochs=5,
            hidden_config=[100],
            use_bn=True, )
        reg.fit(self.data_real_world, self.data_real_world)

        # case6: bad case, invalid dtypes
        with self.assertRaises(ValueError):
            reg = NLinearModel(
                in_chunk_len=7 * 96 + 9 * 4,
                out_chunk_len=96,
                skip_chunk_len=15 * 4,
                eval_metrics=["mse", "mae"],
                batch_size=128,
                max_epochs=5, )
            reg.fit(self.data_int_type, self.data_int_type)

    def test_predict(self):
        reg = NLinearModel(
            in_chunk_len=7 * 96 + 9 * 4,
            out_chunk_len=96,
            skip_chunk_len=15 * 4,
            sampling_stride=96,
            eval_metrics=["mse", "mae"],
            batch_size=128,
            max_epochs=5, )

        # case1: single target, real world data
        reg.fit(self.data_real_world, self.data_real_world)
        res = reg.predict(self.data_real_world)
        self.assertIsInstance(res, TSDataset)
        self.assertEqual(res.get_target().data.shape, (96, 1))

        reg = NLinearModel(
            in_chunk_len=7 * 96 + 9 * 4,
            out_chunk_len=96,
            skip_chunk_len=15 * 4,
            eval_metrics=["mse", "mae"],
            batch_size=128,
            max_epochs=5, )

        # case2: multi-target
        reg.fit(self.data_multi_target, self.data_multi_target)
        res = reg.predict(self.data_multi_target)
        self.assertIsInstance(res, TSDataset)
        self.assertEqual(res.get_target().data.shape, (96, 2))


if __name__ == "__main__":
    unittest.main()
