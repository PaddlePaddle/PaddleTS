# !/usr/bin/env python3
# -*- coding:utf-8 -*-

from unittest import TestCase, mock
import unittest
import random
import time

import pandas as pd
import numpy as np

from paddlets.datasets import TimeSeries, TSDataset
from paddlets.models.common.callbacks import Callback
from paddlets.models.forecasting import RNNBlockRegressor


class TestRNNRegressor(TestCase):
    """
    TestRNNRegressor
    """

    def setUp(self):
        """
        unittest function
        """
        np.random.seed(2023)
        single_target = TimeSeries.load_from_dataframe(
            pd.Series(
                np.random.randn(2000).astype(np.float32),
                index=pd.date_range("2022-01-01", periods=2000, freq="15T"),
                name="a"
            ))
        multi_target = TimeSeries.load_from_dataframe(
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

        int_target = TimeSeries.load_from_dataframe(
            pd.Series(
                np.random.randint(0, 10, 2000).astype(np.int32),
                index=pd.date_range("2022-01-01", periods=2000, freq="15T"),
                name="a"
            ))
        category_known_cov = TimeSeries.load_from_dataframe(
            pd.DataFrame(
                np.random.choice([0], [2500, 2]),
                index=pd.date_range("2022-01-01", periods=2500, freq="15T"),
                columns=["c1", "c2"]
            ))

        static_cov = {"f": 0, "g": 2.0}
        self.tsdataset1 = TSDataset(single_target, observed_cov, known_cov, static_cov)
        self.tsdataset2 = TSDataset(multi_target, observed_cov, category_known_cov, static_cov)
        self.tsdataset3 = TSDataset(int_target, None, category_known_cov, None)
        super().setUp()

    def test_init_model(self):
        """
        test init model
        """
        # case1
        param1 = {
            "rnn_type_or_module": "LSTM",
            "dropout": 0.1,
            "batch_size": 1,
            "max_epochs": 1,
            "verbose": 1,
            "patience": 1
        }

        model = RNNBlockRegressor(
            in_chunk_len=10,
            out_chunk_len=5,
            skip_chunk_len=4 * 4,
            eval_metrics=["mse", "mae"],
            **param1
        )

        # case 2: rnn type invalid
        param2 = {
            "rnn_type_or_module": "lstm",
            "batch_size": 1,
            "max_epochs": 1,
            "verbose": 1,
            "patience": 1,
        }
        with self.assertRaises(ValueError):
            model = RNNBlockRegressor(
                in_chunk_len=10,
                out_chunk_len=5,
                skip_chunk_len=4 * 4,
                eval_metrics=["mse", "mae"],
                **param2
            )

    def test_fit(self):
        """
        test fit:

        """
        # case1: single_target
        reg1 = RNNBlockRegressor(
            in_chunk_len=10,
            out_chunk_len=5,
            skip_chunk_len=4 * 4,
            eval_metrics=["mse", "mae"]
        )
        reg1.fit(self.tsdataset1)
        reg1.fit(self.tsdataset1, self.tsdataset1)

        # case2: single_target, known_cov, observed_cov
        reg2 = RNNBlockRegressor(
            in_chunk_len=10,
            out_chunk_len=5,
            rnn_type_or_module="LSTM",
            fcn_out_config=[64],
            skip_chunk_len=4 * 4,
            eval_metrics=["mse", "mae"]
        )
        reg2.fit(self.tsdataset1, self.tsdataset1)

        # case3: multi_target
        reg3 = RNNBlockRegressor(
            in_chunk_len=10,
            out_chunk_len=5,
            skip_chunk_len=4 * 4,
            eval_metrics=["mse", "mae"]
        )
        reg3.fit(self.tsdataset2, self.tsdataset2)

        # case4: multi_target, known_cov, observed_cov
        reg4 = RNNBlockRegressor(
            in_chunk_len=10,
            out_chunk_len=5,
            rnn_type_or_module="GRU",
            fcn_out_config=[64],
            skip_chunk_len=4 * 4,
            eval_metrics=["mse", "mae"]
        )
        reg4.fit(self.tsdataset2, self.tsdataset2)

        # case5: invalid dtypes
        with self.assertRaises(ValueError):
            reg5 = RNNBlockRegressor(
                in_chunk_len=10,
                out_chunk_len=5,
                rnn_type_or_module="GRU",
                fcn_out_config=[64],
                skip_chunk_len=4 * 4,
                eval_metrics=["mse", "mae"],
            )
            reg5.fit(self.tsdataset3, self.tsdataset3)

    def test_predict(self):
        """
        test predict
        """
        reg = RNNBlockRegressor(
            in_chunk_len=10,
            out_chunk_len=5,
            rnn_type_or_module="SimpleRNN",
            fcn_out_config=[32],
            skip_chunk_len=4 * 4,
            eval_metrics=["mse", "mae"]
        )
        # case1: single_target
        reg.fit(self.tsdataset1, self.tsdataset1)
        res = reg.predict(self.tsdataset1)
        self.assertIsInstance(res, TSDataset)
        self.assertEqual(res.get_target().data.shape, (5, 1))

        # case2: multi_target
        reg.fit(self.tsdataset2, self.tsdataset2)
        res = reg.predict(self.tsdataset2)

        self.assertIsInstance(res, TSDataset)
        self.assertEqual(res.get_target().data.shape, (5, 2))


if __name__ == "__main__":
    unittest.main()
