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
from paddlets.models.forecasting import DeepARModel


class TestDeepARModel(TestCase):
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

        self.tsdataset1 = TSDataset(single_target, observed_cov, known_cov, static_cov)
        self.tsdataset2 = TSDataset(multi_target, observed_cov, known_cov, static_cov)
        super().setUp()

    def test_init_model(self):
        """
        test init model
        """
        # case1
        param1 = {
                "rnn_type_or_module": "GRU",
                "dropout": 0.1,
                "batch_size": 1,
                "max_epochs": 1,
                "verbose": 1,
                "patience": 1
                }

        model = DeepARModel(
            in_chunk_len=10,
            out_chunk_len=5,
            skip_chunk_len=4 * 4,
            eval_metrics=["mse", "mae"],
            **param1
            )
        
        # case 2: rnn type invalid
        param2 = {
                "rnn_type_or_module": "rnn",
                "batch_size": 1,
                "max_epochs": 1,
                "verbose": 1,
                "patience": 1,
                }
        with self.assertRaises(ValueError):
            model = DeepARModel(
                in_chunk_len= 10,
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
        reg1 = DeepARModel(
            in_chunk_len= 10,
            out_chunk_len=5,
            skip_chunk_len=4 * 4,
            eval_metrics=["quantile_loss"],
            batch_size=512,
            max_epochs=5,
        )
        reg1.fit(self.tsdataset1)
        reg1.fit(self.tsdataset1, self.tsdataset1)
    
        # case2: single_target, known_cov, observed_cov
        reg2 = DeepARModel(
            in_chunk_len= 10,
            out_chunk_len=5,
            skip_chunk_len=4 * 4,
            batch_size=512,
            max_epochs=5,
        )
        reg2.fit(self.tsdataset1, self.tsdataset1)
    
        # case3: multi_target, known_cov, observed_cov
        reg3 = DeepARModel(
            in_chunk_len= 10,
            out_chunk_len=5,
            skip_chunk_len=4 * 4,
            eval_metrics=["quantile_loss"],
            batch_size=512,
            max_epochs=5
        )
        reg3.fit(self.tsdataset2, self.tsdataset2)

    def test_predict(self):
        """
        test predict
        """
        # case 1: sampling & quantiles
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

        reg.fit(self.tsdataset2, self.tsdataset2)
        res = reg.predict(self.tsdataset2)
        self.assertIsInstance(res, TSDataset)
        self.assertEqual(res.get_target().data.shape, (5, 2*101))

        # case 2: mean & quantiles
        reg2 = DeepARModel(
            in_chunk_len=10,
            out_chunk_len=5,
            skip_chunk_len=4 * 4,
            eval_metrics=["mse", "mae"],
            batch_size=512,
            num_samples = 101,
            regression_mode="mean",
            output_mode="quantiles",
            max_epochs=5
        )

        reg2.fit(self.tsdataset2, self.tsdataset2)
        res = reg2.predict(self.tsdataset2)
        self.assertIsInstance(res, TSDataset)
        self.assertEqual(res.get_target().data.shape, (5, 2*101))

        # case 3: sampling & predictions
        reg3 = DeepARModel(
            in_chunk_len=10,
            out_chunk_len=5,
            skip_chunk_len=4 * 4,
            eval_metrics=["mse", "mae"],
            batch_size=512,
            num_samples = 101,
            regression_mode="sampling",
            output_mode="predictions",
            max_epochs=5
        )

        reg3.fit(self.tsdataset2, self.tsdataset2)
        res = reg3.predict(self.tsdataset2)
        self.assertIsInstance(res, TSDataset)
        self.assertEqual(res.get_target().data.shape, (5, 2))

        #case 4: mean & predictions
        reg4 = DeepARModel(
            in_chunk_len=10,
            out_chunk_len=5,
            skip_chunk_len=4 * 4,
            eval_metrics=["mse", "mae"],
            batch_size=512,
            num_samples = 101,
            regression_mode="mean",
            output_mode="predictions",
            max_epochs=5
        )
        reg4.fit(self.tsdataset2, self.tsdataset2)
        res = reg4.predict(self.tsdataset2)
        self.assertIsInstance(res, TSDataset)
        self.assertEqual(res.get_target().data.shape, (5, 2))


if __name__ == "__main__":
    unittest.main()

