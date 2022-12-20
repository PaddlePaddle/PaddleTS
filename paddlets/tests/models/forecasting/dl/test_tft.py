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
from paddlets.models.forecasting import TFTModel as TFT


class TestTFT(TestCase):
    """
    TestTFT
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
                "output_quantiles": [0.5, 0.05, 0.95],
                "batch_size": 128,
                "max_epochs": 2,
                "patience": 1
                }

        model = TFT(
            in_chunk_len=10,
            out_chunk_len=5,
            skip_chunk_len=4 * 4,
            **param1
            )
    
    def test_fit(self):
        """
        test fit:
        """
        # case1: single_target
        reg1 = TFT(
            in_chunk_len= 10,
            out_chunk_len=5,
            skip_chunk_len=4 * 4,
            output_quantiles = [0.5, 0.05, 0.95],
            batch_size=512,
            max_epochs=3,
        )
        reg1.fit(self.tsdataset1)
        reg1.fit(self.tsdataset1, self.tsdataset1)
    
        # case2: multi_target, known_cov, observed_cov
        reg3 = TFT(
            in_chunk_len= 10,
            out_chunk_len=5,
            skip_chunk_len=4 * 4,
            batch_size=512,
            max_epochs=3
        )
        reg1.fit(self.tsdataset1)
        reg3.fit(self.tsdataset2, self.tsdataset2)

    def test_predict(self):
        """
        test predict
        """
        # case 1: ordinal result
        reg = TFT(
            in_chunk_len=10,
            out_chunk_len=5,
            skip_chunk_len=4 * 4,
            batch_size=512,
            max_epochs=1
        )

        reg.fit(self.tsdataset2, self.tsdataset2)
        res = reg.predict(self.tsdataset2)
        self.assertIsInstance(res, TSDataset)        
        self.assertEqual(res.get_target().data.shape, (5, 2*3))
        
        # case 2: interpretable result
        reg2 = TFT(
            in_chunk_len=10,
            out_chunk_len=5,
            skip_chunk_len=4 * 4,
            batch_size=512,
            max_epochs=1
        )

        reg2.fit(self.tsdataset2, self.tsdataset2)
        res = reg2.predict_interpretable(self.tsdataset2)
        self.assertIsInstance(res, dict)


if __name__ == "__main__":
    unittest.main()

