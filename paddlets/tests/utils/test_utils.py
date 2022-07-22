#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import sys
sys.path.append(".")
from unittest import TestCase
import unittest

import pandas as pd
import numpy as np

from paddlets.models.dl.paddlepaddle import LSTNetRegressor
from paddlets.datasets import TimeSeries, TSDataset
from paddlets.utils.utils import check_model_fitted
from paddlets.pipeline.pipeline import Pipeline
from paddlets.utils import get_uuid

class TestUtils(TestCase):
    def setUp(self):
        """unittest function
        """
        super().setUp()

    def test_check_model_fitted(self):
        """unittest function
        """
        np.random.seed(2022)
        target1 = TimeSeries.load_from_dataframe(
            pd.Series(
                np.random.randn(2000).astype(np.float32),
                index=pd.date_range("2022-01-01", periods=2000, freq="15T"),
                name="a"
            ))
        target2 = TimeSeries.load_from_dataframe(
            pd.DataFrame(
                np.random.randn(2000, 2).astype(np.float32),
                index=pd.date_range("2022-01-01", periods=2000, freq="15T"),
                columns=["a1", "a2"]
            ))
        observed_cov = TimeSeries.load_from_dataframe(
            pd.DataFrame(
                np.random.randn(2000, 2).astype(np.float32),
                index=pd.date_range("2022-01-01", periods=2000, freq="15T"),
                columns=["b", "c"]
            ))
        known_cov = TimeSeries.load_from_dataframe(
            pd.DataFrame(
                np.random.randn(2500, 2).astype(np.float32),
                index=pd.date_range("2022-01-01", periods=2500, freq="15T"),
                columns=["b1", "c1"]
            ))
        static_cov = {"f": 1, "g": 2}
        self.tsdataset1 = TSDataset(target1, observed_cov, known_cov, static_cov)

        # case1 fitted paddle
        lstnet = LSTNetRegressor(
            in_chunk_len=1 * 96 + 20 * 4,
            out_chunk_len=96,
            max_epochs=1
        )
        lstnet.fit(self.tsdataset1)
        check_model_fitted(lstnet)

        # case2 fitted pipeline
        param = {"in_chunk_len": 1 * 96 + 20 * 4, "out_chunk_len": 96, "max_epochs": 1}
        pipe = Pipeline([(LSTNetRegressor, param)])
        pipe.fit(self.tsdataset1, self.tsdataset1)
        check_model_fitted(lstnet)

        #case3 not fit paddle
        lstnet = LSTNetRegressor(
            in_chunk_len=1 * 96 + 20 * 4,
            out_chunk_len=96,
            max_epochs=1
        )
        with self.assertRaises(ValueError):
            check_model_fitted(lstnet)

        # case4 not init
        with self.assertRaises(ValueError):
            check_model_fitted(LSTNetRegressor)

        # case5 not fit paddle, add self defined msg
        lstnet = LSTNetRegressor(
            in_chunk_len=1 * 96 + 20 * 4,
            out_chunk_len=96,
            max_epochs=1
        )
        with self.assertRaises(ValueError):
            check_model_fitted(lstnet, msg=" %(name)s test")

    def test_get_uuid(self):
        """
        unittest function
        """
        uuid = get_uuid("hello-", "-world")
        self.assertEqual(len(uuid), 28)

if __name__ == "__main__":
    unittest.main()

