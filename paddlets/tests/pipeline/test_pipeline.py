# !/usr/bin/env python3
# -*- coding:utf-8 -*-

import unittest
from unittest import TestCase

import pandas as pd
import numpy as np

from bts.models.dl.paddlepaddle.callbacks import Callback
from bts.models.dl.paddlepaddle import MLPRegressor
from bts.transform.ksigma import KSigma
from bts.datasets.tsdataset import TimeSeries, TSDataset
from bts.pipeline.pipeline import Pipeline

class TestPipeline(TestCase):
    def setUp(self):
        """
        unittest function
        """
        super().setUp()

    def test_fit(self):
        """
        unittest function
        """

        np.random.seed(2022)

        target = TimeSeries.load_from_dataframe(
                pd.Series(np.random.randn(2000).astype(np.float32),
                index=pd.date_range("2022-01-01", periods=2000, freq="15T"),
                name="a"
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
        tsdataset = TSDataset(target, observed_cov, known_cov, static_cov)
        transform_params = {"cols":['b1'], "k": 0.5}
        transform_params_1 = {"cols":['c1'], "k": 0.7}
        nn_params = {
            'in_chunk_len': 7 * 96 + 20 * 4,
            'out_chunk_len': 96,
            'skip_chunk_len': 4 * 4,
            'eval_metrics': ["mse", "mae"]
        }

        try:
            pipe = Pipeline([(KSigma)])
        except Exception as e:
            succeed = False
            self.assertFalse(succeed)

        pipe = Pipeline([(KSigma, transform_params), (KSigma, transform_params_1), (MLPRegressor, nn_params)])
        pipe.fit(tsdataset, tsdataset)
        self.assertEqual(len(pipe._transform_list), 2)
        
    def test_predict(self):
        """
        unittest function
        """

        np.random.seed(2022)

        target = TimeSeries.load_from_dataframe(
                pd.Series(np.random.randn(2000).astype(np.float32),
                index=pd.date_range("2022-01-01", periods=2000, freq="15T"),
                name="a"
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
        tsdataset = TSDataset(target, observed_cov, known_cov, static_cov)
        transform_params = {"cols":['b1'], "k": 0.5}
        transform_params_1 = {"cols":['c1'], "k": 0.7}
        nn_params = {
            'in_chunk_len': 7 * 96 + 20 * 4,
            'out_chunk_len': 96,
            'skip_chunk_len': 4 * 4,
            'eval_metrics': ["mse", "mae"]
        }

        pipe = Pipeline([(KSigma, transform_params), (KSigma, transform_params_1), (MLPRegressor, nn_params)])
        pipe.fit(tsdataset, tsdataset)
        res = pipe.predict(tsdataset)
        self.assertIsInstance(res, TSDataset)
        self.assertEqual(res.get_target().data.shape, (96, 1))

    def test_predict_proba(self):
        """
        unittest function
        """
        pass

    def test_save(self):
        """
        unittest function
        """
        pass

    def test_load(self):
        """
        unittest function
        """
        pass

if __name__ == "__main__":
    unittest.main()