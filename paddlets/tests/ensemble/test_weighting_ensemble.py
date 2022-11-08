# !/usr/bin/env python3
# -*- coding:utf-8 -*-

from xml.etree.ElementPath import prepare_predicate
import pandas as pd
import numpy as np

import unittest
from unittest import TestCase

from paddlets.models.forecasting import MLPRegressor
from paddlets import TimeSeries, TSDataset
from paddlets.ensemble import WeightingEnsembleForecaster, StackingEnsembleForecaster


class TestEnsembleBase(TestCase):
    def setUp(self):
        """unittest function
        """
        super().setUp()

    def test_init(self):
        # case1
        mlp_params = {
            'eval_metrics': ["mse", "mae"]
        }

        model1 = WeightingEnsembleForecaster(
            in_chunk_len=7 * 96 + 20 * 4,
            out_chunk_len=96,
            skip_chunk_len=4 * 4,
            estimators=[(MLPRegressor, mlp_params)])
        assert model1 is not None

        # case2
        mlp_params = {
            'in_chunk_len': 7 * 96 + 20 * 4,
            'out_chunk_len': 96,
            'skip_chunk_len': 4 * 4,
            'eval_metrics': ["mse", "mae"]
        }

        model2 = WeightingEnsembleForecaster(
            in_chunk_len=7 * 96 + 20 * 4,
            out_chunk_len=96,
            skip_chunk_len=4 * 4,
            estimators=[(MLPRegressor, mlp_params)])
        assert model2 is not None

        # case3
        mlp1_params = {
            'eval_metrics': ["mse", "mae"]
        }

        mlp2_params = {
            'eval_metrics': ["mse", "mae"]
        }

        mlp3_params = {
            'eval_metrics': ["mse", "mae"]
        }

        model2 = WeightingEnsembleForecaster(
            in_chunk_len=7 * 96 + 20 * 4,
            out_chunk_len=96,
            skip_chunk_len=4 * 4,
            estimators=[(MLPRegressor, mlp1_params), (MLPRegressor, mlp2_params), (MLPRegressor, mlp3_params)])
        assert model2 is not None

    def test_fit(self):
        np.random.seed(2022)

        target = TimeSeries.load_from_dataframe(
            pd.Series(np.random.randn(200).astype(np.float32),
                      index=pd.date_range("2022-01-01", periods=200, freq="15T"),
                      name="a"
                      ))

        observed_cov = TimeSeries.load_from_dataframe(
            pd.DataFrame(
                np.random.randn(200, 2).astype(np.float32),
                index=pd.date_range("2022-01-01", periods=200, freq="15T"),
                columns=["b", "c"]
            ))
        known_cov = TimeSeries.load_from_dataframe(
            pd.DataFrame(
                np.random.randn(250, 2).astype(np.float32),
                index=pd.date_range("2022-01-01", periods=250, freq="15T"),
                columns=["b1", "c1"]
            ))
        static_cov = None
        tsdataset = TSDataset(target, observed_cov, known_cov, static_cov)

        # case1
        mlp1_params = {
            'eval_metrics': ["mse", "mae"]
        }

        mlp2_params = {
            'eval_metrics': ["mse", "mae"]
        }

        mlp3_params = {
            'eval_metrics': ["mse", "mae"]
        }

        model1 = WeightingEnsembleForecaster(
            in_chunk_len=16,
            out_chunk_len=96,
            skip_chunk_len=4 * 4,
            estimators=[(MLPRegressor, mlp1_params), (MLPRegressor, mlp2_params), (MLPRegressor, mlp3_params)])

        model1.fit(tsdataset)

    def test_predict(self):
        np.random.seed(2022)

        # case1 single-target
        target = TimeSeries.load_from_dataframe(
            pd.Series(np.random.randn(200).astype(np.float32),
                      index=pd.date_range("2022-01-01", periods=200, freq="15T"),
                      name="a"
                      ))

        observed_cov = TimeSeries.load_from_dataframe(
            pd.DataFrame(
                np.random.randn(200, 2).astype(np.float32),
                index=pd.date_range("2022-01-01", periods=200, freq="15T"),
                columns=["b", "c"]
            ))
        known_cov = TimeSeries.load_from_dataframe(
            pd.DataFrame(
                np.random.randn(250, 2).astype(np.float32),
                index=pd.date_range("2022-01-01", periods=250, freq="15T"),
                columns=["b1", "c1"]
            ))
        static_cov = None
        tsdataset = TSDataset(target, observed_cov, known_cov, static_cov)

        mlp1_params = {
            'eval_metrics': ["mse", "mae"]
        }

        mlp2_params = {
            'eval_metrics': ["mse", "mae"]
        }

        mlp3_params = {
            'eval_metrics': ["mse", "mae"]
        }

        model1 = WeightingEnsembleForecaster(
            in_chunk_len=16,
            out_chunk_len=16,
            skip_chunk_len=4 * 4,
            estimators=[(MLPRegressor, mlp1_params), (MLPRegressor, mlp2_params), (MLPRegressor, mlp3_params)])

        model1.fit(tsdataset)
        predcitions = model1.predict(tsdataset)
        assert (len(predcitions.target) == 16)

        # case2 multi-target
        target = TimeSeries.load_from_dataframe(
            pd.DataFrame(
                np.random.randn(200, 2).astype(np.float32),
                index=pd.date_range("2022-01-01", periods=200, freq="15T"),
                columns=["a1", "a2"]
            ))

        observed_cov = TimeSeries.load_from_dataframe(
            pd.DataFrame(
                np.random.randn(200, 2).astype(np.float32),
                index=pd.date_range("2022-01-01", periods=200, freq="15T"),
                columns=["b", "c"]
            ))
        known_cov = TimeSeries.load_from_dataframe(
            pd.DataFrame(
                np.random.randn(250, 2).astype(np.float32),
                index=pd.date_range("2022-01-01", periods=250, freq="15T"),
                columns=["b1", "c1"]
            ))
        static_cov = None
        tsdataset = TSDataset(target, observed_cov, known_cov, static_cov)

        mlp1_params = {
            'eval_metrics': ["mse", "mae"]
        }

        mlp2_params = {
            'eval_metrics': ["mse", "mae"]
        }

        mlp3_params = {
            'eval_metrics': ["mse", "mae"]
        }

        model1 = WeightingEnsembleForecaster(
            in_chunk_len=16,
            out_chunk_len=16,
            skip_chunk_len=4 * 4,
            estimators=[(MLPRegressor, mlp1_params), (MLPRegressor, mlp2_params), (MLPRegressor, mlp3_params)])

        model1.fit(tsdataset)
        predcitions = model1.predict(tsdataset)
        assert (len(predcitions.target) == 16)

    def test_get_support_modes(self):
        WeightingEnsembleForecaster.get_support_modes()

    def test_save_and_load(self):
        np.random.seed(2022)

        target = TimeSeries.load_from_dataframe(
            pd.Series(np.random.randn(200).astype(np.float32),
                      index=pd.date_range("2022-01-01", periods=200, freq="15T"),
                      name="a"
                      ))

        observed_cov = TimeSeries.load_from_dataframe(
            pd.DataFrame(
                np.random.randn(200, 2).astype(np.float32),
                index=pd.date_range("2022-01-01", periods=200, freq="15T"),
                columns=["b", "c"]
            ))
        known_cov = TimeSeries.load_from_dataframe(
            pd.DataFrame(
                np.random.randn(250, 2).astype(np.float32),
                index=pd.date_range("2022-01-01", periods=250, freq="15T"),
                columns=["b1", "c1"]
            ))
        static_cov = None
        tsdataset = TSDataset(target, observed_cov, known_cov, static_cov)

        # case1
        mlp1_params = {
            'eval_metrics': ["mse", "mae"]
        }

        mlp2_params = {
            'eval_metrics': ["mse", "mae"]
        }

        mlp3_params = {
            'eval_metrics': ["mse", "mae"],
        }

        model1 = WeightingEnsembleForecaster(
            in_chunk_len=16,
            out_chunk_len=16,
            skip_chunk_len=4 * 4,
            estimators=[(MLPRegressor, mlp1_params), (MLPRegressor, mlp2_params), (MLPRegressor, mlp3_params)])

        model1.fit(tsdataset)
        model1.save(path="/tmp/ensemble_test3/")

        model1 = model1.load(path="/tmp/ensemble_test3/")
        import shutil
        shutil.rmtree("/tmp/ensemble_test3/")
        prediction = model1.predict(tsdataset)
        assert (len(prediction.target) == 16)
