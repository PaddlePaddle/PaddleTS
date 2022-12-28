# !/usr/bin/env python3
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np

import unittest
from unittest import TestCase
from typing import Callable, List, Optional, Tuple, Union

from paddlets.models.forecasting import MLPRegressor
from paddlets.models.forecasting import NHiTSModel
from paddlets import TimeSeries, TSDataset
from paddlets.ensemble.base import EnsembleBase


class MockEnsemble(EnsembleBase):
    def fit(self,
            train_tsdataset: TSDataset,
            valid_tsdataset: Optional[TSDataset] = None) -> None:
        pass

    def predict(self,
                tsdataset: TSDataset) -> None:
        pass


class TestEnsembleBase(TestCase):
    def setUp(self):
        """unittest function
        """
        super().setUp()

    def test_init(self):
        # case1
        mlp_params = {
            'in_chunk_len': 4 * 4,
            'out_chunk_len': 1,
            'skip_chunk_len': 0,
        }

        model1 = MockEnsemble(
            estimators=[(MLPRegressor, mlp_params)])
        assert model1 is not None

        # case2
        mlp_params = {
            'in_chunk_len': 4 * 4,
            'out_chunk_len': 1,
            'skip_chunk_len': 0,
            'eval_metrics': ["mse", "mae"]
        }

        model2 = MockEnsemble(
            estimators=[(MLPRegressor, mlp_params)])
        assert model2 is not None

        # case3
        mlp1_params = {
            'in_chunk_len': 4 * 4,
            'out_chunk_len': 1,
            'skip_chunk_len': 0,
            'eval_metrics': ["mse", "mae"]
        }

        mlp2_params = {
            'in_chunk_len': 4 * 4,
            'out_chunk_len': 1,
            'skip_chunk_len': 0,
            'eval_metrics': ["mse", "mae"]
        }

        nhits_params = {
            'in_chunk_len': 4 * 4,
            'out_chunk_len': 1,
            'skip_chunk_len': 0,
            'eval_metrics': ["mse", "mae"]
        }

        model2 = MockEnsemble(
            estimators=[(MLPRegressor, mlp1_params), (MLPRegressor, mlp2_params)])
        assert model2 is not None


    def test_fit_estimators(self):
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
            'in_chunk_len': 4 * 4,
            'out_chunk_len': 1,
            'skip_chunk_len': 0,
            'eval_metrics': ["mse", "mae"]
        }

        mlp2_params = {
            'in_chunk_len': 4 * 4,
            'out_chunk_len': 1,
            'skip_chunk_len': 0,
            'eval_metrics': ["mse", "mae"]
        }

        nhits_params = {
            'in_chunk_len': 4 * 4,
            'out_chunk_len': 1,
            'skip_chunk_len': 0,
            'eval_metrics': ["mse", "mae"]
        }

        model1 = MockEnsemble(
            estimators=[(MLPRegressor, mlp1_params), (MLPRegressor, mlp2_params)])

        model1._fit_estimators(tsdataset)

    def test_predict_estimators(self):
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
            'in_chunk_len': 4 * 4,
            'out_chunk_len': 1,
            'skip_chunk_len': 0,
            'eval_metrics': ["mse", "mae"]
        }

        mlp2_params = {
            'in_chunk_len': 4 * 4,
            'out_chunk_len': 1,
            'skip_chunk_len': 0,
            'eval_metrics': ["mse", "mae"]
        }

        nhits_params = {
            'in_chunk_len': 4 * 4,
            'out_chunk_len': 1,
            'skip_chunk_len': 0,
            'eval_metrics': ["mse", "mae"]
        }
        model1 = MockEnsemble(
            estimators=[(MLPRegressor, mlp1_params), (MLPRegressor, mlp2_params)])

        model1._fit_estimators(tsdataset)
        model1._predict_estimators(tsdataset)

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
            'in_chunk_len': 4 * 4,
            'out_chunk_len': 1,
            'skip_chunk_len': 0,
            'eval_metrics': ["mse", "mae"]
        }

        mlp2_params = {
            'in_chunk_len': 4 * 4,
            'out_chunk_len': 1,
            'skip_chunk_len': 0,
            'eval_metrics': ["mse", "mae"]
        }

        nhits_params = {
            'in_chunk_len': 4 * 4,
            'out_chunk_len': 1,
            'skip_chunk_len': 0,
            'eval_metrics': ["mse", "mae"]
        }

        model1 = MockEnsemble(
            estimators=[(MLPRegressor, mlp1_params), (MLPRegressor, mlp2_params)])

        model1._fit_estimators(tsdataset)
        model1.save(path="/tmp/ensemble_test1/")

        model1.load(path="/tmp/ensemble_test1/")
        import shutil
        shutil.rmtree("/tmp/ensemble_test1/")
        model1._predict_estimators(tsdataset)
