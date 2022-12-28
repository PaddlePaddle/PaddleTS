# !/usr/bin/env python3
# -*- coding:utf-8 -*-

import pandas as pd
from paddlets.pipeline.pipeline import Pipeline
from paddlets.transform.fill import Fill
import numpy as np
from sklearn.linear_model import LinearRegression

import unittest
from unittest import TestCase

from paddlets.models.forecasting import MLPRegressor
from paddlets.models.forecasting import NHiTSModel
from paddlets import TimeSeries, TSDataset
from paddlets.ensemble import StackingEnsembleForecaster


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

        model1 = StackingEnsembleForecaster(
            in_chunk_len=7 * 96 + 20 * 4,
            out_chunk_len=96,
            skip_chunk_len=4 * 4,
            estimators=[(MLPRegressor, mlp_params)])
        assert model1 is not None

        # case2
        mlp1_params = {
            'eval_metrics': ["mse", "mae"]
        }

        mlp2_params = {
            'eval_metrics': ["mse", "mae"]
        }

        nhits_params = {
            'eval_metrics': ["mse", "mae"]
        }

        model2 = StackingEnsembleForecaster(
            in_chunk_len=7 * 96 + 20 * 4,
            out_chunk_len=96,
            skip_chunk_len=4 * 4,
            estimators=[(MLPRegressor, mlp1_params), (MLPRegressor, mlp2_params)])
        assert model2 is not None

        # case3
        mlp1_params = {
            'eval_metrics': ["mse", "mae"]
        }

        mlp2_params = {
            'eval_metrics': ["mse", "mae"]
        }

        nhits_params = {
            'eval_metrics': ["mse", "mae"]
        }

        model2 = StackingEnsembleForecaster(
            in_chunk_len=7 * 96 + 20 * 4,
            out_chunk_len=96,
            skip_chunk_len=4 * 4,
            estimators=[(MLPRegressor, mlp1_params), (MLPRegressor, mlp2_params)],
            final_learner=LinearRegression())
        assert model2 is not None

        # case4 badcase (wrong final learner)

        mlp1_params = {
            'eval_metrics': ["mse", "mae"]
        }

        mlp2_params = {
            'eval_metrics': ["mse", "mae"]
        }

        nhits_params = {
            'eval_metrics': ["mse", "mae"]
        }

        with self.assertRaises(ValueError):
            model2 = StackingEnsembleForecaster(
                in_chunk_len=7 * 96 + 20 * 4,
                out_chunk_len=96,
                skip_chunk_len=4 * 4,
                estimators=[(MLPRegressor, mlp1_params), (MLPRegressor, mlp2_params)],
                final_learner=NHiTSModel)

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

        nhits_params = {
            'eval_metrics': ["mse", "mae"]
        }

        model1 = StackingEnsembleForecaster(
            in_chunk_len=16,
            out_chunk_len=16,
            skip_chunk_len=4,
            estimators=[(MLPRegressor, mlp1_params), (MLPRegressor, mlp2_params)])

        model1.fit(tsdataset)

        # case2
        mlp1_params = {
            'eval_metrics': ["mse", "mae"]
        }

        mlp2_params = {
            'eval_metrics': ["mse", "mae"]
        }

        nhits_params = {
            'eval_metrics': ["mse", "mae"]
        }

        model1 = StackingEnsembleForecaster(
            in_chunk_len=16,
            out_chunk_len=16,
            skip_chunk_len=4,
            estimators=[(MLPRegressor, mlp1_params), (MLPRegressor, mlp2_params)],
            final_learner=LinearRegression())

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

        nhits_params = {
            'eval_metrics': ["mse", "mae"]
        }

        model1 = StackingEnsembleForecaster(
            in_chunk_len=16,
            out_chunk_len=16,
            skip_chunk_len=4 * 4,
            estimators=[(MLPRegressor, mlp1_params), (MLPRegressor, mlp2_params)])

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

        nhits_params = {
            'eval_metrics': ["mse", "mae"]
        }

        model1 = StackingEnsembleForecaster(
            in_chunk_len=16,
            out_chunk_len=16,
            skip_chunk_len=0,
            estimators=[(MLPRegressor, mlp1_params), (MLPRegressor, mlp2_params)])

        model1.fit(tsdataset)
        predcitions = model1.predict(tsdataset)
        print(predcitions)
        predcitions2 = model1.recursive_predict(tsdataset, predict_length=100)
        assert (len(predcitions.target) == 16)
        assert (len(predcitions2.target) == 100)

        #case3 fit with valid_dataset
        model1 = StackingEnsembleForecaster(
            in_chunk_len=16,
            out_chunk_len=16,
            skip_chunk_len=0,
            estimators=[(MLPRegressor, mlp1_params), (MLPRegressor, mlp2_params)])

        model1.fit(tsdataset,tsdataset)
        predcitions = model1.predict(tsdataset)
        print(predcitions)
        predcitions2 = model1.recursive_predict(tsdataset, predict_length=100)
        assert (len(predcitions.target) == 16)


        #case4 fit with valid_dataset  , houldout
        model1 = StackingEnsembleForecaster(
            in_chunk_len=16,
            out_chunk_len=16,
            skip_chunk_len=0,
            resampling_strategy="holdout",
            estimators=[(MLPRegressor, mlp1_params), (MLPRegressor, mlp2_params)])

        test,val = tsdataset.split(0.2)
        model1.fit(test,val)
        predcitions = model1.predict(tsdataset)
        print(predcitions)
        predcitions2 = model1.recursive_predict(tsdataset, predict_length=100)
        assert (len(predcitions.target) == 16)


        #case5 pipeline
        pipe_params = {
            "steps":[(Fill,{"cols":"a1"}),(MLPRegressor,mlp2_params)]
        }
        model1 = StackingEnsembleForecaster(
            in_chunk_len=16,
            out_chunk_len=16,
            skip_chunk_len=0,
            resampling_strategy="holdout",
            estimators=[(MLPRegressor, mlp1_params), (MLPRegressor, mlp2_params), (Pipeline, pipe_params)])

        test,val = tsdataset.split(0.2)
        model1.fit(test,val)
        predcitions = model1.predict(tsdataset)
        print(predcitions)
        predcitions2 = model1.recursive_predict(tsdataset, predict_length=100)
        assert (len(predcitions.target) == 16)
        
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

        nhits_params = {
            'eval_metrics': ["mse", "mae"],
        }

        model1 = StackingEnsembleForecaster(
            in_chunk_len=16,
            out_chunk_len=16,
            skip_chunk_len=4 * 4,
            estimators=[(MLPRegressor, mlp1_params), (MLPRegressor, mlp2_params)])

        model1.fit(tsdataset)
        model1.save(path="/tmp/ensemble_test2/")

        model1 = model1.load(path="/tmp/ensemble_test2/")
        import shutil
        shutil.rmtree("/tmp/ensemble_test2/")
        predictions = model1.predict(tsdataset)
        assert (len(predictions.target) == 16)
