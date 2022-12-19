# !/usr/bin/env python3
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np

import unittest
from unittest import TestCase

from paddlets.models.forecasting import MLPRegressor
from paddlets import TimeSeries, TSDataset
from paddlets.ensemble import WeightingEnsembleForecaster, WeightingEnsembleAnomaly


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

        nhits_params = {
            'eval_metrics': ["mse", "mae"]
        }

        model2 = WeightingEnsembleForecaster(
            in_chunk_len=7 * 96 + 20 * 4,
            out_chunk_len=96,
            skip_chunk_len=4 * 4,
            estimators=[(MLPRegressor, mlp1_params), (MLPRegressor, mlp2_params)])
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

        nhits_params = {
            'eval_metrics': ["mse", "mae"]
        }

        model1 = WeightingEnsembleForecaster(
            in_chunk_len=16,
            out_chunk_len=96,
            skip_chunk_len=4 * 4,
            estimators=[(MLPRegressor, mlp1_params), (MLPRegressor, mlp2_params)])

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

        model1 = WeightingEnsembleForecaster(
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

        model1 = WeightingEnsembleForecaster(
            in_chunk_len=16,
            out_chunk_len=16,
            skip_chunk_len=4 * 4,
            estimators=[(MLPRegressor, mlp1_params), (MLPRegressor, mlp2_params)])

        model1.fit(tsdataset)
        predcitions = model1.predict(tsdataset)
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

        model1 = WeightingEnsembleForecaster(
            in_chunk_len=16,
            out_chunk_len=16,
            skip_chunk_len=4 * 4,
            estimators=[(MLPRegressor, mlp1_params), (MLPRegressor, mlp2_params)])

        model1.fit(tsdataset)
        model1.save(path="/tmp/ensemble_test3/")

        model1 = model1.load(path="/tmp/ensemble_test3/")
        import shutil
        shutil.rmtree("/tmp/ensemble_test3/")
        prediction = model1.predict(tsdataset)
        assert (len(prediction.target) == 16)

from paddlets.models.anomaly import AutoEncoder
class TestWeightingEnsembleAnomly(TestCase):

    def setUp(self):
        """unittest function
        """
        np.random.seed(2022)
        label = pd.Series(
                np.random.randint(0, 2, 200),
                index=pd.date_range("2022-01-01", periods=200, freq="15T", name='timestamp'),
                name="label")
        feature = pd.DataFrame(
                np.random.randn(200, 2).astype(np.float32),
                index=pd.date_range("2022-01-01", periods=200, freq="15T", name='timestamp'),
                columns=["a", "b"])
        
        # index is DatetimeIndex
        self.tsdataset1 = TSDataset.load_from_dataframe(pd.concat([label,feature],axis=1), 
                target_cols='label', feature_cols='a')
        
        # There is no target in tsdataset
        self.tsdataset2 = TSDataset.load_from_dataframe(pd.concat([label,feature],axis=1), 
                feature_cols=['a', 'b'])     
        
        # index is RangeIndex
        index = pd.RangeIndex(0, 200, 1)
        label = label.reset_index(drop=True).reindex(index)
        feature = feature.reset_index(drop=True).reindex(index)
        self.tsdataset3 = TSDataset.load_from_dataframe(pd.concat([label,feature],axis=1), 
                label_col='label', feature_cols='a')
        # There is no target in tsdataset
        self.tsdataset4 = TSDataset.load_from_dataframe(pd.concat([label,feature],axis=1), 
                feature_cols=['a', 'b']) 
        self.test_init()     
        super().setUp()

    def test_init(self):
        """unittest function for init
        """
        # case1 
        ae_param = {
            "batch_size": 1,
            "max_epochs": 1,
            "verbose": 1,
            "patience": 1,
        }
        
        estimators = [(AutoEncoder, ae_param)]
        self._model1 = WeightingEnsembleAnomaly(1,estimators)


        # case2
        #from paddlets.models.anomaly import IForest, PCA, LOF, KNN, OCSVM
        param = { 
                "sampling_stride" : 1,
                "contamination" : 0.2}
        estimators = [(AutoEncoder, ae_param),(AutoEncoder, ae_param)]
        self._model2 = WeightingEnsembleAnomaly(1,estimators)

        # case3
        param = { 
                "sampling_stride" : 1,
                "contamination" : 0.2}
        estimators = [(AutoEncoder, ae_param)]
        self._model3 = WeightingEnsembleAnomaly(1,estimators,mode="voting")

        # case4
        param = { 
                "sampling_stride" : 1,
                "contamination" : 0.2}
        estimators = [(AutoEncoder, ae_param),(AutoEncoder, ae_param)]
        self._model4 = WeightingEnsembleAnomaly(1,estimators,standardization=False)

        # case5
        param = { 
                "sampling_stride" : 1,
                "contamination" : 0.2}
        estimators = [(AutoEncoder, ae_param)]
        self._model5 = WeightingEnsembleAnomaly(1,estimators,contamination=0.2)

    def test_fit(self):
        """
        unittest function for fit
        """
        # case1 
        self._model1.fit(self.tsdataset1)
        self._model1.fit(self.tsdataset2)
        self._model1.fit(self.tsdataset3)
        self._model1.fit(self.tsdataset4)

        # case2
        self._model2.fit(self.tsdataset1)
        self._model2.fit(self.tsdataset2)
        self._model2.fit(self.tsdataset3)
        self._model2.fit(self.tsdataset4)

        # case3
        self._model3.fit(self.tsdataset1)
        self._model3.fit(self.tsdataset2)
        self._model3.fit(self.tsdataset3)
        self._model3.fit(self.tsdataset4)


        # case4
        self._model4.fit(self.tsdataset1)
        self._model4.fit(self.tsdataset2)
        self._model4.fit(self.tsdataset3)
        self._model4.fit(self.tsdataset4)

        # case5
        self._model5.fit(self.tsdataset1)
        self._model5.fit(self.tsdataset2)
        self._model5.fit(self.tsdataset3)
        self._model5.fit(self.tsdataset4)


    def test_predict_score(self):
        # model1 
        self._model1.fit(self.tsdataset1)
        res1 = self._model1.predict_score(self.tsdataset1) 
        assert(len(res1.target) == 200)

        self._model1.fit(self.tsdataset2)
        res2 = self._model1.predict_score(self.tsdataset2) 
        assert(len(res2.target) == 200)

        self._model1.fit(self.tsdataset3)
        res3 = self._model1.predict_score(self.tsdataset3) 
        assert(len(res3.target) == 200)

        self._model1.fit(self.tsdataset4)
        res4 = self._model1.predict_score(self.tsdataset4) 
        assert(len(res4.target) == 200)

        # model2
        self._model2.fit(self.tsdataset1)
        res1 = self._model2.predict_score(self.tsdataset1) 
        assert(len(res1.target) == 200)

        self._model2.fit(self.tsdataset2)
        res2 = self._model2.predict_score(self.tsdataset2) 
        assert(len(res2.target) == 200)

        self._model2.fit(self.tsdataset3)
        res3 = self._model2.predict_score(self.tsdataset3) 
        assert(len(res3.target) == 200)

        self._model2.fit(self.tsdataset4)
        res4 = self._model2.predict_score(self.tsdataset4) 
        assert(len(res4.target) == 200)

        # model3
        # self._model3.fit(self.tsdataset1)
        # res1 = self._model3.predict_score(self.tsdataset1) 
        # assert(len(res1.target) == 200)

        # self._model3.fit(self.tsdataset2)
        # res2 = self._model3.predict_score(self.tsdataset2) 
        # assert(len(res2.target) == 200)

        # self._model3.fit(self.tsdataset3)
        # res3 = self._model3.predict_score(self.tsdataset3) 
        # assert(len(res3.target) == 200)

        # self._model3.fit(self.tsdataset4)
        # res4 = self._model3.predict_score(self.tsdataset4) 
        # assert(len(res4.target) == 200)

        # model4
        self._model4.fit(self.tsdataset1)
        res1 = self._model4.predict_score(self.tsdataset1) 
        assert(len(res1.target) == 200)

        self._model4.fit(self.tsdataset2)
        res2 = self._model4.predict_score(self.tsdataset2) 
        assert(len(res2.target) == 200)

        self._model4.fit(self.tsdataset3)
        res3 = self._model4.predict_score(self.tsdataset3) 
        assert(len(res3.target) == 200)

        self._model4.fit(self.tsdataset4)
        res4 = self._model4.predict_score(self.tsdataset4) 
        assert(len(res4.target) == 200)

        # model5
        self._model5.fit(self.tsdataset1)
        res1 = self._model5.predict_score(self.tsdataset1) 
        assert(len(res1.target) == 200)

        self._model5.fit(self.tsdataset2)
        res2 = self._model5.predict_score(self.tsdataset2) 
        assert(len(res2.target) == 200)

        self._model5.fit(self.tsdataset3)
        res3 = self._model5.predict_score(self.tsdataset3) 
        assert(len(res3.target) == 200)

        self._model5.fit(self.tsdataset4)
        res4 = self._model5.predict_score(self.tsdataset4) 
        assert(len(res4.target) == 200)


    def test_predict(self):
        # model1 
        self._model1.fit(self.tsdataset1)
        res1 = self._model1.predict(self.tsdataset1) 
        assert(len(res1.target) == 200)

        self._model1.fit(self.tsdataset2)
        res2 = self._model1.predict(self.tsdataset2) 
        assert(len(res2.target) == 200)

        self._model1.fit(self.tsdataset3)
        res3 = self._model1.predict(self.tsdataset3) 
        assert(len(res3.target) == 200)

        self._model1.fit(self.tsdataset4)
        res4 = self._model1.predict(self.tsdataset4) 
        assert(len(res4.target) == 200)

        # model2
        self._model2.fit(self.tsdataset1)
        res1 = self._model2.predict(self.tsdataset1) 
        assert(len(res1.target) == 200)

        self._model2.fit(self.tsdataset2)
        res2 = self._model2.predict(self.tsdataset2) 
        assert(len(res2.target) == 200)

        self._model2.fit(self.tsdataset3)
        res3 = self._model2.predict(self.tsdataset3) 
        assert(len(res3.target) == 200)

        self._model2.fit(self.tsdataset4)
        res4 = self._model2.predict(self.tsdataset4) 
        assert(len(res4.target) == 200)

        # model3
        self._model3.fit(self.tsdataset1)
        res1 = self._model3.predict(self.tsdataset1) 
        assert(len(res1.target) == 200)

        self._model3.fit(self.tsdataset2)
        res2 = self._model3.predict(self.tsdataset2) 
        assert(len(res2.target) == 200)

        self._model3.fit(self.tsdataset3)
        res3 = self._model3.predict(self.tsdataset3) 
        assert(len(res3.target) == 200)

        self._model3.fit(self.tsdataset4)
        res4 = self._model3.predict(self.tsdataset4) 
        assert(len(res4.target) == 200)

        # model4
        self._model4.fit(self.tsdataset1)
        res1 = self._model4.predict(self.tsdataset1) 
        assert(len(res1.target) == 200)

        self._model4.fit(self.tsdataset2)
        res2 = self._model4.predict(self.tsdataset2) 
        assert(len(res2.target) == 200)

        self._model4.fit(self.tsdataset3)
        res3 = self._model4.predict(self.tsdataset3) 
        assert(len(res3.target) == 200)

        self._model4.fit(self.tsdataset4)
        res4 = self._model4.predict(self.tsdataset4) 
        assert(len(res4.target) == 200)

        # model5
        self._model5.fit(self.tsdataset1)
        res1 = self._model5.predict(self.tsdataset1) 
        assert(len(res1.target) == 200)

        self._model5.fit(self.tsdataset2)
        res2 = self._model5.predict(self.tsdataset2) 
        assert(len(res2.target) == 200)

        self._model5.fit(self.tsdataset3)
        res3 = self._model5.predict(self.tsdataset3) 
        assert(len(res3.target) == 200)

        self._model5.fit(self.tsdataset4)
        res4 = self._model5.predict(self.tsdataset4) 
        assert(len(res4.target) == 200)

    def test_save_and_load(self):
        self._model1.fit(self.tsdataset1)
        res1 = self._model1.predict(self.tsdataset1) 
        assert(len(res1.target) == 200)

        self._model1.save(path="/tmp/ensemble_test2/")

        model1 = self._model1.load(path="/tmp/ensemble_test2/")
        import shutil
        shutil.rmtree("/tmp/ensemble_test2/")
        predictions = model1.predict(self.tsdataset1)
        assert (len(predictions.target) == 200)
