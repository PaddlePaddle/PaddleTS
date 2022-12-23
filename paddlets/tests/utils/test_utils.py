# !/usr/bin/env python3
# -*- coding:utf-8 -*-
import sys
import os
sys.path.append(".")
from unittest import TestCase
import unittest
import json

import pandas as pd
import numpy as np

from paddlets.models.forecasting import LSTNetRegressor
from paddlets.datasets import TimeSeries, TSDataset
from paddlets.datasets.repository import get_dataset
from paddlets.utils.utils import check_model_fitted, repr_results_to_tsdataset, build_ts_infer_input
from paddlets.pipeline.pipeline import Pipeline
from paddlets.utils import get_uuid, plot_anoms
from paddlets.models.forecasting import MLPRegressor
from paddlets.models.forecasting import NHiTSModel
from paddlets.ensemble import StackingEnsembleForecaster

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
        static_cov = {"f": 1.0, "g": 2.0}
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

        # case6 fitted Ensemble
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
            estimators=[(MLPRegressor, mlp1_params), (MLPRegressor, mlp2_params), (NHiTSModel, nhits_params)])

        model1.fit(self.tsdataset1)
        check_model_fitted(model1)

    def test_get_uuid(self):
        """
        unittest function
        """
        uuid = get_uuid("hello-", "-world")
        self.assertEqual(len(uuid), 28)

    def test_repr_results_to_tsdataset(self):
        """
        unittest function
        """
        data_array = np.random.randn(2,4)
        data_df = pd.DataFrame(data_array, columns=['a','b','c','y'])
        tsdataset = TSDataset.load_from_dataframe(
                                df=data_df, 
                                observed_cov_cols=['a', 'b', 'c'],
                                target_cols='y')
        result = repr_results_to_tsdataset(np.random.randn(1, 2, 4), tsdataset)
        self.assertEqual(result.to_dataframe().columns.tolist(), ['y', 'repr_0', 'repr_1', 'repr_2', 'repr_3'])

    def test_plot_anoms(self):
        label = pd.Series(
                np.random.randint(0, 2, 200),
                index=pd.date_range("2022-01-01", periods=200, freq="15T", name='timestamp'),
                name="label")
        feature = pd.DataFrame(
                np.random.randn(200, 2).astype(np.float32),
                index=pd.date_range("2022-01-01", periods=200, freq="15T", name='timestamp'),
                columns=["a", "b"])
  

        # index is RangeIndex
        index = pd.RangeIndex(0, 200, 1)
        label = label.reset_index(drop=True).reindex(index)
        feature = feature.reset_index(drop=True).reindex(index)
        tsdataset = TSDataset.load_from_dataframe(pd.concat([label,feature],axis=1), 
                label_col='label', feature_cols='a')
                
        from paddlets.models.anomaly import AutoEncoder
        ae = AutoEncoder(
            in_chunk_len=32,
            eval_metrics=["mse", "mae"],
            max_epochs=1,
            patience=1
        )
        ae.fit(tsdataset, tsdataset)
        predict = ae.predict(tsdataset)
        plot_anoms(predict,tsdataset,"a")
        plot_anoms(predict,tsdataset)
        plot_anoms(predict)

    def test_build_ts_infer_input(self):
        dataset = get_dataset('WTH')

        #case1: good case for forecasting
        meta_data = {
            "model_type": "forecasting",
            "size": {
                "in_chunk_len": 100,
                "out_chunk_len": 20,
                "skip_chunk_len": 1,
            },
            "input_data": {
                "past_target": [None, 100, 1],
                "observed_cov_numeric": [None, 100, 11]
            },
        }
        file_name = f"/tmp/test_meta_file_{np.random.randint(1000, 200000)}"
        with open(file_name, mode='w') as f:
            json.dump(meta_data, f)
        res = build_ts_infer_input(dataset, file_name)
        self.assertEqual(res['past_target'].shape, (1, 100, 1))
        self.assertTrue((res['past_target'] == dataset.target.to_numpy()[-100:].reshape(1, 100, 1)).all())
        self.assertEqual(res['observed_cov_numeric'].shape, (1, 100, 11))
        self.assertTrue((res['observed_cov_numeric'] == dataset.observed_cov.to_numpy()[-100:].reshape(1, 100, 11)).all())

        #case2: good case for anomaly
        meta_data = {
            "model_type": "anomaly",
            "size": {
                "in_chunk_len": 100,
            },
            "input_data": {
                "observed_cov_numeric": [None, 100, 11]
            },
        }
        file_name = f"/tmp/test_meta_file_{np.random.randint(1000, 200000)}"
        with open(file_name, mode='w') as f:
            json.dump(meta_data, f)
        res = build_ts_infer_input(dataset, file_name)
        self.assertEqual(res['observed_cov_numeric'].shape, (len(dataset.observed_cov) - 100 + 1, 100, 11))

        #case3: bad case, dataset and meta_file not match
        meta_data = {
            "model_type": "forecasting",
            "size": {
                "in_chunk_len": 100,
                "out_chunk_len": 20,
                "skip_chunk_len": 1,
            },
            "input_data": {
                "past_target": [None, 100, 1],
                "observed_cov_numeric": [None, 100, 11],
                "known_cov_numeric": [None, 120, 11],
            },
        }
        file_name = f"/tmp/test_meta_file_{np.random.randint(1000, 200000)}"
        with open(file_name, mode='w') as f:
            json.dump(meta_data, f)
        with self.assertRaises(ValueError):
            res = build_ts_infer_input(dataset, file_name)

        #case4: bad case, meta_file is not right
        meta_data = {
            "model_type": "forecasting",
        }
        file_name = f"/tmp/test_meta_file_{np.random.randint(1000, 200000)}"
        with open(file_name, mode='w') as f:
            json.dump(meta_data, f)
        with self.assertRaises(ValueError):
            res = build_ts_infer_input(dataset, file_name)


if __name__ == "__main__":
    unittest.main()
