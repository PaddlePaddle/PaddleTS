# !/usr/bin/env python3
# -*- coding:utf-8 -*-
import os
import unittest
import shutil
import copy

import pandas as pd
import numpy as np
from unittest import TestCase

from paddlets.models.forecasting import MLPRegressor
from paddlets.models.anomaly import AutoEncoder
from paddlets.transform import KSigma, TimeFeatureGenerator, StandardScaler, StatsTransform, Fill
from paddlets.datasets.tsdataset import TimeSeries, TSDataset
from paddlets.pipeline.pipeline import Pipeline
from paddlets.utils import get_uuid
from paddlets.datasets.repository import get_dataset


class TestPipeline(TestCase):
    def setUp(self):
        """
        unittest function
        """
        self.tmp_dir = "/tmp/paddlets_pipeline_test" + get_uuid()
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
        static_cov = {"f": 1., "g": 2.}
        tsdataset = TSDataset(target, observed_cov, known_cov, static_cov)
        transform_params = {"cols": ['b1'], "k": 0.5}
        transform_params_1 = {"cols": ['c1'], "k": 0.7}
        nn_params = {
            'in_chunk_len': 7 * 96 + 20 * 4,
            'out_chunk_len': 96,
            'skip_chunk_len': 4 * 4,
            'eval_metrics': ["mse", "mae"],
            'max_epochs': 3
        }

        try:
            pipe = Pipeline([(KSigma)])
        except Exception as e:
            succeed = False
            self.assertFalse(succeed)

        pipe = Pipeline([(KSigma, transform_params), (KSigma, transform_params_1), \
                         (MLPRegressor, nn_params)])
        pipe.fit(tsdataset, tsdataset)
        self.assertEqual(len(pipe._transform_list), 2)

        pipe.fit([tsdataset, tsdataset], tsdataset)
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
        static_cov = {"f": 1., "g": 2.}
        tsdataset = TSDataset(target, observed_cov, known_cov, static_cov)
        transform_params = {"cols": ['b1'], "k": 0.5}
        transform_params_1 = {"cols": ['c1'], "k": 0.7}
        transform_params_2 = {"cols": ['b', 'c']}

        nn_params = {
            'in_chunk_len': 7 * 96 + 20 * 4,
            'out_chunk_len': 96,
            'skip_chunk_len': 4 * 4,
            'eval_metrics': ["mse", "mae"],
            'max_epochs': 3,
        }

        pipe = Pipeline([(KSigma, transform_params), (KSigma, transform_params_1), (MLPRegressor, nn_params)])
        pipe.fit(tsdataset, tsdataset)
        res = pipe.predict(tsdataset)
        self.assertIsInstance(res, TSDataset)
        self.assertEqual(res.get_target().data.shape, (96, 1))

        nn_params = {
            'in_chunk_len': 10,
            'max_epochs': 3,
        }
        anomaly_pipe = Pipeline(
            [(KSigma, transform_params), (StandardScaler, transform_params_2), (AutoEncoder, nn_params)])
        anomaly_tsdataset = tsdataset.copy()
        anomaly_tsdataset['a'] = anomaly_tsdataset['a'].astype(int)
        anomaly_pipe.fit(anomaly_tsdataset)
        res = anomaly_pipe.predict(anomaly_tsdataset)
        self.assertIsInstance(res, TSDataset)
        self.assertEqual(res.get_target().data.shape, (1991, 1))
        self.assertEqual(res.get_target().data.columns[0], 'a')

        with self.assertRaises(ValueError):
            res = anomaly_pipe.recursive_predict(anomaly_tsdataset, 10)

    def test_transform(self):
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
        static_cov = {"f": 1., "g": 2.}
        tsdataset = TSDataset(target, observed_cov, known_cov, static_cov)
        transform_params = {"cols": ['b1'], "k": 0.5}
        transform_params_1 = {"cols": ['c1'], "k": 0.7}

        pipe = Pipeline([(KSigma, transform_params), (TimeFeatureGenerator, {}), (KSigma, transform_params_1)])
        pipe.fit(tsdataset, tsdataset)
        res = pipe.transform(tsdataset)
        self.assertIsInstance(res, TSDataset)
        self.assertEqual(res.get_all_cov().to_dataframe().shape, (2500, 14))
        # bad case
        with self.assertRaises(RuntimeError):
            res = pipe.recursive_predict(tsdataset, 10)
        with self.assertRaises(RuntimeError):
            res = pipe.predict(tsdataset)
        with self.assertRaises(RuntimeError):
            res = pipe.predict_proba(tsdataset)
        with self.assertRaises(RuntimeError):
            res = pipe.predict_score(tsdataset)
        # transform when model exist
        nn_params = {
            'in_chunk_len': 7 * 96 + 20 * 4,
            'out_chunk_len': 96,
            'skip_chunk_len': 4 * 4,
            'eval_metrics': ["mse", "mae"],
            'max_epochs': 3
        }
        pipe = Pipeline([(KSigma, transform_params), (KSigma, transform_params_1), \
                         (MLPRegressor, nn_params)])
        pipe.fit(tsdataset, tsdataset)
        res = pipe.transform(tsdataset)
        self.assertIsInstance(res, TSDataset)
        self.assertEqual(res.get_all_cov().to_dataframe().shape, (2500, 4))
        # test inverse transform( this pipe would do nothing)
        inverse_res = pipe.inverse_transform(res)
        self.assertIsInstance(inverse_res, TSDataset)
        self.assertEqual(inverse_res.get_all_cov().to_dataframe().shape, (2500, 4))
        # test normaliztion case
        pipe = Pipeline([(KSigma, transform_params), (StandardScaler, {})])
        pipe.fit(tsdataset, tsdataset)
        res = pipe.transform(tsdataset)
        self.assertIsInstance(res, TSDataset)
        self.assertEqual(res.get_all_cov().to_dataframe().shape, (2500, 4))
        inverse_res = pipe.inverse_transform(res)
        self.assertIsInstance(inverse_res, TSDataset)
        self.assertEqual(inverse_res.get_all_cov().to_dataframe().shape, (2500, 4))

    def test_recursive_predict(self):
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
                np.random.randn(2192, 2).astype(np.float32),
                index=pd.date_range("2022-01-01", periods=2192, freq="15T"),
                columns=["b", "c"]
            ))
        known_cov = TimeSeries.load_from_dataframe(
            pd.DataFrame(
                np.random.randn(2500, 2).astype(np.float32),
                index=pd.date_range("2022-01-01", periods=2500, freq="15T"),
                columns=["b1", "c1"]
            ))
        static_cov = {"f": 1., "g": 2.}
        tsdataset = TSDataset(target, observed_cov, known_cov, static_cov)
        transform_params = {"cols": ['b1'], "k": 0.5}
        transform_params_1 = {"cols": ['c1']}
        nn_params = {
            'in_chunk_len': 7 * 96 + 20 * 4,
            'out_chunk_len': 96,
            'skip_chunk_len': 0,
            'eval_metrics': ["mse", "mae"],
            'max_epochs': 3,
        }

        # case1
        pipe = Pipeline([(KSigma, transform_params), (StatsTransform, transform_params_1), \
                         (MLPRegressor, nn_params)])
        pipe.fit(tsdataset, tsdataset)
        res = pipe.recursive_predict(tsdataset, 201)
        self.assertIsInstance(res, TSDataset)
        self.assertEqual(res.get_target().to_dataframe().shape, (201, 1))
        # test recursive predict proba bad case
        with self.assertRaises(ValueError):
            res = pipe.recursive_predict_proba(tsdataset, 202)
        # recursive predict bad case
        # unsupported index type
        tsdataset.get_target().reindex(
            pd.CategoricalIndex(["a", "b", "c", "a", "b", "c"]),
            fill_value=np.nan
        )
        with self.assertRaises(Exception):
            res = pipe.recursive_predict(tsdataset, 201)

        # case2
        np.random.seed(2022)

        target = TimeSeries.load_from_dataframe(
            pd.Series(np.random.randn(2000).astype(np.float32),
                      index=pd.date_range("2022-01-01", periods=2000, freq="15T"),
                      name="a"
                      ))

        observed_cov = TimeSeries.load_from_dataframe(
            pd.DataFrame(
                np.random.randn(2192, 2).astype(np.float32),
                index=pd.date_range("2022-01-01", periods=2192, freq="15T"),
                columns=["b", "c"]
            ))
        known_cov = TimeSeries.load_from_dataframe(
            pd.DataFrame(
                np.random.randn(2500, 2).astype(np.float32),
                index=pd.date_range("2022-01-01", periods=2500, freq="15T"),
                columns=["b1", "c1"]
            ))
        static_cov = {"f": 1., "g": 2.}
        tsdataset = TSDataset(target, observed_cov, known_cov, static_cov)
        transform_params = {"cols": ['b1'], "k": 0.5}
        nn_params = {
            'in_chunk_len': 7 * 96 + 20 * 4,
            'out_chunk_len': 96,
            'skip_chunk_len': 0,
            'eval_metrics': ["mse", "mae"],
            'max_epochs': 3
        }

        transform_params_1 = {"cols": ['c1'], "end": 10}
        pipe = Pipeline(
            [(StatsTransform, transform_params_1), (KSigma, transform_params), (StatsTransform, transform_params_1), \
             (MLPRegressor, nn_params)])
        pipe.fit(tsdataset, tsdataset)
        res = pipe.recursive_predict(tsdataset, 201)
        self.assertIsInstance(res, TSDataset)
        self.assertEqual(res.get_target().to_dataframe().shape, (201, 1))

        # case3
        np.random.seed(2022)

        target = TimeSeries.load_from_dataframe(
            pd.Series(np.random.randn(2000).astype(np.float32),
                      index=pd.date_range("2022-01-01", periods=2000, freq="15T"),
                      name="a"
                      ))

        observed_cov = TimeSeries.load_from_dataframe(
            pd.DataFrame(
                np.random.randn(2192, 2).astype(np.float32),
                index=pd.date_range("2022-01-01", periods=2192, freq="15T"),
                columns=["b", "c"]
            ))
        known_cov = TimeSeries.load_from_dataframe(
            pd.DataFrame(
                np.random.randn(2500, 2).astype(np.float32),
                index=pd.date_range("2022-01-01", periods=2500, freq="15T"),
                columns=["b1", "c1"]
            ))
        static_cov = {"f": 1., "g": 2.}
        tsdataset = TSDataset(target, observed_cov, known_cov, static_cov)
        transform_params = {"cols": ['b1'], "k": 0.5}
        transform_params_1 = {"cols": ['c1']}
        nn_params = {
            'in_chunk_len': 7 * 96 + 20 * 4,
            'out_chunk_len': 96,
            'skip_chunk_len': 0,
            'eval_metrics': ["mse", "mae"],
            'max_epochs': 3
        }
        pipe = Pipeline([(StatsTransform, transform_params_1), \
                         (MLPRegressor, nn_params)])
        pipe.fit(tsdataset, tsdataset)
        res = pipe.recursive_predict(tsdataset, 201)
        self.assertIsInstance(res, TSDataset)
        self.assertEqual(res.get_target().to_dataframe().shape, (201, 1))
        # test recursive predict proba bad case
        with self.assertRaises(ValueError):
            res = pipe.recursive_predict_proba(tsdataset, 202)
        # recursive predict bad case
        # unsupported index type
        tsdataset.get_target().reindex(
            pd.CategoricalIndex(["a", "b", "c", "a", "b", "c"]),
            fill_value=np.nan
        )
        with self.assertRaises(Exception):
            res = pipe.recursive_predict(tsdataset, 201)

        # case4
        np.random.seed(2022)

        target = TimeSeries.load_from_dataframe(
            pd.Series(np.random.randn(2000).astype(np.float32),
                      index=pd.date_range("2022-01-01", periods=2000, freq="15T"),
                      name="a"
                      ))

        observed_cov = TimeSeries.load_from_dataframe(
            pd.DataFrame(
                np.random.randn(2192, 2).astype(np.float32),
                index=pd.date_range("2022-01-01", periods=2192, freq="15T"),
                columns=["b", "c"]
            ))
        known_cov = TimeSeries.load_from_dataframe(
            pd.DataFrame(
                np.random.randn(2500, 2).astype(np.float32),
                index=pd.date_range("2022-01-01", periods=2500, freq="15T"),
                columns=["b1", "c1"]
            ))
        static_cov = {"f": 1., "g": 2.}
        tsdataset = TSDataset(target, observed_cov, known_cov, static_cov)
        transform_params = {"cols": ['b1'], "k": 0.5}
        transform_params_1 = {"cols": ['c1']}
        nn_params = {
            'in_chunk_len': 7 * 96 + 20 * 4,
            'out_chunk_len': 96,
            'skip_chunk_len': 0,
            'eval_metrics': ["mse", "mae"],
            'max_epochs': 3
        }
        pipe = Pipeline([(StatsTransform, transform_params_1), (KSigma, transform_params), (Fill, {"cols": ['c1']}), \
                         (MLPRegressor, nn_params)])
        pipe.fit(tsdataset, tsdataset)
        res = pipe.recursive_predict(tsdataset, 201)
        self.assertIsInstance(res, TSDataset)
        self.assertEqual(res.get_target().to_dataframe().shape, (201, 1))
        # test recursive predict proba bad case
        with self.assertRaises(ValueError):
            res = pipe.recursive_predict_proba(tsdataset, 202)
        # recursive predict bad case
        # unsupported index type
        tsdataset.get_target().reindex(
            pd.CategoricalIndex(["a", "b", "c", "a", "b", "c"]),
            fill_value=np.nan
        )
        with self.assertRaises(Exception):
            res = pipe.recursive_predict(tsdataset, 201)

        # case4
        np.random.seed(2022)

        target = TimeSeries.load_from_dataframe(
            pd.Series(np.random.randn(2000).astype(np.float32),
                      index=pd.date_range("2022-01-01", periods=2000, freq="15T"),
                      name="a"
                      ))

        observed_cov = TimeSeries.load_from_dataframe(
            pd.DataFrame(
                np.random.randn(2192, 2).astype(np.float32),
                index=pd.date_range("2022-01-01", periods=2192, freq="15T"),
                columns=["b", "c"]
            ))
        known_cov = TimeSeries.load_from_dataframe(
            pd.DataFrame(
                np.random.randn(2500, 2).astype(np.float32),
                index=pd.date_range("2022-01-01", periods=2500, freq="15T"),
                columns=["b1", "c1"]
            ))
        static_cov = {"f": 1., "g": 2.}
        tsdataset = TSDataset(target, observed_cov, known_cov, static_cov)
        transform_params = {"cols": ['b1'], "k": 0.5}
        transform_params_1 = {"cols": ['c1']}
        nn_params = {
            'in_chunk_len': 7 * 96 + 20 * 4,
            'out_chunk_len': 96,
            'skip_chunk_len': 0,
            'eval_metrics': ["mse", "mae"],
            'max_epochs': 3
        }
        pipe = Pipeline([(StatsTransform, transform_params_1), (KSigma, transform_params), (Fill, {"cols": ['c1']}), \
                         (MLPRegressor, nn_params)])
        pipe.fit(tsdataset, tsdataset)
        res = pipe.recursive_predict(tsdataset, 201)
        self.assertIsInstance(res, TSDataset)
        self.assertEqual(res.get_target().to_dataframe().shape, (201, 1))
        # test recursive predict proba bad case
        with self.assertRaises(ValueError):
            res = pipe.recursive_predict_proba(tsdataset, 202)
        # recursive predict bad case
        # unsupported index type
        tsdataset.get_target().reindex(
            pd.CategoricalIndex(["a", "b", "c", "a", "b", "c"]),
            fill_value=np.nan
        )
        with self.assertRaises(Exception):
            res = pipe.recursive_predict(tsdataset, 201)

        # case5
        target = TimeSeries.load_from_dataframe(
            pd.Series(np.random.randn(2000).astype(np.float32),
                      index=pd.date_range("2022-01-01", periods=2000, freq="15T"),
                      name="a"
                      ))

        observed_cov = TimeSeries.load_from_dataframe(
            pd.DataFrame(
                np.random.randn(2192, 2).astype(np.float32),
                index=pd.date_range("2022-01-01", periods=2192, freq="15T"),
                columns=["b", "c"]
            ))
        known_cov = TimeSeries.load_from_dataframe(
            pd.DataFrame(
                np.random.randn(2500, 2).astype(np.float32),
                index=pd.date_range("2022-01-01", periods=2500, freq="15T"),
                columns=["b1", "c1"]
            ))
        static_cov = {"f": 1., "g": 2.}
        tsdataset = TSDataset(target, observed_cov, known_cov, static_cov)
        tsdataset.get_observed_cov()
        transform_params = {"cols": ['b1'], "k": 0.5}
        transform_params_1 = {"cols": ['c1']}
        nn_params = {
            'in_chunk_len': 7 * 96 + 20 * 4,
            'out_chunk_len': 96,
            'skip_chunk_len': 0,
            'eval_metrics': ["mse", "mae"],
            'max_epochs': 3
        }
        pipe = Pipeline(
            [(StatsTransform, transform_params_1), (KSigma, transform_params), (Fill, {"cols": ['c1']}), \
             (MLPRegressor, nn_params)])
        pipe.fit(tsdataset, tsdataset)
        res = pipe.recursive_predict(tsdataset, 201)
        with self.assertRaises(RuntimeError):
            res = pipe.recursive_predict(tsdataset, 289)

        target = TimeSeries.load_from_dataframe(
            pd.Series(np.random.randn(2000).astype(np.float32),
                      index=pd.date_range("2022-01-01", periods=2000, freq="15T"),
                      name="a"
                      ))

        observed_cov = TimeSeries.load_from_dataframe(
            pd.DataFrame(
                np.random.randn(2191, 2).astype(np.float32),
                index=pd.date_range("2022-01-01", periods=2191, freq="15T"),
                columns=["b", "c"]
            ))
        known_cov = TimeSeries.load_from_dataframe(
            pd.DataFrame(
                np.random.randn(2500, 2).astype(np.float32),
                index=pd.date_range("2022-01-01", periods=2500, freq="15T"),
                columns=["b1", "c1"]
            ))
        static_cov = {"f": 1., "g": 2.}
        tsdataset = TSDataset(target, observed_cov, known_cov, static_cov)
        with self.assertRaises(RuntimeError):
            res = pipe.recursive_predict(tsdataset, 201)
        res = pipe.recursive_predict(tsdataset, 192)

        target = TimeSeries.load_from_dataframe(
            pd.Series(np.random.randn(2000).astype(np.float32),
                      index=pd.date_range("2022-01-01", periods=2000, freq="15T"),
                      name="a"
                      ))

        observed_cov = TimeSeries.load_from_dataframe(
            pd.DataFrame(
                np.random.randn(2192, 2).astype(np.float32),
                index=pd.date_range("2022-01-01", periods=2192, freq="15T"),
                columns=["b", "c"]
            ))
        known_cov = TimeSeries.load_from_dataframe(
            pd.DataFrame(
                np.random.randn(2096, 2).astype(np.float32),
                index=pd.date_range("2022-01-01", periods=2096, freq="15T"),
                columns=["b1", "c1"]
            ))
        static_cov = {"f": 1., "g": 2.}
        tsdataset = TSDataset(target, observed_cov, known_cov, static_cov)
        res = pipe.recursive_predict(tsdataset, 95)
        res = pipe.recursive_predict(tsdataset, 96)
        with self.assertRaises(RuntimeError):
            res = pipe.recursive_predict(tsdataset, 97)
        with self.assertRaises(RuntimeError):
            res = pipe.recursive_predict(tsdataset, 300)

    def test_predict_proba(self):
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
        static_cov = {"f": 1., "g": 2.}
        tsdataset = TSDataset(target, observed_cov, known_cov, static_cov)
        transform_params = {"cols": ['b1'], "k": 0.5}
        transform_params_1 = {"cols": ['c1'], "k": 0.7}
        nn_params = {
            'in_chunk_len': 7 * 96 + 20 * 4,
            'out_chunk_len': 96,
            'skip_chunk_len': 4 * 4,
            'eval_metrics': ["mse", "mae"],
            'max_epochs': 3
        }

        pipe = Pipeline([(KSigma, transform_params), (KSigma, transform_params_1), (MLPRegressor, nn_params)])
        pipe.fit(tsdataset, tsdataset)
        # test bad case
        with self.assertRaises(ValueError):
            res = pipe.predict_proba(tsdataset)
        # todo: test good case

    def test_predict_score(self):
        """
        unittest function
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

        tsdataset = TSDataset.load_from_dataframe(pd.concat([label, feature], axis=1),
                                                  target_cols='label', feature_cols=['a', 'b'])

        transform_params = {"cols": ['a'], "k": 0.5}
        transform_params_1 = {"cols": ['a', 'b']}

        nn_params = {
            'in_chunk_len': 10,
            'max_epochs': 3
        }
        pipe = Pipeline([(KSigma, transform_params), (StandardScaler, transform_params_1), (AutoEncoder, nn_params)])
        pipe.fit(tsdataset)
        res = pipe.predict_score(tsdataset)
        self.assertIsInstance(res, TSDataset)
        self.assertEqual(res.get_target().data.shape, (191, 1))
        self.assertEqual(res.get_target().data.columns[0], 'label_score')

    def test_save_and_load(self):
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
        static_cov = {"f": 1., "g": 2.}
        tsdataset = TSDataset(target, observed_cov, known_cov, static_cov)
        transform_params = {"cols": ['b1'], "k": 0.5}
        transform_params_1 = {"cols": ['c1'], "k": 0.7}
        transform_params_2 = {"cols": ['b', 'c']}

        nn_params = {
            'in_chunk_len': 7 * 96 + 20 * 4,
            'out_chunk_len': 96,
            'skip_chunk_len': 0,
            'eval_metrics': ["mse", "mae"],
            'max_epochs': 3
        }
        # case1(forecasting)
        pipe = Pipeline([(KSigma, transform_params), (KSigma, transform_params_1), \
                         (MLPRegressor, nn_params)])
        pipe.fit(tsdataset)
        res_before_save_load = pipe.predict(tsdataset)
        # make tmp dir
        # os.mkdir(self.tmp_dir)
        pipe.save(self.tmp_dir)
        # save again
        with self.assertRaises(FileExistsError):
            pipe.save(self.tmp_dir)
        # path is not a directory
        with self.assertRaises(ValueError):
            pipe.save(os.path.join(self.tmp_dir, "pipeline-partial.pkl"))
        # load pipeline bad case
        with self.assertRaises(FileNotFoundError):
            Pipeline.load(self.tmp_dir + "hello")
        # load pipeline
        pipeline_after_save_load = Pipeline.load(self.tmp_dir)
        # test predict
        res_after_save_load = pipeline_after_save_load.predict(tsdataset)
        # clear file
        # shutil.rmtree(self.tmp_dir)
        self.assertTrue(res_before_save_load.get_target().to_dataframe() \
                        .equals(res_after_save_load.get_target().to_dataframe()))

        # case2 (dl anomaly)
        nn_params = {
            'in_chunk_len': 10,
            'max_epochs': 3
        }
        anomaly_pipe = Pipeline(
            [(KSigma, transform_params), (StandardScaler, transform_params_2), (AutoEncoder, nn_params)])
        anomaly_tsdataset = tsdataset.copy()
        anomaly_tsdataset['a'] = anomaly_tsdataset['a'].astype(int)
        anomaly_pipe.fit(anomaly_tsdataset)
        res_before_save_load = anomaly_pipe.predict(anomaly_tsdataset)
        shutil.rmtree(self.tmp_dir)
        anomaly_pipe.save(self.tmp_dir)
        # save again
        with self.assertRaises(FileExistsError):
            anomaly_pipe.save(self.tmp_dir)
        # path is not a directory
        with self.assertRaises(ValueError):
            anomaly_pipe.save(os.path.join(self.tmp_dir, "pipeline-partial.pkl"))
        # load pipeline bad case
        with self.assertRaises(FileNotFoundError):
            Pipeline.load(self.tmp_dir + "hello")
        # load pipeline
        pipeline_after_save_load = Pipeline.load(self.tmp_dir)
        # test predict
        res_after_save_load = pipeline_after_save_load.predict(anomaly_tsdataset)
        # clear file
        # shutil.rmtree(self.tmp_dir)
        self.assertTrue(res_before_save_load.get_target().to_dataframe() \
                        .equals(res_after_save_load.get_target().to_dataframe()))


    def test_multiple_datasets_fit(self):

        # load multi time series
        tsdataset_1 = get_dataset("UNI_WTH")
        _, tsdataset_1 = tsdataset_1.split(int(len(tsdataset_1.get_target()) * 0.95))
        tsdataset_2 = copy.deepcopy(tsdataset_1)
        valid_tsdataset = copy.deepcopy(tsdataset_1)
        tsdatasets = [tsdataset_1, tsdataset_2]
        self.assertEqual(len(tsdatasets), 2)

        transform_params = {"cols": ['WetBulbCelsius'], "k": 0.5}
        transform_params_1 = {"cols": ['WetBulbCelsius'], "k": 0.7}
        nn_params = {
            'in_chunk_len': 7 * 96 + 20 * 4,
            'out_chunk_len': 96,
            'skip_chunk_len': 4 * 4,
            'eval_metrics': ["mse", "mae"],
            'max_epochs': 3,
        }

        pipe = Pipeline([(KSigma, transform_params), (KSigma, transform_params_1), (MLPRegressor, nn_params)])
        pipe.fit(tsdatasets, valid_tsdataset)
        pipe.fit(tsdatasets, tsdatasets)
        pipe.predict(valid_tsdataset)
        with self.assertRaises(AttributeError):
            pipe.predict(tsdatasets)
        pipe.transform(valid_tsdataset)
        pipe.transform(tsdatasets)


if __name__ == "__main__":
    unittest.main()
