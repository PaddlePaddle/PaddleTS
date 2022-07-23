# !/usr/bin/env python3
# -*- coding:utf-8 -*-
import os
import unittest
import shutil

import pandas as pd
import numpy as np
from unittest import TestCase

from paddlets.models.dl.paddlepaddle.callbacks import Callback
from paddlets.models.dl.paddlepaddle import MLPRegressor
from paddlets.transform import KSigma, TimeFeatureGenerator, StandardScaler
from paddlets.datasets.tsdataset import TimeSeries, TSDataset
from paddlets.pipeline.pipeline import Pipeline
from paddlets.utils import get_uuid


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
        static_cov = {"f": 1, "g": 2}
        tsdataset = TSDataset(target, observed_cov, known_cov, static_cov)
        transform_params = {"cols": ['b1'], "k": 0.5}
        transform_params_1 = {"cols": ['c1'], "k": 0.7}
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

        pipe = Pipeline([(KSigma, transform_params), (KSigma, transform_params_1), \
                         (TimeFeatureGenerator, {}), (MLPRegressor, nn_params)])
        pipe.fit(tsdataset, tsdataset)
        self.assertEqual(len(pipe._transform_list), 3)

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
        transform_params = {"cols": ['b1'], "k": 0.5}
        transform_params_1 = {"cols": ['c1'], "k": 0.7}
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
        static_cov = {"f": 1, "g": 2}
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
        # transform when model exist
        nn_params = {
            'in_chunk_len': 7 * 96 + 20 * 4,
            'out_chunk_len': 96,
            'skip_chunk_len': 4 * 4,
            'eval_metrics': ["mse", "mae"]
        }
        pipe = Pipeline([(KSigma, transform_params), (TimeFeatureGenerator, {}), (KSigma, transform_params_1), \
                         (MLPRegressor, nn_params)])
        pipe.fit(tsdataset, tsdataset)
        res = pipe.transform(tsdataset)
        self.assertIsInstance(res, TSDataset)
        self.assertEqual(res.get_all_cov().to_dataframe().shape, (2500, 14))
        #test inverse transform( this pipe would do nothing)
        inverse_res = pipe.inverse_transform(res)
        self.assertIsInstance(inverse_res, TSDataset)
        self.assertEqual(inverse_res.get_all_cov().to_dataframe().shape, (2500, 14))
        #test normaliztion case
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
        transform_params = {"cols": ['b1'], "k": 0.5}
        transform_params_1 = {"cols": ['c1'], "k": 0.7}
        nn_params = {
            'in_chunk_len': 7 * 96 + 20 * 4,
            'out_chunk_len': 96,
            'skip_chunk_len': 0,
            'eval_metrics': ["mse", "mae"]
        }
        pipe = Pipeline([(KSigma, transform_params), (TimeFeatureGenerator, {}), (KSigma, transform_params_1), \
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
        static_cov = {"f": 1, "g": 2}
        tsdataset = TSDataset(target, observed_cov, known_cov, static_cov)
        transform_params = {"cols": ['b1'], "k": 0.5}
        transform_params_1 = {"cols": ['c1'], "k": 0.7}
        nn_params = {
            'in_chunk_len': 7 * 96 + 20 * 4,
            'out_chunk_len': 96,
            'skip_chunk_len': 4 * 4,
            'eval_metrics': ["mse", "mae"]
        }

        pipe = Pipeline([(KSigma, transform_params), (KSigma, transform_params_1), (MLPRegressor, nn_params)])
        pipe.fit(tsdataset, tsdataset)
        # test bad case
        with self.assertRaises(ValueError):
            res = pipe.predict_proba(tsdataset)
        # todo: test good case

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
        static_cov = {"f": 1, "g": 2}
        tsdataset = TSDataset(target, observed_cov, known_cov, static_cov)
        transform_params = {"cols": ['b1'], "k": 0.5}
        transform_params_1 = {"cols": ['c1'], "k": 0.7}
        nn_params = {
            'in_chunk_len': 7 * 96 + 20 * 4,
            'out_chunk_len': 96,
            'skip_chunk_len': 0,
            'eval_metrics': ["mse", "mae"]
        }
        pipe = Pipeline([(KSigma, transform_params), (TimeFeatureGenerator, {}), (KSigma, transform_params_1), \
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


if __name__ == "__main__":
    unittest.main()