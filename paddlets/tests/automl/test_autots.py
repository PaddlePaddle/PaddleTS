# !/usr/bin/env python3
# -*- coding:utf-8 -*-
import unittest
from unittest import TestCase

import copy

from paddlets.automl.autots import AutoTS, DEFAULT_K_FOLD, DEFAULT_SPLIT_RATIO
from paddlets.models.forecasting import MLPRegressor
from paddlets.transform import Fill
from paddlets.datasets.repository import get_dataset
from paddlets.metrics import MAE

class TestOptimizeRunner(TestCase):
    def setUp(self):
        """
        unittest function
        """
        super().setUp()

    def test_init(self):
        """
        unittest function
        """
        # default setting
        autots_model = AutoTS(MLPRegressor, 96, 2)
        self.assertEqual(autots_model._in_chunk_len, 96)
        self.assertEqual(autots_model._out_chunk_len, 2)
        self.assertEqual(autots_model._skip_chunk_len, 0)
        self.assertEqual(autots_model._search_space, autots_model.search_space())
        self.assertEqual(autots_model._search_alg, "TPE")
        self.assertEqual(autots_model._resampling_strategy, "holdout")
        self.assertEqual(autots_model._split_ratio, DEFAULT_SPLIT_RATIO)
        self.assertEqual(autots_model._k_fold, DEFAULT_K_FOLD)
        self.assertEqual(autots_model._metric, MAE)
        self.assertEqual(autots_model._mode, "min")
        self.assertEqual(autots_model._ensemble, False)
        self.assertEqual(autots_model._refit, True)

        with self.assertRaises(NotImplementedError):
            autots_model = AutoTS([Fill, MLPRegressor], 96, 2)

    def test_fit_predict(self):
        """

        unittest function

        """
        tsdataset = get_dataset("UNI_WTH")
        _, tsdataset = tsdataset.split(int(len(tsdataset.get_target())*0.99))
        autots_model = AutoTS(MLPRegressor, 25, 2, sampling_stride=25, local_dir="./")
        autots_model.fit(tsdataset, n_trials=1)
        sp = autots_model.search_space()
        predicted = autots_model.predict(tsdataset)
        predicted = autots_model.recursive_predict(tsdataset, 5)
        best_param = autots_model.best_param
        #test传入valid 的情况
        train, valid = tsdataset.split(int(len(tsdataset.get_target()) * 0.5))
        autots_model.fit(train, valid, n_trials=1)

        from ray.tune import uniform, qrandint, choice
        sp = {
            "Fill": {
                "cols": ['WetBulbCelsius'],
                "method": choice(['max', 'min', 'mean', 'median', 'pre', 'next', 'zero']),
                "value": uniform(0.1, 0.9),
                "window_size": qrandint(20, 50, q=1)
            },
            "MLPRegressor": {
                "batch_size": qrandint(16, 64, q=16),
                "use_bn": choice([True, False]),
                "max_epochs": qrandint(10, 50, q=10)
            }
        }
        autots_model = AutoTS([Fill, MLPRegressor], 25, 2, search_space=sp, sampling_stride=25, local_dir="./")
        autots_model.fit(tsdataset, n_trials=1)
        sp = autots_model.search_space()
        predicted = autots_model.predict(tsdataset)
        predicted = autots_model.recursive_predict(tsdataset, 5)
        best_param = autots_model.best_param
        #test传入valid 的情况
        train, valid = tsdataset.split(int(len(tsdataset.get_target()) * 0.5))
        autots_model = AutoTS([Fill, MLPRegressor], 25, 2, search_space=sp, local_dir="./")
        autots_model.fit(train, valid, n_trials=1)
        #get best estimator
        best_estimator = autots_model.best_estimator()
        predicted = autots_model.predict(tsdataset)
        predicted = autots_model.recursive_predict(tsdataset, 5)


    def test_defaut_search_space_fit(self):
        """

        unittest function

        """
        from paddlets.models.forecasting import MLPRegressor, RNNBlockRegressor, LSTNetRegressor, NHiTSModel, \
            TransformerModel, InformerModel, DeepARModel
        from paddlets.automl.search_space_configer import SearchSpaceConfiger
        from ray.tune import qrandint
        paddlets_configer = SearchSpaceConfiger()
        dl = [RNNBlockRegressor, NHiTSModel, TransformerModel, MLPRegressor, LSTNetRegressor, InformerModel, DeepARModel]
        tsdataset = get_dataset("WTH")
        _, tsdataset = tsdataset.split(int(len(tsdataset.get_target())*0.95))
        #数据归一化，若不归一化，可能出现模型训练梯度消失，或者爆炸问题。
        from paddlets.transform import StandardScaler
        scaler = StandardScaler()
        scaler.fit(tsdataset)
        tsdataset = scaler.transform(tsdataset)

        for e in dl:
            sp = paddlets_configer.get_default_search_space(e)
            if "max_epochs" in sp:
                sp['max_epochs'] = qrandint(2, 3, q=1)
            autots_model = AutoTS(e, 15, 2, search_space=sp, sampling_stride=25, local_dir="./")
            autots_model.fit(tsdataset, n_trials=2)
            sp = autots_model.search_space()
            predicted = autots_model.predict(tsdataset)

    def test_multiple_datasets_fit(self):

        # load multi time series
        tsdataset_1 = get_dataset("UNI_WTH")
        _, tsdataset_1 = tsdataset_1.split(int(len(tsdataset_1.get_target()) * 0.99))
        tsdataset_2 = copy.deepcopy(tsdataset_1)
        valid_tsdataset = copy.deepcopy(tsdataset_1)
        tsdatasets = [tsdataset_1, tsdataset_2]
        self.assertEqual(len(tsdatasets), 2)
        autots_model = AutoTS(MLPRegressor, 25, 2, sampling_stride=25, local_dir="./")
        autots_model.fit(tsdatasets, valid_tsdataset, n_trials=1)
        autots_model.fit(tsdatasets, tsdatasets, n_trials=1)
        autots_model.fit(valid_tsdataset, tsdatasets, n_trials=1)


if __name__ == "__main__":
    unittest.main()