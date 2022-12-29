# !/usr/bin/env python3
# -*- coding:utf-8 -*-

import unittest
import os
import random
import json
import shutil
import datetime
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from itertools import product

from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.cd import CD
from pyod.models.cof import COF
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
from pyod.models.gmm import GMM
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.inne import INNE
from pyod.models.kde import KDE
from pyod.models.knn import KNN
from pyod.models.lmdd import LMDD
from pyod.models.loci import LOCI
from pyod.models.loda import LODA
from pyod.models.lof import LOF
from pyod.models.lscp import LSCP
from pyod.models.mad import MAD
from pyod.models.mcd import MCD

from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.rgraph import RGraph
from pyod.models.rod import ROD
from pyod.models.sampling import Sampling
from pyod.models.sod import SOD
from pyod.models.sos import SOS

# below imports requires tensorflow/torch/keras, thus commented.
# from pyod.models.anogan import AnoGAN
# from pyod.models.auto_encoder import AutoEncoder
# from pyod.models.auto_encoder_torch import AutoEncoder as AutoEncoderTorch
# from pyod.models.deep_svdd import DeepSVDD
# from pyod.models.feature_bagging import FeatureBagging
# from pyod.models.lunar import LUNAR
# from pyod.models.mo_gaal import MO_GAAL
# from pyod.models.so_gaal import SO_GAAL
# from pyod.models.suod import SUOD
# from pyod.models.vae import VAE
# from pyod.models.xgbod import XGBOD

from paddlets.models.forecasting.ml.ml_base import MLBaseModel
from paddlets.models.ml_model_wrapper import PyodModelWrapper, make_ml_model
from paddlets.datasets import TSDataset, TimeSeries
from paddlets.models.data_adapter import MLDataLoader


class MockNotPyodModel(object):
    """Mock class that is NOT from pyod."""
    pass


class TestPyodModelWrapper(unittest.TestCase):
    """
    test PyodModelWrapper.
    """
    def setUp(self):
        """
        unittest setup.

        _good_to_init_pyod_model_list: can be init, but not sure if good to fit / predict. This list is for
            unittest GOOD cases.
        _bad_to_fit_pyod_model_set: Models in this list will raise error while fitting. This list is for
            unittest BAD cases.
        _bad_to_predict_pyod_model_set: Models in this list will raise error while making prediction. This list is for
            unittest BAD cases.
        _good_to_fit_and_predict_and_predict_score_pyod_model_list: Models in this list will be good to fit and
            make prediction, this list is for unittest GOOD cases.
        """
        self._default_modelname = "modelname"
        self._default_model_class = OCSVM
        self._default_in_chunk_len = 3

        self._empty_init_params_template = dict()

        self._good_to_init_pyod_model_class_list = [
            ABOD,
            CBLOF,
            CD,
            COF,
            COPOD,
            ECOD,
            GMM,
            HBOS,
            IForest,
            INNE,
            KDE,
            KNN,
            LMDD,
            LOCI,
            LODA,
            LOF,
            MAD,
            MCD,
            OCSVM,
            PCA,
            RGraph,
            ROD,
            Sampling,
            SOD,
            SOS
        ]

        # good cases for __init__ method.
        self._good_to_init_pyod_model_list = []
        for model_class in self._good_to_init_pyod_model_class_list:
            # init param = None, predict param = None
            dict1 = {
                "clazz": model_class,
                "init_params": dict(),
                "predict_params": dict()
            }
            self._good_to_init_pyod_model_list.append(dict1)

            if model_class not in {MAD}:
                # init param != None, predict param = None
                dict2 = {
                    "clazz": model_class,
                    "init_params": {"contamination": 0.1},
                    "predict_params": dict()
                }
                self._good_to_init_pyod_model_list.append(dict2)

            # init param = None, predict param != None
            dict3 = {
                "clazz": model_class,
                "init_params": dict(),
                "predict_params": {"return_confidence": False}
            }
            self._good_to_init_pyod_model_list.append(dict3)

            if model_class not in {MAD}:
                # init param != None, predict param != None
                dict4 = {
                    "clazz": model_class,
                    "init_params": {"contamination": 0.1},
                    "predict_params": {"return_confidence": False}
                }
                self._good_to_init_pyod_model_list.append(dict4)

        # Bad cases for fit method.
        self._bad_to_fit_pyod_model_set = {
            # subset_size=20 must be between 0 and n_samples=10.
            Sampling,

            # Expected n_neighbors <= n_samples,  but n_samples = 10, n_neighbors = 21
            SOD,

            # MAD algorithm is just for uni-variate data. Got Data with 2 Dimensions.
            MAD,

            # k must be less than or equal to the number of training points
            LSCP,

            # Input contains NaN, infinity or a value too large for dtype('float64').
            CD,

            # Could not form valid cluster separation.Please change n_clusters or change clustering method
            CBLOF
        }

        # Bad cases for predict method (it is empty because currently all models in pyod are good to predict.)
        self._bad_to_predict_pyod_model_set = set()

        # Good cases for fit and predict and predict_score methods.
        self._good_to_fit_and_predict_and_predict_score_pyod_model_list = list()
        for m in self._good_to_init_pyod_model_list:
            if (m["clazz"] not in self._bad_to_fit_pyod_model_set) and \
                    (m["clazz"] not in self._bad_to_predict_pyod_model_set):
                self._good_to_fit_and_predict_and_predict_score_pyod_model_list.append(m)
        super().setUp()

    def test_init_model(self):
        """
        test SklearnModelWrapper::__init__

        Note: as all pyod models are inherited from pyod.models.base.BaseDetector, meanwhile, the fit, predict and
        predict_score methods are all decorated by abc.abstractmethod, so pyod automatically helped us avoid the
        child classes from forgetting to implement these methods, thus, no need to test cases where classes inherited
        from BaseDetector are not implement these methods, because abc.abstractmethod will detect it and raises an error
        in prior to out unittest function.
        """
        #####################################################################
        # case 0 (good case)                                                #
        # 1) models are inherited from pyod.models.base.BaseDetector.       #
        # 2) models has implemented fit, predict and predict_score methods. #
        # 3) no init_params                                                 #
        #####################################################################
        for model in self._good_to_init_pyod_model_list:
            if model["init_params"] != dict():
                continue
            succeed = True
            try:
                _ = PyodModelWrapper(
                    in_chunk_len=self._default_in_chunk_len,
                    model_class=model["clazz"],
                    model_init_params=model["init_params"]
                )
            except ValueError:
                succeed = False
            self.assertTrue(succeed)

        ###########################
        # case 1 (bad case)       #
        # 1) model_class is None. #
        ###########################
        bad_model_class = None
        succeed = True
        try:
            _ = PyodModelWrapper(
                in_chunk_len=self._default_in_chunk_len,
                model_class=bad_model_class
            )
        except ValueError:
            succeed = False
        self.assertFalse(succeed)

        ##############################################
        # case 2 (bad case)                          #
        # 1) isinstance(model_class, type) is False. #
        ##############################################
        bad_model_class = MockNotPyodModel()
        succeed = True
        try:
            _ = PyodModelWrapper(
                in_chunk_len=self._default_in_chunk_len,
                model_class=bad_model_class
            )
        except ValueError:
            succeed = False
        self.assertFalse(succeed)

        ################################################
        # case 3 (bad case)                            #
        # 1) model is NOT inherited from BaseDetector. #
        ################################################
        bad_model_class = MockNotPyodModel
        succeed = True
        try:
            _ = PyodModelWrapper(
                in_chunk_len=self._default_in_chunk_len,
                model_class=bad_model_class
            )
        except ValueError:
            succeed = False
        self.assertFalse(succeed)

        #######################
        # case 4 (bad case)   #
        # 1) in_chunk_len < 0 #
        #######################
        bad_in_chunk_len = -1

        succeed = True
        try:
            _ = PyodModelWrapper(
                in_chunk_len=bad_in_chunk_len,
                model_class=self._default_model_class
            )
        except ValueError:
            succeed = False
        self.assertFalse(succeed)

        ###########################
        # case 5 (bad case)       #
        # 1) sampling_stride == 0 #
        ###########################
        bad_sampling_stride = 0
        succeed = True
        try:
            _ = PyodModelWrapper(
                in_chunk_len=self._default_in_chunk_len,
                sampling_stride=bad_sampling_stride,
                model_class=self._default_model_class
            )
        except ValueError:
            succeed = False
        self.assertFalse(succeed)

        ###########################
        # case 6 (bad case)       #
        # 1) sampling_stride < 0  #
        ###########################
        bad_sampling_stride = -1
        succeed = True
        try:
            _ = PyodModelWrapper(
                in_chunk_len=self._default_in_chunk_len,
                sampling_stride=bad_sampling_stride,
                model_class=self._default_model_class
            )
        except ValueError:
            succeed = False
        self.assertFalse(succeed)

        ################################
        # case 7 (bad case)            #
        # 1) invalid_model_init_params #
        ################################
        bad_model_init_params = {"mock_bad_key": "mock_bad_value"}
        succeed = True
        try:
            _ = PyodModelWrapper(
                in_chunk_len=self._default_in_chunk_len,
                model_class=self._default_model_class,
                model_init_params=bad_model_init_params
            )
        except ValueError:
            succeed = False
        self.assertFalse(succeed)

        #####################################################################
        # case 8 (bad case)                                                 #
        # 1) predict_params["return_confidence"] is True, which is invalid. #
        #####################################################################
        bad_model_predict_params = {"return_confidence": True}
        succeed = True
        try:
            _ = PyodModelWrapper(
                model_class=self._default_model_class,
                in_chunk_len=self._default_in_chunk_len,
                predict_params=bad_model_predict_params
            )
        except ValueError:
            succeed = False
        self.assertFalse(succeed)

    def test_fit_and_predict_and_predict_score(self):
        """
        test PyodModelWrapper::fit and PyodModelWrapper::predict.
        """
        ########################
        # case 0 (good case)   #
        # 1) no predict_params #
        ########################
        target_periods = 0
        known_periods = 0
        observed_periods = 10

        for model in self._good_to_fit_and_predict_and_predict_score_pyod_model_list:
            if model["predict_params"] != dict():
                continue
            ds = self._build_mock_ts_dataset(
                target_periods=target_periods,
                known_periods=known_periods,
                observed_periods=observed_periods
            )
            model_wrapper = PyodModelWrapper(
                in_chunk_len=self._default_in_chunk_len,
                model_class=model["clazz"],
                model_init_params=model["init_params"]
            )

            # fit
            model_wrapper.fit(train_data=ds)

            # predict
            predicted_ds = model_wrapper.predict(ds)
            self.assertIsNotNone(predicted_ds.get_target())
            self.assertTrue("anomaly_label" in predicted_ds.get_target().columns)

            # predict_score
            predict_score_ds = model_wrapper.predict_score(ds)
            self.assertIsNotNone(predict_score_ds.get_target())
            self.assertTrue("anomaly_score" in predict_score_ds.get_target().columns)

            # predict_score_ndarray = model_wrapper.predict_score(ds)
            # self.assertTrue(0 < predict_score_ndarray.shape[0] <= observed_periods)

        ##################################################
        # case 1 (good case)                             #
        # 1) predict_params["return_confidence"] = False #
        ##################################################
        target_periods = 0
        known_periods = 0
        observed_periods = 10

        for model in self._good_to_fit_and_predict_and_predict_score_pyod_model_list:
            if model["predict_params"] == dict():
                continue
            self.assertFalse(model["predict_params"]["return_confidence"])
            ds = self._build_mock_ts_dataset(
                target_periods=target_periods,
                known_periods=known_periods,
                observed_periods=observed_periods
            )
            model_wrapper = PyodModelWrapper(
                in_chunk_len=self._default_in_chunk_len,
                model_class=model["clazz"],
                model_init_params=model["init_params"],
                predict_params=model["predict_params"]
            )
            # fit
            model_wrapper.fit(train_data=ds)

            # predict
            predicted_ds = model_wrapper.predict(ds)
            self.assertIsNotNone(predicted_ds.get_target())
            self.assertTrue("anomaly_label" in predicted_ds.get_target().columns)

            # predict_score
            predict_score_ds = model_wrapper.predict_score(ds)
            self.assertIsNotNone(predict_score_ds.get_target())
            self.assertTrue("anomaly_score" in predict_score_ds.get_target().columns)

        #################################################################
        # case 2 (good case)                                            #
        # 1) Non default udf_ml_dataloader_to_fit_ndarray function.     #
        # 2) Non default udf_ml_dataloader_to_predict_ndarray function. #
        # 3) others optional params are default.                        #
        #################################################################
        target_periods = 0
        known_periods = 0
        observed_periods = 10

        for model in self._good_to_fit_and_predict_and_predict_score_pyod_model_list:
            ds = self._build_mock_ts_dataset(
                target_periods=target_periods,
                known_periods=known_periods,
                observed_periods=observed_periods
            )

            model_wrapper = PyodModelWrapper(
                in_chunk_len=self._default_in_chunk_len,
                model_class=model["clazz"],
                model_init_params=model["init_params"],
                udf_ml_dataloader_to_fit_ndarray=self.udf_ml_dataloader_to_fit_ndarray,
                udf_ml_dataloader_to_predict_ndarray=self.udf_ml_dataloader_to_predict_ndarray
            )
            # fit
            model_wrapper.fit(train_data=ds)

            # predict
            predicted_ds = model_wrapper.predict(ds)
            self.assertIsNotNone(predicted_ds.get_target())
            self.assertTrue("anomaly_label" in predicted_ds.get_target().columns)

            # predict_score
            predict_score_ds = model_wrapper.predict_score(ds)
            self.assertIsNotNone(predict_score_ds.get_target())
            self.assertTrue("anomaly_score" in predict_score_ds.get_target().columns)

        ############################
        # case 3 (bad case)        #
        # 1) observed_cov is None. #
        ############################
        ds = self._build_mock_ts_dataset(
            target_periods=10,
            known_periods=10,
            observed_periods=10
        )
        ds.observed_cov = None

        model_wrapper = PyodModelWrapper(
            model_class=self._default_model_class,
            in_chunk_len=self._default_in_chunk_len
        )

        succeed = True
        try:
            model_wrapper.fit(train_data=ds)
        except ValueError:
            succeed = False
        self.assertFalse(succeed)

    def test_save(self):
        """
        test PyodModelWrapper::save (inherited from MLBaseModel::save).

        Here will only test good case to ensure that it works for child class PyodModelWrapper. To know more about
        bad cases, see unittest for MLBaseModel (tests.models.ml.test_ml_base.py::test_save).
        """
        ########################################
        # case 0 (good case)                   #
        # 1) path exists.                      #
        # 2) No file conflicts.                #
        ########################################
        model_wrapper = PyodModelWrapper(
            model_class=self._default_model_class,
            in_chunk_len=self._default_in_chunk_len
        )

        observed_periods = 10
        ds = self._build_mock_ts_dataset(
            target_periods=0,
            known_periods=0,
            observed_periods=observed_periods
        )

        # fit
        model_wrapper.fit(train_data=ds)

        # predict
        predicted_ds = model_wrapper.predict(ds)
        self.assertIsNotNone(predicted_ds.get_target())
        self.assertTrue("anomaly_label" in predicted_ds.get_target().columns)

        # predict_score
        predict_score_ds = model_wrapper.predict_score(ds)
        self.assertIsNotNone(predict_score_ds.get_target())
        self.assertTrue("anomaly_score" in predict_score_ds.get_target().columns)

        # save
        path = os.path.join(os.getcwd(), str(random.randint(1, 10000000)))
        os.mkdir(path)
        model_wrapper.save(os.path.join(path, self._default_modelname))

        files = set(os.listdir(path))
        internal_filename_map = {
            "model_meta": "%s_%s" % (self._default_modelname, "model_meta")
        }
        self.assertEqual(files, {self._default_modelname, *internal_filename_map.values()})

        # model type
        with open(os.path.join(path, internal_filename_map["model_meta"]), "r") as f:
            model_meta = json.load(f)
        self.assertTrue(PyodModelWrapper.__name__ in model_meta["ancestor_classname_set"])
        self.assertTrue(MLBaseModel.__name__ in model_meta["ancestor_classname_set"])
        self.assertEqual(PyodModelWrapper.__module__, model_meta["modulename"])
        shutil.rmtree(path)

        #########################################################
        # case 1 (good case)                                    #
        # 1) path exists.                                       #
        # 2) No file conflicts.                                 #
        # 3) the same model can be saved twice at the same dir. #
        #########################################################
        # save the same model instance twice with different name at same path.
        path = os.path.join(os.getcwd(), str(random.randint(1, 10000000)))
        os.mkdir(path)

        # build and fit the only model instance.
        model_wrapper = PyodModelWrapper(
            model_class=self._default_model_class,
            in_chunk_len=self._default_in_chunk_len
        )

        observed_periods = 10
        ds = self._build_mock_ts_dataset(
            target_periods=0,
            known_periods=0,
            observed_periods=observed_periods
        )
        model_wrapper.fit(train_data=ds)

        # save the first one.
        model_1_name = "a"
        model_1_internal_filename_map = {"model_meta": "%s_%s" % (model_1_name, "model_meta")}
        model_wrapper.save(os.path.join(path, model_1_name))

        # save the second one.
        model_2_name = "b"
        model_2_internal_filename_map = {"model_meta": "%s_%s" % (model_2_name, "model_meta")}
        model_wrapper.save(os.path.join(path, model_2_name))

        files = set(os.listdir(path))
        self.assertEqual(
            files,
            {
                model_1_name,
                *model_1_internal_filename_map.values(),
                model_2_name,
                *model_2_internal_filename_map.values()
            }
        )

        shutil.rmtree(path)

    def test_load(self):
        """
        test SklearnModelWrapper::load (inherited from MLBaseModel::load).

        Here will only test good case to ensure that it works for child class SklearnModelWrapper. To know more about
        bad cases, see unittest for MLBaseModel (tests.models.ml.test_ml_base.py::test_load).
        """
        ####################################################################################
        # case 0 (good case)                                                               #
        # 1) model exists in the given path.                                               #
        # 2) the saved model is fitted before loading.                                     #
        # 3) the predicted result for loaded model is identical to the one before loading. #
        ####################################################################################
        # build + fit + save
        model_wrapper = PyodModelWrapper(
            model_class=self._default_model_class,
            in_chunk_len=self._default_in_chunk_len
        )

        observed_periods = 10
        ds = self._build_mock_ts_dataset(
            target_periods=0,
            known_periods=0,
            observed_periods=observed_periods
        )
        model_wrapper.fit(train_data=ds)

        # store predict result before save & load
        # predict()
        pred_ds_before_load = model_wrapper.predict(ds)
        self.assertIsNotNone(pred_ds_before_load.get_target())
        self.assertTrue("anomaly_label" in pred_ds_before_load.get_target().columns)

        # predict_score()
        pred_score_ds_before_load = model_wrapper.predict_score(ds)
        self.assertIsNotNone(pred_score_ds_before_load.get_target())
        self.assertTrue("anomaly_score" in pred_score_ds_before_load.get_target().columns)

        # save
        path = os.path.join(os.getcwd(), str(random.randint(1, 10000000)))
        os.mkdir(path)
        abs_model_path = os.path.join(path, self._default_modelname)
        model_wrapper.save(abs_model_path)

        # load model
        loaded_model_wrapper = MLBaseModel.load(abs_model_path)

        # model type expected
        assert isinstance(loaded_model_wrapper, PyodModelWrapper)
        self.assertTrue(isinstance(loaded_model_wrapper, PyodModelWrapper))

        # predict using loaded model
        pred_ds_after_load = loaded_model_wrapper.predict(ds)
        pred_score_ds_after_load = loaded_model_wrapper.predict_score(ds)

        # compare predicted result
        self.assertTrue(np.alltrue(
            pred_ds_after_load.get_target().to_numpy(False) == pred_score_ds_after_load.get_target().to_numpy(False)
        ))
        shutil.rmtree(path)

        #############################################################################################################
        # case 1 (good case)                                                                                        #
        # 1) model exists in the given path.                                                                        #
        # 2) the saved model is fitted before loading.                                                              #
        # 3) will load 2 model instances from the same saved model file, namely, loaded_model_1 and loaded_model_2. #
        # 4) guarantee that loaded_model_1.predict(data) == loaded_model_2.predict(data)                            #
        #############################################################################################################
        # build + fit + save
        model_wrapper = PyodModelWrapper(
            model_class=self._default_model_class,
            in_chunk_len=self._default_in_chunk_len
        )

        observed_periods = 10
        ds = self._build_mock_ts_dataset(
            target_periods=0,
            known_periods=0,
            observed_periods=observed_periods
        )
        model_wrapper.fit(train_data=ds)

        # store predicted dataset before load
        pred_ds_before_load = model_wrapper.predict(ds)
        self.assertIsNotNone(pred_ds_before_load.get_target())
        self.assertTrue("anomaly_label" in pred_ds_before_load.get_target().columns)

        # predict_score()
        pred_score_ds_before_load = model_wrapper.predict_score(ds)
        self.assertIsNotNone(pred_score_ds_before_load.get_target())
        self.assertTrue("anomaly_score" in pred_score_ds_before_load.get_target().columns)

        # save
        path = os.path.join(os.getcwd(), str(random.randint(1, 10000000)))
        os.mkdir(path)

        # use same mode instance to save the first model file.
        model_1_name = "a"
        abs_model_1_path = os.path.join(path, model_1_name)
        model_wrapper.save(abs_model_1_path)

        # use same mode instance to save the second model file.
        model_2_name = "b"
        abs_model_2_path = os.path.join(path, model_2_name)
        model_wrapper.save(abs_model_2_path)

        # load 2 models
        loaded_model_1 = MLBaseModel.load(abs_model_1_path)
        loaded_model_2 = MLBaseModel.load(abs_model_2_path)

        # model type expected
        assert isinstance(loaded_model_1, PyodModelWrapper)
        assert isinstance(loaded_model_2, PyodModelWrapper)
        self.assertEqual(model_wrapper.__class__, loaded_model_1.__class__)
        self.assertEqual(model_wrapper.__class__, loaded_model_2.__class__)

        # assert predict() expected
        loaded_model_1_pred_ds = loaded_model_1.predict(ds)
        loaded_model_2_pred_ds = loaded_model_2.predict(ds)

        self.assertTrue(np.alltrue(
            loaded_model_1_pred_ds.get_target().to_numpy(False) == pred_ds_before_load.get_target().to_numpy(False)
        ))
        self.assertTrue(np.alltrue(
            loaded_model_2_pred_ds.get_target().to_numpy(False) == pred_ds_before_load.get_target().to_numpy(False)
        ))

        # assert predict_score() expected
        loaded_model_1_pred_score_ds = loaded_model_1.predict_score(ds)
        loaded_model_2_pred_score_ds = loaded_model_2.predict_score(ds)

        loaded_model_1_pred_score_ds_target = loaded_model_1_pred_score_ds.get_target().to_numpy(False)
        loaded_model_2_pred_score_ds_target = loaded_model_2_pred_score_ds.get_target().to_numpy(False)
        self.assertTrue(np.alltrue(
            loaded_model_1_pred_score_ds_target == pred_score_ds_before_load.get_target().to_numpy()
        ))
        self.assertTrue(np.alltrue(
            loaded_model_2_pred_score_ds_target == pred_score_ds_before_load.get_target().to_numpy()
        ))

        shutil.rmtree(path)

    def test_make_ml_model(self):
        """
        test ml_model_wrapper::make_ml_model.
        """
        ######################
        # case 0 (good case) #
        # 1) pyod model.  #
        ######################
        observed_periods = 10
        for model in self._good_to_fit_and_predict_and_predict_score_pyod_model_list:
            ds = self._build_mock_ts_dataset(
                target_periods=0,
                known_periods=0,
                observed_periods=observed_periods
            )

            model_wrapper = make_ml_model(
                in_chunk_len=self._default_in_chunk_len,
                model_class=model["clazz"],
                model_init_params=model["init_params"]
            )
            assert isinstance(model_wrapper, PyodModelWrapper)

            # fit
            model_wrapper.fit(train_data=ds)

            # predict
            predicted_ds = model_wrapper.predict(ds)
            self.assertIsNotNone(predicted_ds.get_target())
            self.assertTrue("anomaly_label" in predicted_ds.get_target().columns)

            # predict_score
            predict_score_ds = model_wrapper.predict_score(ds)
            self.assertIsNotNone(predict_score_ds.get_target())
            self.assertTrue("anomaly_score" in predict_score_ds.get_target().columns)

        ###########################
        # case 1 (bad case)       #
        # 1) model_class is None. #
        ###########################
        bad_model_class = None
        succeed = True
        try:
            _ = make_ml_model(
                model_class=bad_model_class,
                in_chunk_len=self._default_in_chunk_len
            )
        except ValueError:
            succeed = False
        self.assertFalse(succeed)

        ##############################################
        # case 2 (bad case)                          #
        # 1) isinstance(model_class, type) is False. #
        ##############################################
        bad_model_class = MockNotPyodModel()
        succeed = True
        try:
            _ = make_ml_model(
                model_class=bad_model_class,
                in_chunk_len=self._default_in_chunk_len
            )
        except ValueError:
            succeed = False
        self.assertFalse(succeed)

    def _build_mock_ts_dataset(
        self,
        target_periods: int = 10,
        known_periods: int = 10,
        observed_periods: int = 10,
        target_start_timestamp: pd.Timestamp = pd.Timestamp(datetime.datetime.now().date()),
        known_start_timestamp: pd.Timestamp = pd.Timestamp(datetime.datetime.now().date()),
        observed_start_timestamp: pd.Timestamp = pd.Timestamp(datetime.datetime.now().date()),
        freq: str = "1D",
        numeric: bool = True,
        categorical: bool = True
    ):
        """
        Build mock dataset.

        all timeseries must have same freq.
        """
        numeric_dtype = np.float32
        categorical_dtype = np.int64

        target_ts = None
        if target_periods > 0:
            # target (pyod model wrapper requires target col num == 1, thus cannot both contain numeric + categorical).
            target_df = pd.Series(
                [i for i in range(target_periods)],
                index=pd.date_range(start=target_start_timestamp, periods=target_periods, freq=freq),
                name="target0",
                dtype=numeric_dtype
            )
            target_ts = TimeSeries.load_from_dataframe(data=target_df)

        # known
        known_cov_ts = None
        if known_periods > 0:
            known_raw_data = [(i * 10, i * 100) for i in range(known_periods)]
            known_numeric_df = None
            if numeric:
                # numeric
                known_numeric_data = np.array(known_raw_data, dtype=numeric_dtype)
                known_numeric_df = pd.DataFrame(
                    data=known_numeric_data,
                    index=pd.date_range(start=known_start_timestamp, periods=known_periods, freq=freq),
                    columns=["known_numeric_0", "known_numeric_1"]
                )

            known_categorical_df = None
            if categorical:
                # categorical
                known_categorical_data = np.array(known_raw_data, dtype=categorical_dtype)
                known_categorical_df = pd.DataFrame(
                    data=known_categorical_data,
                    index=pd.date_range(start=known_start_timestamp, periods=known_periods, freq=freq),
                    columns=["known_categorical_0", "known_categorical_1"]
                )
            if (known_numeric_df is None) and (known_categorical_df is None):
                raise Exception(f"failed to build known cov data, both numeric df and categorical df are all None.")
            if (known_numeric_df is not None) and (known_categorical_df is not None):
                # both are NOT None.
                known_cov_df = pd.concat([known_numeric_df, known_categorical_df], axis=1)
            else:
                known_cov_df = [known_numeric_df, known_categorical_df][1 if known_numeric_df is None else 0]
            known_cov_ts = TimeSeries.load_from_dataframe(data=known_cov_df)

        # observed
        observed_cov_ts = None
        if observed_periods > 0:
            observed_raw_data = [(i * -1, i * -10) for i in range(observed_periods)]
            observed_numeric_df = None
            if numeric:
                # numeric
                observed_numeric_data = np.array(observed_raw_data, dtype=numeric_dtype)
                observed_numeric_df = pd.DataFrame(
                    data=observed_numeric_data,
                    index=pd.date_range(start=observed_start_timestamp, periods=observed_periods, freq=freq),
                    columns=["observed_numeric_0", "observed_numeric_1"]
                )

            observed_categorical_df = None
            if categorical:
                # categorical
                observed_categorical_data = np.array(observed_raw_data, dtype=categorical_dtype)
                observed_categorical_df = pd.DataFrame(
                    data=observed_categorical_data,
                    index=pd.date_range(start=observed_start_timestamp, periods=observed_periods, freq=freq),
                    columns=["observed_categorical_0", "observed_categorical_1"]
                )

            if (observed_numeric_df is None) and (observed_categorical_df is None):
                raise Exception(f"failed to build observed cov data, both numeric df and categorical df are all None.")
            if (observed_numeric_df is not None) and (observed_categorical_df is not None):
                # both are NOT None.
                observed_cov_df = pd.concat([observed_numeric_df, observed_categorical_df], axis=1)
            else:
                observed_cov_df = [observed_numeric_df, observed_categorical_df][
                    1 if observed_numeric_df is None else 0]
            observed_cov_ts = TimeSeries.load_from_dataframe(data=observed_cov_df)

        # static
        static = dict()
        if numeric:
            # numeric
            static["static_numeric"] = np.float32(1)
        if categorical:
            # categorical
            static["static_categorical"] = np.int64(2)

        return TSDataset(
            target=target_ts,
            known_cov=known_cov_ts,
            observed_cov=observed_cov_ts,
            static_cov=static
        )

    @staticmethod
    def udf_ml_dataloader_to_fit_ndarray(
        ml_dataloader: MLDataLoader,
        model_init_params: Dict[str, Any],
        in_chunk_len: int
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        data = next(ml_dataloader)

        # Please note that anomaly samples will NEVER contain the following keys:
        # "past_target_*"
        # "future_target_*"
        # "known_cov_*"
        # Refers to models.anomaly.ml.adapter.ml_dataset.MLDataset::_build_samples() to get more details.
        sample_x_keys = data.keys()
        # concatenated ndarray will follow the below ordered list rule:
        # [rule 1] left -> right = observed_cov_features, ..., static_cov_features.
        # [rule 2] left -> right = numeric features, ..., categorical features.
        product_keys = product(["numeric", "categorical"], ["observed_cov", "static_cov"])
        full_ordered_x_key_list = [f"{t[1]}_{t[0]}" for t in product_keys]

        # For example, given:
        # sample_keys (un-ordered) = {"observed_cov_categorical", "static_cov_numeric", "observed_cov_numeric"}
        # full_ordered_x_key_list = [
        #   "observed_cov_numeric",
        #   "static_cov_numeric",
        #   "observed_cov_categorical",
        #   "static_cov_categorical"
        # ]
        # Thus, actual_ordered_x_key_list = [
        #   "observed_cov_numeric",
        #   "static_cov_numeric",
        #   "observed_cov_categorical"
        # ]
        # The built sample ndarray will be like below:
        # [
        #   [
        #       observed_cov_numeric_feature, static_cov_numeric_feature, observed_cov_categorical_feature
        #   ],
        #   [
        #       observed_cov_numeric_feature, static_cov_numeric_feature, observed_cov_categorical_feature
        #   ],
        #   ...
        # ]
        actual_ordered_x_key_list = []
        for k in full_ordered_x_key_list:
            if k in sample_x_keys:
                actual_ordered_x_key_list.append(k)

        reshaped_x_ndarray_list = []
        for k in actual_ordered_x_key_list:
            ndarray = data[k]
            # 3-dim -> 2-dim
            reshaped_ndarray = ndarray.reshape(ndarray.shape[0], ndarray.shape[1] * ndarray.shape[2])
            reshaped_x_ndarray_list.append(reshaped_ndarray)
        # Note: if a_ndarray.dtype = np.int64, b_ndarray.dtype = np.float32, then
        # np.hstack(tup=(a_ndarray, b_ndarray)).dtype will ALWAYS BE np.float32
        x = np.hstack(tup=reshaped_x_ndarray_list)
        return x, None

    @staticmethod
    def udf_ml_dataloader_to_predict_ndarray(
        ml_dataloader: MLDataLoader,
        model_init_params: Dict[str, Any],
        in_chunk_len: int
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        return TestPyodModelWrapper.udf_ml_dataloader_to_fit_ndarray(ml_dataloader, model_init_params, in_chunk_len)
