# !/usr/bin/env python3
# -*- coding:utf-8 -*-

import pickle
import numpy as np
import paddle
import unittest
import os
from typing import Optional, Dict, List
import random
import shutil
import pandas as pd
import json

from typing import List, Dict, Any, Callable, Optional
from paddlets import TSDataset, TimeSeries
from paddlets.models.classify.dl.paddle_base import PaddleBaseClassifier
from paddlets.models.classify.dl.cnn import CNNClassifier


class TestClassifyBaseModel(unittest.TestCase):
    """
    ClassifyBaseModel unittest

    Currently, no need to test optimizer related logic.
    """
    def setUp(self):
        """
        unittest setup
        """
        self.default_modelname = "model"
        super().setUp()

    def test_save(self):
        """Test ClassifyBaseModel.save()"""
        ############################################
        # case 1 (good case)                       #
        # 1) Model path exists.                    #
        # 2) No filename conflicts.                #
        # 3) Not use valid data when fit.          #
        # 4) Use built-in model CNNClassifier. #
        ############################################
        model = self._build_cnn_model()

        paddlets_ds, labels = self._build_mock_data_and_label(random_data=True)
        # no validation dataset
        model.fit(train_tsdatasets=paddlets_ds, train_labels=labels)
        self.assertTrue(model._network is not None)
        self.assertTrue(model._optimizer is not None)

        path = os.path.join(os.getcwd(), str(random.randint(1, 10000000)))
        os.mkdir(path)

        internal_filename_map = {
            "model_meta": "%s_%s" % (self.default_modelname, "model_meta"),
            "network_statedict": "%s_%s" % (self.default_modelname, "network_statedict"),
            # currently ignore optimizer.
            # "optimizer_statedict": "%s_%s" % (modelname, "optimizer_statedict"),
        }
        model.save(os.path.join(path, self.default_modelname))
        self.assertTrue(model._network is not None)
        self.assertTrue(model._optimizer is not None)

        files = set(os.listdir(path))
        self.assertEqual(files, {self.default_modelname, *internal_filename_map.values()})

        # mode type CNNClassifier
        with open(os.path.join(path, internal_filename_map["model_meta"]), "r") as f:
            model_meta = json.load(f)
        # CNNClassifier,ClassifyBaseModel,Trainable,ABC,object
        self.assertTrue(CNNClassifier.__name__ in model_meta["ancestor_classname_set"])
        self.assertTrue(PaddleBaseClassifier.__name__ in model_meta["ancestor_classname_set"])
        # paddlets.models.anomaly.dl.CNNClassifier
        self.assertEqual(CNNClassifier.__module__, model_meta["modulename"])
        shutil.rmtree(path)

        ############################################
        # case 1 (good case)                       #
        # 1) Model path exists.                    #
        # 2) No filename conflicts.                #
        # 3) Use valid data when fit.              #
        # 4) Use built-in model CNNClassifier. #
        ############################################
        model = self._build_cnn_model()

        train_paddlets_ds, train_labels = self._build_mock_data_and_label(random_data=True)
        valid_paddlets_ds, valid_labels = self._build_mock_data_and_label(random_data=True)
        # use validation dataset.
        model.fit(train_tsdatasets=train_paddlets_ds, train_labels=train_labels,
                  valid_tsdatasets=valid_paddlets_ds, valid_labels=valid_labels)
        self.assertTrue(model._network is not None)
        self.assertTrue(model._optimizer is not None)

        path = os.path.join(os.getcwd(), str(random.randint(1, 10000000)))
        os.mkdir(path)

        internal_filename_map = {
            "model_meta": "%s_%s" % (self.default_modelname, "model_meta"),
            "network_statedict": "%s_%s" % (self.default_modelname, "network_statedict"),
            # currently ignore optimizer.
            # "optimizer_statedict": "%s_%s" % (modelname, "optimizer_statedict"),
        }
        model.save(os.path.join(path, self.default_modelname))
        self.assertTrue(model._network is not None)
        self.assertTrue(model._optimizer is not None)

        files = set(os.listdir(path))
        self.assertEqual(files, {self.default_modelname, *internal_filename_map.values()})

        # mode type CNNClassifier
        with open(os.path.join(path, internal_filename_map["model_meta"]), "r") as f:
            model_meta = json.load(f)
        # CNNClassifier,ClassifyBaseModel,Trainable,ABC,object
        self.assertTrue(CNNClassifier.__name__ in model_meta["ancestor_classname_set"])
        self.assertTrue(PaddleBaseClassifier.__name__ in model_meta["ancestor_classname_set"])
        # paddlets.models.dl.anomaly.CNNClassifier
        self.assertEqual(CNNClassifier.__module__, model_meta["modulename"])
        shutil.rmtree(path)

        ############################################
        # case 2 (good case)                       #
        # 1) Model path exists.                    #
        # 2) No filename conflicts.                #
        # 3) Save the same model twice.            #
        # 4) Use built-in model CNNClassifier. #
        ############################################
        # Note the following:
        # 1) this case is to guarantee that internal files from different models will NOT cause conflict because their
        # filenames are bounded with modelname (more precisely, each internal filename uses modelname as prefix).
        # 2) as case-0 and case-1 already guarantee that save() api works well regardless of whether the validation
        # dataset is passed into fit() api or not, thus the current test case won't test these 2 scenarios (i.e.
        # passing / not passing validation dataset to fit() api) separately.
        # 3) this case is to guarantee that a single same model instance can be saved for multiple times.

        # save the same model instance twice with different name at same path.
        path = os.path.join(os.getcwd(), str(random.randint(1, 10000000)))
        os.mkdir(path)

        # build and fit the only model instance.
        model = self._build_cnn_model()

        paddlets_ds, labels = self._build_mock_data_and_label(random_data=True)
        model.fit(train_tsdatasets=paddlets_ds, train_labels=labels)
        self.assertTrue(model._network is not None)
        self.assertTrue(model._optimizer is not None)

        # save the first one.
        model_1_name = "a"
        model_1_internal_filename_map = {
            "model_meta": "%s_%s" % (model_1_name, "model_meta"),
            "network_statedict": "%s_%s" % (model_1_name, "network_statedict"),
            # currently ignore optimizer.
            # "optimizer_statedict": "%s_%s" % (model_1_name, "optimizer_statedict"),
        }
        model.save(os.path.join(path, model_1_name))

        # save the second one.
        model_2_name = "b"
        model_2_internal_filename_map = {
            "model_meta": "%s_%s" % (model_2_name, "model_meta"),
            "network_statedict": "%s_%s" % (model_2_name, "network_statedict"),
            # currently ignore optimizer.
            # "optimizer_statedict": "%s_%s" % (model_2_name, "optimizer_statedict"),
        }
        model.save(os.path.join(path, model_2_name))

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

        ############################################
        # case 3 (bad case) Model path NOT exists. #
        ############################################
        model = self._build_cnn_model()

        paddlets_ds, labels = self._build_mock_data_and_label(random_data=True)
        model.fit(train_tsdatasets=paddlets_ds, train_labels=labels)
        self.assertTrue(model._network is not None)

        path = os.path.join(os.getcwd(), str(random.randint(1, 10000000)))

        succeed = True
        try:
            # path not exists.
            model.save(os.path.join(path, "void_model"))
        except Exception as e:
            succeed = False

        self.assertFalse(succeed)

        ###############################################################
        # case 4 (bad case) Model path is a directory, but NOT a file.#
        ###############################################################
        model = self._build_cnn_model()

        paddlets_ds, labels = self._build_mock_data_and_label(random_data=True)
        model.fit(train_tsdatasets=paddlets_ds, train_labels=labels)
        self.assertTrue(model._network is not None)

        path = os.path.join(os.getcwd(), str(random.randint(1, 10000000)))
        os.mkdir(path)

        succeed = True
        try:
            # path exists, but is a dir, not a file.
            model.save(path)
        except Exception as e:
            succeed = False

        self.assertFalse(succeed)
        shutil.rmtree(path)

        ##########################################################################
        # case 5 (bad case) Modelname conflicts with caller's existing filename. #
        ##########################################################################
        model = self._build_cnn_model()

        paddlets_ds, labels = self._build_mock_data_and_label(random_data=True)
        model.fit(train_tsdatasets=paddlets_ds, train_labels=labels)
        self.assertTrue(model._network is not None)

        path = os.path.join(os.getcwd(), str(random.randint(1, 10000000)))
        os.mkdir(path)

        # create a dup file
        abs_model_path = os.path.join(path, self.default_modelname)
        dup_file_content = "this is a dup file."
        with open(abs_model_path, "w") as f:
            f.write(dup_file_content)

        succeed = True
        try:
            # modelname conflicts with user's existing files.
            model.save(abs_model_path)
        except Exception as e:
            succeed = False

        self.assertFalse(succeed)

        # assert that the pre-existed conflict file content will (and should) not be overwritten.
        files = os.listdir(path)
        self.assertEqual(1, len(files))
        self.assertEqual(files[0], self.default_modelname)
        with open(os.path.join(path, self.default_modelname), "r") as f:
            tmp_data = f.read().strip()
        self.assertEqual(dup_file_content, tmp_data)
        shutil.rmtree(path)

        #############################################
        # case 6 (bad case) model._network is None. #
        #############################################
        model = self._build_cnn_model()
        model._network = None

        path = os.path.join(os.getcwd(), str(random.randint(1, 10000000)))
        os.mkdir(path)

        succeed = True
        try:
            # path exists, but _network is None
            model.save(os.path.join(path, self.default_modelname))
        except Exception as e:
            succeed = False

        self.assertFalse(succeed)
        shutil.rmtree(path)

        #################################################################################
        # case 7 (bad case) Internal filename conflicts with callers' existing filename #
        #################################################################################
        model = self._build_cnn_model()

        paddlets_ds, labels = self._build_mock_data_and_label(random_data=True)
        model.fit(train_tsdatasets=paddlets_ds, train_labels=labels)
        self.assertTrue(model._network is not None)

        path = os.path.join(os.getcwd(), str(random.randint(1, 10000000)))
        os.mkdir(path)

        modelname = self.default_modelname
        internal_filename_map = {
            "model_meta": "%s_%s" % (modelname, "model_meta"),
            "network_statedict": "%s_%s" % (modelname, "network_statedict"),
            # currently ignore optimizer.
            # "optimizer_statedict": "%s_%s" % (modelname, "optimizer_statedict"),
        }
        # create some dup files conflict with internal files.
        dup_model_meta_content = "this is a file dup with model_meta."
        with open(os.path.join(path, internal_filename_map["model_meta"]), "w") as f:
            f.write(dup_model_meta_content)

        succeed = True
        try:
            # internal files conflict with caller's existing files.
            model.save(os.path.join(path, modelname))
        except Exception as e:
            succeed = False
        self.assertFalse(succeed)
        shutil.rmtree(path)

    def test_load(self):
        """Test ClassifyBaseModel.load()"""
        ###################################
        # case 0 (good case)              #
        # 1) Model exists.                #
        # 2) Not use valid data when fit. #
        ###################################
        # build + fit + save an cnn Model.
        model = self._build_cnn_model()

        paddlets_ds, labels = self._build_mock_data_and_label(random_data=True)
        model.fit(train_tsdatasets=paddlets_ds, train_labels=labels)
        model_network = model._network

        preds = model.predict(paddlets_ds)
        self.assertEqual(len(paddlets_ds), len(preds))

        path = os.path.join(os.getcwd(), str(random.randint(1, 10000000)))
        os.mkdir(path)

        abs_model_path = os.path.join(path, self.default_modelname)
        model.save(abs_model_path)

        # load cnn model
        loaded_model = PaddleBaseClassifier.load(abs_model_path)

        # model type expected
        self.assertTrue(isinstance(loaded_model, CNNClassifier))
        # network type expected
        self.assertTrue(isinstance(loaded_model._network, model_network.__class__))

        # network state_dict expected
        self.assertEqual(
            model_network.state_dict().keys(),
            loaded_model._network.state_dict().keys()
        )
        common_network_state_dict_keys = model_network.state_dict().keys()
        for k in common_network_state_dict_keys:
            # {"_nn.0.weight": Tensor(shape=(xx, xx)), ...}
            raw = model_network.state_dict()[k]
            loaded = loaded_model._network.state_dict()[k]
            if isinstance(raw, paddle.Tensor):
                # convert tensor to numpy and call np.alltrue() to compare.
                self.assertTrue(np.alltrue(raw.numpy().astype(np.float64) == loaded.numpy().astype(np.float64)))

        # prediction results expected.
        loaded_model_preds = loaded_model.predict(paddlets_ds)
        self.assertTrue(np.alltrue(
            preds == loaded_model_preds
        ))
        shutil.rmtree(path)

        ###############################
        # case 1 (good case)          #
        # 1) Model exists.            #
        # 2) Use valid data when fit. #
        ###############################
        # build + fit + save an CNN Model.
        model = self._build_cnn_model()

        train_paddlets_ds, train_labels = self._build_mock_data_and_label(random_data=True)
        valid_paddlets_ds, valid_labels = self._build_mock_data_and_label(random_data=True)
        model.fit(train_tsdatasets=train_paddlets_ds, train_labels=train_labels,
                  valid_tsdatasets=valid_paddlets_ds, valid_labels=valid_labels)
        model_network = model._network

        preds = model.predict(paddlets_ds)
        self.assertEqual(len(preds), len(train_paddlets_ds))

        path = os.path.join(os.getcwd(), str(random.randint(1, 10000000)))
        os.mkdir(path)

        abs_model_path = os.path.join(path, self.default_modelname)
        model.save(abs_model_path)

        # load cnn model
        loaded_model = PaddleBaseClassifier.load(abs_model_path)

        # model type expected
        self.assertTrue(isinstance(loaded_model, CNNClassifier))
        # network type expected
        self.assertTrue(isinstance(loaded_model._network, model_network.__class__))

        # network state_dict expected
        self.assertEqual(
            model_network.state_dict().keys(),
            loaded_model._network.state_dict().keys()
        )
        common_network_state_dict_keys = model_network.state_dict().keys()
        for k in common_network_state_dict_keys:
            # {"_nn.0.weight": Tensor(shape=(xx, xx)), ...}
            raw = model_network.state_dict()[k]
            loaded = loaded_model._network.state_dict()[k]
            if isinstance(raw, paddle.Tensor):
                # convert tensor to numpy and call np.alltrue() to compare.
                self.assertTrue(np.alltrue(raw.numpy().astype(np.float64) == loaded.numpy().astype(np.float64)))

        # prediction results expected.
        loaded_model_preds = loaded_model.predict(paddlets_ds)
        self.assertTrue(np.alltrue(
            preds == loaded_model_preds
        ))
        shutil.rmtree(path)

        ############################################################
        # case 2 (good case) Two model exists under the same path. #
        ############################################################
        # build + fit + save an AE Model.
        model = self._build_cnn_model()

        train_paddlets_ds, train_labels = self._build_mock_data_and_label(random_data=True)
        valid_paddlets_ds, valid_labels = self._build_mock_data_and_label(random_data=True)
        model.fit(train_tsdatasets=train_paddlets_ds, train_labels=train_labels,
                  valid_tsdatasets=valid_paddlets_ds, valid_labels=valid_labels)
        model_network = model._network

        preds = model.predict(paddlets_ds)
        self.assertEqual(len(preds), len(train_paddlets_ds))

        path = os.path.join(os.getcwd(), str(random.randint(1, 10000000)))
        os.mkdir(path)

        # use same mode instance to save the first model file.
        model_1_name = "a"
        abs_model_1_path = os.path.join(path, model_1_name)
        model.save(abs_model_1_path)

        # use same mode instance to save the second model file.
        model_2_name = "b"
        abs_model_2_path = os.path.join(path, model_2_name)
        model.save(abs_model_2_path)

        # load the 2 ae models
        loaded_model_1 = PaddleBaseClassifier.load(abs_model_1_path)
        loaded_model_2 = PaddleBaseClassifier.load(abs_model_2_path)

        # model type expected
        self.assertEqual(model.__class__, loaded_model_1.__class__)
        self.assertEqual(model.__class__, loaded_model_2.__class__)

        # network type expected
        self.assertEqual(model_network.__class__, loaded_model_1._network.__class__)
        self.assertEqual(model_network.__class__, loaded_model_2._network.__class__)

        # network state_dict expected
        self.assertEqual(model_network.state_dict().keys(), loaded_model_1._network.state_dict().keys())
        self.assertEqual(model_network.state_dict().keys(), loaded_model_2._network.state_dict().keys())
        common_network_state_dict_keys = model_network.state_dict().keys()
        for k in common_network_state_dict_keys:
            # {"_nn.0.weight": Tensor(shape=(xx, xx)), ...}
            raw = model_network.state_dict()[k]
            loaded_1 = loaded_model_1._network.state_dict()[k]
            loaded_2 = loaded_model_2._network.state_dict()[k]
            if isinstance(raw, paddle.Tensor):
                # convert tensor to numpy and call np.alltrue() to compare.
                self.assertTrue(np.alltrue(raw.numpy().astype(np.float64) == loaded_1.numpy().astype(np.float64)))
                self.assertTrue(np.alltrue(raw.numpy().astype(np.float64) == loaded_2.numpy().astype(np.float64)))

        # prediction results expected.
        loaded_model_1_preds = loaded_model_1.predict(paddlets_ds)
        loaded_model_2_preds = loaded_model_2.predict(paddlets_ds)
        self.assertTrue(np.alltrue(
            preds == loaded_model_1_preds
        ))
        self.assertTrue(np.alltrue(
            preds == loaded_model_2_preds
        ))
        shutil.rmtree(path)

        ############################################################
        # case 3 (bad case) Model not exists under the given path. #
        ############################################################
        path = os.path.join(os.getcwd(), str(random.randint(1, 10000000)))

        succeed = True
        try:
            # path not exist.
            PaddleBaseClassifier.load(path)
        except Exception as e:
            succeed = False
        self.assertFalse(succeed)

        ##########################################################
        # case 4 (bad case) Path is a directory, but NOT a file. #
        ##########################################################
        path = os.path.join(os.getcwd(), str(random.randint(1, 10000000)))
        os.mkdir(path)

        succeed = True
        try:
            PaddleBaseClassifier.load(path)
        except Exception as e:
            succeed = False
        self.assertFalse(succeed)
        shutil.rmtree(path)

    def _build_cnn_model(
        self
    ) -> CNNClassifier:
        """
        Internal-only method, used for building an CNN model. The model is inherited from ClassifyBaseModel.

        Returns:
            CNNClassifier: the built cnn model instance.
        """
        return CNNClassifier()

    @staticmethod
    def _build_mock_data_and_label(
            target_periods: int = 200,
            target_dims: int = 5,
            n_classes: int = 4,
            instance_cnt: int = 100,
            random_data: bool = True,
            range_index: bool = False,
            seed: bool = False
    ):
        """
        build train datasets and labels.
        todo:not equal target_periods?
        """
        if seed:
            np.random.seed(2022)

        target_cols = [f"dim_{k}" for k in range(target_dims)]
        labels = [f"class" + str(item) for item in np.random.randint(0, n_classes, instance_cnt)]

        ts_datasets = []
        for i in range(instance_cnt):
            if random_data:
                target_data = np.random.randint(0, 10, (target_periods, target_dims))
            else:
                target_data = target_periods * [target_dims * [0]]
            if range_index:
                target_df = pd.DataFrame(
                    target_data,
                    index=pd.RangeIndex(0, target_periods, 1),
                    columns=target_cols
                )
            else:
                target_df = pd.DataFrame(
                    target_data,
                    index=pd.date_range("2022-01-01", periods=target_periods, freq="1D"),
                    columns=target_cols
                )
            ts_datasets.append(
                TSDataset(target=TimeSeries.load_from_dataframe(data=target_df).astype(np.float32))
            )

        return ts_datasets, labels


if __name__ == "__main__":
    unittest.main()
