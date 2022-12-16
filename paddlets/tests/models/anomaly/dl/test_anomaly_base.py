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

from paddlets import TSDataset, TimeSeries
from paddlets.models.anomaly.dl.anomaly_base import AnomalyBaseModel
from paddlets.models.anomaly import AutoEncoder


class _MockAnomalyNetwork(paddle.nn.Layer):
    """A Mock paddle.nn.Layer used for testing.
    Args:
        in_chunk_dim(int): The size of the loopback window, i.e. the number of time steps feed to the model.
        feature_dim(int): The numer of features.
        hidden_config(List[int]|None): The ith element represents the number of neurons in the ith hidden layer.
        use_bn(bool): Whether to use batch normalization.

    Attributes:
        _nn(paddle.nn.Sequential): Dynamic graph LayerList.
    """
    def __init__(
        self,
        in_chunk_dim: int = 1,
        feature_dim: int = 1,
        hidden_config: List[int] = None,
        use_bn: bool = False
    ):
        super(_MockAnomalyNetwork, self).__init__()
        hidden_config = [8, 16] if (hidden_config is None or len(hidden_config) == 0) else hidden_config
        dims, layers = [in_chunk_dim] + hidden_config, []
        for i in range(1, len(dims)):
            layers.append(paddle.nn.Linear(dims[i - 1], dims[i]))
            if use_bn:
                layers.append(paddle.nn.BatchNorm1D(feature_dim))
        self._encoder = paddle.nn.Sequential(*layers)
        
        dims = hidden_config[::-1] + [in_chunk_dim]
        layers = []
        for i in range(1, len(dims)):
            layers.append(paddle.nn.Linear(dims[i - 1], dims[i]))
            if use_bn:
                layers.append(paddle.nn.BatchNorm1D(feature_dim))
        self._decoder = paddle.nn.Sequential(*layers)

    def forward(
        self,
        x: Dict[str, paddle.Tensor]
    ) -> paddle.Tensor:
        x = paddle.transpose(x["observed_cov"], perm=[0, 2, 1])
        x = self._encoder(x)
        x = self._decoder(x)
        return paddle.transpose(x, perm=[0, 2, 1])


class _MockNotAnomalyBaseModel(object):
    """
    Test bad case for AnomalyBaseModel.load() method.

    This class implements save / load / _init_network / _init_optimizer, but not inherited from AnomalyBaseModel.
    """
    def __init__(
        self,
        in_chunk_len: int = 1,
        hidden_config: List[int] = None
    ):
        self._in_chunk_len = in_chunk_len
        self._fit_params = {"observed_dim": 1}
        self._hidden_config = [8, 16] if hidden_config is None else hidden_config
        self._network = None
        self._optimizer = None
        self._callback_container = None

    def fit(self, train_tsdataset: TSDataset, valid_tsdataset: Optional[TSDataset] = None):
        self._network = self._init_network()
        self._optimizer = self._init_optimizer()

    def predict(self, data: TSDataset) -> TSDataset:
        pass

    def _init_network(self) -> paddle.nn.Layer:
        return _MockAnomalyNetwork(
            in_chunk_dim=self._in_chunk_len,
            feature_dim=self._fit_params['observed_dim'],
            hidden_config=self._hidden_config
        )

    def _init_optimizer(self) -> paddle.optimizer.Optimizer:
        return paddle.optimizer.Adam(parameters=self._network.parameters())

    def save(self, path: str) -> None:
        """
        Simulate the simplified-version of the AnomalyBaseModel.save() logic.

        This class is only used for testing, so only the core save logic are preserved, other logics (e.g. type check,
        name conflict check, etc.) are skipped.
        """
        abs_path = os.path.abspath(path)
        abs_root_path = os.path.dirname(abs_path)

        network_ref_copy = self._network
        self._network = None
        optimizer_ref_copy = self._optimizer
        self._optimizer = None
        callback_container_ref_copy = self._callback_container
        self._callback_container = None

        # save model
        with open(abs_path, "wb") as f:
            pickle.dump(self, f)
        model_meta = {
            # ChildModel,AnomalyBaseModel,Trainable,ABC,object
            "ancestor_classname_set": [clazz.__name__ for clazz in self.__class__.mro()],
            # paddlets.models.anomaly.dl.paddlepaddle.xxx
            "modulename": self.__module__
        }

        modelname = os.path.basename(abs_path)
        internal_filename_map = {
            "model_meta": "%s_%s" % (modelname, "model_meta"),
            "network_statedict": "%s_%s" % (modelname, "network_statedict"),
            # currently ignore optimizer.
            # "optimizer_statedict": "%s_%s" % (modelname, "optimizer_statedict"),
        }
        with open(os.path.join(abs_root_path, internal_filename_map["model_meta"]), "w") as f:
            json.dump(model_meta, f, ensure_ascii=False)
        paddle.save(
            obj=network_ref_copy.state_dict(),
            path=os.path.join(abs_root_path, internal_filename_map["network_statedict"])
        )
        self._network = network_ref_copy
        self._optimizer = optimizer_ref_copy
        self._callback_container = callback_container_ref_copy
        return


class TestAnomalyBaseModel(unittest.TestCase):
    """
    AnomalyBaseModel unittest

    Currently, no need to test optimizer related logic.
    """
    def setUp(self):
        """
        unittest setup
        """
        self.default_modelname = "model"
        super().setUp()
        
    def test_check_tsdataset(self):
        """Test check_tsdataset"""
        ############################################
        # case 1 (bad case)                        #
        # target is float.                         #
        ############################################
        model = self._build_ae_model()
        paddlets_ds = self._build_mock_ts_dataset(random_data=True)
        paddlets_ds['label'] = paddlets_ds['label'].astype(float)
        with self.assertRaises(ValueError):
            model.fit(paddlets_ds)
            
        ############################################
        # case 2 (bad case)                        #
        # known is not None.                       #
        ############################################
        model = self._build_ae_model()
        paddlets_ds = self._build_mock_ts_dataset(random_data=True)
        known_df = pd.DataFrame(
            np.random.randn(200, 2), 
            index=pd.date_range('2022-01-01', periods=200, freq='1D')
        )
        known_ts = TimeSeries.load_from_dataframe(data=known_df)
        paddlets_ds.known_cov = known_ts
        with self.assertLogs("paddlets", level="WARNING") as captured:
            model.fit(paddlets_ds)
            self.assertEqual(
                captured.records[0].getMessage(),
                "Input tsdataset contains known cov `0,1` which will be ignored in anomaly detection scenario."
            )
            
        ############################################
        # case 3 (bad case)                        #
        # static is not None.                      #
        ############################################
        model = self._build_ae_model()
        paddlets_ds = self._build_mock_ts_dataset(random_data=True)
        static_dict = {'static': 1}
        paddlets_ds.static_cov = static_dict
        with self.assertLogs("paddlets", level="WARNING") as captured:
            model.fit(paddlets_ds)
            self.assertEqual(
                captured.records[0].getMessage(),
                "Input tsdataset contains static cov `static` which will be ignored in anomaly detection scenario."
            )
            
        ######################################################
        # case 4 (bad case)                                  #
        # There are non numerical types in the observed.     #
        ######################################################
        model = self._build_ae_model()
        paddlets_ds = self._build_mock_ts_dataset(random_data=True)
        paddlets_ds['a'] = paddlets_ds['a'].astype(object)
        with self.assertRaises(ValueError):
            model.fit(paddlets_ds)
            
        ############################################
        # case 5 (bad case)                        #
        # The observed are all integers.           #
        ############################################
        model = self._build_ae_model()
        paddlets_ds = self._build_mock_ts_dataset(random_data=True)
        observed = paddlets_ds.get_observed_cov()
        for col, dtype in observed.dtypes.items():
            paddlets_ds[col] = paddlets_ds[col].astype(int)
        with self.assertRaises(ValueError):
            model.fit(paddlets_ds)
        

    def test_save(self):
        """Test AnomalyBaseModel.save()"""
        ############################################
        # case 1 (good case)                       #
        # 1) Model path exists.                    #
        # 2) No filename conflicts.                #
        # 3) Not use valid data when fit.          #
        # 4) Use built-in model AutoEncoder. #
        ############################################
        model = self._build_ae_model()

        paddlets_ds = self._build_mock_ts_dataset(random_data=True)
        # no validation dataset
        model.fit(train_tsdataset=paddlets_ds)
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

        # mode type AutoEncoder
        with open(os.path.join(path, internal_filename_map["model_meta"]), "r") as f:
            model_meta = json.load(f)
        # AutoEncoder,AnomalyBaseModel,Trainable,ABC,object
        self.assertTrue(AutoEncoder.__name__ in model_meta["ancestor_classname_set"])
        self.assertTrue(AnomalyBaseModel.__name__ in model_meta["ancestor_classname_set"])
        # paddlets.models.anomaly.dl.autoencoder
        self.assertEqual(AutoEncoder.__module__, model_meta["modulename"])
        shutil.rmtree(path)

        ############################################
        # case 1 (good case)                       #
        # 1) Model path exists.                    #
        # 2) No filename conflicts.                #
        # 3) Use valid data when fit.              #
        # 4) Use built-in model AutoEncoder. #
        ############################################
        model = self._build_ae_model()

        train_paddlets_ds = self._build_mock_ts_dataset(random_data=True)
        valid_paddlets_ds = self._build_mock_ts_dataset(random_data=True)
        # use validation dataset.
        model.fit(train_tsdataset=train_paddlets_ds, valid_tsdataset=valid_paddlets_ds)
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

        # mode type AutoEncoder
        with open(os.path.join(path, internal_filename_map["model_meta"]), "r") as f:
            model_meta = json.load(f)
        # AutoEncoder,AnomalyBaseModel,Trainable,ABC,object
        self.assertTrue(AutoEncoder.__name__ in model_meta["ancestor_classname_set"])
        self.assertTrue(AnomalyBaseModel.__name__ in model_meta["ancestor_classname_set"])
        # paddlets.models.dl.anomaly.autoencoder
        self.assertEqual(AutoEncoder.__module__, model_meta["modulename"])
        shutil.rmtree(path)

        ############################################
        # case 2 (good case)                       #
        # 1) Model path exists.                    #
        # 2) No filename conflicts.                #
        # 3) Save the same model twice.            #
        # 4) Use built-in model AutoEncoder. #
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
        model = self._build_ae_model()

        paddlets_ds = self._build_mock_ts_dataset(random_data=True)
        model.fit(train_tsdataset=paddlets_ds)
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
        model = self._build_ae_model()

        paddlets_ds = self._build_mock_ts_dataset(random_data=True)
        model.fit(train_tsdataset=paddlets_ds)
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
        model = self._build_ae_model()

        paddlets_ds = self._build_mock_ts_dataset(random_data=True)
        model.fit(train_tsdataset=paddlets_ds)
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
        model = self._build_ae_model()

        paddlets_ds = self._build_mock_ts_dataset(random_data=True)
        model.fit(train_tsdataset=paddlets_ds)
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
        model = self._build_ae_model()
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
        model = self._build_ae_model()

        paddlets_ds = self._build_mock_ts_dataset(random_data=True)
        model.fit(train_tsdataset=paddlets_ds)
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

        ############################################
        # case 8 (good case)                       
        # 1) network_model == True
        # 2) dygraph_to_static == True
        ############################################
        model = self._build_ae_model()

        paddlets_ds = self._build_mock_ts_dataset(random_data=True)
        # no validation dataset
        model.fit(train_tsdataset=paddlets_ds)
        self.assertTrue(model._network is not None)
        self.assertTrue(model._optimizer is not None)

        path = os.path.join(os.getcwd(), str(random.randint(1, 10000000)))
        os.mkdir(path)

        internal_filename_map = {
            "network_model":"%s.pdmodel" % (self.default_modelname),
            "network_model_params":"%s.pdiparams" % (self.default_modelname),
            "network_model_params_info":"%s.pdiparams.info" % (self.default_modelname),
            "model_meta": "%s_%s" % (self.default_modelname, "model_meta"),
            "network_statedict": "%s_%s" % (self.default_modelname, "network_statedict"),
            # currently ignore optimizer.
            # "optimizer_statedict": "%s_%s" % (modelname, "optimizer_statedict"),
        }
        model.save(os.path.join(path, self.default_modelname), network_model=True, dygraph_to_static=True)
        self.assertTrue(model._network is not None)
        self.assertTrue(model._optimizer is not None)

        files = set(os.listdir(path))
        self.assertEqual(files, {self.default_modelname, *internal_filename_map.values()})

        # mode type AutoEncoder
        with open(os.path.join(path, internal_filename_map["model_meta"]), "r") as f:
            model_meta = json.load(f)
        # AutoEncoder,AnomalyBaseModel,Trainable,ABC,object
        self.assertTrue(AutoEncoder.__name__ in model_meta["ancestor_classname_set"])
        self.assertTrue(AnomalyBaseModel.__name__ in model_meta["ancestor_classname_set"])
        # paddlets.models.anomaly.dl.autoencoder
        self.assertEqual(AutoEncoder.__module__, model_meta["modulename"])
        self.assertEqual("anomaly", model_meta['model_type'])
        self.assertTrue("model_threshold" in model_meta)
        self.assertEqual({"in_chunk_len": 10}, model_meta["size"])
        self.assertEqual({"observed_cov_numeric": [None, 10, 2]}, model_meta["input_data"])
        shutil.rmtree(path)

        ############################################
        # case 9 (good case)                       
        # 1) network_model == True
        # 2) dygraph_to_static == False
        ############################################
        model = self._build_ae_model()

        paddlets_ds = self._build_mock_ts_dataset(random_data=True)
        # no validation dataset
        model.fit(train_tsdataset=paddlets_ds)
        self.assertTrue(model._network is not None)
        self.assertTrue(model._optimizer is not None)

        path = os.path.join(os.getcwd(), str(random.randint(1, 10000000)))
        os.mkdir(path)

        internal_filename_map = {
            "network_model":"%s.pdmodel" % (self.default_modelname),
            "network_model_params":"%s.pdiparams" % (self.default_modelname),
            "network_model_params_info":"%s.pdiparams.info" % (self.default_modelname),
            "model_meta": "%s_%s" % (self.default_modelname, "model_meta"),
            "network_statedict": "%s_%s" % (self.default_modelname, "network_statedict"),
            # currently ignore optimizer.
            # "optimizer_statedict": "%s_%s" % (modelname, "optimizer_statedict"),
        }
        model.save(os.path.join(path, self.default_modelname), network_model=True, dygraph_to_static=False)
        self.assertTrue(model._network is not None)
        self.assertTrue(model._optimizer is not None)

        files = set(os.listdir(path))
        self.assertEqual(files, {self.default_modelname, *internal_filename_map.values()})

        # mode type AutoEncoder
        with open(os.path.join(path, internal_filename_map["model_meta"]), "r") as f:
            model_meta = json.load(f)
        # AutoEncoder,AnomalyBaseModel,Trainable,ABC,object
        self.assertTrue(AutoEncoder.__name__ in model_meta["ancestor_classname_set"])
        self.assertTrue(AnomalyBaseModel.__name__ in model_meta["ancestor_classname_set"])
        # paddlets.models.anomaly.dl.autoencoder
        self.assertEqual(AutoEncoder.__module__, model_meta["modulename"])
        self.assertEqual("anomaly", model_meta['model_type'])
        self.assertTrue("model_threshold" in model_meta)
        self.assertEqual({"in_chunk_len": 10}, model_meta["size"])
        self.assertEqual({"observed_cov_numeric": [None, 10, 2]}, model_meta["input_data"])
        shutil.rmtree(path)

    def test_load(self):
        """Test AnomalyBaseModel.load()"""
        ###################################
        # case 0 (good case)              #
        # 1) Model exists.                #
        # 2) Not use valid data when fit. #
        ###################################
        # build + fit + save an RNN Model.
        in_chunk_len = 10
        model = self._build_ae_model(in_chunk_len=in_chunk_len)

        paddlets_ds = self._build_mock_ts_dataset(random_data=True)
        model.fit(train_tsdataset=paddlets_ds)
        model_network = model._network

        predicted_paddlets_ds = model.predict(paddlets_ds)
        self.assertEqual(
            ((len(paddlets_ds.get_observed_cov())-in_chunk_len+1), len(paddlets_ds.get_target().data.columns)),
            predicted_paddlets_ds.get_target().data.shape
        )

        path = os.path.join(os.getcwd(), str(random.randint(1, 10000000)))
        os.mkdir(path)

        abs_model_path = os.path.join(path, self.default_modelname)
        model.save(abs_model_path)

        # load rnn model
        loaded_model = AnomalyBaseModel.load(abs_model_path)

        # model type expected
        self.assertTrue(isinstance(loaded_model, AutoEncoder))
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
        loaded_model_predicted_paddlets_ds = loaded_model.predict(paddlets_ds)
        self.assertTrue(np.alltrue(
            predicted_paddlets_ds.get_target().to_numpy(False) == loaded_model_predicted_paddlets_ds.get_target().to_numpy(False)
        ))
        shutil.rmtree(path)

        ###############################
        # case 1 (good case)          #
        # 1) Model exists.            #
        # 2) Use valid data when fit. #
        ###############################
        # build + fit + save an AE Model.
        in_chunk_len = 10
        model = self._build_ae_model(in_chunk_len=in_chunk_len)

        train_paddlets_ds = self._build_mock_ts_dataset(random_data=True)
        valid_paddlets_ds = self._build_mock_ts_dataset(random_data=True)
        model.fit(train_tsdataset=train_paddlets_ds, valid_tsdataset=valid_paddlets_ds)
        model_network = model._network

        predicted_paddlets_ds = model.predict(paddlets_ds)
        self.assertEqual(
            ((len(paddlets_ds.get_observed_cov())-in_chunk_len+1), len(paddlets_ds.get_target().data.columns)),
            predicted_paddlets_ds.get_target().data.shape
        )

        path = os.path.join(os.getcwd(), str(random.randint(1, 10000000)))
        os.mkdir(path)

        abs_model_path = os.path.join(path, self.default_modelname)
        model.save(abs_model_path)

        # load rnn model
        loaded_model = AnomalyBaseModel.load(abs_model_path)

        # model type expected
        self.assertTrue(isinstance(loaded_model, AutoEncoder))
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
        loaded_model_predicted_paddlets_ds = loaded_model.predict(paddlets_ds)
        self.assertTrue(np.alltrue(
            predicted_paddlets_ds.get_target().to_numpy(False) == loaded_model_predicted_paddlets_ds.get_target().to_numpy(False)
        ))
        shutil.rmtree(path)

        ############################################################
        # case 2 (good case) Two model exists under the same path. #
        ############################################################
        # build + fit + save an AE Model.
        in_chunk_len = 10
        model = self._build_ae_model(in_chunk_len=in_chunk_len)

        train_paddlets_ds = self._build_mock_ts_dataset(random_data=True)
        valid_paddlets_ds = self._build_mock_ts_dataset(random_data=True)
        model.fit(train_tsdataset=train_paddlets_ds, valid_tsdataset=valid_paddlets_ds)
        model_network = model._network

        predicted_paddlets_ds = model.predict(paddlets_ds)
        self.assertEqual(
            ((len(paddlets_ds.get_observed_cov())-in_chunk_len+1), len(paddlets_ds.get_target().data.columns)),
            predicted_paddlets_ds.get_target().data.shape
        )

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
        loaded_model_1 = AnomalyBaseModel.load(abs_model_1_path)
        loaded_model_2 = AnomalyBaseModel.load(abs_model_2_path)

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
        loaded_model_1_predicted_paddlets_ds = loaded_model_1.predict(paddlets_ds)
        loaded_model_2_predicted_paddlets_ds = loaded_model_2.predict(paddlets_ds)
        self.assertTrue(np.alltrue(
            predicted_paddlets_ds.get_target().to_numpy(False) == loaded_model_1_predicted_paddlets_ds.get_target().to_numpy(False)
        ))
        self.assertTrue(np.alltrue(
            predicted_paddlets_ds.get_target().to_numpy(False) == loaded_model_2_predicted_paddlets_ds.get_target().to_numpy(False)
        ))
        shutil.rmtree(path)

        ##################################################################
        # case 3 (bad case) Model does NOT inherit from  AnomalyBaseModel #
        ##################################################################
        # model not inherited from AnomalyBaseModel
        model = _MockNotAnomalyBaseModel()

        paddlets_ds = self._build_mock_ts_dataset(random_data=True)
        model.fit(train_tsdataset=paddlets_ds)
        self.assertTrue(model._network is not None)
        self.assertTrue(model._optimizer is not None)

        path = os.path.join(os.getcwd(), str(random.randint(1, 10000000)))
        os.mkdir(path)

        abs_model_path = os.path.join(path, self.default_modelname)
        model.save(abs_model_path)

        succeed = True
        try:
            loaded_model = AnomalyBaseModel.load(abs_model_path)
        except Exception as e:
            succeed = False
        self.assertFalse(succeed)
        shutil.rmtree(path)

        ############################################################
        # case 4 (bad case) Model not exists under the given path. #
        ############################################################
        path = os.path.join(os.getcwd(), str(random.randint(1, 10000000)))

        succeed = True
        try:
            # path not exist.
            AnomalyBaseModel.load(path)
        except Exception as e:
            succeed = False
        self.assertFalse(succeed)

        ##########################################################
        # case 5 (bad case) Path is a directory, but NOT a file. #
        ##########################################################
        path = os.path.join(os.getcwd(), str(random.randint(1, 10000000)))
        os.mkdir(path)

        succeed = True
        try:
            AnomalyBaseModel.load(path)
        except Exception as e:
            succeed = False
        self.assertFalse(succeed)
        shutil.rmtree(path)

        ###############################
        # case 6 (good case)          
        # 1) paddle inference.
        # 2) dygraph_to_static == True
        ###############################
        # build + fit + save an AE Model.
        in_chunk_len = 10
        model = self._build_ae_model(in_chunk_len=in_chunk_len)

        train_paddlets_ds = self._build_mock_ts_dataset(random_data=True)
        valid_paddlets_ds = self._build_mock_ts_dataset(random_data=True)
        model.fit(train_tsdataset=train_paddlets_ds, valid_tsdataset=valid_paddlets_ds)
        model_network = model._network

        predicted_paddlets_ds = model.predict(paddlets_ds)
        self.assertEqual(
            ((len(paddlets_ds.get_observed_cov())-in_chunk_len+1), len(paddlets_ds.get_target().data.columns)),
            predicted_paddlets_ds.get_target().data.shape
        )
        _, input_data_ts = paddlets_ds.split(len(paddlets_ds.observed_cov) - 10)
        input_data_ts_score = model.predict_score(input_data_ts)

        path = os.path.join(os.getcwd(), str(random.randint(1, 10000000)))
        os.mkdir(path)

        internal_filename_map = {
            "network_model":"%s.pdmodel" % (self.default_modelname),
            "network_model_params":"%s.pdiparams" % (self.default_modelname),
            "network_model_params_info":"%s.pdiparams.info" % (self.default_modelname),
            "model_meta": "%s_%s" % (self.default_modelname, "model_meta"),
            "network_statedict": "%s_%s" % (self.default_modelname, "network_statedict"),
            # currently ignore optimizer.
            # "optimizer_statedict": "%s_%s" % (modelname, "optimizer_statedict"),
        }

        abs_model_path = os.path.join(path, self.default_modelname)
        model.save(abs_model_path, network_model=True, dygraph_to_static=True)

        with open(os.path.join(path, internal_filename_map["model_meta"]), "r") as f:
            model_meta = json.load(f)

        import paddle.inference as paddle_infer
        config = paddle_infer.Config(abs_model_path + ".pdmodel", abs_model_path + ".pdiparams")
        predictor = paddle_infer.create_predictor(config)
        input_names = predictor.get_input_names()
        self.assertEqual(len(input_names), len(model_meta['input_data']))
        self.assertEqual(input_names[0], list(model_meta['input_data'].keys())[0])

        input_handle = predictor.get_input_handle(input_names[0])
        input_handle.reshape([1, list(model_meta['input_data'].values())[0][-2], list(model_meta['input_data'].values())[0][-1]])

        input_data = input_data_ts.observed_cov.to_numpy()
        input_data = input_data.reshape([1, list(model_meta['input_data'].values())[0][-2], list(model_meta['input_data'].values())[0][-1]]).astype("float32")
        input_handle.copy_from_cpu(input_data)

        predictor.run()
        output_names = predictor.get_output_names()
        output_handle = predictor.get_output_handle(output_names[0])
        output_data = output_handle.copy_to_cpu()

        self.assertEqual(output_data.shape, (1, model_meta['size']['in_chunk_len'], list(model_meta['input_data'].values())[0][-1]))
        loss = model._get_loss(paddle.to_tensor(output_data), paddle.to_tensor(input_data))
        self.assertAlmostEqual(input_data_ts_score.target.data.iloc[0, 0], loss[0], delta=0.001)

        shutil.rmtree(path)

        ###############################
        # case 6 (good case)          
        # 1) paddle inference.
        # 2) dygraph_to_static == False
        ###############################
        # build + fit + save an AE Model.
        in_chunk_len = 10
        model = self._build_ae_model(in_chunk_len=in_chunk_len)

        train_paddlets_ds = self._build_mock_ts_dataset(random_data=True)
        valid_paddlets_ds = self._build_mock_ts_dataset(random_data=True)
        model.fit(train_tsdataset=train_paddlets_ds, valid_tsdataset=valid_paddlets_ds)
        model_network = model._network

        predicted_paddlets_ds = model.predict(paddlets_ds)
        self.assertEqual(
            ((len(paddlets_ds.get_observed_cov())-in_chunk_len+1), len(paddlets_ds.get_target().data.columns)),
            predicted_paddlets_ds.get_target().data.shape
        )
        _, input_data_ts = paddlets_ds.split(len(paddlets_ds.observed_cov) - 10)
        input_data_ts_score = model.predict_score(input_data_ts)

        path = os.path.join(os.getcwd(), str(random.randint(1, 10000000)))
        os.mkdir(path)

        internal_filename_map = {
            "network_model":"%s.pdmodel" % (self.default_modelname),
            "network_model_params":"%s.pdiparams" % (self.default_modelname),
            "network_model_params_info":"%s.pdiparams.info" % (self.default_modelname),
            "model_meta": "%s_%s" % (self.default_modelname, "model_meta"),
            "network_statedict": "%s_%s" % (self.default_modelname, "network_statedict"),
            # currently ignore optimizer.
            # "optimizer_statedict": "%s_%s" % (modelname, "optimizer_statedict"),
        }

        abs_model_path = os.path.join(path, self.default_modelname)
        model.save(abs_model_path, network_model=True, dygraph_to_static=False)

        with open(os.path.join(path, internal_filename_map["model_meta"]), "r") as f:
            model_meta = json.load(f)

        import paddle.inference as paddle_infer
        config = paddle_infer.Config(abs_model_path + ".pdmodel", abs_model_path + ".pdiparams")
        predictor = paddle_infer.create_predictor(config)
        input_names = predictor.get_input_names()
        self.assertEqual(len(input_names), len(model_meta['input_data']))
        self.assertEqual(input_names[0], list(model_meta['input_data'].keys())[0])

        input_handle = predictor.get_input_handle(input_names[0])
        input_handle.reshape([1, list(model_meta['input_data'].values())[0][-2], list(model_meta['input_data'].values())[0][-1]])

        input_data = input_data_ts.observed_cov.to_numpy()
        input_data = input_data.reshape([1, list(model_meta['input_data'].values())[0][-2], list(model_meta['input_data'].values())[0][-1]]).astype("float32")
        input_handle.copy_from_cpu(input_data)

        predictor.run()
        output_names = predictor.get_output_names()
        output_handle = predictor.get_output_handle(output_names[0])
        output_data = output_handle.copy_to_cpu()

        self.assertEqual(output_data.shape, (1, model_meta['size']['in_chunk_len'], list(model_meta['input_data'].values())[0][-1]))
        loss = model._get_loss(paddle.to_tensor(output_data), paddle.to_tensor(input_data))
        self.assertAlmostEqual(input_data_ts_score.target.data.iloc[0, 0], loss[0], delta=0.001)

        shutil.rmtree(path)

    def _build_ae_model(
        self,
        in_chunk_len: int = 10,
        max_epochs: int = 1,
    ) -> AutoEncoder:
        """
        Internal-only method, used for building an AutoEncoder model. The model is inherited from AnomalyBaseModel.

        Args:
            in_chunk_len(int, optional): AutoEncoder model required param.

        Returns:
            AutoEncoder: the built autoencoder model instance.
        """
        return AutoEncoder(
            in_chunk_len=in_chunk_len,
            max_epochs=max_epochs
        )

    @staticmethod
    def _build_mock_ts_dataset(
            target_periods: int = 200,
            observed_periods: int = 200,
            random_data: bool = False,
            seed: bool = False
    ):
        """
        build paddlets TSDataset instance.

        Note that random_data must set to True if the returned TSDataset is used for fitting models, otherwise paddle
        will raise the following exception:
        RuntimeError: (NotFound) There are no kernels which are registered in the ae operator.
        """
        if seed:
            np.random.seed(2023)

        target_cols = ["label"]
        if random_data:
            target_data = np.random.randint(0, 2, target_periods)
        else:
            target_data = target_periods * [0]
        target_df = pd.DataFrame(
            target_data,
            index=pd.date_range("2022-01-01", periods=target_periods, freq="1D"),
            columns=target_cols
        )

        observed_cols = ["a", "b"]
        if random_data:
            observed_data = np.random.randn(observed_periods, len(observed_cols)).astype(np.float64)
        else:
            observed_data = [(i * -1 for _ in range(len(observed_cols))) for i in range(observed_periods)]
        observed_cov_df = pd.DataFrame(
            observed_data,
            index=pd.date_range("2022-01-01", periods=observed_periods, freq="1D"),
            columns=observed_cols
        )

        return TSDataset(
            target=TimeSeries.load_from_dataframe(data=target_df),
            observed_cov=TimeSeries.load_from_dataframe(data=observed_cov_df),
        )

if __name__ == "__main__":
    unittest.main()
