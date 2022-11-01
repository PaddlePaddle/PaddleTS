# !/usr/bin/env python3
# -*- coding:utf-8 -*-

import pickle
import numpy as np
import paddle
import unittest
import os
from typing import Optional, Dict, List, Callable
import random
import shutil
import pandas as pd
import json
from copy import deepcopy

from paddlets import TSDataset, TimeSeries
from paddlets.models.representation.dl.repr_base import ReprBaseModel
from paddlets.models.representation.dl.ts2vec import TS2Vec


class _MockPaddleNetwork(paddle.nn.Layer):
    """Paddle layer implementing averaged model for Stochastic Weight Averaging (SWA).

    Args:
        network(paddle.nn.Layer): The network to use with SWA.
        avg_fn(Callable[..., paddle.Tensor]|None): The averaging function used to update parameters.

    Attributes:
        _network(paddle.nn.Layer): The network to use with SWA.
        _avg_fn(Callable[..., paddle.Tensor]): The averaging function used to update parameters.
    """
    def __init__(
        self,
        network: paddle.nn.Layer,
        avg_fn: Optional[Callable[..., paddle.Tensor]] = None,
    ):
        super(_MockPaddleNetwork, self).__init__()
        self._network = deepcopy(network)
        self.register_buffer("_num_averaged", paddle.to_tensor(0))

        def default_avg_fn(averaged_model_params, model_params, num_averaged):
            return averaged_model_params + (model_params - averaged_model_params) / (num_averaged + 1)
        self._avg_fn = default_avg_fn if avg_fn is None else avg_fn

    def forward(self, *args, **kwargs) -> paddle.Tensor:
        """Forward.

        Returns:
            paddle.Tensor: Output of model.
        """
        return self._network(*args, **kwargs)


class _MockNotPaddleModel(object):
    """
    Test bad case for ReprBaseModel.load() method.

    This class implements save / load / _init_network / _init_optimizer, but not inherited from ReprBaseModel.
    """
    def __init__(
        self,
        in_chunk_len: int = 1,
        out_chunk_len: int = 1,
        skip_chunk_len: int = 0,
        hidden_config: List[int] = None
    ):
        self._in_chunk_len = in_chunk_len
        self._out_chunk_len = out_chunk_len
        self._skip_chunk_len = skip_chunk_len
        self._fit_params = {"target_dim": 10}
        self._hidden_config = [100] if hidden_config is None else hidden_config
        self._network = None
        self._optimizer = None
        self._callback_container = None

    def fit(self, train_tsdataset: TSDataset, valid_tsdataset: Optional[TSDataset] = None):
        self._network = self._init_network()
        self._optimizer = self._init_optimizer()

    def predict(self, data: TSDataset) -> TSDataset:
        pass

    def encode(self, data: TSDataset) -> np.ndarray:
        pass

    def _init_network(self) -> paddle.nn.Layer:
        return _MockPaddleNetwork(network=self._network)

    def _init_optimizer(self) -> paddle.optimizer.Optimizer:
        return paddle.optimizer.Adam(parameters=self._network.parameters())

    def save(self, path: str) -> None:
        """
        Simulate the simplified-version of the ReprBaseModel.save() logic.

        This class is only used for testing, so only the core save logic are preserved, other logics (e.g. type check,
        name conflict check, etc.) are skipped.
        """
        abs_path = os.path.abspath(path)
        abs_root_path = os.path.dirname(abs_path)
        # network_statedict = self._network.state_dict()

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
            "ancestor_classname_set": [clazz.__name__ for clazz in self.__class__.mro()],
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


class TestReprBaseModel(unittest.TestCase):
    """
    ReprBaseModel unittest

    Currently, no need to test optimizer related logic.
    """
    def setUp(self):
        """
        unittest setup
        """
        self.default_modelname = "model"
        super().setUp()

    def test_save(self):
        """Test ReprBaseModel.save()"""
        ###################################
        # case 0 (good case)              #
        # 1) Model path exists.           #
        # 2) No filename conflicts.       #
        # 3) Not use valid data when fit. #
        # 4) Use built-in model TS2Vec.   #
        ###################################
        segment_size = 300
        sampling_stride = 300
        model = self._build_ts2vec_model(segment_size=segment_size, sampling_stride=sampling_stride)

        tsdataset = self._build_mock_ts_dataset(
            target_periods=segment_size,
            known_periods=segment_size,
            observed_periods=segment_size,
            random_data=True
        )
        # no validation dataset
        model.fit(train_tsdataset=tsdataset)
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

        # mode type
        with open(os.path.join(path, internal_filename_map["model_meta"]), "r") as f:
            model_meta = json.load(f)
        self.assertTrue(TS2Vec.__name__ in model_meta["ancestor_classname_set"])
        self.assertTrue(ReprBaseModel.__name__ in model_meta["ancestor_classname_set"])
        self.assertEqual(TS2Vec.__module__, model_meta["modulename"])
        shutil.rmtree(path)

        #################################
        # case 1 (good case)            #
        # 1) Model path exists.         #
        # 2) No filename conflicts.     #
        # 3) Save the same model twice. #
        # 4) Use built-in model TS2Vec. #
        #################################
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
        segment_size = 300
        sampling_stride = 300
        model = self._build_ts2vec_model(segment_size=segment_size, sampling_stride=sampling_stride)

        tsdataset = self._build_mock_ts_dataset(
            target_periods=segment_size,
            known_periods=segment_size,
            observed_periods=segment_size,
            random_data=True
        )
        model.fit(train_tsdataset=tsdataset)
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
        # case 2 (bad case) Model path NOT exists. #
        ############################################
        segment_size = 300
        sampling_stride = 300
        model = self._build_ts2vec_model(segment_size=segment_size, sampling_stride=sampling_stride)

        tsdataset = self._build_mock_ts_dataset(
            target_periods=segment_size,
            known_periods=segment_size,
            observed_periods=segment_size,
            random_data=True
        )
        model.fit(train_tsdataset=tsdataset)
        self.assertTrue(model._network is not None)

        path = os.path.join(os.getcwd(), str(random.randint(1, 10000000)))

        with self.assertRaises(ValueError):
            # path not exists.
            model.save(os.path.join(path, "void_model"))

        ###############################################################
        # case 3 (bad case) Model path is a directory, but NOT a file.#
        ###############################################################
        segment_size = 300
        sampling_stride = 300
        model = self._build_ts2vec_model(segment_size=segment_size, sampling_stride=sampling_stride)

        tsdataset = self._build_mock_ts_dataset(
            target_periods=segment_size,
            known_periods=segment_size,
            observed_periods=segment_size,
            random_data=True
        )
        model.fit(train_tsdataset=tsdataset)
        self.assertTrue(model._network is not None)

        path = os.path.join(os.getcwd(), str(random.randint(1, 10000000)))
        os.mkdir(path)

        with self.assertRaises(ValueError):
            # path exists, but is a dir, not a file.
            model.save(path)
        shutil.rmtree(path)

        ##########################################################################
        # case 4 (bad case) Modelname conflicts with caller's existing filename. #
        ##########################################################################
        segment_size = 300
        sampling_stride = 300
        model = self._build_ts2vec_model(segment_size=segment_size, sampling_stride=sampling_stride)

        tsdataset = self._build_mock_ts_dataset(
            target_periods=segment_size,
            known_periods=segment_size,
            observed_periods=segment_size,
            random_data=True
        )

        model.fit(train_tsdataset=tsdataset)
        self.assertTrue(model._network is not None)

        path = os.path.join(os.getcwd(), str(random.randint(1, 10000000)))
        os.mkdir(path)

        # create a dup file
        abs_model_path = os.path.join(path, self.default_modelname)
        dup_file_content = "this is a dup file."
        with open(abs_model_path, "w") as f:
            f.write(dup_file_content)

        with self.assertRaises(ValueError):
            # modelname conflicts with user's existing files.
            model.save(abs_model_path)

        # assert that the pre-existed conflict file content will (and should) not be overwritten.
        files = os.listdir(path)
        self.assertEqual(1, len(files))
        self.assertEqual(files[0], self.default_modelname)
        with open(os.path.join(path, self.default_modelname), "r") as f:
            tmp_data = f.read().strip()
        self.assertEqual(dup_file_content, tmp_data)
        shutil.rmtree(path)

        #############################################
        # case 5 (bad case) model._network is None. #
        #############################################
        model = self._build_ts2vec_model()
        model._network = None

        path = os.path.join(os.getcwd(), str(random.randint(1, 10000000)))
        os.mkdir(path)

        with self.assertRaises(ValueError):
            # path exists, but _network is None
            model.save(os.path.join(path, self.default_modelname))
        shutil.rmtree(path)

        #################################################################################
        # case 6 (bad case) Internal filename conflicts with callers' existing filename #
        #################################################################################
        segment_size = 300
        sampling_stride = 300
        model = self._build_ts2vec_model(segment_size=segment_size, sampling_stride=sampling_stride)

        tsdataset = self._build_mock_ts_dataset(
            target_periods=segment_size,
            known_periods=segment_size,
            observed_periods=segment_size,
            random_data=True
        )
        model.fit(train_tsdataset=tsdataset)
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

        with self.assertRaises(ValueError):
            # internal files conflict with caller's existing files.
            model.save(os.path.join(path, modelname))
        shutil.rmtree(path)

    def test_load(self):
        """Test ReprBaseModel.load()"""
        ###################################
        # case 0 (good case)              #
        # 1) Model exists.                #
        # 2) Not use valid data when fit. #
        ###################################
        # build + fit + save
        segment_size = 300
        sampling_stride = 300
        model = self._build_ts2vec_model(segment_size=segment_size, sampling_stride=sampling_stride)

        tsdataset = self._build_mock_ts_dataset(
            target_periods=segment_size,
            known_periods=segment_size,
            observed_periods=segment_size,
            random_data=True
        )
        model.fit(train_tsdataset=tsdataset)
        model_network = model._network

        encoded_ndarray = model.encode(tsdataset)

        path = os.path.join(os.getcwd(), str(random.randint(1, 10000000)))
        os.mkdir(path)

        abs_model_path = os.path.join(path, self.default_modelname)
        model.save(abs_model_path)

        # load model
        loaded_model = ReprBaseModel.load(abs_model_path)

        # model type expected
        self.assertIsInstance(loaded_model, TS2Vec)
        # network type expected
        self.assertIsInstance(loaded_model._network, model_network.__class__)

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

        # encoded result expected.
        loaded_model_encoded_ndarray = loaded_model.encode(tsdataset)
        self.assertTrue(np.alltrue(encoded_ndarray == loaded_model_encoded_ndarray))
        shutil.rmtree(path)

        # ############################################################
        # # case 1 (good case) Two model exists under the same path. #
        # ############################################################
        # build + fit + save
        segment_size = 300
        sampling_stride = 300
        model = self._build_ts2vec_model(segment_size=segment_size, sampling_stride=sampling_stride)

        tsdataset = self._build_mock_ts_dataset(
            target_periods=segment_size,
            known_periods=segment_size,
            observed_periods=segment_size,
            random_data=True
        )

        model.fit(tsdataset)
        model_network = model._network

        encoded_ndarray = model.encode(tsdataset)

        path = os.path.join(os.getcwd(), str(random.randint(1, 10000000)))
        os.mkdir(path)

        # use same mode instance to save the 1st model file.
        model_1_name = "a"
        abs_model_1_path = os.path.join(path, model_1_name)
        model.save(abs_model_1_path)

        # use same mode instance to save the 2nd model file.
        model_2_name = "b"
        abs_model_2_path = os.path.join(path, model_2_name)
        model.save(abs_model_2_path)

        # load 2 models
        loaded_model_1 = ReprBaseModel.load(abs_model_1_path)
        loaded_model_2 = ReprBaseModel.load(abs_model_2_path)

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
        loaded_model_1_encoded_ndarray = loaded_model_1.encode(tsdataset)
        loaded_model_2_encoded_ndarray = loaded_model_2.encode(tsdataset)
        self.assertTrue(np.alltrue(encoded_ndarray == loaded_model_1_encoded_ndarray))
        self.assertTrue(np.alltrue(encoded_ndarray == loaded_model_2_encoded_ndarray))
        shutil.rmtree(path)

        ###############################################################
        # case 2 (bad case) Model does NOT inherit from ReprBaseModel #
        ###############################################################
        model = _MockNotPaddleModel()

        tsdataset = self._build_mock_ts_dataset(random_data=True)
        model.fit(train_tsdataset=tsdataset)
        self.assertTrue(model._network is not None)
        self.assertTrue(model._optimizer is not None)

        path = os.path.join(os.getcwd(), str(random.randint(1, 10000000)))
        os.mkdir(path)

        abs_model_path = os.path.join(path, self.default_modelname)
        model.save(abs_model_path)

        with self.assertRaises(ValueError):
            _ = ReprBaseModel.load(abs_model_path)
        shutil.rmtree(path)

        ############################################################
        # case 3 (bad case) Model not exists under the given path. #
        ############################################################
        path = os.path.join(os.getcwd(), str(random.randint(1, 10000000)))

        with self.assertRaises(ValueError):
            # path not exist.
            ReprBaseModel.load(path)

        ##########################################################
        # case 4 (bad case) Path is a directory, but NOT a file. #
        ##########################################################
        path = os.path.join(os.getcwd(), str(random.randint(1, 10000000)))
        os.mkdir(path)

        with self.assertRaises(ValueError):
            _ = ReprBaseModel.load(path)
        shutil.rmtree(path)
    
    def _build_ts2vec_model(
        self,
        segment_size=300,
        sampling_stride=300,
        max_epochs=1
    ) -> TS2Vec:
        """
        Internal-only method, used for building a model. The model is inherited from ReprBaseModel.

        Returns:
            TS2Vec: the built model instance.
        """
        return TS2Vec(segment_size=segment_size, sampling_stride=sampling_stride, max_epochs=max_epochs)

    @staticmethod
    def _build_mock_ts_dataset(
        target_periods: int = 200,
        known_periods: int = 300,
        observed_periods: int = 200,
        random_data: bool = False,
        seed: bool = False
    ):
        """
        build bts TSDataset instance.

        Note that random_data must set to True if the returned TSDataset is used for fitting models, otherwise paddle
        will raise the following exception:
        RuntimeError: (NotFound) There are no kernels which are registered in the xxx operator.
        """
        if seed:
            np.random.seed(2023)

        target_cols = ["target0"]
        if random_data:
            target_data = np.random.randn(target_periods, len(target_cols)).astype(np.float64)
        else:
            target_data = [[i for _ in range(len(target_cols))] for i in range(target_periods)]
        target_df = pd.DataFrame(
            target_data,
            index=pd.date_range("2022-01-01", periods=target_periods, freq="1D"),
            columns=target_cols
        )

        known_cols = ["known0", "known1"]
        if random_data:
            known_data = np.random.randn(known_periods, len(known_cols)).astype(np.float64)
        else:
            known_data = [(i * 10 for _ in range(len(known_cols))) for i in range(known_periods)]
        known_cov_df = pd.DataFrame(
            known_data,
            index=pd.date_range("2022-01-01", periods=known_periods, freq="1D"),
            columns=known_cols
        )

        observed_cols = ["past0", "past1"]
        if random_data:
            known_data = np.random.randn(observed_periods, len(observed_cols)).astype(np.float64)
        else:
            known_data = [(i * -1 for _ in range(len(observed_cols))) for i in range(observed_periods)]
        observed_cov_df = pd.DataFrame(
            known_data,
            index=pd.date_range("2022-01-01", periods=observed_periods, freq="1D"),
            columns=observed_cols
        )

        return TSDataset(
            target=TimeSeries.load_from_dataframe(data=target_df),
            known_cov=TimeSeries.load_from_dataframe(data=known_cov_df),
            observed_cov=TimeSeries.load_from_dataframe(data=observed_cov_df),
            static_cov={"static0": 1.0, "static1": 2.0}
        )
