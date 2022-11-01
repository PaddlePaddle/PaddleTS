# !/usr/bin/env python3
# -*- coding:utf-8 -*-

import paddle
import shutil
import unittest
import os
import pandas as pd
import numpy as np
import random
from typing import List, Dict, Optional
import pickle
import json

import paddlets
from paddlets import TSDataset, TimeSeries
from paddlets.models.forecasting import RNNBlockRegressor
from paddlets.models.representation import TS2Vec


class _MockNotMLModel(object):
    """This class simulates a model that is not inherited from MLBaseModel"""
    def save(self, path: str):
        """a minimized save implementation."""
        abs_path = os.path.abspath(path)
        with open(abs_path, "wb") as f:
            # path contains model file name.
            pickle.dump(self, f)

        model_meta = {
            # ChildModel,MLBaseModel,BaseModel,Trainable,object
            "ancestor_classname_set": [clazz.__name__ for clazz in self.__class__.mro()],
            # test_ml_base
            "modulename": self.__module__
        }
        modelname = os.path.basename(abs_path)
        internal_filename_map = {"model_meta": "%s_%s" % (modelname, "model_meta")}
        abs_root_path = os.path.dirname(abs_path)
        with open(os.path.join(abs_root_path, internal_filename_map["model_meta"]), "w") as f:
            json.dump(model_meta, f, ensure_ascii=False)


class _MockPaddleNetwork(paddle.nn.Layer):
    """
    A Mock paddle.nn.Layer used for testing.

    Args:
        in_chunk_dim(int): The length of past target time series chunk for a single sample.
        out_chunk_dim(int): The length of future target time series chunk for a single sample.
        target_dim(int): The numer of targets.
        hidden_config(List[int]|None): The ith element represents the number of neurons in the ith hidden layer.
        use_bn(bool): Whether to use batch normalization.

    Attributes:
        _nn(paddle.nn.Sequential): Dynamic graph LayerList.
    """
    def __init__(
        self,
        in_chunk_dim: int = 1,
        out_chunk_dim: int = 1,
        target_dim: int = 1,
        hidden_config: List[int] = None,
        use_bn: bool = False
    ):
        super(_MockPaddleNetwork, self).__init__()
        hidden_config = [99] if (hidden_config is None or len(hidden_config) == 0) else hidden_config
        dimensions, layers = [in_chunk_dim] + hidden_config, []
        for in_dim, out_dim in zip(dimensions[:-1], dimensions[1:]):
            layers.append(paddle.nn.Linear(in_dim, out_dim))
            if use_bn:
                layers.append(paddle.nn.BatchNorm1D(target_dim))
            layers.append(paddle.nn.ReLU())
        layers.append(
            paddle.nn.Linear(dimensions[-1], out_chunk_dim)
        )
        self._nn = paddle.nn.Sequential(*layers)

    def forward(
        self,
        x: Dict[str, paddle.Tensor]
    ) -> paddle.Tensor:
        x = paddle.transpose(x["past_target"], perm=[0, 2, 1])
        return paddle.transpose(self._nn(x), perm=[0, 2, 1])


class _MockNotPaddleModel(object):
    """
    test bad case for PaddleBaseModel.load() method.

    This class implements save / load / _init_network / _init_optimizer, but not inherited from PaddleBaseModel.
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

    def _init_network(self) -> paddle.nn.Layer:
        return _MockPaddleNetwork(
            in_chunk_dim=self._in_chunk_len,
            out_chunk_dim=self._out_chunk_len,
            target_dim=self._fit_params["target_dim"],
            hidden_config=self._hidden_config
        )

    def _init_optimizer(self) -> paddle.optimizer.Optimizer:
        return paddle.optimizer.Adam(parameters=self._network.parameters())

    def save(self, path: str) -> None:
        """
        simulate the simplified-version of the PaddleBaseModel.save() logic.

        this class is only used for testing, so only the core save logic are preserved, other logics (e.g. type check,
        name conflict check, etc.) are skipped.
        """
        abs_path = os.path.abspath(path)
        abs_root_path = os.path.dirname(abs_path)
        network_statedict = self._network.state_dict()
        self._network = None
        self._optimizer = None
        self._callback_container = None

        # save model
        with open(abs_path, "wb") as f:
            pickle.dump(self, f)
        model_meta = {
            # ChildModel,PaddleBaseModelImpl,PaddleBaseModel,BaseModel,Trainable,ABC,object
            "ancestor_classname_set": [clazz.__name__ for clazz in self.__class__.mro()],
            # paddlets.models.dl.paddlepaddle.xxx
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
            obj=network_statedict,
            path=os.path.join(abs_root_path, internal_filename_map["network_statedict"])
        )


class TestModelLoader(unittest.TestCase):
    def setUp(self) -> None:
        self.default_modelname = "model"
        super().setUp()

    def test_load(self):
        """tst paddlets.models.load()"""
        ##########################################################################
        # case 1 (good case) The loaded model is inherited from PaddleBaseModel. #
        ##########################################################################
        in_chunk_len = 10
        out_chunk_len = 5
        model = self._build_rnn_model(in_chunk_len=in_chunk_len, out_chunk_len=out_chunk_len)

        paddlets_ds = self._build_mock_ts_dataset(random_data=True)
        model.fit(train_tsdataset=paddlets_ds)
        model_network = model._network

        predicted_paddlets_ds = model.predict(paddlets_ds)
        self.assertEqual(
            (out_chunk_len, len(paddlets_ds.get_target().data.columns)),
            predicted_paddlets_ds.get_target().data.shape
        )

        path = os.path.join(os.getcwd(), str(random.randint(1, 10000000)))
        os.mkdir(path)

        abs_model_path = os.path.join(path, self.default_modelname)
        model.save(abs_model_path)

        # load rnn model
        loaded_model = paddlets.models.load(abs_model_path)

        # model type expected
        self.assertTrue(isinstance(loaded_model, RNNBlockRegressor))
        assert isinstance(loaded_model, RNNBlockRegressor)
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

        #################################################################################
        # case 2 (bad case)                                                             #
        # 1) Model exists under the given path.                                         #
        # 1) (bad) The model is neither inherited from MLBaseModel not PaddleBaseModel. #
        #################################################################################
        # Explicitly init a mock model that is Neither inherited from MLBaseModel not PaddleBaseModel.
        model = _MockNotMLModel()

        path = os.path.join(os.getcwd(), str(random.randint(1, 10000000)))
        os.mkdir(path)

        abs_model_path = os.path.join(path, self.default_modelname)
        model.save(abs_model_path)

        succeed = True
        try:
            loaded_model = paddlets.models.load(abs_model_path)
        except Exception as e:
            succeed = False
        self.assertFalse(succeed)
        shutil.rmtree(path)

        ######################################
        # case 3 (bad case) Path NOT exists. #
        ######################################
        # no such path
        path = os.path.join(os.getcwd(), str(random.randint(1, 10000000)))

        succeed = True
        try:
            loaded_model = paddlets.models.load(path)
        except Exception as e:
            succeed = False

        self.assertFalse(succeed)

        ##########################################################
        # case 4 (bad case) path is a directory, but NOT a file. #
        ##########################################################
        # path is dir, not a file.
        path = os.path.join(os.getcwd(), str(random.randint(1, 10000000)))
        os.mkdir(path)

        succeed = True
        try:
            loaded_model = paddlets.models.load(path)
        except Exception as e:
            succeed = False

        self.assertFalse(succeed)
        shutil.rmtree(path)

    def _build_rnn_model(
            self,
            in_chunk_len: int = 10,
            out_chunk_len: int = 5,
            skip_chunk_len: int = 16,
            rnn_type_or_module: str = "SimpleRNN",
            fcn_out_config: List = None,
            eval_metrics: List[str] = None
    ) -> RNNBlockRegressor:
        """
        Internal method, used for building an RNN model. The model is inherited from PaddleBaseModel.

        Args:
            in_chunk_len(int, optional): RNNBlockRegressor model required param.
            out_chunk_len(int, optional): RNNBlockRegressor model required param.
            skip_chunk_len(int, optional): RNNBlockRegressor model required param.
            rnn_type_or_module(str, optional): RNNBlockRegressor model required param.
            fcn_out_config(List[int], optional): RNNBlockRegressor model required param.
            eval_metrics(List[str], optional): RNNBlockRegressor model required param.

        Returns:
            RNNBlockRegressor: the built rnn model instance.
        """
        return RNNBlockRegressor(
            in_chunk_len=in_chunk_len,
            out_chunk_len=out_chunk_len,
            skip_chunk_len=skip_chunk_len,
            rnn_type_or_module=rnn_type_or_module,
            fcn_out_config=[32] if fcn_out_config is None else fcn_out_config,
            eval_metrics=["mse", "mae"] if eval_metrics is None else eval_metrics
        )

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
        build paddlets TSDataset instance.

        Note that random_data must set to True if the returned TSDataset is used for fitting models, otherwise paddle
        will raise the following exception:
        RuntimeError: (NotFound) There are no kernels which are registered in the rnn operator.
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
