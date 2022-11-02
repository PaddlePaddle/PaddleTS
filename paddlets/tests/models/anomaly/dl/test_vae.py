# !/usr/bin/env python3
# -*- coding:utf-8 -*-

from unittest import TestCase
from unittest import mock
import unittest
import random

import pandas as pd
import numpy as np
import paddle

from paddlets.models.common.callbacks import Callback
from paddlets.models.anomaly import VAE
from paddlets.datasets import TimeSeries
from paddlets.datasets import TSDataset


class TestVAE(TestCase):
    def setUp(self):
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

        # index is DatetimeIndex
        self.tsdataset1 = TSDataset.load_from_dataframe(pd.concat([label, feature], axis=1), 
                target_cols='label', feature_cols='a')

        # There is no target in tsdataset
        self.tsdataset2 = TSDataset.load_from_dataframe(pd.concat([label, feature], axis=1), 
                feature_cols=['a', 'b'])     

        # index is RangeIndex
        index = pd.RangeIndex(0, 200, 1)
        label = label.reset_index(drop=True).reindex(index)
        feature = feature.reset_index(drop=True).reindex(index)
        self.tsdataset3 = TSDataset.load_from_dataframe(pd.concat([label, feature], axis=1), 
                target_cols='label', feature_cols='a')
        # There is no target in tsdataset
        self.tsdataset4 = TSDataset.load_from_dataframe(pd.concat([label, feature], axis=1), 
                feature_cols=['a', 'b'])      
        super().setUp()

    def test_init(self):
        """unittest function for init
        """
        # case1 (All parameters are valid)
        param1 = {
            "batch_size": 1,
            "max_epochs": 1,
            "verbose": 1,
            "patience": 1,
            "base_en": 'LSTM', 
            "base_de": 'LSTM', 
            "hidden_config": [128, 64, 32],
            "use_bn": True, 
            "use_drop": True, 
            "dropout_rate": 0.3, 
        }
        vae = VAE(
            in_chunk_len=1,
            **param1
        )

        # case2 (batch_size is illegal)
        param2 = {
            "batch_size": 0,
            "max_epochs": 1,
            "verbose": 1,
            "patience": 1,
            "hidden_config": [32],
        }
        with self.assertRaises(ValueError):
            vae = VAE(
                in_chunk_len=1,
                **param2
            )

        # case3 (max_epochs is illegal)
        param3 = {
            "batch_size": 1,
            "max_epochs": 0,
            "verbose": 1,
            "patience": 1,
            "hidden_config": [32],
        }
        with self.assertRaises(ValueError):
            vae = VAE(
                in_chunk_len=1,
                **param3
            )

        # case4 (verbose is illegal)
        param4 = {
            "batch_size": 1,
            "max_epochs": 1,
            "verbose": 0,
            "patience": 1,
            "hidden_config": [32],
        }
        with self.assertRaises(ValueError):
            vae = VAE(
                in_chunk_len=1,
                **param4
            )

        # case5 (patience is illegal)
        param5 = {
            "batch_size": 1,
            "max_epochs": 1,
            "verbose": 1,
            "patience": -1,
            "hidden_config": [32],
        }
        with self.assertRaises(ValueError):
            vae = VAE(
                in_chunk_len=1,
                **param5
            )

        # case6 base_en is illegal
        param6 = {
            "batch_size": 128,
            "max_epochs": 1,
            "verbose": 1,
            "patience": 1,
            "hidden_config": [32],
            "base_en": "bruce",
        }
        with self.assertRaises(ValueError):
            vae = VAE(
                in_chunk_len=1,
                **param6
            )
            vae.fit(self.tsdataset1)

        # case7 base_de is illegal
        param7 = {
            "batch_size": 128,
            "max_epochs": 1,
            "verbose": 1,
            "patience": 1,
            "hidden_config": [32],
            "base_de": "superman",
        }
        with self.assertRaises(ValueError):
            vae = VAE(
                in_chunk_len=1,
                **param7
            )
            vae.fit(self.tsdataset1)

        # case8 (use_drop = True)
        param8 = {
            "batch_size": 128,
            "max_epochs": 1,
            "verbose": 1,
            "patience": 1,
            "hidden_config": [32],
            'use_drop': True, 
        }
        vae = VAE(
            in_chunk_len=1,
            **param8
                )
        vae.fit(self.tsdataset1)

        # case9 (use_bn = True)
        param9 = {
            "batch_size": 128,
            "max_epochs": 1,
            "verbose": 1,
            "patience": 1,
            "hidden_config": [32],
            "use_bn": True,
            "use_drop": False, 
            "dropout_rate": 0.3, 
        }
        vae = VAE(
            in_chunk_len=1,
            **param9
        )
        vae.fit(self.tsdataset1)
        self.assertIsInstance(vae._network.encoder._nn._nn[1], paddle.nn.BatchNorm1D)
        
        #case10 (CNN_CNN)
        param10 = {
            "batch_size": 128,
            "max_epochs": 1,
            "verbose": 1,
            "patience": 1,
            "hidden_config": [32],
            "base_en": 'CNN',
            "base_de": 'CNN',
        }
        vae = VAE(
            in_chunk_len=1,
            **param10
        )
        vae.fit(self.tsdataset1)
        
        #case11 (MLP_MLP)
        param11 = {
            "batch_size": 128,
            "max_epochs": 1,
            "verbose": 1,
            "patience": 1,
            "hidden_config": [32],
            "base_en": 'MLP',
            "base_de": 'MLP',
        }
        vae = VAE(
            in_chunk_len=1,
            **param11
        )
        vae.fit(self.tsdataset1)
        
        #case12 (LSTM_LSTM)
        param12 = {
            "batch_size": 128,
            "max_epochs": 1,
            "verbose": 1,
            "patience": 1,
            "hidden_config": [32],
            "base_en": 'LSTM',
            "base_de": 'LSTM',
        }
        vae = VAE(
            in_chunk_len=1,
            **param12
        )
        vae.fit(self.tsdataset1)
        
    def test_init_dataloader(self):
        """unittest function for init dataloader
        """
        vae = VAE(
            in_chunk_len=16,
            max_epochs=1
        )
        #case 1 (valid dataset is empty)
        _, valid_dataloaders = vae._init_fit_dataloaders(self.tsdataset1)
        self.assertEqual(len(valid_dataloaders), 0)

        #case 2 (valid dataset is not empty)
        _, valid_dataloaders = vae._init_fit_dataloaders(self.tsdataset1, self.tsdataset1)
        self.assertNotEqual(len(valid_dataloaders), 0)

        #case 3 (The training set contains illegal data types, such as string type)
        tsdataset = self.tsdataset1.copy()
        tsdataset.astype({"a": "O"})
        with self.assertRaises(TypeError):
            vae.fit(tsdataset, tsdataset)

    def test_init_metrics(self):
        """unittest function for init metrics
        """
        #case 1 (The User passed metrics)
        vae = VAE(
            in_chunk_len=16,
            max_epochs=1,
            eval_metrics=["mae"]
        )
        _, metrics_names, _ = vae._init_metrics(["val"])
        self.assertEqual(metrics_names[-1], "val_mae")

        #case 2 (The User not passed metrics)
        ae = VAE(
            in_chunk_len=16,
            max_epochs=1,
        )
        _, metrics_names, _ = ae._init_metrics(["val"])
        self.assertEqual(metrics_names[-1], "val_mse") 

    def test_init_callbacks(self):
        """unittest function for init callbacks
        """
        # case1 (patience = 0)
        vae = VAE(
            in_chunk_len=16,
            patience=0
        )
        vae._metrics, vae._metrics_names, _ = vae._init_metrics(["val"])
        with self.assertLogs("paddlets", level="WARNING") as captured:
            vae._init_callbacks()
            self.assertEqual(len(captured.records), 1) # check that there is only one log message
            self.assertEqual(
                captured.records[0].getMessage(), 
                "No early stopping will be performed, last training weights will be used."
            )

        # case2 (patience > 0)
        vae = VAE(
            in_chunk_len=16,
            patience=1
        )
        vae._metrics, vae._metrics_names, _ = vae._init_metrics(["val"])
        _, callback_container = vae._init_callbacks()

        # case3 (The User passed callbacks)
        self.assertEqual(len(callback_container._callbacks), 2)

        # case4 (The User not passed callbacks)
        callback = Callback()
        vae = VAE(
            in_chunk_len=16,
            callbacks=[callback]
        )
        vae._metrics, vae._metrics_names, _ = vae._init_metrics(["val"])
        _, callback_container = vae._init_callbacks()
        self.assertEqual(len(callback_container._callbacks), 3)

    def test_fit(self):
        """unittest function for fit
        """
        #case 1 (The user only inputs the training set, reaching the end of the maximum epochs training)
        vae = VAE(
            in_chunk_len=1,
            optimizer_params=dict(learning_rate=1e-1),
            eval_metrics=["mse", "mae"],
            batch_size=512,
            max_epochs=10,
            patience=1
        )
        vae.fit(self.tsdataset1)

        #case 2 (The user inputs the training set and valid set, early_stopping works)
        vae = VAE(
            in_chunk_len=1,
            optimizer_params=dict(learning_rate=1e-1),
            eval_metrics=["mse", "mae"],
            batch_size=512,
            max_epochs=20,
            patience=1
        )
        vae.fit(self.tsdataset1, self.tsdataset1)
        self.assertEqual(vae._stop_training, True)

    def test_predict(self):
        """unittest function for predict
        """
        # case1 (index is DatetimeIndex and the feature is Univariate)
        vae = VAE(
            in_chunk_len=1,
            eval_metrics=["mse", "mae"],
            max_epochs=1,
            patience=1
        )
        vae.fit(self.tsdataset1, self.tsdataset1)
        res = vae.predict(self.tsdataset1)
        self.assertIsInstance(res, TSDataset)
        self.assertIsInstance(res.get_target().data.index, pd.DatetimeIndex)
        self.assertEqual(res.get_target().data.shape, (200, 1))
        self.assertEqual(res.get_target().data.columns[0], 'label')

        # case2 (index is DatetimeIndex and the feature is Multivariable and there is no label in rawtsdataset)
        vae = VAE(
            in_chunk_len=16,
            eval_metrics=["mse", "mae"],
            max_epochs=1,
            patience=1,
            base_en='CNN',
            base_de='CNN',
        )
        vae.fit(self.tsdataset2, self.tsdataset2)
        res = vae.predict(self.tsdataset2)
        self.assertIsInstance(res, TSDataset)
        self.assertIsInstance(res.get_target().data.index, pd.DatetimeIndex)
        self.assertEqual(res.get_target().data.shape, (185, 1))
        self.assertEqual(res.get_target().data.columns[0], 'anomaly_label')

        # case3 (index is RangeIndex and the feature is Univariate)
        vae = VAE(
            in_chunk_len=32,
            eval_metrics=["mse", "mae"],
            max_epochs=1,
            patience=1
        )
        vae.fit(self.tsdataset3, self.tsdataset3)
        res = vae.predict(self.tsdataset3)
        self.assertIsInstance(res, TSDataset)
        self.assertIsInstance(res.get_target().data.index, pd.RangeIndex)
        self.assertEqual(res.get_target().data.shape, (169, 1))
        self.assertEqual(res.get_target().data.columns[0], 'label')

        # case4 (index is RangeIndex and the feature is Multivariable and there is no label in rawtsdataset)
        vae = VAE(
            in_chunk_len=32,
            eval_metrics=["mse", "mae"],
            max_epochs=1,
            patience=1,
        )
        vae.fit(self.tsdataset4, self.tsdataset4)
        res = vae.predict(self.tsdataset4)
        self.assertIsInstance(res, TSDataset)
        self.assertIsInstance(res.get_target().data.index, pd.RangeIndex)
        self.assertEqual(res.get_target().data.shape, (169, 1))
        self.assertEqual(res.get_target().data.columns[0], 'anomaly_label')

    def test_predict_score(self):
        """unittest function for predict_score
        """
        # case1 (index is DatetimeIndex and the feature is Univariate)
        vae = VAE(
            in_chunk_len=16,
            eval_metrics=["mse", "mae"],
            max_epochs=1,
            patience=1
        )
        vae.fit(self.tsdataset1, self.tsdataset1)
        res = vae.predict_score(self.tsdataset1)
        self.assertIsInstance(res, TSDataset)
        self.assertIsInstance(res.get_target().data.index, pd.DatetimeIndex)
        self.assertEqual(res.get_target().data.shape, (185, 1))
        self.assertEqual(res.get_target().data.columns[0], 'label_score')

        # case2 (index is DatetimeIndex and the feature is Multivariable and there is no label in rawtsdataset)
        vae = VAE(
            in_chunk_len=16,
            eval_metrics=["mse", "mae"],
            max_epochs=1,
            patience=1
        )
        vae.fit(self.tsdataset2, self.tsdataset2)
        res = vae.predict_score(self.tsdataset2)
        self.assertIsInstance(res, TSDataset)
        self.assertIsInstance(res.get_target().data.index, pd.DatetimeIndex)
        self.assertEqual(res.get_target().data.shape, (185, 1))
        self.assertEqual(res.get_target().data.columns[0], 'anomaly_score')

        # case3 (index is RangeIndex and the feature is Univariate)
        vae = VAE(
            in_chunk_len=32,
            eval_metrics=["mse", "mae"],
            max_epochs=1,
            patience=1
        )
        vae.fit(self.tsdataset3, self.tsdataset3)
        res = vae.predict_score(self.tsdataset3)
        self.assertIsInstance(res, TSDataset)
        self.assertIsInstance(res.get_target().data.index, pd.RangeIndex)
        self.assertEqual(res.get_target().data.shape, (169, 1))
        self.assertEqual(res.get_target().data.columns[0], 'label_score')

        # case4 (index is RangeIndex and the feature is Multivariable and there is no label in rawtsdataset)
        vae = VAE(
            in_chunk_len=32,
            eval_metrics=["mse", "mae"],
            max_epochs=1,
            patience=1,
            base_en='CNN',
            base_de='CNN',
        )
        vae.fit(self.tsdataset4, self.tsdataset4)
        res = vae.predict_score(self.tsdataset4)
        self.assertIsInstance(res, TSDataset)
        self.assertIsInstance(res.get_target().data.index, pd.RangeIndex)
        self.assertEqual(res.get_target().data.shape, (169, 1))
        self.assertEqual(res.get_target().data.columns[0], 'anomaly_score')

        # case5 (index is RangeIndex and the feature is Multivariable and there is no label in rawtsdataset)
        # in_chunk_len = 199
        vae = VAE(
            in_chunk_len=199,
            eval_metrics=["mse", "mae"],
            max_epochs=1,
            patience=1,
            base_en='CNN',
            base_de='CNN',
        )
        vae.fit(self.tsdataset4, self.tsdataset4)
        res = vae.predict_score(self.tsdataset4)

        self.assertIsInstance(res, TSDataset)
        self.assertIsInstance(res.get_target().data.index, pd.RangeIndex)
        self.assertEqual(res.get_target().data.shape, (2, 1))
        self.assertEqual(res.get_target().data.columns[0], 'anomaly_score')


if __name__ == "__main__":
    unittest.main()
