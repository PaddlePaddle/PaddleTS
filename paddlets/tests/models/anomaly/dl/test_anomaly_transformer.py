# !/usr/bin/env python3
# -*- coding:utf-8 -*-

from unittest import TestCase, mock
import unittest
import random

import pandas as pd
import numpy as np
import paddle

from paddlets.models.common.callbacks import Callback
from paddlets.models.anomaly import AnomalyTransformer
from paddlets.datasets import TimeSeries, TSDataset


class TestAnomalyTransformer(TestCase):
    def setUp(self):
        """unittest function
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
        self.tsdataset1 = TSDataset.load_from_dataframe(pd.concat([label,feature],axis=1), 
                target_cols='label', feature_cols='a')
        
        # There is no target in tsdataset
        self.tsdataset2 = TSDataset.load_from_dataframe(pd.concat([label,feature],axis=1), 
                feature_cols=['a', 'b'])     
        
        # index is RangeIndex
        index = pd.RangeIndex(0, 200, 1)
        label = label.reset_index(drop=True).reindex(index)
        feature = feature.reset_index(drop=True).reindex(index)
        self.tsdataset3 = TSDataset.load_from_dataframe(pd.concat([label,feature],axis=1), 
                label_col='label', feature_cols='a')
        # There is no target in tsdataset
        self.tsdataset4 = TSDataset.load_from_dataframe(pd.concat([label,feature],axis=1), 
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
        }
        model = AnomalyTransformer(
            in_chunk_len=16,
            **param1
        )
        
        # case2 (batch_size is illegal)
        param2 = {
            "batch_size": 0,
            "max_epochs": 1,
            "verbose": 1,
            "patience": 1,
        }
        with self.assertRaises(ValueError):
            model = AnomalyTransformer(
                in_chunk_len=16,
                **param2
            )
            
        # case3 (max_epochs is illegal)
        param3 = {
            "batch_size": 1,
            "max_epochs": 0,
            "verbose": 1,
            "patience": 1,
        }
        with self.assertRaises(ValueError):
            model = AnomalyTransformer(
                in_chunk_len=16,
                **param3
            )
            
        # case4 (verbose is illegal)
        param4 = {
            "batch_size": 1,
            "max_epochs": 1,
            "verbose": 0,
            "patience": 1,
        }
        with self.assertRaises(ValueError):
            model = AnomalyTransformer(
                in_chunk_len=16,
                **param4
            )
            
        # case5 (patience is illegal)
        param5 = {
            "batch_size": 1,
            "max_epochs": 1,
            "verbose": 1,
            "patience": -1,
        }
        with self.assertRaises(ValueError):
            model = AnomalyTransformer(
                in_chunk_len=16,
                **param5
            )

    def test_init_dataloader(self):
        """unittest function for init dataloader
        """
        model = AnomalyTransformer(
            in_chunk_len=16,
            max_epochs = 1
        )
        #case 1 (valid dataset is empty)
        _, valid_dataloaders = model._init_fit_dataloaders(self.tsdataset1)
        self.assertEqual(len(valid_dataloaders), 0)
        
        #case 2 (valid dataset is not empty)
        _, valid_dataloaders = model._init_fit_dataloaders(self.tsdataset1, self.tsdataset1)
        self.assertNotEqual(len(valid_dataloaders), 0)
        
        #case 3 (The training set contains illegal data types, such as string type)
        tsdataset = self.tsdataset1.copy()
        tsdataset.astype({"a": "O"})
        with self.assertRaises(TypeError):
             model.fit(tsdataset, tsdataset)
            
    def test_init_metrics(self):
        """unittest function for init metrics
        """
        #case 1 (The User passed metrics)
        model = AnomalyTransformer(
            in_chunk_len=16,
            max_epochs = 1,
            eval_metrics = ["mae"]
        )
        _, metrics_names, _ = model._init_metrics(["val"])
        self.assertEqual(metrics_names[-1], "val_mae")
            
        #case 2 (The User not passed metrics)
        model = AnomalyTransformer(
            in_chunk_len=16,
            max_epochs = 1,
        )
        _, metrics_names, _ = model._init_metrics(["val"])
        self.assertEqual(metrics_names[-1], "val_mse") 
        
    def test_init_callbacks(self):
        """unittest function for init callbacks
        """
        # case1 (patience = 0)
        model = AnomalyTransformer(
            in_chunk_len=16,
            patience=0
        )
        model._metrics, model._metrics_names, _ = model._init_metrics(["val"])
        with self.assertLogs("paddlets", level="WARNING") as captured:
            model._init_callbacks()
            self.assertEqual(len(captured.records), 1) # check that there is only one log message
            self.assertEqual(
                captured.records[0].getMessage(), 
                "No early stopping will be performed, last training weights will be used."
            )

        # case2 (patience > 0)
        model = AnomalyTransformer(
            in_chunk_len=16,
            patience=1
        )
        model._metrics, model._metrics_names, _ = model._init_metrics(["val"])
        _, callback_container = model._init_callbacks()

        # case3 (The User passed callbacks)
        self.assertEqual(len(callback_container._callbacks), 2)

        # case4 (The User not passed callbacks)
        callback = Callback()
        model = AnomalyTransformer(
            in_chunk_len=16,
            callbacks=[callback]
        )
        model._metrics, model._metrics_names, _ = model._init_metrics(["val"])
        _, callback_container = model._init_callbacks()
        self.assertEqual(len(callback_container._callbacks), 3)
            
    def test_fit(self):
        """unittest function for fit
        """
        #case 1 (The user only inputs the training set, reaching the end of the maximum epochs training)
        model = AnomalyTransformer(
            in_chunk_len=20,
            optimizer_params=dict(learning_rate=1e-1),
            batch_size=16,
            max_epochs=10,
            patience=1
        )
        model.fit(self.tsdataset1, self.tsdataset1)
        
        #case 2 (The user inputs the training set and valid set, early_stopping works)
        model = AnomalyTransformer(
            in_chunk_len=20,
            optimizer_params=dict(learning_rate=1e-1),
            batch_size=16,
            max_epochs=20,
            patience=1
        )
        model.fit(self.tsdataset1, self.tsdataset1)
        self.assertEqual(model._stop_training, True)
        
    def test_predict(self):
        # case1 (index is DatetimeIndex and the feature is Univariate)
        model = AnomalyTransformer(
            in_chunk_len=10,
            max_epochs=1,
            patience=1,
        )
        model.fit(self.tsdataset1, self.tsdataset1)
        res = model.predict(self.tsdataset1, self.tsdataset1, res_adjust=True)
        self.assertIsInstance(res, TSDataset)
        self.assertIsInstance(res.get_target().data.index, pd.DatetimeIndex)
        self.assertEqual(res.get_target().data.shape, (200, 1))
        self.assertEqual(res.get_target().data.columns[0], 'label')
        
        # case2 (index is DatetimeIndex and the feature is Multivariable and there is no label in rawtsdataset)
        model = AnomalyTransformer(
            in_chunk_len=10,
            max_epochs=1,
            patience=1,
        )
        model.fit(self.tsdataset2, self.tsdataset2)
        res = model.predict(self.tsdataset2, self.tsdataset2, res_adjust=False)
        self.assertIsInstance(res, TSDataset)
        self.assertIsInstance(res.get_target().data.index, pd.DatetimeIndex)
        self.assertEqual(res.get_target().data.shape, (200, 1))
        self.assertEqual(res.get_target().data.columns[0], 'anomaly_label')
        
        # case3 (index is RangeIndex and the feature is Univariate)
        model = AnomalyTransformer(
            in_chunk_len=20,
            max_epochs=1,
            patience=1
        )
        model.fit(self.tsdataset3, self.tsdataset3)
        res = model.predict(self.tsdataset3, self.tsdataset3, res_adjust=True)
        self.assertIsInstance(res, TSDataset)
        self.assertIsInstance(res.get_target().data.index, pd.RangeIndex)
        self.assertEqual(res.get_target().data.shape, (200, 1))
        self.assertEqual(res.get_target().data.columns[0], 'label')
        
        # case4 (index is RangeIndex and the feature is Multivariable and there is no label in rawtsdataset)
        model = AnomalyTransformer(
            in_chunk_len=20,
            max_epochs=1,
            patience=1
        )
        model.fit(self.tsdataset4, self.tsdataset4)
        res = model.predict(self.tsdataset4, self.tsdataset4, res_adjust=False)
        self.assertIsInstance(res, TSDataset)
        self.assertIsInstance(res.get_target().data.index, pd.RangeIndex)
        self.assertEqual(res.get_target().data.shape, (200, 1))
        self.assertEqual(res.get_target().data.columns[0], 'anomaly_label')

    def test_predict_score(self):
        """unittest function for predict_score
        """
        # case1 (index is DatetimeIndex and the feature is Univariate)
        model = AnomalyTransformer(
            in_chunk_len=10,
            max_epochs=1,
            patience=1
        )
        model.fit(self.tsdataset1, self.tsdataset1)
        res = model.predict_score(self.tsdataset1)
        self.assertIsInstance(res, TSDataset)
        self.assertIsInstance(res.get_target().data.index, pd.DatetimeIndex)
        self.assertEqual(res.get_target().data.shape, (200, 1))
        self.assertEqual(res.get_target().data.columns[0], 'label_score')
        
        # case2 (index is DatetimeIndex and the feature is Multivariable and there is no label in rawtsdataset)
        model = AnomalyTransformer(
            in_chunk_len=10,
            max_epochs=1,
            patience=1
        )
        model.fit(self.tsdataset2, self.tsdataset2)
        res = model.predict_score(self.tsdataset2)
        self.assertIsInstance(res, TSDataset)
        self.assertIsInstance(res.get_target().data.index, pd.DatetimeIndex)
        self.assertEqual(res.get_target().data.shape, (200, 1))
        self.assertEqual(res.get_target().data.columns[0], 'anomaly_score')
        
        # case3 (index is RangeIndex and the feature is Univariate)
        model = AnomalyTransformer(
            in_chunk_len=20,
            max_epochs=1,
            patience=1
        )
        model.fit(self.tsdataset3, self.tsdataset3)
        res = model.predict_score(self.tsdataset3)
        self.assertIsInstance(res, TSDataset)
        self.assertIsInstance(res.get_target().data.index, pd.RangeIndex)
        self.assertEqual(res.get_target().data.shape, (200, 1))
        self.assertEqual(res.get_target().data.columns[0], 'label_score')
        
        # case4 (index is RangeIndex and the feature is Multivariable and there is no label in rawtsdataset)
        model = AnomalyTransformer(
            in_chunk_len=20,
            max_epochs=1,
            patience=1,
        )
        model.fit(self.tsdataset4, self.tsdataset4)
        res = model.predict_score(self.tsdataset4)
        self.assertIsInstance(res, TSDataset)
        self.assertIsInstance(res.get_target().data.index, pd.RangeIndex)
        self.assertEqual(res.get_target().data.shape, (200, 1))
        self.assertEqual(res.get_target().data.columns[0], 'anomaly_score')
        
        # case5 (index is RangeIndex and the feature is Multivariable and there is no label in rawtsdataset)
        # in_chunk_len = 199
        model = AnomalyTransformer(
            in_chunk_len=199,
            max_epochs=1,
            patience=1,
        )
        model.fit(self.tsdataset4, self.tsdataset4)
        res = model.predict_score(self.tsdataset4)
        self.assertIsInstance(res, TSDataset)
        self.assertIsInstance(res.get_target().data.index, pd.RangeIndex)
        self.assertEqual(res.get_target().data.shape, (199, 1))
        self.assertEqual(res.get_target().data.columns[0], 'anomaly_score')
        
if __name__ == "__main__":
    unittest.main()