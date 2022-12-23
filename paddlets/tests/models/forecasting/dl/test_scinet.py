# !/usr/bin/env python3
# -*- coding:utf-8 -*-

from unittest import TestCase
import unittest

import pandas as pd
import numpy as np

from paddlets.models.forecasting import SCINetModel
from paddlets.datasets import TimeSeries, TSDataset


class TestSCINetModel(TestCase):
    def setUp(self):
        """
        unittest function
        """
        np.random.seed(2022)
        target1 = pd.Series(
            np.random.randn(2000).astype(np.float32),
            index=pd.date_range("2022-01-01", periods=2000, freq="15T"),
            name="a")
        target2 = pd.DataFrame(
            np.random.randn(2000, 2).astype(np.float32),
            index=pd.date_range("2022-01-01", periods=2000, freq="15T"),
            columns=["a1", "a2"])
        observed_cov = pd.DataFrame(
            np.random.randn(2000, 2).astype(np.float32),
            index=pd.date_range("2022-01-01", periods=2000, freq="15T"),
            columns=["b", "c"])
        categorical_observed_cov = observed_cov.astype(np.int64)
        known_cov = pd.DataFrame(
            np.random.randn(2500, 2).astype(np.float32),
            index=pd.date_range("2022-01-01", periods=2500, freq="15T"),
            columns=["b1", "c1"])
        static_cov = {"f": 1.0, "g": 2.0}

        # index type = DatetimeIndex
        self.tsdataset1 = TSDataset(
            TimeSeries.load_from_dataframe(target1),
            TimeSeries.load_from_dataframe(observed_cov),
            TimeSeries.load_from_dataframe(known_cov),
            static_cov
        )
        self.tsdataset2 = TSDataset(
            TimeSeries.load_from_dataframe(target2),
            TimeSeries.load_from_dataframe(observed_cov),
            TimeSeries.load_from_dataframe(known_cov),
            static_cov
        )
        # bad tsdataset, contains categorical data.
        self.categorical_tsdataset3 = TSDataset(
            TimeSeries.load_from_dataframe(target2),
            TimeSeries.load_from_dataframe(categorical_observed_cov),
            TimeSeries.load_from_dataframe(known_cov),
            static_cov
        )

        # index type = RangeIndex
        index = pd.RangeIndex(0, 2000, 2)
        index2 = pd.RangeIndex(0, 2500, 2)
        target2 = target2.reset_index(drop=True).reindex(index)
        observed_cov = observed_cov.reset_index(drop=True).reindex(index)
        known_cov = known_cov.reset_index(drop=True).reindex(index2)
        self.tsdataset3 = TSDataset(
            TimeSeries.load_from_dataframe(target2, freq=index.step),
            TimeSeries.load_from_dataframe(observed_cov, freq=index.step),
            TimeSeries.load_from_dataframe(known_cov, freq=index2.step),
            static_cov)
        super().setUp()

    def test_init(self):
        """
        unittest function
        """
        #################
        # (good) case 1 #
        #################
        _ = SCINetModel(
            in_chunk_len=48,
            out_chunk_len=24
        )

        #####################
        # (bad) case 2      #
        # invalid num_stack #
        #####################
        with self.assertRaises(ValueError):
            _ = SCINetModel(
                in_chunk_len=48,
                out_chunk_len=24,
                # must > 0
                num_stack=0
            )

        with self.assertRaises(ValueError):
            _ = SCINetModel(
                in_chunk_len=48,
                out_chunk_len=24,
                # must <= 2
                num_stack=3
            )

        ######################################
        # (bad) case 3                       #
        # invalid in_chunk_len and num_level #
        ######################################
        # self.in_chunk_len % (np.power(2, self.num_level)) != 0 must be True
        param = {
            "in_chunk_len": 47,
            "out_chunk_len": 24,
            # 47 % 2**3 = 7 != 0, so this is a bad case.
            "num_level": 3
        }
        with self.assertRaises(ValueError):
            _ = SCINetModel(**param)

        param = {
            "in_chunk_len": 48,
            "out_chunk_len": 24,
            # 48 % 2**5= 16 != 0, so this is a bad case too.
            "num_level": 5
        }
        with self.assertRaises(ValueError):
            _ = SCINetModel(**param)

    def test_fit(self):
        """
        unittest function
        """
        ##########################
        # (good) case 1          #
        # fit without valid data #
        ##########################
        model = SCINetModel(
            in_chunk_len=48,
            out_chunk_len=24,
            batch_size=8,
            max_epochs=1,
            patience=1
        )
        model.fit(train_tsdataset=self.tsdataset1)

        #######################
        # (good) case 2       #
        # fit with valid data #
        #######################
        model = SCINetModel(
            in_chunk_len=48,
            out_chunk_len=24,
            batch_size=8,
            max_epochs=1,
            patience=1
        )
        model.fit(train_tsdataset=self.tsdataset1, valid_tsdataset=self.tsdataset1)

        ############################################
        # (good) case 3                            #
        # fit with different parameter combination #
        ############################################
        param = {
            "in_chunk_len": 48,
            "out_chunk_len": 24,
            "batch_size": 8,
            "max_epochs": 1,
            "hidden_size": 4,
            "num_stack": 1,
            "num_level": 3,
            "num_decoder_layer": 1,
            "kernel_size": 3,
            "dropout_rate": 0.5
        }
        model = SCINetModel(**param)
        model.fit(train_tsdataset=self.tsdataset1, valid_tsdataset=self.tsdataset1)

        param = {
            "in_chunk_len": 48,
            "out_chunk_len": 24,
            "batch_size": 8,
            "max_epochs": 1,
            "hidden_size": 4,
            "num_stack": 2,
            "num_level": 3,
            "num_decoder_layer": 1,
            "kernel_size": 3,
            "dropout_rate": 0.5
        }
        model = SCINetModel(**param)
        model.fit(train_tsdataset=self.tsdataset1, valid_tsdataset=self.tsdataset1)

        param = {
            "in_chunk_len": 736,
            "out_chunk_len": 720,
            "batch_size": 256,
            "max_epochs": 1,
            "hidden_size": 4,
            "num_stack": 1,
            "num_level": 5,
            "num_decoder_layer": 1,
            "kernel_size": 5,
            "dropout_rate": 0.5
        }
        model = SCINetModel(**param)
        model.fit(train_tsdataset=self.tsdataset1, valid_tsdataset=self.tsdataset1)

        ###############################################################
        # (bad) case 4                                                #
        # train tsdataset unexpectedly contains categorical variates. #
        ###############################################################
        param = {
            "in_chunk_len": 48,
            "out_chunk_len": 24
        }
        model = SCINetModel(**param)
        with self.assertRaises(ValueError):
            model.fit(train_tsdataset=self.categorical_tsdataset3)

    def test_predict(self):
        """
        unittest function
        """
        ########################
        # (good) case1         #
        # uni-variates predict #
        ########################
        param = {
            "in_chunk_len": 48,
            "out_chunk_len": 24,
            "batch_size": 8,
            "max_epochs": 1
        }
        model = SCINetModel(**param)
        model.fit(train_tsdataset=self.tsdataset1, valid_tsdataset=self.tsdataset1)
        res = model.predict(tsdataset=self.tsdataset1)

        self.assertIsInstance(res, TSDataset)
        self.assertTrue(self.tsdataset1.target.data.shape[1] == 1)
        self.assertEqual(res.get_target().data.shape, (param["out_chunk_len"], self.tsdataset1.target.data.shape[1]))

        ##########################
        # (good) case2           #
        # multi-variates predict #
        ##########################
        model.fit(train_tsdataset=self.tsdataset2, valid_tsdataset=self.tsdataset2)
        res = model.predict(tsdataset=self.tsdataset2)
        self.assertIsInstance(res, TSDataset)
        self.assertTrue(self.tsdataset2.target.data.shape[1] > 1)
        self.assertEqual(res.get_target().data.shape, (param["out_chunk_len"], self.tsdataset2.target.data.shape[1]))

    def test_recursive_predict(self):
        """
        unittest function
        """
        param = {
            "in_chunk_len": 48,
            "out_chunk_len": 24,
            "batch_size": 8,
            "eval_metrics": ["mse", "mae"],
            "max_epochs": 1
        }
        pred_len = 4 * param["out_chunk_len"]
        ########################
        # (good) case1         #
        # uni-variates predict #
        ########################
        model = SCINetModel(**param)
        model.fit(train_tsdataset=self.tsdataset1, valid_tsdataset=self.tsdataset1)

        res = model.recursive_predict(tsdataset=self.tsdataset1, predict_length=pred_len)
        self.assertIsInstance(res, TSDataset)
        self.assertTrue(self.tsdataset1.target.data.shape[1] == 1)
        self.assertEqual(res.get_target().data.shape, (pred_len, self.tsdataset1.target.data.shape[1]))

        ##########################
        # (good) case2           #
        # multi-variates predict #
        ##########################
        model.fit(train_tsdataset=self.tsdataset2, valid_tsdataset=self.tsdataset2)
        res = model.recursive_predict(tsdataset=self.tsdataset2, predict_length=pred_len)
        self.assertIsInstance(res, TSDataset)
        self.assertTrue(self.tsdataset2.target.data.shape[1] == 2)
        self.assertEqual(res.get_target().data.shape, (pred_len, self.tsdataset2.target.data.shape[1]))

        ################################
        # (good) case3                 #
        # multi-variates predict       #
        # time index type = RangeIndex #
        ################################
        model.fit(train_tsdataset=self.tsdataset3, valid_tsdataset=self.tsdataset3)
        res = model.recursive_predict(tsdataset=self.tsdataset3, predict_length=pred_len)
        self.assertIsInstance(res, TSDataset)
        self.assertTrue(self.tsdataset3.target.data.shape[1] == 2)
        self.assertEqual(res.get_target().data.shape, (pred_len, self.tsdataset3.target.data.shape[1]))


if __name__ == "__main__":
    unittest.main()
