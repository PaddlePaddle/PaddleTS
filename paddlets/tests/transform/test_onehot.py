# !/usr/bin/env python3
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np

import os
import sys
import time
import unittest
from unittest import TestCase

from paddlets.transform.sklearn_transforms import OneHot
from paddlets.datasets.tsdataset import TimeSeries, TSDataset

class TestOneHot(TestCase):
    def setUp(self):
        """
        unittest function
        """
        super().setUp()

    def test_fit(self):
        """
        unittest function
        """
        ts_df = pd.DataFrame(np.array([['male', 10, 'e'], ['female', 20, 'e'], ['male', 10, 'd'], \
                ['female', 20, 'd'], ['female', 15, 'f']]), columns = ['Sex', 'Type', 'Test'])
        
        ts = TSDataset.load_from_dataframe(df=ts_df, known_cov_cols=['Sex', 'Type'], observed_cov_cols = ['Test'])
        ob = OneHot(['Test', 'Sex'], handle_unknown='ignore', drop=True)
        result = ob.fit(ts)
        results = ob.fit([ts, ts])

    def test_transform(self):
        """
        unittest function
        """

        ts_df = pd.DataFrame(np.array([['male', 10, 'e'], ['female', 20, 'e'], ['male', 10, 'd'], \
                ['female', 20, 'd'], ['female', 15, 'f']]), columns = ['Sex', 'Type', 'Test'])

        known_true_df = pd.DataFrame([['10', 0.0, 1.0], ['20', 1.0, 0.0], \
                                ['10', 0.0, 1.0], ['20', 1.0, 0.0],\
                                ['15', 1.0, 0.0]], columns=['Type', 'Sex_0', 'Sex_1'])
     

        observed_true_df = pd.DataFrame([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], \
                                [1.0, 0.0, 0.0], [1.0, 0.0, 0.0],\
                                [0.0, 0.0, 1.0]], columns=['Test_0', 'Test_1', 'Test_2'])

        ts = TSDataset.load_from_dataframe(df=ts_df, known_cov_cols=['Sex', 'Type'], observed_cov_cols = ['Test'])
        params = {'cols':['Test', 'Sex'], 'handle_unknown': 'ignore',  "drop": True}
        ob = eval("OneHot")(**params)
        ob.fit(ts)
        result = ob.transform(ts, False)
        known = result.get_known_cov().data
        observed = result.get_observed_cov().data
        self.assertEqual(np.array(known_true_df).tolist(), np.array(known).tolist())
        self.assertEqual(np.array(observed_true_df).tolist(), np.array(observed).tolist())

        ob.fit([ts, ts])
        results = ob.transform([ts, ts], False)
        self.assertEqual(len(results), 2)
        for result in results:
            known = result.get_known_cov().data
            observed = result.get_observed_cov().data
            self.assertEqual(np.array(known_true_df).tolist(), np.array(known).tolist())
            self.assertEqual(np.array(observed_true_df).tolist(), np.array(observed).tolist())

    def test_fit_transform(self):
        """
        unittest function
        """

        ts_df = pd.DataFrame(np.array([['male', 10, 'e'], ['female', 20, 'e'], ['male', 10, 'd'], \
                ['female', 20, 'd'], ['female', 15, 'f']]), columns = ['Sex', 'Type', 'Test'])
        known_true_df = pd.DataFrame([['10', 0.0, 1.0], ['20', 1.0, 0.0], \
                                ['10', 0.0, 1.0], ['20', 1.0, 0.0],\
                                ['15', 1.0, 0.0]], columns=['Type', 'Sex_0', 'Sex_1'])
     

        observed_true_df = pd.DataFrame([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], \
                                [1.0, 0.0, 0.0], [1.0, 0.0, 0.0],\
                                [0.0, 0.0, 1.0]], columns=['Test_0', 'Test_1', 'Test_2'])

        ts = TSDataset.load_from_dataframe(df=ts_df, known_cov_cols=['Sex', 'Type'], observed_cov_cols = ['Test'])
        params = {'cols':['Test', 'Sex'], 'handle_unknown': 'ignore',  "drop": True}
        ob = eval("OneHot")(**params)
        result = ob.fit_transform(ts, False)
        known = result.get_known_cov().data
        observed = result.get_observed_cov().data
        self.assertEqual(np.array(known_true_df).tolist(), np.array(known).tolist())
        self.assertEqual(np.array(observed_true_df).tolist(), np.array(observed).tolist())

        results = ob.fit_transform([ts, ts], False)
        self.assertEqual(len(results), 2)
        for result in results:
            known = result.get_known_cov().data
            observed = result.get_observed_cov().data
            self.assertEqual(np.array(known_true_df).tolist(), np.array(known).tolist())
            self.assertEqual(np.array(observed_true_df).tolist(), np.array(observed).tolist())
   
if __name__ == "__main__":
    unittest.main()
