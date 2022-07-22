# !/usr/bin/env python3
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np

import os
import sys
import time
import unittest
from unittest import TestCase

from paddlets.transform.ordinal import Ordinal
from paddlets.datasets.tsdataset import TimeSeries, TSDataset

class TestOrdinal(TestCase):
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
        params = params = {'cols':['Test', 'Sex'], "drop": True}
        ob = Ordinal(**params)
        result = ob.fit(ts)


    def test_transform(self):
        """
        unittest function
        """

        ts_df = pd.DataFrame(np.array([['male', 10, 'e'], ['female', 20, 'e'], ['male', 10, 'd'], \
                ['female', 20, 'd'], ['female', 15, 'f']]), columns = ['Sex', 'Type', 'Test'])

        known_true_df = pd.DataFrame([['10', 1.0], ['20', 0.0], ['10', 1.0], ['20', 0.0], ['15', 0.0]], \
                                     columns=['Type', 'Sex_encoder'])
     

        observed_true_df = pd.DataFrame([1.0, 1.0, 0.0, 0.0, 2.0], columns=['Test_encoder'])

        ts = TSDataset.load_from_dataframe(df=ts_df, known_cov_cols=['Sex', 'Type'], observed_cov_cols = ['Test'])
        params = {'cols':['Test', 'Sex'], "drop": True}
        ob = Ordinal(**params)
        ob.fit(ts)
        result = ob.transform(ts, False)
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

        known_true_df = pd.DataFrame([['10', 1.0], ['20', 0.0], ['10', 1.0], ['20', 0.0], ['15', 0.0]], \
                                     columns=['Type', 'Sex_encoder'])
     

        observed_true_df = pd.DataFrame([1.0, 1.0, 0.0, 0.0, 2.0], columns=['Test_encoder'])

        ts = TSDataset.load_from_dataframe(df=ts_df, known_cov_cols=['Sex', 'Type'], observed_cov_cols = ['Test'])
        params = params = {'cols':['Test', 'Sex'], "drop": True}
        ob = Ordinal(**params)
        ob.fit(ts)
        result = ob.transform(ts, False)
        known = result.get_known_cov().data
        observed = result.get_observed_cov().data
        self.assertEqual(np.array(known_true_df).tolist(), np.array(known).tolist())
        self.assertEqual(np.array(observed_true_df).tolist(), np.array(observed).tolist())
   
if __name__ == "__main__":
    unittest.main()                                    
