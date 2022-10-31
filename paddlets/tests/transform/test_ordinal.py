# !/usr/bin/env python3
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np

import os
import sys
import time
import unittest
from unittest import TestCase

from paddlets.transform.sklearn_transforms import Ordinal
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
        params = {'cols':['Test', 'Sex'], "drop": True}
        ob = Ordinal(**params)
        result = ob.fit(ts)
        results = ob.fit([ts, ts])
        
        ts1 = ts.copy()
        ts2 = ts.copy()
        ts.static_cov = {'a': 'a1', 'b': 'b1'}
        ts1.static_cov = {'a': 'a1', 'b': 'b2'}
        ts2.static_cov = {'a': 'a2', 'b': 'b3'}
        params = {'cols':['Test', 'Sex', 'a', 'b'], "drop": True}
        ob = Ordinal(**params)
        results = ob.fit([ts, ts1, ts2])

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
        known = result.get_known_cov().data[['Type', 'Sex']]
        observed = result.get_observed_cov().data
        self.assertEqual(np.array(known_true_df).tolist(), np.array(known).tolist())
        self.assertEqual(np.array(observed_true_df).tolist(), np.array(observed).tolist())

        ob.fit([ts, ts])
        results = ob.transform([ts, ts], False)
        self.assertEqual(len(results), 2)
        for result in results:
            known = result.get_known_cov().data[['Type', 'Sex']]
            observed = result.get_observed_cov().data
            self.assertEqual(np.array(known_true_df).tolist(), np.array(known).tolist())
            self.assertEqual(np.array(observed_true_df).tolist(), np.array(observed).tolist())

        ts1 = ts.copy()
        ts2 = ts.copy()
        ts.static_cov = {'a': 'a1', 'b': 'b1'}
        ts1.static_cov = {'a': 'a1', 'b': 'b2'}
        ts2.static_cov = {'a': 'a2', 'b': 'b3'}
        params = {'cols':['Test', 'Sex', 'a', 'b'], "drop": True}
        ob = Ordinal(**params)
        ob.fit([ts, ts1, ts2])
        results = ob.transform([ts, ts1, ts2], False)
        self.assertEqual(len(results), 3)
        for result in results:
            known = result.get_known_cov().data[['Type', 'Sex']]
            observed = result.get_observed_cov().data
            self.assertEqual(np.array(known_true_df).tolist(), np.array(known).tolist())
            self.assertEqual(np.array(observed_true_df).tolist(), np.array(observed).tolist())
        self.assertEqual(results[0].static_cov, {'a': 0.0, 'b': 0.0})
        self.assertEqual(results[1].static_cov, {'a': 0.0, 'b': 1.0})
        self.assertEqual(results[2].static_cov, {'a': 1.0, 'b': 2.0})

        results = ob.transform([ts, ts1], False)
        self.assertEqual(len(results), 2)
        for result in results:
            known = result.get_known_cov().data[['Type', 'Sex']]
            observed = result.get_observed_cov().data
            self.assertEqual(np.array(known_true_df).tolist(), np.array(known).tolist())
            self.assertEqual(np.array(observed_true_df).tolist(), np.array(observed).tolist())
        self.assertEqual(results[0].static_cov, {'a': 0.0, 'b': 0.0})
        self.assertEqual(results[1].static_cov, {'a': 0.0, 'b': 1.0})

        ts1.static_cov = {'a': 'a1'}
        params = {'cols':['Test', 'Sex', 'a', 'b'], "drop": True}
        ob = Ordinal(**params)
        with self.assertRaises(ValueError):
            ob.fit([ts, ts1, ts2])

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
        params = {'cols':['Test', 'Sex'], "drop": True}
        ob = Ordinal(**params)
        ob.fit(ts)
        result = ob.transform(ts, False)
        known = result.get_known_cov().data[['Type', 'Sex']]
        observed = result.get_observed_cov().data
        self.assertEqual(np.array(known_true_df).tolist(), np.array(known).tolist())
        self.assertEqual(np.array(observed_true_df).tolist(), np.array(observed).tolist())

        results = ob.fit_transform([ts, ts], False)
        self.assertEqual(len(results), 2)
        for result in results:
            known = result.get_known_cov().data[['Type', 'Sex']]
            observed = result.get_observed_cov().data
            self.assertEqual(np.array(known_true_df).tolist(), np.array(known).tolist())
            self.assertEqual(np.array(observed_true_df).tolist(), np.array(observed).tolist())

        ts1 = ts.copy()
        ts2 = ts.copy()
        ts.static_cov = {'a': 'a1', 'b': 'b1'}
        ts1.static_cov = {'a': 'a1', 'b': 'b2'}
        ts2.static_cov = {'a': 'a2', 'b': 'b3'}
        params = {'cols':['Test', 'Sex', 'a', 'b'], "drop": True}
        ob = Ordinal(**params)
        results = ob.fit_transform([ts, ts1, ts2], False)
        self.assertEqual(len(results), 3)
        for result in results:
            known = result.get_known_cov().data[['Type', 'Sex']]
            observed = result.get_observed_cov().data
            self.assertEqual(np.array(known_true_df).tolist(), np.array(known).tolist())
            self.assertEqual(np.array(observed_true_df).tolist(), np.array(observed).tolist())
        self.assertEqual(results[0].static_cov, {'a': 0.0, 'b': 0.0})
        self.assertEqual(results[1].static_cov, {'a': 0.0, 'b': 1.0})
        self.assertEqual(results[2].static_cov, {'a': 1.0, 'b': 2.0})

        ts1.static_cov = {'a': 'a1'}
        params = {'cols':['Test', 'Sex', 'a', 'b'], "drop": True}
        ob = Ordinal(**params)
        with self.assertRaises(ValueError):
            ob.fit_transform([ts, ts1, ts2])
   
if __name__ == "__main__":
    unittest.main()                                    
