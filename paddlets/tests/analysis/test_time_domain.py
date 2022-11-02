# !/usr/bin/env python3
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np

import os
import sys
import time
import unittest
from unittest import TestCase

from paddlets.datasets.tsdataset import TimeSeries, TSDataset
from paddlets.analysis import Seasonality, Acf, Correlation

class TestFrequencyDomain(TestCase):
    def setUp(self):
        """
        unittest function
        """
        super().setUp()

    def test_Seasonality(self):
        """
        unittest function
        """
        #case1, illegal data format
        s = [1, 2, 3, 4]
        flag = False
        try: 
            res = Seasonality().analyze(s)
        except:
            flag = True
        self.assertTrue(flag)
        
        #case2, input data is pd.Series
        data = np.sin(np.pi * 2 / 100 * np.arange(1000))
        s = pd.Series(data)
        res = Seasonality().analyze(s)
        self.assertEqual(np.shape(res), (2, ))
        self.assertEqual(res[0], {0: 100, })
        
        #case3, input data contains illegal characters
        s = pd.Series(range(10))
        s[0] = 's'
        flag = False
        try:
            res = Seasonality().analyze(s)
        except:
            flag = True
        self.assertTrue(flag)
        
        #case4, input data is dataframe
        df = pd.DataFrame({'target': np.sin(np.pi * 2 / 100 * np.arange(1000))})
        res = Seasonality().analyze(df)
        self.assertEqual(np.shape(res), (2, ))
        self.assertEqual(res[0], {'target': 100, })
        
        #case5, input data is tsdataset
        df = pd.DataFrame({'target': np.sin(np.pi * 2 / 100 * np.arange(1000)), 'cov': np.sin(np.pi * 2 / 10 * np.arange(1000))})
        ts = TSDataset.load_from_dataframe(df, target_cols="target", observed_cov_cols='cov')
        res = Seasonality()(ts)
        self.assertEqual(np.shape(res), (2, ))
        self.assertEqual(res[0], {'target': 100, 'cov': 10})
        
        #case6, input data is tsdataset, select feature
        df = pd.DataFrame({'target': np.sin(np.pi * 2 / 100 * np.arange(1000)), 'cov': np.sin(np.pi * 2 / 10 * np.arange(1000))})
        ts = TSDataset.load_from_dataframe(df, target_cols="target", observed_cov_cols='cov')
        res = Seasonality()(ts, columns=['target'])
        self.assertEqual(np.shape(res), (2, ))
        self.assertEqual(res[0], {'target': 100, })
        
        #case7, test_get_properties()
        res = Seasonality().get_properties().get("name")
        self.assertEqual(res, "seasonality")
        
        #case8, test plot
        df = pd.DataFrame({'target': np.sin(np.pi * 2 / 100 * np.arange(1000)), 'cov': np.sin(np.pi * 2 / 10 * np.arange(1000))})
        ts = TSDataset.load_from_dataframe(df, target_cols="target", observed_cov_cols='cov')
        sea = Seasonality()
        res = sea(ts)
        plot = sea.plot()
        plot.savefig('/tmp/seasonality.png')
        
    def test_Acf(self):
        """
        unittest function
        """
        #case1, illegal data format or len(s) < nlags
        s = [1, 2, 3, 4]
        flag = False
        try: 
            res = Acf().analyze(s)
        except:
            flag = True
        self.assertTrue(flag)
        
        #case2, input data is pd.Series
        data = np.sin(np.pi * 2 / 100 * np.arange(1000))
        s = pd.Series(data)
        res = Acf().analyze(s)
        self.assertEqual(len(res[0]), 2)
        
        #case3, input data contains illegal characters
        s = pd.Series(range(1000))
        s[0] = 's'
        flag = False
        try:
            res = Acf().analyze(s)
        except:
            flag = True
        self.assertTrue(flag)
        
        #case4, input data is dataframe
        df = pd.DataFrame({'target': np.sin(np.pi * 2 / 100 * np.arange(1000))})
        res = Acf().analyze(df)
        self.assertEqual(len(res['target']), 2)
        
        #case5, input data is tsdataset
        df = pd.DataFrame({'target': np.sin(np.pi * 2 / 100 * np.arange(1000)), 'cov': np.sin(np.pi * 2 / 10 * np.arange(1000))})
        ts = TSDataset.load_from_dataframe(df, target_cols="target", observed_cov_cols='cov')
        res = Acf()(ts)
        self.assertEqual(len(res), 2)
        self.assertEqual(len(res['target']), 2)
        
        #case6, input data is tsdataset, select feature
        df = pd.DataFrame({'target': np.sin(np.pi * 2 / 100 * np.arange(1000)), 'cov': np.sin(np.pi * 2 / 10 * np.arange(1000))})
        ts = TSDataset.load_from_dataframe(df, target_cols="target", observed_cov_cols='cov')
        res = Acf()(ts, columns=['target'])
        self.assertEqual(len(res), 1)
        self.assertEqual(len(res['target']), 2)
        
        #case7, test_get_properties()
        res = Acf().get_properties().get("name")
        self.assertEqual(res, "acf")
        
        #case8, test plot
        df = pd.DataFrame({'target': np.sin(np.pi * 2 / 100 * np.arange(1000)), 'cov': np.sin(np.pi * 2 / 10 * np.arange(1000))})
        ts = TSDataset.load_from_dataframe(df, target_cols="target", observed_cov_cols='cov')
        sea = Acf()
        res = sea(ts)
        plot = sea.plot()
        plot.savefig('/tmp/acf.png')
        
    def test_Correlation(self):
        """
        unittest function
        """
        #case1, illegal data format or len(s) < nlags
        s = [1, 2, 3, 4]
        flag = False
        try: 
            res = Correlation().analyze(s)
        except:
            flag = True
        self.assertTrue(flag)
        
        #case2, input data is pd.Series
        data = np.sin(np.pi * 2 / 100 * np.arange(1000))
        s = pd.Series(data)
        flag = False
        try:
            res = Correlation().analyze(s)
        except:
            flag = True
        self.assertTrue(flag)
        
        #case3, input data contains illegal characters
        s = pd.Series(range(1000))
        s[0] = 's'
        flag = False
        try:
            res = Correlation().analyze(s)
        except:
            flag = True
        self.assertTrue(flag)
        
        #case4, input data is dataframe and columns number = 1
        df = pd.DataFrame({'target': np.sin(np.pi * 2 / 100 * np.arange(1000))})
        flag = False
        try:
            res = Correlation().analyze(df)
        except:
            flag= True
        self.assertTrue(flag)
        
        #case5, input data is tsdataset and columns number > 1
        df = pd.DataFrame({'target': np.sin(np.pi * 2 / 100 * np.arange(1000)), 'cov': np.sin(np.pi * 2 / 10 * np.arange(1000))})
        ts = TSDataset.load_from_dataframe(df, target_cols="target", observed_cov_cols='cov')
        res = Correlation()(ts)
        self.assertEqual(len(res), 2)
        self.assertEqual(len(res['target']), 2)
        
        #case6, input data is tsdataset and columns number > 1 and lag
        df = pd.DataFrame({'target': np.sin(np.pi * 2 / 100 * np.arange(1000)), 'cov': np.sin(np.pi * 2 / 10 * np.arange(1000))})
        ts = TSDataset.load_from_dataframe(df, target_cols="target", observed_cov_cols='cov')
        res = Correlation(lag=3, lag_cols=['cov'])(ts)
        self.assertEqual(len(res), 2)
        self.assertEqual(len(res['target']), 2)
        
        #case7, input data is tsdataset, select feature
        df = pd.DataFrame({'target': np.sin(np.pi * 2 / 100 * np.arange(1000)), 'cov': np.sin(np.pi * 2 / 10 * np.arange(1000)),
                          'cov1': np.sin(np.pi * 2 / 10 * np.arange(1000))})
        ts = TSDataset.load_from_dataframe(df, target_cols="target", observed_cov_cols='cov')
        res = Correlation()(ts, columns=['target', 'cov'])
        self.assertEqual(len(res), 2)
        self.assertEqual(len(res['target']), 2)
        
        #case8, test_get_properties()
        res = Correlation().get_properties().get("name")
        self.assertEqual(res, "correlation")
        
        #case9, test plot
        df = pd.DataFrame({'target': np.sin(np.pi * 2 / 100 * np.arange(1000)), 'cov': np.sin(np.pi * 2 / 10 * np.arange(1000))})
        ts = TSDataset.load_from_dataframe(df, target_cols="target", observed_cov_cols='cov')
        sea = Correlation()
        res = sea(ts)
        plot = sea.plot()
        plot.savefig('/tmp/correlation.png')
        
        
if __name__ == "__main__":
    unittest.main()