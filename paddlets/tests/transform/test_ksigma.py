# !/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import sys

import time
import unittest
import pandas as pd
import numpy as np
from unittest import TestCase

from paddlets.transform.ksigma import KSigma
from paddlets.datasets.tsdataset import TimeSeries, TSDataset

class TestKSigma(TestCase):
    def setUp(self):
        """
        unittest function
        """
        super().setUp()

    def test_fit(self):
        """
        unittest function
        """
        #sample of test data
        ts_df = pd.DataFrame(([[90, 10, 'e'], [80, 20, 'e'], [70, 10, 'd'], \
                [80, 20, 'd'], [85, 15, 'f']]), columns = ['Math', 'Eng', 'Name'])
        ts = TSDataset.load_from_dataframe(df=ts_df, known_cov_cols=['Math', 'Eng'], observed_cov_cols = ['Name'])
        
        #case1, a column
        ob = KSigma(['Math'])
        result = ob.fit(ts)
        expect_cols_stats_dict = {'Math': [58.75140453871301, 103.24859546128698, 81.0]}
        real_cols_stats_dict = result._cols_stats_dict
        for i in range(len(expect_cols_stats_dict['Math'])):
            self.assertAlmostEqual(expect_cols_stats_dict['Math'][i], real_cols_stats_dict['Math'][i])
        
        #case2, two column
        ob = KSigma(['Math', 'Eng'])
        result = ob.fit(ts)
        expect_cols_stats_dict = {'Math': [58.75140453871301, 103.24859546128698, 81.0], 'Eng': [0.0, 30.0, 15.0]}
        real_cols_stats_dict = result._cols_stats_dict
        for col in ['Math', 'Eng']:
            for i in range(len(expect_cols_stats_dict[col])):
                self.assertAlmostEqual(expect_cols_stats_dict[col][i], real_cols_stats_dict[col][i])
        
        #case3, three column, and the third column is string
        ob = KSigma(['Math', 'Eng', 'Name'])
        result = ob.fit(ts)
        expect_cols_stats_dict = {'Math': [58.75140453871301, 103.24859546128698, 81.0], 'Eng': [0.0, 30.0, 15.0]}
        real_cols_stats_dict = result._cols_stats_dict
        for col in ['Math', 'Eng']:
            for i in range(len(expect_cols_stats_dict[col])):
                self.assertAlmostEqual(expect_cols_stats_dict[col][i], real_cols_stats_dict[col][i])
        
        #case4, a column and change the k value
        ob = KSigma(['Math'], 0)
        result = ob.fit(ts)
        expect_cols_stats_dict = {'Math': [81.0, 81.0, 81.0]}
        real_cols_stats_dict = result._cols_stats_dict
        for i in range(len(expect_cols_stats_dict['Math'])):
            self.assertAlmostEqual(expect_cols_stats_dict['Math'][i], real_cols_stats_dict['Math'][i])

        result = ob.fit([ts, ts])
        expect_cols_stats_dict = {'Math': [81.0, 81.0, 81.0]}
        real_cols_stats_dict = result._cols_stats_dict
        for i in range(len(expect_cols_stats_dict['Math'])):
            self.assertAlmostEqual(expect_cols_stats_dict['Math'][i], real_cols_stats_dict['Math'][i])

    def test_transform(self):
        """
        unittest function
        """
        #sample of test data
        ts_df = pd.DataFrame(([[90, 10, 'e'], [80, 20, 'e'], [70, 10, 'd'], \
                [80, 20, 'd'], [85, 15, 'f']]), columns = ['Math', 'Eng', 'Name'])
        ts = TSDataset.load_from_dataframe(df=ts_df, known_cov_cols=['Math', 'Eng'], observed_cov_cols = ['Name'])
        
        #case1, a column
        ob = KSigma(['Math'])
        result = ob.fit(ts).transform(ts)
        expect_known_df = pd.DataFrame([90, 80, 70, 80, 85], columns=['Math'], dtype=float)
        real_known_df = result.get_known_cov().data
        self.assertEqual(np.array(expect_known_df['Math']).tolist(), np.array(real_known_df['Math']).tolist())
        
        #case2, two column
        ob = KSigma(['Math', 'Eng'])
        result = ob.fit(ts).transform(ts)
        expect_known_df = pd.DataFrame([[90, 10], [80, 20], [70, 10], [80, 20], [85, 15]], columns=['Math', 'Eng'], dtype=float)
        real_known_df = result.get_known_cov().data
        for col in ['Math', 'Eng']:
            self.assertEqual(np.array(expect_known_df[col]).tolist(), np.array(real_known_df[col]).tolist())
        
        #case3, three column, and the third column is string
        ob = KSigma(['Math', 'Eng', 'Name'])
        result = ob.fit(ts).transform(ts)
        expect_known_df = pd.DataFrame([[90, 10], [80, 20], [70, 10], [80, 20], [85, 15]], columns=['Math', 'Eng'], dtype=float)
        expect_observed_df = pd.DataFrame(['e', 'e', 'd', 'd', 'f'], columns=['Name'])
        real_known_df = result.get_known_cov().data
        real_observed_df = result.get_observed_cov().data
        for col in ['Math', 'Eng']:
            self.assertEqual(np.array(expect_known_df[col]).tolist(), np.array(real_known_df[col]).tolist())
        self.assertEqual(np.array(expect_observed_df['Name']).tolist(), np.array(real_observed_df['Name']).tolist())
        
        #case4, a column and change the k value
        ts = TSDataset.load_from_dataframe(df=ts_df, known_cov_cols=['Math', 'Eng'], observed_cov_cols = ['Name'])
        ob = KSigma(['Math'], 0)
        result = ob.fit(ts).transform(ts)
        expect_known_df = pd.DataFrame([81, 81, 81, 81, 81], columns=['Math'], dtype=float)
        real_known_df = result.get_known_cov().data
        self.assertEqual(np.array(expect_known_df['Math']).tolist(), np.array(real_known_df['Math']).tolist())

        results = ob.fit(ts).transform([ts, ts])
        self.assertEqual(len(results), 2)
        for resutl in results:
            expect_known_df = pd.DataFrame([81, 81, 81, 81, 81], columns=['Math'], dtype=float)
            real_known_df = result.get_known_cov().data
            self.assertEqual(np.array(expect_known_df['Math']).tolist(), np.array(real_known_df['Math']).tolist())

    def test_fit_transform(self):
        """
        unittest function
        """
        #sample of test data
        ts_df = pd.DataFrame(([[90, 10, 'e'], [80, 20, 'e'], [70, 10, 'd'], \
                [80, 20, 'd'], [85, 15, 'f']]), columns = ['Math', 'Eng', 'Name'])
        ts = TSDataset.load_from_dataframe(df=ts_df, known_cov_cols=['Math', 'Eng'], observed_cov_cols = ['Name'])
        
        #case1, a column
        ob = KSigma(['Math'])
        result = ob.fit_transform(ts)
        expect_known_df = pd.DataFrame([90, 80, 70, 80, 85], columns=['Math'], dtype=float)
        real_known_df = result.get_known_cov().data
        self.assertEqual(np.array(expect_known_df['Math']).tolist(), np.array(real_known_df['Math']).tolist())
        
        #case2, two column
        ob = KSigma(['Math', 'Eng'])
        result = ob.fit_transform(ts)
        expect_known_df = pd.DataFrame([[90, 10], [80, 20], [70, 10], [80, 20], [85, 15]], columns=['Math', 'Eng'], dtype=float)
        real_known_df = result.get_known_cov().data
        for col in ['Math', 'Eng']:
            self.assertEqual(np.array(expect_known_df[col]).tolist(), np.array(real_known_df[col]).tolist())
        
        #case3, three column, and the third column is string
        ob = KSigma(['Math', 'Eng', 'Name'])
        result = ob.fit_transform(ts)
        expect_known_df = pd.DataFrame([[90, 10], [80, 20], [70, 10], [80, 20], [85, 15]], columns=['Math', 'Eng'], dtype=float)
        expect_observed_df = pd.DataFrame(['e', 'e', 'd', 'd', 'f'], columns=['Name'])
        real_known_df = result.get_known_cov().data
        real_observed_df = result.get_observed_cov().data
        for col in ['Math', 'Eng']:
            self.assertEqual(np.array(expect_known_df[col]).tolist(), np.array(real_known_df[col]).tolist())
        self.assertEqual(np.array(expect_observed_df['Name']).tolist(), np.array(real_observed_df['Name']).tolist())
        
        #case4, a column and change the k value
        ts = TSDataset.load_from_dataframe(df=ts_df, known_cov_cols=['Math', 'Eng'], observed_cov_cols = ['Name'])
        ob = KSigma(['Math'], 0)
        result = ob.fit_transform(ts)
        expect_known_df = pd.DataFrame([81, 81, 81, 81, 81], columns=['Math'], dtype=float)
        real_known_df = result.get_known_cov().data
        self.assertEqual(np.array(expect_known_df['Math']).tolist(), np.array(real_known_df['Math']).tolist())

        results = ob.fit_transform([ts, ts])
        self.assertEqual(len(results), 2)
        for resutl in results:
            expect_known_df = pd.DataFrame([81, 81, 81, 81, 81], columns=['Math'], dtype=float)
            real_known_df = result.get_known_cov().data
            self.assertEqual(np.array(expect_known_df['Math']).tolist(), np.array(real_known_df['Math']).tolist())

if __name__ == "__main__":
    unittest.main()
