# !/usr/bin/env python3
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np

import os
import sys
import time
import unittest
from unittest import TestCase

from paddlets.transform.difference import DifferenceFeatureGenerator
from paddlets.datasets.tsdataset import TimeSeries, TSDataset

class TestDifferenceFeature(TestCase):
    def setUp(self):
        """
        unittest function
        """
        super().setUp()

    def test_transform(self):
        """
        unittest function
        """
        base_date = pd.date_range(start='2021-01-01', end='2021-01-05', freq='D')
        true_base_date = pd.date_range(start='2021-01-01', end='2021-01-05', freq='D')
        ts_df = pd.DataFrame({'DATE_TIME': base_date})
        ts_df['load'] = list(range(1, 6))
        ts_df_know = pd.DataFrame({'DATE_TIME': true_base_date})
        ts_df_know['other'] = list(range(1, 6))
        
        known_true_df = pd.DataFrame({'DATE_TIME': true_base_date, 'other': list(range(1, 6)), 'other_diff_2': [np.nan, np.nan, 2., 2., 2.], 'other_diff_4': [np.nan, np.nan, np.nan, np.nan, 4.]}).set_index('DATE_TIME')
        
        ts = TSDataset.load_from_dataframe(df=ts_df, time_col='DATE_TIME', target_cols='load')
        known_cov = TimeSeries.load_from_dataframe(data=ts_df_know, time_col='DATE_TIME', value_cols='other')
        ts.set_known_cov(known_cov)
        ob = DifferenceFeatureGenerator(cols='other', difference_points=4, down_samples=2)
        result = ob.fit(ts)
        
        result = ob.transform(ts, False)
        known = result.get_known_cov().data
        known[known.isnull()] = np.nan
        known_true_df[known_true_df.isnull()] = np.nan

        self.assertTrue(known_true_df.equals(known))

        ob.fit([ts, ts])
        results = ob.transform([ts, ts])
        self.assertEqual(len(results), 2)
        for result in results:
            known = result.get_known_cov().data
            known[known.isnull()] = np.nan
            known_true_df[known_true_df.isnull()] = np.nan
            self.assertTrue(known_true_df.equals(known))

    def test_fit_transform(self):
        """
        unittest function
        """
        base_date = pd.date_range(start='2021-01-01', end='2021-01-05', freq='D')
        true_base_date = pd.date_range(start='2021-01-01', end='2021-01-05', freq='D')
        ts_df = pd.DataFrame({'DATE_TIME': base_date})
        ts_df['load'] = list(range(1, 6))
        ts_df_know = pd.DataFrame({'DATE_TIME': true_base_date})
        ts_df_know['other'] = list(range(1, 6))
        
        known_true_df = pd.DataFrame({'DATE_TIME': true_base_date, 'other': list(range(1, 6)), 'other_diff_2': [np.nan, np.nan, 2., 2., 2.], 'other_diff_4': [np.nan, np.nan, np.nan, np.nan, 4.]}).set_index('DATE_TIME')
        
        ts = TSDataset.load_from_dataframe(df=ts_df, time_col='DATE_TIME', target_cols='load')
        known_cov = TimeSeries.load_from_dataframe(data=ts_df_know, time_col='DATE_TIME', value_cols='other')
        ts.set_known_cov(known_cov)
        ob = DifferenceFeatureGenerator(cols='other', difference_points=4, down_samples=2)
        result = ob.fit(ts)
        
        result = ob.transform(ts, False)
        known = result.get_known_cov().data
        known[known.isnull()] = np.nan
        known_true_df[known_true_df.isnull()] = np.nan

        self.assertTrue(known_true_df.equals(known))

        ob.fit([ts, ts])
        results = ob.transform([ts, ts])
        self.assertEqual(len(results), 2)
        for result in results:
            known = result.get_known_cov().data
            known[known.isnull()] = np.nan
            known_true_df[known_true_df.isnull()] = np.nan
            self.assertTrue(known_true_df.equals(known))
        

if __name__ == "__main__":
    unittest.main()
