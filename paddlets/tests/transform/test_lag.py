# !/usr/bin/env python3
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np

import os
import sys
import time
import unittest
from unittest import TestCase

from paddlets.transform.lag import LagFeatureGenerator
from paddlets.datasets.tsdataset import TimeSeries, TSDataset

class TestLagFeature(TestCase):
    def setUp(self):
        """
        unittest function
        """
        super().setUp()

    def test_transform(self):
        """
        unittest function
        """
        #feature names is None
        base_date = pd.date_range(start='2021-01-01', end='2021-01-05', freq='D')
        true_base_date = pd.date_range(start='2021-01-01', end='2021-01-05', freq='D')
        ts_df = pd.DataFrame({'DATE_TIME': base_date})
        ts_df['load'] = list(range(1, 6))
        
        known_true_df = pd.DataFrame({'DATE_TIME': true_base_date, 'load_before_2': [np.nan, np.nan, 1., 2., 3.], 'load_before_4': [np.nan, np.nan, np.nan, np.nan, 1.]}).set_index('DATE_TIME')
        
        ts = TSDataset.load_from_dataframe(df=ts_df, time_col='DATE_TIME', target_cols='load')
        ob = LagFeatureGenerator(lag_points=4, down_samples=2)
        result = ob.fit(ts)
        
        result = ob.transform(ts, False)
        known = result.get_observed_cov().data
        known[known.isnull()] = np.nan
        known_true_df[known_true_df.isnull()] = np.nan

        self.assertTrue(known_true_df.equals(known))
        
        #feature names is not None
        base_date = pd.date_range(start='2021-01-01', end='2021-01-05', freq='D')
        true_base_date = pd.date_range(start='2021-01-01', end='2021-01-05', freq='D')
        ts_df = pd.DataFrame({'DATE_TIME': base_date})
        ts_df['load'] = list(range(1, 6))
        ts_df_know = pd.DataFrame({'DATE_TIME': true_base_date})
        ts_df_know['other'] = list(range(1, 6))
        
        known_true_df = pd.DataFrame({'DATE_TIME': true_base_date, 'other': list(range(1, 6)), 'other_before_2': [np.nan, np.nan, 1., 2., 3.], 'other_before_4': [np.nan, np.nan, np.nan, np.nan, 1.]}).set_index('DATE_TIME')
        
        ts = TSDataset.load_from_dataframe(df=ts_df, time_col='DATE_TIME', target_cols='load')
        known_cov = TimeSeries.load_from_dataframe(data=ts_df_know, time_col='DATE_TIME', value_cols='other')
        ts.set_known_cov(known_cov)
        ob = LagFeatureGenerator(cols='other', lag_points=4, down_samples=2)
        result = ob.fit(ts)
        
        result = ob.transform(ts, False)
        known = result.get_known_cov().data
        known[known.isnull()] = np.nan
        known_true_df[known_true_df.isnull()] = np.nan

        self.assertTrue(known_true_df.equals(known))

        results = ob.transform([ts, ts], False)
        self.assertEqual(len(results), 2)
        for resutl in results:
            known = result.get_known_cov().data
            known[known.isnull()] = np.nan
            known_true_df[known_true_df.isnull()] = np.nan
            self.assertTrue(known_true_df.equals(known))

    def test_fit_transform(self):
        """
        unittest function
        """
        #feature names is None
        base_date = pd.date_range(start='2021-01-01', end='2021-01-05', freq='D')
        true_base_date = pd.date_range(start='2021-01-01', end='2021-01-05', freq='D')
        ts_df = pd.DataFrame({'DATE_TIME': base_date})
        ts_df['load'] = list(range(1, 6))
        
        known_true_df = pd.DataFrame({'DATE_TIME': true_base_date, 'load_before_2': [np.nan, np.nan, 1., 2., 3.], 'load_before_4': [np.nan, np.nan, np.nan, np.nan, 1.]}).set_index('DATE_TIME')
        
        ts = TSDataset.load_from_dataframe(df=ts_df, time_col='DATE_TIME', target_cols='load')
        ob = LagFeatureGenerator(lag_points=4, down_samples=2)
        result = ob.fit(ts)
        
        result = ob.transform(ts, False)
        known = result.get_observed_cov().data
        known[known.isnull()] = np.nan
        known_true_df[known_true_df.isnull()] = np.nan

        self.assertTrue(known_true_df.equals(known))
        
        #feature names is not None
        base_date = pd.date_range(start='2021-01-01', end='2021-01-05', freq='D')
        true_base_date = pd.date_range(start='2021-01-01', end='2021-01-05', freq='D')
        ts_df = pd.DataFrame({'DATE_TIME': base_date})
        ts_df['load'] = list(range(1, 6))
        ts_df_know = pd.DataFrame({'DATE_TIME': true_base_date})
        ts_df_know['other'] = list(range(1, 6))
        
        known_true_df = pd.DataFrame({'DATE_TIME': true_base_date, 'other': list(range(1, 6)), 'other_before_2': [np.nan, np.nan, 1., 2., 3.], 'other_before_4': [np.nan, np.nan, np.nan, np.nan, 1.]}).set_index('DATE_TIME')
        
        ts = TSDataset.load_from_dataframe(df=ts_df, time_col='DATE_TIME', target_cols='load')
        known_cov = TimeSeries.load_from_dataframe(data=ts_df_know, time_col='DATE_TIME', value_cols='other')
        ts.set_known_cov(known_cov)
        ob = LagFeatureGenerator(cols='other', lag_points=4, down_samples=2)
        result = ob.fit(ts)
        
        result = ob.transform(ts, False)
        known = result.get_known_cov().data
        known[known.isnull()] = np.nan
        known_true_df[known_true_df.isnull()] = np.nan

        self.assertTrue(known_true_df.equals(known))

        results = ob.fit_transform([ts, ts], False)
        self.assertEqual(len(results), 2)
        for resutl in results:
            known = result.get_known_cov().data
            known[known.isnull()] = np.nan
            known_true_df[known_true_df.isnull()] = np.nan

            self.assertTrue(known_true_df.equals(known))

if __name__ == "__main__":
    unittest.main()
