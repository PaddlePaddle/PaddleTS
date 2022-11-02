# !/usr/bin/env python3
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np

import os
import sys
import time
import unittest
from unittest import TestCase

from paddlets.transform.time_feature import TimeFeatureGenerator
from paddlets.datasets.tsdataset import TimeSeries, TSDataset

class TestTimeFeature(TestCase):
    def setUp(self):
        """
        unittest function
        """
        super().setUp()

    def test_transform(self):
        """
        unittest function
        """
        #known cov为None时
        base_date = pd.date_range(start='2021-01-01', end='2021-01-05', freq='D')
        true_base_date = pd.date_range(start='2021-01-01', end='2021-01-08', freq='D')
        ts_df = pd.DataFrame({'DATE_TIME': base_date})
        ts_df['load'] = 10
        
        known_true_df = pd.DataFrame({'DATE_TIME': true_base_date, 'year': [2021] * 8, 'month': [1] * 8, 'day': [1, 2, 3, 4, 5, 6, 7, 8], 
                                     'weekday': [4, 5, 6, 0, 1, 2, 3, 4], 'hour': [0] * 8, 'quarter': [1] * 8, 'dayofyear': [1, 2, 3, 4, 5, 6, 7, 8], 
                                     'weekofyear': [53, 53, 53, 1, 1, 1, 1, 1], 'is_holiday': [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'is_workday': [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]}).set_index('DATE_TIME')
        
        ts = TSDataset.load_from_dataframe(df=ts_df, time_col='DATE_TIME', target_cols='load')
        ob = TimeFeatureGenerator(extend_points=3)
        result = ob.fit(ts)
        
        result = ob.transform(ts, False)
        known = result.get_known_cov().data

        self.assertEqual(np.array(known_true_df).tolist(), np.array(known).tolist())
        
        #know cov不为None时
        base_date = pd.date_range(start='2021-01-01', end='2021-01-05', freq='D')
        true_base_date = pd.date_range(start='2021-01-01', end='2021-01-08', freq='D')
        ts_df = pd.DataFrame({'DATE_TIME': base_date})
        ts_df['load'] = 10
        ts_df_know = pd.DataFrame({'DATE_TIME': true_base_date})
        ts_df_know['other'] = 11
        
        known_true_df = pd.DataFrame({'DATE_TIME': true_base_date, 'other': [11] * 8, 'year': [2021] * 8, 'month': [1] * 8, 'day': [1, 2, 3, 4, 5, 6, 7, 8], 
                                     'weekday': [4, 5, 6, 0, 1, 2, 3, 4], 'hour': [0] * 8, 'quarter': [1] * 8, 'dayofyear': [1, 2, 3, 4, 5, 6, 7, 8], 
                                     'weekofyear': [53, 53, 53, 1, 1, 1, 1, 1], 'is_holiday': [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'is_workday': [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]}).set_index('DATE_TIME')
        
        ts = TSDataset.load_from_dataframe(df=ts_df, time_col='DATE_TIME', target_cols='load')
        known_cov = TimeSeries.load_from_dataframe(data=ts_df_know, time_col='DATE_TIME', value_cols='other')
        ts.set_known_cov(known_cov)
        ob = TimeFeatureGenerator()
        result = ob.fit(ts)
        
        result = ob.transform(ts, False)
        known = result.get_known_cov().data

        self.assertEqual(np.array(known_true_df).tolist(), np.array(known).tolist())

        results = ob.fit([ts, ts])
        results= ob.transform([ts, ts], False)
        self.assertEqual(len(results), 2)
        for result in results:
            known = result.get_known_cov().data
            self.assertEqual(np.array(known_true_df).tolist(), np.array(known).tolist())

    def test_fit_transform(self):
        """
        unittest function
        """
        #known cov为None时
        base_date = pd.date_range(start='2021-01-01', end='2021-01-05', freq='D')
        true_base_date = pd.date_range(start='2021-01-01', end='2021-01-08', freq='D')
        ts_df = pd.DataFrame({'DATE_TIME': base_date})
        ts_df['load'] = 10
        
        known_true_df = pd.DataFrame({'DATE_TIME': true_base_date, 'year': [2021] * 8, 'month': [1] * 8, 'day': [1, 2, 3, 4, 5, 6, 7, 8], 
                                     'weekday': [4, 5, 6, 0, 1, 2, 3, 4], 'hour': [0] * 8, 'quarter': [1] * 8, 'dayofyear': [1, 2, 3, 4, 5, 6, 7, 8], 
                                     'weekofyear': [53, 53, 53, 1, 1, 1, 1, 1], 'is_holiday': [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'is_workday': [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]}).set_index('DATE_TIME')
        
        ts = TSDataset.load_from_dataframe(df=ts_df, time_col='DATE_TIME', target_cols='load')
        ob = TimeFeatureGenerator(extend_points=3)
        result = ob.fit(ts)
        
        result = ob.transform(ts, False)
        known = result.get_known_cov().data

        self.assertEqual(np.array(known_true_df).tolist(), np.array(known).tolist())
        
        #know cov不为None时
        base_date = pd.date_range(start='2021-01-01', end='2021-01-05', freq='D')
        true_base_date = pd.date_range(start='2021-01-01', end='2021-01-08', freq='D')
        ts_df = pd.DataFrame({'DATE_TIME': base_date})
        ts_df['load'] = 10
        ts_df_know = pd.DataFrame({'DATE_TIME': true_base_date})
        ts_df_know['other'] = 11
        
        known_true_df = pd.DataFrame({'DATE_TIME': true_base_date, 'other': [11] * 8, 'year': [2021] * 8, 'month': [1] * 8, 'day': [1, 2, 3, 4, 5, 6, 7, 8], 
                                     'weekday': [4, 5, 6, 0, 1, 2, 3, 4], 'hour': [0] * 8, 'quarter': [1] * 8, 'dayofyear': [1, 2, 3, 4, 5, 6, 7, 8], 
                                     'weekofyear': [53, 53, 53, 1, 1, 1, 1, 1], 'is_holiday': [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'is_workday': [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]}).set_index('DATE_TIME')
        
        ts = TSDataset.load_from_dataframe(df=ts_df, time_col='DATE_TIME', target_cols='load')
        known_cov = TimeSeries.load_from_dataframe(data=ts_df_know, time_col='DATE_TIME', value_cols='other')
        ts.set_known_cov(known_cov)
        ob = TimeFeatureGenerator()
        result = ob.fit(ts)
        
        result = ob.transform(ts, False)
        known = result.get_known_cov().data

        self.assertEqual(np.array(known_true_df).tolist(), np.array(known).tolist())

        results= ob.fit_transform([ts, ts], False)
        self.assertEqual(len(results), 2)
        for result in results:
            known = result.get_known_cov().data
            self.assertEqual(np.array(known_true_df).tolist(), np.array(known).tolist())
   

if __name__ == "__main__":
    unittest.main()
