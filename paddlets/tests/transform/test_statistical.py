# !/usr/bin/env python3
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np

import unittest
from unittest import TestCase

from paddlets.transform.statistical import StatsTransform
from paddlets.datasets.tsdataset import TimeSeries, TSDataset


class TestStatsTransform(TestCase):
    def setUp(self):
        """
        unittest function
        """
        super().setUp()

    def test_fit(self):
        """
        unittest function
        """
        pass


    def test_transform(self):
        """
        unittest function
        """

        fake_df = pd.DataFrame(np.arange(12.5, 32.5).reshape((5, 4)), columns=["WW", "XX", "YY", "ZZ"])
        ts = TSDataset.load_from_dataframe(df=fake_df, known_cov_cols=['WW', 'XX', 'YY', 'ZZ'])
        try:
            ob = StatsTransform(cols = [], start=0, end=2, statistics=['median', 'mean', 'min', 'max'])
        except Exception as e:
            succeed = False
            self.assertFalse(succeed)

        try:
            ob = StatsTransform(cols = ["WW"], start=0, end=2, statistics=['median', 'mean', 'min', 'max'])
        except Exception as e:
            succeed = False
            self.assertFalse(succeed)

        try:
            ob = StatsTransform(cols = ["WW"], start=0, end=2, statistics=['med', 'mean'])
        except Exception as e:
            succeed = False
            self.assertFalse(succeed)

        try:
            ob = StatsTransform(cols = ["WW"], start=3, end=2, statistics=['median', 'mean', 'min', 'max'])
        except Exception as e:
            succeed = False
            self.assertFalse(succeed)

        try:
            ob = StatsTransform(cols = ["WW"], start=0, end=-1, statistics=['median', 'mean', 'min', 'max'])
        except Exception as e:
            succeed = False
            self.assertFalse(succeed)

        ob = StatsTransform(cols = ["WW"], start=0, end=2, statistics=['median', 'mean', 'min', 'max'])
        ob.fit(ts)
        new_ts = ob.transform(ts)
        known = new_ts.get_known_cov().data

        known_true_df = pd.DataFrame([[float('NaN'), float('NaN'), float('NaN'), float('NaN')], [14.5, 14.5, 12.5, 16.5], \
                                    [18.5, 18.5, 16.5, 20.5], [22.5, 22.5, 20.5, 24.5],\
                                    [26.5, 26.5, 24.5, 28.5]], columns=["WW_median", "WW_mean", "WW_min", "WW_max"])
        self.assertEqual(np.array(known_true_df).tolist()[1:], np.array(known[["WW_median", "WW_mean", "WW_min", "WW_max"]]).tolist()[1:])
        
        #��target�н���ʱ��ͳ��
        fake_input = pd.DataFrame([[0.72, 0.29, 0.55, "2021-01-01 00:00:00"], 
                                   [0.80, -0.84, -0.7, "2021-01-01 00:00:01"],
                                   [-1.2, 0.6, 2.1, "2021-01-01 00:00:02"], 
                                   [0.6, 1.3, -0.4, "2021-01-01 00:00:03"]], columns=['value', 'cov1', 'cov2', 'time'])
        ts = TSDataset.load_from_dataframe(df=fake_input, time_col='time', \
                observed_cov_cols=['cov1', 'cov2', 'time'], target_cols=['value'])
        ob = StatsTransform(cols = ["value"], start=0, end=1, statistics=['min'])
        ob.fit(ts)
        new_ts = ob.transform(ts)
        observed = new_ts.get_observed_cov().data
        self.assertEqual(observed['value_min'].tolist(), [0.72, 0.8, -1.2, 0.6])

        ob.fit([ts, ts])
        new_tss = ob.transform([ts, ts])
        self.assertEqual(len(new_tss), 2)
        for new_ts in new_tss:
            observed = new_ts.get_observed_cov().data
            self.assertEqual(observed['value_min'].tolist(), [0.72, 0.8, -1.2, 0.6])
    
    def test_fit_transform(self):
        """
        unittest function
        """

        fake_df = pd.DataFrame(np.arange(12.5, 32.5).reshape((5, 4)), columns=["WW", "XX", "YY", "ZZ"])
        ts = TSDataset.load_from_dataframe(df=fake_df, known_cov_cols=['WW', 'XX', 'YY', 'ZZ']) 
        ob = StatsTransform(cols = ["WW"], start=0, end=2, statistics=['median', 'mean', 'min', 'max'])
        ob.fit(ts)
        new_ts = ob.fit_transform(ts)
        known = new_ts.get_known_cov().data

        known_true_df = pd.DataFrame([[float('NaN'), float('NaN'), float('NaN'), float('NaN')], [14.5, 14.5, 12.5, 16.5], \
                                    [18.5, 18.5, 16.5, 20.5], [22.5, 22.5, 20.5, 24.5],\
                                    [26.5, 26.5, 24.5, 28.5]], columns=["WW_median", "WW_mean", "WW_min", "WW_max"])
        self.assertEqual(np.array(known_true_df).tolist()[1:], np.array(known[["WW_median", "WW_mean", "WW_min", "WW_max"]]).tolist()[1:])

        new_tss = ob.fit_transform([ts, ts])
        self.assertEqual(len(new_tss), 2)
        for new_ts in new_tss:
            known = new_ts.get_known_cov().data

            known_true_df = pd.DataFrame([[float('NaN'), float('NaN'), float('NaN'), float('NaN')], [14.5, 14.5, 12.5, 16.5], \
                                        [18.5, 18.5, 16.5, 20.5], [22.5, 22.5, 20.5, 24.5],\
                                        [26.5, 26.5, 24.5, 28.5]], columns=["WW_median", "WW_mean", "WW_min", "WW_max"])
            self.assertEqual(np.array(known_true_df).tolist()[1:], np.array(known[["WW_median", "WW_mean", "WW_min", "WW_max"]]).tolist()[1:])


   
if __name__ == "__main__":
    unittest.main()
