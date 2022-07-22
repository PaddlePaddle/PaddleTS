# !/usr/bin/env python3
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np

import os
import sys
import time
import unittest
from unittest import TestCase

from paddlets import TimeSeries, TSDataset
from paddlets.analysis import summary
from paddlets.analysis import max 


class TestAnalysis(TestCase):
    def setUp(self):
        """
        unittest function
        """
        super().setUp()

    def test_Summarizer(self):
        """
        unittest function
        """
        #case1
        a = pd.Series(np.random.randn(5))
        res = summary.analyze(a)
        self.assertEqual(res.shape, (9,))
     
        #case2
        periods = 100
        df = pd.DataFrame(
            [1 for i in range(periods)],
            index = pd.date_range('2022-01-01', periods=periods, freq='1D'),
            columns=['target']
        )
        ts = TSDataset.load_from_dataframe(df, target_cols="target")
        ts['target2'] = ts['target'] + 1
        self.assertEqual(summary(ts).shape, (9, 2))

        #case3:
        self.assertEqual(summary(ts, 'target').shape, (9, ))
        self.assertEqual(summary(ts, ['target', 'target2']).shape, (9, 2))

        #case4: 
        self.assertEqual(ts.summary('target').shape, (9, ))
        self.assertEqual(ts.summary(['target', 'target2']).shape, (9, 2))
        flag = False
        try:
            self.assertEqual(ts.summary_no_exists(['target', 'target2']).shape, (9, 2))
        except:
            flag = True
        self.assertTrue(flag)

        #case5: 
        res = summary.get_properties().get("name")
        self.assertEqual(res, "summary")


    def test_Max(self):
        """
        unittest function
        """
        #case1
        s = pd.Series(range(100))
        res = max.analyze(s)
        self.assertEqual(res, 99)
        
        #case2
        df = pd.DataFrame(np.array(range(100)).reshape(20, 5))
        res = max.analyze(df)
        self.assertCountEqual(res, pd.Series([95, 96, 97, 98, 99]))

        #case3
        cov_cols = ['c%d' % i for i in range(4)] 
        df = pd.DataFrame(np.array(range(100)).reshape(20, 5), columns=['target'] + cov_cols)
        ts = TSDataset.load_from_dataframe(df, target_cols="target", observed_cov_cols=cov_cols)
        res = max(ts, ['target', 'c0', 'c3'])
        self.assertCountEqual(res, pd.Series([95, 96, 99]))

        #case4: 
        self.assertEqual(ts.max('target'), 95)
        self.assertCountEqual(ts.max(['target', 'c0']), pd.Series([95, 96]))
        flag = False
        try:
            self.assertEqual(ts.max_no_exists('target').shape, (9, 2))
        except:
            flag = True
        self.assertTrue(flag)
                    
        #case5: 
        res = max.get_properties().get("name")
        self.assertEqual(res, "max")


if __name__ == "__main__":
    unittest.main()
