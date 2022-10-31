# !/usr/bin/env python3
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import os
import sys
import time
import unittest
from unittest import TestCase
from paddlets.transform.fill import Fill
from paddlets.datasets.tsdataset import TimeSeries
from paddlets.datasets.tsdataset import TSDataset


class TestFill(TestCase):
    def setUp(self):
        """
        unittest function
        """
        super().setUp()

    def test_fit(self):
        """
        unittest function
        """
        print(sys.stderr, "test_fit()...")
        a = [1, np.nan, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, np.nan]
        a1 = [1, np.nan, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, np.nan]
        b = [1, np.nan, 3, 4, 5, 6, 7, 8, 9, 10, 11, np.nan, np.nan, 14, 15, 16]
        b1 = [1, np.nan, 3, 4, 5, 6, 7, 8, 9, 10, 11, np.nan, np.nan, 14, 15, 16]
        fake_input = pd.DataFrame(columns=["X", "Y", "X1", "Y1"])
        fake_input['X'] = a
        fake_input['Y'] = b
        fake_input['X1'] = a1
        fake_input['Y1'] = b1
        ts = TSDataset.load_from_dataframe(df=fake_input, known_cov_cols=['X', 'X1'], 
                                                       observed_cov_cols=['Y', 'Y1'])
        params = {'cols': ['X', 'X1', 'Y', 'Y1']} 
        ob = eval("Fill")(**params)
        result = ob.fit(ts)
        results = ob.fit([ts, ts])

    def test_transform(self):
        """
        unittest function
        """
        print(sys.stderr, "test_transform()...")
        #测试数据样例
        a = [1, np.nan, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, np.nan]
        a1 = [1, np.nan, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, np.nan]
        b = [1, np.nan, 3, 4, 5, 6, 7, 8, 9, 10, 11, np.nan, np.nan, 14, 15, 16]
        b1 = [1, np.nan, 3, 4, 5, 6, 7, 8, 9, 10, 11, np.nan, np.nan, 14, 15, 16]
        fake_input = pd.DataFrame(columns=["X", "Y", "X1", "Y1"])
        fake_input['X'] = a
        fake_input['Y'] = b
        fake_input['X1'] = a1
        fake_input['Y1'] = b1
        ts = TSDataset.load_from_dataframe(df=fake_input, known_cov_cols=['X', 'X1'], 
                                                       observed_cov_cols=['Y', 'Y1'])

        # test -- pre
        a = [1, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15]
        a1 = [1, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15]
        b = [1, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11, 11, 14, 15, 16]
        b1 = [1, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11, 11, 14, 15, 16]
        expect_output = pd.DataFrame(columns=["X", "Y", "X1", "Y1"])
        expect_output['X'] = a
        expect_output['Y'] = b
        expect_output['X1'] = a1
        expect_output['Y1'] = b1
        expect_output = expect_output.astype('float')
        eo = TSDataset.load_from_dataframe(df=expect_output, known_cov_cols=['X', 'X1'], 
                                                          observed_cov_cols=['Y', 'Y1'])
        params = {'cols': ['X', 'X1', 'Y', 'Y1'], 'method': 'pre'} 
        ob = eval("Fill")(**params)
        at = ob.transform(ts)
        self.assertTrue(eo.get_known_cov().data.equals(at.get_known_cov().data))
        self.assertTrue(eo.get_observed_cov().data.equals(at.get_observed_cov().data))

        # test -- next
        a = [1, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, np.nan]
        a1 = [1, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, np.nan]
        b = [1, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 14, 14, 15, 16]
        b1 = [1, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 14, 14, 15, 16]
        expect_output = pd.DataFrame(columns=["X", "Y", "X1", "Y1"])
        expect_output['X'] = a
        expect_output['Y'] = b
        expect_output['X1'] = a1
        expect_output['Y1'] = b1
        expect_output = expect_output.astype('float')
        eo = TSDataset.load_from_dataframe(df=expect_output, known_cov_cols=['X', 'X1'], 
                                                        observed_cov_cols=['Y', 'Y1'])
        params = {'cols': ['X', 'X1', 'Y', 'Y1'], 'method': 'next'} 
        ob = eval("Fill")(**params)
        at = ob.transform(ts)
        self.assertTrue(eo.get_known_cov().data.equals(at.get_known_cov().data))
        self.assertTrue(eo.get_observed_cov().data.equals(at.get_observed_cov().data))        

        # test --  zero
        a = [1, 0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0]
        a1 = [1, 0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0]
        b = [1, 0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 0, 14, 15, 16]
        b1 = [1, 0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 0, 14, 15, 16]
        expect_output = pd.DataFrame(columns=["X", "Y", "X1", "Y1"])
        expect_output['X'] = a
        expect_output['Y'] = b
        expect_output['X1'] = a1
        expect_output['Y1'] = b1
        expect_output = expect_output.astype('float')
        eo = TSDataset.load_from_dataframe(df=expect_output, known_cov_cols=['X', 'X1'], 
                                                          observed_cov_cols=['Y', 'Y1'])
        params = {'cols': ['X', 'X1', 'Y', 'Y1'], 'method': 'zero'} 
        ob = eval("Fill")(**params)
        at = ob.transform(ts)
        self.assertTrue(eo.get_known_cov().data.equals(at.get_known_cov().data))
        self.assertTrue(eo.get_observed_cov().data.equals(at.get_observed_cov().data))

        # test --  opt
        a = [1, 3.1415926, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 3.1415926]
        a1 = [1, 3.1415926, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 3.1415926]
        b = [1, 3.1415926, 3, 4, 5, 6, 7, 8, 9, 10, 11, 3.1415926, 3.1415926, 14, 15, 16]
        b1 = [1, 3.1415926, 3, 4, 5, 6, 7, 8, 9, 10, 11, 3.1415926, 3.1415926, 14, 15, 16]
        expect_output = pd.DataFrame(columns=["X", "Y", "X1", "Y1"])
        expect_output['X'] = a
        expect_output['Y'] = b
        expect_output['X1'] = a1
        expect_output['Y1'] = b1
        expect_output = expect_output.astype('float')
        eo = TSDataset.load_from_dataframe(df=expect_output, known_cov_cols=['X', 'X1'], 
                                                          observed_cov_cols=['Y', 'Y1'])
        params = {'cols': ['X', 'X1', 'Y', 'Y1'], 'method': 'default', 'value': '3.1415926'} 
        ob = eval("Fill")(**params)
        at = ob.transform(ts)
        self.assertTrue(eo.get_known_cov().data.equals(at.get_known_cov().data.astype('float')))
        self.assertTrue(eo.get_observed_cov().data.equals(at.get_observed_cov().data.astype('float')))

        # test --  max
        a = [1, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15]
        a1 = [1, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15]
        b = [1, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11, 11, 14, 15, 16]
        b1 = [1, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11, 11, 14, 15, 16]
        expect_output = pd.DataFrame(columns=["X", "Y", "X1", "Y1"])
        expect_output['X'] = a
        expect_output['Y'] = b
        expect_output['X1'] = a1
        expect_output['Y1'] = b1
        expect_output = expect_output.astype('float')
        eo = TSDataset.load_from_dataframe(df=expect_output, known_cov_cols=['X', 'X1'], 
                                                          observed_cov_cols=['Y', 'Y1'])
        params = {'cols': ['X', 'X1', 'Y', 'Y1'], 'method': 'max'} 
        ob = eval("Fill")(**params)
        at = ob.transform(ts)
        self.assertTrue(eo.get_known_cov().data.equals(at.get_known_cov().data))
        self.assertTrue(eo.get_observed_cov().data.equals(at.get_observed_cov().data))

        # test --  min
        a = [1, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 7]
        a1 = [1, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 7]
        b = [1, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 3, 3, 14, 15, 16]
        b1 = [1, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 3, 3, 14, 15, 16]
        expect_output = pd.DataFrame(columns=["X", "Y", "X1", "Y1"])
        expect_output['X'] = a
        expect_output['Y'] = b
        expect_output['X1'] = a1
        expect_output['Y1'] = b1
        expect_output = expect_output.astype('float')
        eo = TSDataset.load_from_dataframe(df=expect_output, known_cov_cols=['X', 'X1'], 
                                                          observed_cov_cols=['Y', 'Y1'])
        params = {'cols': ['X', 'X1', 'Y', 'Y1'], 'method': 'min', 'min_num_non_missing_values': 1} 
        ob = eval("Fill")(**params)
        at = ob.transform(ts)
        self.assertTrue(eo.get_known_cov().data.equals(at.get_known_cov().data))
        self.assertTrue(eo.get_observed_cov().data.equals(at.get_observed_cov().data))

        # test --  avg
        a = [1, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 14.5]
        a1 = [1, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 14.5]
        b = [1, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 10.5, 10.75, 14, 15, 16]
        b1 = [1, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 10.5, 10.75, 14, 15, 16]
        expect_output = pd.DataFrame(columns=["X", "Y", "X1", "Y1"])
        expect_output['X'] = a
        expect_output['Y'] = b
        expect_output['X1'] = a1
        expect_output['Y1'] = b1
        expect_output = expect_output.astype('float')
        eo = TSDataset.load_from_dataframe(df=expect_output, known_cov_cols=['X', 'X1'], 
                                                          observed_cov_cols=['Y', 'Y1'])
        params = {'cols': ['X', 'X1', 'Y', 'Y1'], 'method': 'mean', 'window_size': 3} 
        ob = eval("Fill")(**params)
        at = ob.transform(ts)
        self.assertTrue(eo.get_known_cov().data.equals(at.get_known_cov().data))
        self.assertTrue(eo.get_observed_cov().data.equals(at.get_observed_cov().data))

        # test --  median
        a = [1, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 14]
        a1 = [1, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 14]
        b = [1, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 10, 10, 14, 15, 16]
        b1 = [1, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 10, 10, 14, 15, 16]
        expect_output = pd.DataFrame(columns=["X", "Y", "X1", "Y1"])
        expect_output['X'] = a
        expect_output['Y'] = b
        expect_output['X1'] = a1
        expect_output['Y1'] = b1
        expect_output = expect_output.astype('float')
        eo = TSDataset.load_from_dataframe(df=expect_output, known_cov_cols=['X', 'X1'], 
                                                          observed_cov_cols=['Y', 'Y1'])
        params = {'cols': ['X', 'X1', 'Y', 'Y1'], 'method': 'median', 'window_size': 4} 
        ob = eval("Fill")(**params)
        at = ob.transform(ts)
        self.assertTrue(eo.get_known_cov().data.equals(at.get_known_cov().data))
        self.assertTrue(eo.get_observed_cov().data.equals(at.get_observed_cov().data))

        ats= ob.transform([ts, ts])
        self.assertEqual(len(ats), 2)
        for at in ats:
            self.assertTrue(eo.get_known_cov().data.equals(at.get_known_cov().data))
            self.assertTrue(eo.get_observed_cov().data.equals(at.get_observed_cov().data))

        return

    def test_fit_transform(self):
        """
        unittest function
        """
        print(sys.stderr, "test_fit_transform()...")
        #测试数据样例
        a = [1, np.nan, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, np.nan]
        a1 = [1, np.nan, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, np.nan]
        b = [1, np.nan, 3, 4, 5, 6, 7, 8, 9, 10, 11, np.nan, np.nan, 14, 15, 16]
        b1 = [1, np.nan, 3, 4, 5, 6, 7, 8, 9, 10, 11, np.nan, np.nan, 14, 15, 16]
        fake_input = pd.DataFrame(columns=["X", "Y", "X1", "Y1"])
        fake_input['X'] = a
        fake_input['Y'] = b
        fake_input['X1'] = a1
        fake_input['Y1'] = b1
        ts = TSDataset.load_from_dataframe(df=fake_input, known_cov_cols=['X', 'X1'], 
                                                     observed_cov_cols=['Y', 'Y1'])

        # test -- pre
        a = [1, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15]
        a1 = [1, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15]
        b = [1, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11, 11, 14, 15, 16]
        b1 = [1, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11, 11, 14, 15, 16]
        expect_output = pd.DataFrame(columns=["X", "Y", "X1", "Y1"])
        expect_output['X'] = a
        expect_output['Y'] = b
        expect_output['X1'] = a1
        expect_output['Y1'] = b1
        expect_output = expect_output.astype('float')
        eo = TSDataset.load_from_dataframe(df=expect_output, known_cov_cols=['X', 'X1'], 
                                                        observed_cov_cols=['Y', 'Y1'])
        params = {'cols': ['X', 'X1', 'Y', 'Y1'], 'method': 'pre'} 
        ob = eval("Fill")(**params)
        at = ob.fit_transform(ts)
        self.assertTrue(eo.get_known_cov().data.equals(at.get_known_cov().data))
        self.assertTrue(eo.get_observed_cov().data.equals(at.get_observed_cov().data))

        # test -- next
        a = [1, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, np.nan]
        a1 = [1, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, np.nan]
        b = [1, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 14, 14, 15, 16]
        b1 = [1, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 14, 14, 15, 16]
        expect_output = pd.DataFrame(columns=["X", "Y", "X1", "Y1"])
        expect_output['X'] = a
        expect_output['Y'] = b
        expect_output['X1'] = a1
        expect_output['Y1'] = b1
        expect_output = expect_output.astype('float')
        eo = TSDataset.load_from_dataframe(df=expect_output, known_cov_cols=['X', 'X1'], 
                                                        observed_cov_cols=['Y', 'Y1'])
        params = {'cols': ['X', 'X1', 'Y', 'Y1'], 'method': 'next'} 
        ob = eval("Fill")(**params)
        at = ob.fit_transform(ts)
        self.assertTrue(eo.get_known_cov().data.equals(at.get_known_cov().data))
        self.assertTrue(eo.get_observed_cov().data.equals(at.get_observed_cov().data))        

        # test --  zero
        a = [1, 0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0]
        a1 = [1, 0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0]
        b = [1, 0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 0, 14, 15, 16]
        b1 = [1, 0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 0, 14, 15, 16]
        expect_output = pd.DataFrame(columns=["X", "Y", "X1", "Y1"])
        expect_output['X'] = a
        expect_output['Y'] = b
        expect_output['X1'] = a1
        expect_output['Y1'] = b1
        expect_output = expect_output.astype('float')
        eo = TSDataset.load_from_dataframe(df=expect_output, known_cov_cols=['X', 'X1'], 
                                                        observed_cov_cols=['Y', 'Y1'])
        params = {'cols': ['X', 'X1', 'Y', 'Y1'], 'method': 'zero'} 
        ob = eval("Fill")(**params)
        at = ob.fit_transform(ts)
        self.assertTrue(eo.get_known_cov().data.equals(at.get_known_cov().data))
        self.assertTrue(eo.get_observed_cov().data.equals(at.get_observed_cov().data))

        # test --  opt
        a = [1, 3.1415926, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 3.1415926]
        a1 = [1, 3.1415926, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 3.1415926]
        b = [1, 3.1415926, 3, 4, 5, 6, 7, 8, 9, 10, 11, 3.1415926, 3.1415926, 14, 15, 16]
        b1 = [1, 3.1415926, 3, 4, 5, 6, 7, 8, 9, 10, 11, 3.1415926, 3.1415926, 14, 15, 16]
        expect_output = pd.DataFrame(columns=["X", "Y", "X1", "Y1"])
        expect_output['X'] = a
        expect_output['Y'] = b
        expect_output['X1'] = a1
        expect_output['Y1'] = b1
        expect_output = expect_output.astype('float')
        eo = TSDataset.load_from_dataframe(df=expect_output, known_cov_cols=['X', 'X1'], 
                                                        observed_cov_cols=['Y', 'Y1'])
        params = {'cols': ['X', 'X1', 'Y', 'Y1'], 'method': 'default', 'value': '3.1415926'} 
        ob = eval("Fill")(**params)
        at = ob.fit_transform(ts)
        self.assertTrue(eo.get_known_cov().data.equals(at.get_known_cov().data.astype('float')))
        self.assertTrue(eo.get_observed_cov().data.equals(at.get_observed_cov().data.astype('float')))

        # test --  max
        a = [1, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15]
        a1 = [1, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15]
        b = [1, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11, 11, 14, 15, 16]
        b1 = [1, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11, 11, 14, 15, 16]
        expect_output = pd.DataFrame(columns=["X", "Y", "X1", "Y1"])
        expect_output['X'] = a
        expect_output['Y'] = b
        expect_output['X1'] = a1
        expect_output['Y1'] = b1
        expect_output = expect_output.astype('float')
        eo = TSDataset.load_from_dataframe(df=expect_output, known_cov_cols=['X', 'X1'], 
                                                        observed_cov_cols=['Y', 'Y1'])
        params = {'cols': ['X', 'X1', 'Y', 'Y1'], 'method': 'max'} 
        ob = eval("Fill")(**params)
        at = ob.fit_transform(ts)
        self.assertTrue(eo.get_known_cov().data.equals(at.get_known_cov().data))
        self.assertTrue(eo.get_observed_cov().data.equals(at.get_observed_cov().data))

        # test --  min
        a = [1, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 7]
        a1 = [1, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 7]
        b = [1, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 3, 3, 14, 15, 16]
        b1 = [1, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 3, 3, 14, 15, 16]
        expect_output = pd.DataFrame(columns=["X", "Y", "X1", "Y1"])
        expect_output['X'] = a
        expect_output['Y'] = b
        expect_output['X1'] = a1
        expect_output['Y1'] = b1
        expect_output = expect_output.astype('float')
        eo = TSDataset.load_from_dataframe(df=expect_output, known_cov_cols=['X', 'X1'], 
                                                        observed_cov_cols=['Y', 'Y1'])
        params = {'cols': ['X', 'X1', 'Y', 'Y1'], 'method': 'min'} 
        ob = eval("Fill")(**params)
        at = ob.fit_transform(ts)
        self.assertTrue(eo.get_known_cov().data.equals(at.get_known_cov().data))
        self.assertTrue(eo.get_observed_cov().data.equals(at.get_observed_cov().data))

        # test --  avg
        a = [1, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 14.5]
        a1 = [1, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 14.5]
        b = [1, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 10.5, 10.75, 14, 15, 16]
        b1 = [1, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 10.5, 10.75, 14, 15, 16]
        expect_output = pd.DataFrame(columns=["X", "Y", "X1", "Y1"])
        expect_output['X'] = a
        expect_output['Y'] = b
        expect_output['X1'] = a1
        expect_output['Y1'] = b1
        expect_output = expect_output.astype('float')
        eo = TSDataset.load_from_dataframe(df=expect_output, known_cov_cols=['X', 'X1'], 
                                                        observed_cov_cols=['Y', 'Y1'])
        params = {'cols': ['X', 'X1', 'Y', 'Y1'], 'method': 'mean', 'window_size': 3} 
        ob = eval("Fill")(**params)
        at = ob.fit_transform(ts)
        self.assertTrue(eo.get_known_cov().data.equals(at.get_known_cov().data))
        self.assertTrue(eo.get_observed_cov().data.equals(at.get_observed_cov().data))

        # test --  median
        a = [1, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 14.5]
        a1 = [1, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 14.5]
        b = [1, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 10.5, 10.75, 14, 15, 16]
        b1 = [1, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 10.5, 10.75, 14, 15, 16]
        expect_output = pd.DataFrame(columns=["X", "Y", "X1", "Y1"])
        expect_output['X'] = a
        expect_output['Y'] = b
        expect_output['X1'] = a1
        expect_output['Y1'] = b1
        expect_output = expect_output.astype('float')
        eo = TSDataset.load_from_dataframe(df=expect_output, known_cov_cols=['X', 'X1'], 
                                                        observed_cov_cols=['Y', 'Y1'])
        params = {'cols': ['X', 'X1', 'Y', 'Y1'], 'method': 'median', 'window_size': 3} 
        ob = eval("Fill")(**params)
        at = ob.fit_transform(ts)
        self.assertTrue(eo.get_known_cov().data.equals(at.get_known_cov().data))
        self.assertTrue(eo.get_observed_cov().data.equals(at.get_observed_cov().data))
        
        ats= ob.fit_transform([ts, ts])
        self.assertEqual(len(ats), 2)
        for at in ats:
            self.assertTrue(eo.get_known_cov().data.equals(at.get_known_cov().data))
            self.assertTrue(eo.get_observed_cov().data.equals(at.get_observed_cov().data))

        return

if __name__ == "__main__":
    unittest.main()
