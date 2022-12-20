# !/usr/bin/env python3
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import os
import sys
import time
import unittest
from unittest import TestCase
from sklearn import preprocessing
from paddlets.transform.sklearn_transforms import MinMaxScaler
from paddlets.transform.sklearn_transforms import StandardScaler
from paddlets.datasets.tsdataset import TimeSeries
from paddlets.datasets.tsdataset import TSDataset


class test_MinMaxScaler(TestCase):
    def setUp(self):
        """
        unittest function
        """
        super().setUp()

    def test_MinMax(self):
        """
        unittest function
        """
        print(sys.stderr, "test_MinMax()...")
        # 测试数据样例
        a = [2, 4, 6, 8]
        b = [10, 8, 6, 4]
        a1 = [1, 2, 3, 4]
        b1 = [5, 4, 3, 2]
        fake_input = pd.DataFrame(columns=["X", "Y", "X1", "Y1"])
        fake_input['X'] = a
        fake_input['Y'] = b
        fake_input['X1'] = a1
        fake_input['Y1'] = b1
        fake_input.index = pd.date_range('2022-01-01', periods=4, freq='1D')
        input1 = TSDataset.load_from_dataframe(df=fake_input, known_cov_cols=['X', 'Y'], 
                                                     observed_cov_cols=['X1', 'Y1'])
        # test -- MinMaxScaler 1
        params1 = params1 = {'f_range': (0, 1), 'clip': False} 
        test_transform1 = eval("MinMaxScaler")(**params1)
        test_transform1.fit(input1)
        transfrom1 = test_transform1.transform(input1)
        inverse1 = test_transform1.inverse_transform(transfrom1)
        fit_trans1 = test_transform1.fit_transform(input1)
        self.assertTrue(inverse1.get_known_cov().data.astype('int').equals(input1.get_known_cov().data.astype('int')))

        # test -- MinMaxScaler 2
        params2 = {'cols': ['X', 'Y'], 'f_range': (0, 1), 'clip': False} 
        test_transform1 = eval("MinMaxScaler")(**params2)
        test_transform1.fit(input1)
        transfrom1 = test_transform1.transform(input1)
        inverse1 = test_transform1.inverse_transform(transfrom1)
        fit_trans1 = test_transform1.fit_transform(input1)
        self.assertTrue(inverse1.get_known_cov().data.astype('int').equals(input1.get_known_cov().data.astype('int')))

        # test -- MinMaxScaler 3
        params2 = {'cols': ['X1', 'Y1'], 'f_range': (0, 1), 'clip': False} 
        test_transform1 = eval("MinMaxScaler")(**params2)
        test_transform1.fit(input1)
        transfrom1 = test_transform1.transform(input1)
        inverse1 = test_transform1.inverse_transform(transfrom1)
        fit_trans1 = test_transform1.fit_transform(input1)
        self.assertTrue(inverse1.get_known_cov().data.astype('int').equals(input1.get_known_cov().data.astype('int')))

        test_transform1.fit([input1, input1])
        transfrom1s = test_transform1.transform([input1, input1])
        self.assertEqual(len(transfrom1s), 2)
        inverse1s = test_transform1.inverse_transform(transfrom1s)
        fit_trans1s = test_transform1.fit_transform([input1, input1])
        self.assertEqual(len(inverse1s), 2)
        self.assertEqual(len(fit_trans1s), 2)
        for inverse1 in inverse1s:
            self.assertTrue(inverse1.get_known_cov().data.astype('int').equals(input1.get_known_cov().data.astype('int')))

    def test_StandardScaler(self):
        """
        unittest function
        """
        print(sys.stderr, "test_StandardScaler()...")

        a = [2, 4, 2, 4, 2]
        b = [1, 2, 1, 2, 1]
        a1 = [1, 1, 1, 1, 1]
        b1 = [2, 2, 2, 2, 2]
        fake_input = pd.DataFrame(columns=["X", "Y", "X1", "Y1"])
        fake_input['X'] = a
        fake_input['Y'] = b
        fake_input['X1'] = a1
        fake_input['Y1'] = b1
        fake_input.index = pd.date_range('2022-01-01', periods=5, freq='1D')
        input1 = TSDataset.load_from_dataframe(df=fake_input, known_cov_cols=['X', 'Y'], 
                                                     observed_cov_cols=['X1', 'Y1'])
        # test -- StandardScaler
        params1 = {'cols': None, 'with_mean': True, 'with_std': True}  
        test_transform1 = eval("StandardScaler")(**params1)
        test_transform1.fit(input1)
        transfrom1 = test_transform1.transform(input1)
        inverse1 = test_transform1.inverse_transform(transfrom1)
        fit_trans1 = test_transform1.fit_transform(input1)
        self.assertTrue(inverse1.get_known_cov().data.astype('int').equals(input1.get_known_cov().data.astype('int')))

        # test -- StandardScaler  2
        params1 = {'cols': ['X', 'Y'], 'with_mean': True, 'with_std': True}  
        test_transform1 = eval("StandardScaler")(**params1)
        test_transform1.fit(input1)
        transfrom1 = test_transform1.transform(input1)
        inverse1 = test_transform1.inverse_transform(transfrom1)
        fit_trans1 = test_transform1.fit_transform(input1)
        self.assertTrue(inverse1.get_known_cov().data.astype('int').equals(input1.get_known_cov().data.astype('int')))

        # test -- StandardScaler  3
        params1 = {'cols': ['X1', 'Y1'], 'with_mean': True, 'with_std': True}  
        test_transform1 = eval("StandardScaler")(**params1)
        test_transform1.fit(input1)
        transfrom1 = test_transform1.transform(input1)
        inverse1 = test_transform1.inverse_transform(transfrom1)
        fit_trans1 = test_transform1.fit_transform(input1)
        self.assertTrue(inverse1.get_known_cov().data.astype('int').equals(input1.get_known_cov().data.astype('int')))

        test_transform1.fit([input1, input1])
        transfrom1s = test_transform1.transform([input1, input1])
        self.assertEqual(len(transfrom1s), 2)
        inverse1s = test_transform1.inverse_transform(transfrom1s)
        fit_trans1s = test_transform1.fit_transform([input1, input1])
        self.assertEqual(len(inverse1s), 2)
        self.assertEqual(len(fit_trans1s), 2)
        for inverse1 in inverse1s:
            self.assertTrue(inverse1.get_known_cov().data.astype('int').equals(input1.get_known_cov().data.astype('int')))

    def test_with_sklearn(self):
        """
        与skearn结果对比
        """
        a = [2, 4, 6, 8, 10, 12]
        b = [100, 80, 60, 40, 20, 10]
        a1 = [-20, -20, -30, -40, -50, -60]
        b1 = [5, 4, 3, 2, 1, 0]
        c = [3, 3, 3, 3, 3, 3]
        fake_input2 = pd.DataFrame(columns=["X", "Y", "X1", "Y1", "C"])
        fake_input2['X'] = a
        fake_input2['Y'] = b
        fake_input2['X1'] = a1
        fake_input2['Y1'] = b1
        fake_input2['C'] = c
        input_ts = TSDataset.load_from_dataframe(df=fake_input2, known_cov_cols=['X', 'X1'], 
                                                            observed_cov_cols=['Y', 'Y1'],
                                                             target_cols='C')
        params1 = {'f_range': (0, 1), 'clip': False} 
        MinMax = eval("MinMaxScaler")(**params1)
        fit_transform_ts = MinMax.fit_transform(input_ts)
        inverse_ts = MinMax.inverse_transform(fit_transform_ts)

        scaler = preprocessing.MinMaxScaler()
        sk_fit_transform = scaler.fit(input_ts.to_numpy()).transform(input_ts.to_numpy())
        np_inverse_sk = scaler.inverse_transform(sk_fit_transform)

        self.assertTrue((inverse_ts.to_numpy().astype('int') == input_ts.to_numpy().astype('int')).all())
        self.assertTrue((fit_transform_ts.to_numpy() == sk_fit_transform).all())
        self.assertTrue((inverse_ts.to_numpy() == np_inverse_sk).all())

        params1 = {'with_mean': True, 'with_std': True} #['X', 'X1', 'Y','Y1']
        Stand = eval("StandardScaler")(**params1)
        fit_transform_ts = Stand.fit_transform(input_ts)
        inverse_ts = Stand.inverse_transform(fit_transform_ts)

        scaler = preprocessing.StandardScaler(with_mean=True, with_std=True)
        sk_fit_transform = scaler.fit(input_ts.to_numpy()).transform(input_ts.to_numpy())
        np_inverse_sk = scaler.inverse_transform(sk_fit_transform)

        self.assertTrue((inverse_ts.to_numpy().astype('int') == input_ts.to_numpy().astype('int')).all())
        self.assertTrue((fit_transform_ts.to_numpy() == sk_fit_transform).all())
        self.assertTrue((inverse_ts.to_numpy() == np_inverse_sk).all())

        
if __name__ == "__main__":
    unittest.main() 