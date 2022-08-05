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
from paddlets.analysis import FFT
from paddlets.analysis import STFT
from paddlets.analysis import CWT


class TestFrequencyDomain(TestCase):
    def setUp(self):
        """
        unittest function
        """
        super().setUp()

    def test_FFT(self):
        """
        unittest function
        """
        #case1, illegal data format
        s = [1, 2, 3, 4]
        flag = False
        try: 
            res = FFT().analyze(s)
        except:
            flag = True
        self.assertTrue(flag)
        
        #case2, input data is pd.Series
        s = pd.Series(range(100))
        res = FFT().analyze(s)
        self.assertEqual(res.shape, (50, 3))
        
        #case3, input data contains illegal characters
        s = pd.Series(range(10))
        s[0] = 's'
        flag = False
        try:
            res = FFT().analyze(s)
        except:
            flag = True
        self.assertTrue(flag)
        
        #case4, input data is dataframe
        df = pd.DataFrame(np.array(range(100)).reshape(20, 5))
        res = FFT().analyze(df)
        self.assertEqual(res.shape, (10, 15))
        
        #case5, input data is tsdataset
        cov_cols = ['c%d' % i for i in range(4)] 
        df = pd.DataFrame(np.array(range(100)).reshape(20, 5), columns=['target'] + cov_cols)
        ts = TSDataset.load_from_dataframe(df, target_cols="target", observed_cov_cols=cov_cols)
        res = FFT()(ts, ['target', 'c0', 'c3'])
        self.assertEqual(res.shape, (10, 9))
        
        #case6, set half=False 
        res = FFT(half=False)(ts, ['target', 'c0', 'c3'])
        self.assertEqual(res.shape, (20, 9))
        
        #case7, test fs parameter
        res = FFT(fs=10)(ts, ['target', 'c0', 'c3'])
        expect_target_x = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
        real_target_x = res['target_x'].tolist()
        self.assertEqual(expect_target_x, real_target_x)
        for i in range(len(expect_target_x)):
            self.assertAlmostEqual(expect_target_x[i], real_target_x[i], 4)
        
        #case8, set norm=False
        res = FFT(norm=False)(ts, ['target', 'c0', 'c3'])
        expect_target_amplitude = [950.000000, 319.622661, 161.803399, 110.134463, 85.065081, 70.710678, 61.803399, 56.116312, 52.573111, 50.623256]
        real_target_amplitude = res['target_amplitude'].tolist()
        for i in range(len(expect_target_amplitude)):
            self.assertAlmostEqual(expect_target_amplitude[i], real_target_amplitude[i], 4)

        #case9, test_get_properties()
        res = FFT().get_properties().get("name")
        self.assertEqual(res, "fft")
        
        #case10, test plot
        fft = FFT()
        res = fft(ts, ['target', 'c0', 'c3'])
        plot = fft.plot()
        plot.savefig('/tmp/fft.png')

    
    def test_STFT(self):
        """
        unittest function
        """
        #case1, illegal data format
        s = [1, 2, 3, 4]
        flag = False
        try: 
            res = STFT().analyze(s)
        except:
            flag = True
        self.assertTrue(flag)
        
        #case2, input data is pd.Series,the data length is less than the default nperseg=256
        s = pd.Series(range(100))
        res = STFT().analyze(s)
        self.assertEqual(res['0_t'].shape, (3, ))
        self.assertEqual(res['0_f'].shape, (51, ))
        self.assertEqual(res['0_Zxx'].shape, (51, 3))
        
        #case3, input data is pd.Series,the data length is less than the default nperseg=256 and set nperseg=10
        s = pd.Series(range(100))
        res = STFT(nperseg=10).analyze(s)
        self.assertEqual(res['0_t'].shape, (21, ))
        self.assertEqual(res['0_f'].shape, (6, ))
        self.assertEqual(res['0_Zxx'].shape, (6, 21))
        
        #case4, input data is pd.Series, the data length is greater than the default nperseg=256
        s = pd.Series(range(1000))
        res = STFT().analyze(s)
        self.assertEqual(res['0_t'].shape, (9, ))
        self.assertEqual(res['0_f'].shape, (129, ))
        self.assertEqual(res['0_Zxx'].shape, (129, 9))
        
        #case5, input data is pd.Series, the data length is greater than the default nperseg=256 and set nperseg=10
        s = pd.Series(range(1000))
        res = STFT(nperseg=10).analyze(s)
        self.assertEqual(res['0_t'].shape, (201, ))
        self.assertEqual(res['0_f'].shape, (6, ))
        self.assertEqual(res['0_Zxx'].shape, (6, 201))
        
        #case5, input data is pd.Series, the data length is greater than the default nperseg=256 and set nperseg<0
        s = pd.Series(range(1000))
        flag = False
        try:
            res = STFT(nperseg=-5).analyze(s)
        except:
            flag = True
        self.assertTrue(flag)
        
        #case5, input data is pd.Series, the data length is greater than the default nperseg=256 and set nperseg=10, fs=10
        s = pd.Series(range(1000))
        res = STFT(nperseg=10, fs=10).analyze(s)
        self.assertEqual(res['0_t'].shape, (201, ))
        self.assertEqual(res['0_f'].shape, (6, ))
        self.assertEqual(res['0_Zxx'].shape, (6, 201))
        
        #case8, input data contains illegal characters
        s = pd.Series(range(10))
        s[0] = 's'
        flag = False
        try:
            res = STFT().analyze(s)
        except:
            flag = True
        self.assertTrue(flag)
        
        #case9, input data is dataframe
        df = pd.DataFrame(np.array(range(100)).reshape(20, 5))
        res = STFT().analyze(df)
        self.assertEqual(res['1_t'].shape, (3, ))
        self.assertEqual(res['1_f'].shape, (11, ))
        self.assertEqual(res['1_Zxx'].shape, (11, 3))
        
        #case10, input data is tsdataset
        cov_cols = ['c%d' % i for i in range(4)] 
        df = pd.DataFrame(np.array(range(100)).reshape(20, 5), columns=['target'] + cov_cols)
        ts = TSDataset.load_from_dataframe(df, target_cols="target", observed_cov_cols=cov_cols)
        res = STFT()(ts, ['target', 'c0', 'c3'])
        self.assertEqual(res['target_t'].shape, (3, ))
        self.assertEqual(res['target_f'].shape, (11, ))
        self.assertEqual(res['target_Zxx'].shape, (11, 3))

        #case11, test_get_properties()
        res = STFT().get_properties().get("name")
        self.assertEqual(res, "stft")
        
        #case12, test plot
        stft = STFT()
        res = stft(ts, 'target')
        plot = stft.plot()
        plot.savefig('/tmp/stft.png')
        
    def test_CWT(self):
        """
        unittest function
        """
        #case1, illegal data format
        s = [1, 2, 3, 4]
        flag = False
        try:
            res = CWT().analyze(s)
        except:
            flag = True
        self.assertTrue(flag)
        
        #case2, input data is pd.Series, set scales=1
        s = pd.Series(range(100))
        flag = False
        try: 
            res = CWT(scales=1).analyze(s)
        except:
            flag = True
        self.assertTrue(flag)
        
        #case3, input data is pd.Series,set scale=10
        s = pd.Series(range(100))
        res = CWT(scales=10).analyze(s)
        self.assertEqual(res['0_t'].shape, (100, ))
        self.assertEqual(res['0_coefs'].shape, (9, 100))
        self.assertEqual(res['0_frequencies'].shape, (9, ))
        
        #case4, input data contains illegal characters
        s = pd.Series(range(10))
        flag = False
        s[0] = 's'
        try:
            res = CWT(scales=5).analyze(s)
        except:
            flag = True
        self.assertTrue(flag)
        
        #case5, input data is dataframe
        df = pd.DataFrame(np.array(range(100)).reshape(20, 5))
        res = CWT(scales=20).analyze(df)
        self.assertEqual(res['1_t'].shape, (20, ))
        self.assertEqual(res['1_coefs'].shape, (19, 20))
        self.assertEqual(res['1_frequencies'].shape, (19, ))
        
        #case6, input data is tsdataset
        cov_cols = ['c%d' % i for i in range(4)] 
        df = pd.DataFrame(np.array(range(100)).reshape(20, 5), columns=['target'] + cov_cols)
        ts = TSDataset.load_from_dataframe(df, target_cols="target", observed_cov_cols=cov_cols)
        res = CWT(scales=100)(ts, ['target', 'c0', 'c3'])
        self.assertEqual(res['target_t'].shape, (20, ))
        self.assertEqual(res['target_coefs'].shape, (99, 20))
        self.assertEqual(res['target_frequencies'].shape, (99, ))

        #case7 test_get_properties()
        res = CWT().get_properties().get("name")
        self.assertEqual(res, "cwt")
        
        #case8, test plot
        cwt = CWT(scales=100)
        res = cwt(ts, ['target', 'c0', 'c3'])
        plot = cwt.plot()
        plot.savefig('/tmp/cwt.png')


if __name__ == "__main__":
    unittest.main()
