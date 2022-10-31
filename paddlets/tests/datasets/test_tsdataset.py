# !/usr/bin/env python3
# -*- coding:utf-8 -*-
import pandas as pd
from paddlets.datasets import tsdataset
import numpy as np

import os
import sys
import time
import unittest
from unittest import TestCase

from paddlets import TimeSeries, TSDataset


class TestTimeSeries(TestCase):
    def setUp(self):
        """
        unittest function
        """
        super().setUp()

    def test_load_from_dataframe(self):
        """
        unittest function
        """
        #case1
        sample1 = pd.DataFrame(
            np.random.randn(200, 2), 
            index=pd.date_range('2022-01-01', periods=200, freq='1D')
        )
        ts1 = TimeSeries.load_from_dataframe(data=sample1)
        self.assertEqual(ts1.freq, 'D')
        self.assertEqual(ts1.data.shape, (200, 2))
        #case2
        sample1 = pd.DataFrame(
            np.random.randn(200, 2), 
            index=pd.date_range('2022-01-01', periods=200, freq='1D'),
            columns=['a', 'b']
        )
        ts1 = TimeSeries.load_from_dataframe(data=sample1, value_cols='a')
        self.assertEqual(ts1.freq, 'D')
        self.assertEqual(ts1.data.shape, (200, 1))
        #case3
        sample1 = pd.DataFrame(
            np.random.randn(200, 3), 
            index=pd.date_range('2022-01-01', periods=200, freq='1D'),
            columns=['a', 'b', 'c']
        )
        ts1 = TimeSeries.load_from_dataframe(data=sample1, value_cols=['a', 'c'])
        self.assertEqual(ts1.freq, 'D')
        self.assertEqual(ts1.data.shape, (200, 2))
        #case4
        sample1 = pd.DataFrame(
            np.random.randn(200, 3), 
            columns=['a', 'b', 'c']
        )
        ts1 = TimeSeries.load_from_dataframe(data=sample1, value_cols=['a', 'c'])
        self.assertEqual(ts1.freq, 1)
        self.assertEqual(ts1.data.shape, (200, 2))
        #case5
        tmp = pd.Series(pd.date_range('2022-01-01', periods=200, freq='1D')).astype(str)
        sample1 = pd.DataFrame(np.random.randn(200, 3), columns=['a', 'c', 'd'])
        sample1['f'] = tmp
        ts1 = TimeSeries.load_from_dataframe(data=sample1, value_cols=['a', 'c'], time_col='f')
        self.assertEqual(ts1.freq, 'D')
        self.assertEqual(ts1.data.shape, (200, 2))
        #case6
        tmp = pd.Series(pd.date_range('2022-01-01', periods=200, freq='1D')).astype(str)
        sample1 = pd.DataFrame(np.random.randn(200, 3), columns=['a', 'c', 'd'])
        sample1['f'] = tmp
        ts1 = TimeSeries.load_from_dataframe(data=sample1, value_cols=['a'], time_col='f')
        self.assertEqual(ts1.freq, 'D')
        self.assertEqual(ts1.data.shape, (200, 1))
        #case7
        tmp = pd.Series(pd.date_range('2022-01-01', periods=200, freq='1D')).astype(str)
        sample1 = pd.DataFrame(np.random.randn(200, 3), columns=['a', 'c', 'd'])
        sample1['f'] = tmp
        ts1 = TimeSeries.load_from_dataframe(data=sample1, value_cols=['a'], time_col='f', freq='2D')
        self.assertEqual(ts1.freq, '2D')
        self.assertEqual(ts1.data.shape, (100, 1))
        ts1 = TimeSeries.load_from_dataframe(data=sample1, value_cols=['a'], time_col='f', freq='12H')
        self.assertEqual(ts1.freq, '12H')
        self.assertEqual(ts1.data.shape, (399, 1))
        #case8
        sample1 = pd.Series(
            np.random.randn(200), 
            index=pd.date_range('2022-01-01', periods=200, freq='1D'),
            name='a'
        )
        ts1 = TimeSeries.load_from_dataframe(data=sample1)
        self.assertEqual(ts1.freq, 'D')
        self.assertEqual(ts1.data.shape, (200, 1))
        #case9
        sample1 = pd.Series(
            np.random.randn(200), 
            name='a'
        )
        ts1 = TimeSeries.load_from_dataframe(data=sample1)
        self.assertEqual(ts1.freq, 1)
        self.assertEqual(ts1.data.shape, (200, 1))
        sample1 = pd.Series(
            np.random.randn(100),
            index=pd.RangeIndex(start=0, stop=200, step=2),
            name='a'
        )
        ts1 = TimeSeries.load_from_dataframe(data=sample1, freq=2)
        self.assertEqual(ts1.freq, 2)
        self.assertEqual(ts1.data.shape, (100, 1))
        self.assertEqual(ts1.dtypes['a'], np.float64)

        ts1 = TimeSeries.load_from_dataframe(data=sample1, freq=2, dtype=np.int64)
        self.assertEqual(ts1.freq, 2)
        self.assertEqual(ts1.data.shape, (100, 1))
        self.assertEqual(ts1.dtypes['a'], np.int64)

        ts1 = TimeSeries.load_from_dataframe(data=sample1, freq=2, dtype={'a': np.int64})
        self.assertEqual(ts1.freq, 2)
        self.assertEqual(ts1.data.shape, (100, 1))
        self.assertEqual(ts1.dtypes['a'], np.int64)

        ts1.to_numeric()
        self.assertEqual(ts1.dtypes['a'], np.float32)
        ts1.to_categorical()
        self.assertEqual(ts1.dtypes['a'], np.int64)
        
        ts1.to_numeric('a')
        self.assertEqual(ts1.dtypes['a'], np.float32)
        ts1.to_categorical(['a'])
        self.assertEqual(ts1.dtypes['a'], np.int64)

    def test_property(self):
        """
        unittest function
        """
        sample1 = pd.DataFrame(
            np.random.randn(200, 2), 
            index=pd.date_range('2022-01-01', periods=200, freq='1D'),
            columns=['a', 'b']
        )
        ts1 = TimeSeries.load_from_dataframe(data=sample1)
        self.assertEqual(list(ts1.time_index), 
                         list(pd.date_range('2022-01-01', periods=200, freq='1D')))
        self.assertEqual(list(ts1.columns), ['a', 'b'])
        self.assertEqual(ts1.start_time, pd.Timestamp('2022-01-01'))
        self.assertEqual(ts1.end_time, pd.Timestamp('2022-07-19'))
        
        #to_XXX add in property test
        self.assertEqual(type(ts1.to_dataframe()), pd.DataFrame)
        self.assertEqual(ts1.to_dataframe().shape, (200, 2))
        self.assertEqual(id(ts1.to_dataframe(copy=False)), id(ts1.data))
        self.assertTrue(id(ts1.to_dataframe()) != id(ts1.data))

        self.assertEqual(type(ts1.to_numpy()), np.ndarray)
        self.assertEqual(ts1.to_numpy().shape, (200, 2))
        #self.assertEqual(id(ts1.to_numpy(copy=False)[0][0]), id(ts1.data.iloc[0, 0]))
        #self.assertTrue(id(ts1.to_numpy()[0][0]) != id(ts1.data.iloc[0, 0]))

        #test dtypes
        self.assertEqual(ts1.dtypes.shape, (2, ))
        self.assertEqual(len(ts1), 200)

        self.assertEqual(repr(ts1.data), repr(ts1))
        self.assertEqual(str(ts1.data), str(ts1))

    def test_split(self):
        """
        unittest function
        """
        sample1 = pd.DataFrame(
            np.random.randn(200, 2), 
            index=pd.date_range('2022-01-01', periods=200, freq='1D'),
            columns=['a', 'b']
        )
        ts1 = TimeSeries.load_from_dataframe(data=sample1)
        self.assertEqual(ts1.get_index_at_point("2022-01-08"), 7)
        self.assertEqual(ts1.get_index_at_point("2022-01-08 12:45:23"), 8)
        self.assertEqual(ts1.get_index_at_point("2022-01-08 12:45:23", after=False), 7)
        try:
            ts1.get_index_at_point('2021-01-08')
        except ValueError as e:
            self.assertEqual(str(e), 'The `point` is out of the valid range.')
        self.assertEqual(ts1.get_index_at_point(10), 10)
        try:
            ts1.get_index_at_point(1000)
        except ValueError as e:
            self.assertEqual(str(e), '`point` (int) should be a valid index in series.')
        self.assertEqual(ts1.get_index_at_point(0.8), 159)

        ts2, ts3 = ts1.split('2022-01-08')
        self.assertEqual(ts2.data.shape, (8, 2))
        self.assertEqual(ts3.data.shape, (192, 2))
        ts2, ts3 = ts1.split(0.8)
        self.assertEqual(ts2.data.shape, (160, 2))
        self.assertEqual(ts3.data.shape, (40, 2))
        ts2, ts3 = ts1.split(100)
        self.assertEqual(ts2.data.shape, (100, 2))
        self.assertEqual(ts3.data.shape, (100, 2))
        ts2, ts3 = ts1.split('2022-01-08 11:00:00')
        self.assertEqual(ts2.data.shape, (9, 2))
        self.assertEqual(ts3.data.shape, (191, 2))
        ts2, ts3 = ts1.split('2022-01-08 11:00:00', after=False)
        self.assertEqual(ts2.data.shape, (8, 2))
        self.assertEqual(ts3.data.shape, (192, 2)) 

    def test_copy(self):
        """
        unittest function
        """
        sample1 = pd.DataFrame(
            np.random.randn(200, 2), 
            index=pd.date_range('2022-01-01', periods=200, freq='1D'),
            columns=['a', 'b']
        )
        ts1 = TimeSeries.load_from_dataframe(data=sample1)
        ts2 = ts1
        self.assertTrue(id(ts1) == id(ts2))
        self.assertTrue(id(ts1.data) == id(ts2.data))        
        ts2 = ts1.copy()
        self.assertTrue(id(ts1) != id(ts2))
        self.assertTrue(id(ts1.data) != id(ts2.data))

    def test_getitem(self):
        """
        unittest function
        """
        sample1 = pd.DataFrame(
            np.random.randn(200, 2),
            index=pd.date_range('2022-01-01', periods=200, freq='1D'),
            columns=['a', 'b']
        )
        ts1 = TimeSeries.load_from_dataframe(data=sample1)

        sample2 = pd.DataFrame(
            np.random.randn(200, 2),
            columns=['a', 'b']
        )
        ts2 = TimeSeries.load_from_dataframe(data=sample2)

        # case1 TimeIndex
        index = pd.date_range('2022-01-02', periods=20, freq='1D')
        res = ts1[index]
        assert len(res) == 20
        assert isinstance(res, TimeSeries)

        # case2 TimeIndex
        index = pd.date_range('2022-01-02', periods=20, freq='2D')
        res = ts1[index]
        assert len(res) == 20
        assert isinstance(res, TimeSeries)
        assert res.freq == index.freq

        # case3 RangeIndex
        index = pd.RangeIndex(0, 20, 2)
        res = ts2[index]
        assert len(res) == 10
        assert isinstance(res, TimeSeries)
        assert res.freq == ts2.freq * 2

        # case4 slice
        res = ts1[10:150]
        assert len(res) == 140
        assert isinstance(res, TimeSeries)

        # case5 badcase
        with self.assertRaises(ValueError):
            res = ts1["bad"]

    def test_concat(self):
        """
        unittest function
        """
        #case1 test axis=1
        sample1 = pd.DataFrame(
            np.random.randn(200, 2), 
            index=pd.date_range('2022-01-01', periods=200, freq='1D'),
            columns=['a', 'b']
        )
        ts1 = TimeSeries.load_from_dataframe(data=sample1)
        sample2 = pd.DataFrame(
            np.random.randn(200, 2), 
            index=pd.date_range('2022-01-01', periods=200, freq='1D'),
            columns=['d', 'e']
        )
        ts2 = TimeSeries.load_from_dataframe(data=sample2)
        ts3 = TimeSeries.concat([ts1, ts2], axis=1)
        self.assertEqual(ts3.data.shape, (200, 4))
            
        #drop duplicated
        ts3 = TimeSeries.load_from_dataframe(data=sample2)
        ts4 = TimeSeries.concat([ts2, ts3], axis=1, drop_duplicates=True)  
        self.assertEqual(ts3.data.shape, (200, 2))
        ts4 = TimeSeries.concat([ts2, ts3], axis=1, drop_duplicates=True,keep='last')
        self.assertEqual(ts3.data.shape, (200, 2))

        #case2 test axis=0
        sample2 = pd.DataFrame(
            np.random.randn(200, 2), 
            index=pd.date_range('2022-07-20', periods=200, freq='1D'),
            columns=['a', 'b']
        )
        ts2 = TimeSeries.load_from_dataframe(data=sample2)  
        ts3 = TimeSeries.concat([ts1, ts2]) #default axis=0
        self.assertEqual(ts3.data.shape, (400, 2))
        sample2 = pd.DataFrame(
            np.random.randn(200, 2), 
            index=pd.date_range('2022-07-30', periods=200, freq='1D'),
            columns=['a', 'b']
        )
        ts2 = TimeSeries.load_from_dataframe(data=sample2)  
        ts3 = TimeSeries.concat([ts1, ts2]) #default axis=0
        self.assertEqual(ts3.data.shape, (410, 2))

        #drop duplicated
        ts2, ts3 = ts1.split(100)
        ts4 = TimeSeries.concat([ts2, ts3], drop_duplicates=True) 
        self.assertEqual(ts4.data.shape, (200, 2))

        ts2, ts3 = ts1.split(100)
        ts4 = TimeSeries.concat([ts2, ts3], drop_duplicates=True,keep='last')   
        self.assertEqual(ts4.data.shape, (200, 2))

    def test_astype(self):
        """
        unittest function
        """
        sample1 = pd.DataFrame(
            np.random.randn(200, 3), 
            index=pd.date_range('2022-01-01', periods=200, freq='1D'),
            columns=['a', 'b', 'c']
        )
        ts1 = TimeSeries.load_from_dataframe(data=sample1)
        ts2 = ts1.copy()
        ts2.astype('float32')
        self.assertEqual(ts2.data.dtypes['a'], 'float32')
        self.assertEqual(ts2.data.dtypes['b'], 'float32')
        self.assertEqual(ts2.data.dtypes['c'], 'float32')
        ts2 = ts1.copy()
        ts2.astype({'a': 'float32'})
        self.assertEqual(ts2.data.dtypes['a'], 'float32')
        self.assertEqual(ts2.data.dtypes['b'], 'float64')
        self.assertEqual(ts2.data.dtypes['c'], 'float64')

    def test_drop_tail_nan(self):
        """
        unittest function
        """
        sample1 = pd.DataFrame(
            np.random.randn(200, 3), 
            index=pd.date_range('2022-01-01', periods=200, freq='1D'),
            columns=['a', 'b', 'c']
        )
        ts1 = TimeSeries.load_from_dataframe(data=sample1)
        ts1.data.iloc[100:] = np.nan
        self.assertEqual(ts1._find_end_index(), 99)
        ts1.drop_tail_nan()
        self.assertEqual(ts1.data.shape, (100, 3))

        sample1 = pd.DataFrame(
            np.random.randn(200, 3), 
            index=pd.date_range('2022-01-01', periods=200, freq='1D'),
            columns=['a', 'b', 'c']
        )
        sample1.iloc[100:] = np.nan
        ts1 = TimeSeries.load_from_dataframe(data=sample1, drop_tail_nan=True)
        self.assertEqual(ts1.data.shape, (100, 3))
        ts1 = TimeSeries.load_from_dataframe(data=sample1)
        self.assertEqual(ts1.data.shape, (200, 3))
        #All nan test
        sample1.iloc[:] = np.nan
        ts1 = TimeSeries.load_from_dataframe(data=sample1, drop_tail_nan=True)
        self.assertEqual(ts1.data.shape, (0, 3))
        ts1 = TimeSeries.load_from_dataframe(data=sample1)
        self.assertEqual(ts1.data.shape, (200, 3))

    def test_json(self):
        """
        unittest function
        """
        sample1 = pd.DataFrame(
            np.random.randn(200, 3), 
            index=pd.date_range('2022-01-01', periods=200, freq='1D'),
            columns=['a', 'b', 'c']
        )
        ts1 = TimeSeries.load_from_dataframe(data=sample1)
        json_data = ts1.to_json()
        ts2 = TimeSeries.load_from_json(json_data)
        self.assertEqual(ts1.data.shape, ts2.data.shape)
        self.assertTrue((ts1.time_index == ts2.time_index).all())

        sample1 = pd.DataFrame(
            np.random.randn(200, 3), 
            columns=['a', 'b', 'c']
        )
        ts1 = TimeSeries.load_from_dataframe(data=sample1)
        json_data = ts1.to_json()
        ts2 = TimeSeries.load_from_json(json_data)
        self.assertEqual(ts1.data.shape, ts2.data.shape)
        self.assertTrue((ts1.time_index == ts2.time_index).all())

        sample1 = pd.DataFrame(
            np.random.randn(200, 3), 
            columns=['a', 'b', '测试列']
        )
        ts1 = TimeSeries.load_from_dataframe(data=sample1)
        json_data = ts1.to_json()
        ts2 = TimeSeries.load_from_json(json_data)
        self.assertEqual(ts1.data.shape, ts2.data.shape)
        self.assertTrue((ts1.time_index == ts2.time_index).all())
        self.assertTrue((ts1.columns == ts2.columns).all())


class TestTSDataset(TestCase): 
    def setUp(self):
        """
        unittest function
        """
        sample1 = pd.Series(
            np.random.randn(200), 
            index=pd.date_range('2022-01-01', periods=200, freq='1D'),
            name='a'
        )
        self.target = TimeSeries.load_from_dataframe(data=sample1)
        
        sample1 = pd.DataFrame(
            np.random.randn(200, 2), 
            index=pd.date_range('2022-01-01', periods=200, freq='1D'),
            columns=['b', 'c']
        )
        self.observed_cov = TimeSeries.load_from_dataframe(data=sample1)
        
        sample1 = pd.DataFrame(
            np.random.randn(200, 2), 
            index=pd.date_range('2022-01-01', periods=200, freq='1D'),
            columns=['b1', 'c1']
        )
        self.known_cov = TimeSeries.load_from_dataframe(data=sample1)
       
        self.static_cov = {'f': 1, 'g': 2}
        super().setUp()

    def test_init(self):
        """
        unittest function
        """
        #case1
        tsdataset = TSDataset(self.target, self.observed_cov, self.known_cov, self.static_cov) 
        self.assertEqual(tsdataset.get_target().data.shape, (200, 1))
        self.assertEqual(tsdataset.get_observed_cov().data.shape, (200, 2))
        self.assertEqual(tsdataset.get_known_cov().data.shape, (200, 2))
        self.assertEqual(tsdataset.get_static_cov(), {'f': 1, 'g': 2})

    def test_load_from_dataframe(self):
        """
        unittest function
        """
        #case1
        sample1 = pd.DataFrame(
            np.random.randn(200, 5), 
            index=pd.date_range('2022-01-01', periods=200, freq='1D'),
            columns=['a', 'b', 'c', 'd', 'e']
        )
        tsdataset = TSDataset.load_from_dataframe(df=sample1)
        self.assertEqual(tsdataset.get_target().data.shape, (200, 5))
        self.assertEqual(tsdataset.get_observed_cov(), None)
        #case2
        sample1['s'] = pd.Series([1 for i in range(200)], index=pd.date_range('2022-01-01', periods=200, freq='1D'))
        tsdataset = TSDataset.load_from_dataframe(
            df=sample1, 
            target_cols='a', 
            observed_cov_cols=['b', 'c', 'd'], 
            known_cov_cols='e', 
            static_cov_cols='s'
        )
        self.assertEqual(tsdataset.get_target().data.shape, (200, 1))
        self.assertEqual(tsdataset.get_observed_cov().data.shape, (200, 3))
        self.assertEqual(tsdataset.get_known_cov().data.shape, (200, 1))
        self.assertEqual(tsdataset.get_static_cov(), {'s': 1})
        #case3
        tmp = pd.Series(pd.date_range('2022-01-01', periods=200, freq='1D')).astype(str)
        sample1 = pd.DataFrame(np.random.randn(200, 3), columns=['a', 'c', 'd'])
        sample1['f'] = tmp
        sample1['s'] = pd.Series([1 for i in range(200)])
        tsdataset = TSDataset.load_from_dataframe(
            df=sample1, 
            target_cols='a', 
            observed_cov_cols=['c', 'd'], 
            static_cov_cols='s', 
            time_col='f'
        )
        self.assertEqual(tsdataset.get_target().data.shape, (200, 1))
        self.assertEqual(tsdataset.get_observed_cov().data.shape, (200, 2))
        self.assertEqual(tsdataset.get_known_cov(), None)
        self.assertEqual(tsdataset.get_static_cov(), {'s': 1})
        #case4 测试缺失值自动填充功能
        tsdataset = TSDataset.load_from_dataframe(
            df=sample1, 
            target_cols='a', 
            observed_cov_cols=['c', 'd'], 
            static_cov_cols='s', 
            time_col='f',
            freq = '12H',
            fill_missing_dates=True
        )
        self.assertEqual(tsdataset.get_all_cov().data.shape, (399, 2))
        self.assertEqual(tsdataset.get_target().data.shape, (399, 1))
        self.assertFalse(tsdataset.get_all_cov().data.isna().values.any())
        self.assertFalse(tsdataset.get_target().data.isna().values.any())
        tsdataset = TSDataset.load_from_dataframe(
            df=sample1, 
            target_cols='a', 
            observed_cov_cols=['c', 'd'], 
            static_cov_cols='s', 
            time_col='f',
            freq = '12H',
            fill_missing_dates=True,
            fillna_method='max'
        )
        self.assertEqual(tsdataset.get_all_cov().data.shape, (399, 2))
        self.assertEqual(tsdataset.get_target().data.shape, (399, 1))
        self.assertFalse(tsdataset.get_all_cov().data.isna().values.any())
        self.assertFalse(tsdataset.get_target().data.isna().values.any())
        tsdataset = TSDataset.load_from_dataframe(
            df=sample1, 
            label_col='a', 
            feature_cols=['c', 'd'], 
            static_cov_cols='s', 
            time_col='f',
            freq = '12H',
            fill_missing_dates=True,
            fillna_method='max'
        )
        self.assertEqual(tsdataset.get_all_cov().data.shape, (399, 2))
        self.assertEqual(tsdataset.get_target().data.shape, (399, 1))
        self.assertFalse(tsdataset.get_all_cov().data.isna().values.any())
        self.assertFalse(tsdataset.get_target().data.isna().values.any())
        self.assertEqual(tsdataset.get_feature().data.shape, (399, 2))
        self.assertEqual(tsdataset.get_label().data.shape, (399, 1))
        self.assertFalse(tsdataset.get_feature().data.isna().values.any())
        self.assertFalse(tsdataset.get_label().data.isna().values.any())

        #load multi time series
        sample1['id'] = pd.Series(
            [0]*100 + [1]*100, 
            name='id'
        )
        tsdatasets = TSDataset.load_from_dataframe(
            df=sample1,
            group_id='id',
            label_col='a', 
            feature_cols=['c', 'd'], 
            static_cov_cols='s', 
            time_col='f',
            freq = '12H',
            fill_missing_dates=True,
            fillna_method='max'
        )
        self.assertEqual(len(tsdatasets), 2)
        for tsdataset in tsdatasets:
            self.assertEqual(tsdataset.get_all_cov().data.shape, (199, 2))
            self.assertEqual(tsdataset.get_target().data.shape, (199, 1))
            self.assertFalse(tsdataset.get_all_cov().data.isna().values.any())
            self.assertFalse(tsdataset.get_target().data.isna().values.any())
            self.assertEqual(tsdataset.get_feature().data.shape, (199, 2))
            self.assertEqual(tsdataset.get_label().data.shape, (199, 1))
            self.assertFalse(tsdataset.get_feature().data.isna().values.any())
            self.assertFalse(tsdataset.get_label().data.isna().values.any())

        tsdatasets = TSDataset.load_from_dataframe(
            df=sample1,
            group_id='id',
            label_col='a', 
            feature_cols=['c', 'd'], 
            static_cov_cols=['s', 'id'], 
            time_col='f',
            freq = '12H',
            fill_missing_dates=True,
            fillna_method='max'
        )
        self.assertEqual(len(tsdatasets), 2)
        for tsdataset in tsdatasets:
            self.assertEqual(list(tsdataset.get_static_cov().keys()), ['s', 'id'])
            self.assertEqual(tsdataset.dtypes['a'], np.float64)

        with self.assertRaises(ValueError):
            tsdatasets = TSDataset.load_from_dataframe(
                df=sample1,
                group_id='id',
                label_col=['a', 'c'], 
                feature_cols=['d'], 
                static_cov_cols='s', 
                time_col='f',
                freq = '12H',
                fill_missing_dates=True,
                fillna_method='max'
            )

        tsdataset = TSDataset.load_from_dataframe(
            df=sample1,
            label_col='a', 
            feature_cols=['c', 'd'], 
            static_cov_cols=['s'], 
            time_col='f',
            freq = '12H',
            fill_missing_dates=True,
            fillna_method='max',
            dtype=np.int64
        )
        self.assertEqual(tsdataset.dtypes['a'], np.int64)
        self.assertEqual(tsdataset.dtypes['c'], np.int64)

        tsdataset = TSDataset.load_from_dataframe(
            df=sample1,
            label_col='a', 
            feature_cols=['c', 'd'], 
            static_cov_cols=['s'], 
            time_col='f',
            freq = '12H',
            fill_missing_dates=True,
            fillna_method='max',
            dtype={'a': np.int64}
        )
        self.assertEqual(tsdataset.dtypes['a'], np.int64)
        self.assertEqual(tsdataset.dtypes['c'], np.float64)

        tsdataset.to_categorical()
        self.assertEqual(tsdataset.dtypes['c'], np.int64)
        tsdataset.to_numeric()
        self.assertEqual(tsdataset.dtypes['a'], np.float32)

        tsdataset.to_categorical('c')
        self.assertEqual(tsdataset.dtypes['c'], np.int64)
        tsdataset.to_numeric(['a', 'c'])
        self.assertEqual(tsdataset.dtypes['c'], np.float32)

        dataset = pd.DataFrame()
        dataset['time'] = pd.Series(['2022-01-04', '2022-01-03', '2022-01-02', '2022-01-01'])
        dataset['target'] = pd.Series([1, 2, 3, 4])
        tsdataset = TSDataset.load_from_dataframe(
            dataset,
            time_col='time',
            target_cols='target'
        )
        self.assertEqual(tsdataset.target.data.shape, (4, 1))

    def test_load_from_csv(self):
        """
        unittest function
        """
        #case1
        tmp = pd.Series(pd.date_range('2022-01-01', periods=200, freq='1D')).astype(str)
        sample1 = pd.DataFrame(np.random.randn(200, 3), columns=['a', 'c', 'd'])
        sample1['f'] = tmp
        sample1['s'] = pd.Series([1 for i in range(200)])
        sample1.to_csv("/tmp/sample1.csv")
        tsdataset = TSDataset.load_from_csv(
            "/tmp/sample1.csv",
            target_cols='a', 
            observed_cov_cols=['c', 'd'], 
            static_cov_cols='s', 
            time_col='f'
        )
        self.assertEqual(tsdataset.get_target().data.shape, (200, 1))
        self.assertEqual(tsdataset.get_observed_cov().data.shape, (200, 2))
        self.assertEqual(tsdataset.get_known_cov(), None)
        self.assertEqual(tsdataset.get_static_cov(), {'s': 1})
        self.assertTrue((tsdataset.get_target().time_index == \
            pd.date_range('2022-01-01', periods=200, freq='1D')).all())
        #case2
        tsdataset = TSDataset.load_from_csv(
            "/tmp/sample1.csv",
            target_cols='a', 
            observed_cov_cols=['c', 'd'], 
            static_cov_cols='s', 
            known_cov_cols='f'
        )
        self.assertEqual(tsdataset.get_target().data.shape, (200, 1))
        self.assertEqual(tsdataset.get_observed_cov().data.shape, (200, 2))
        self.assertEqual(tsdataset.get_known_cov().data.shape, (200, 1))
        self.assertEqual(tsdataset.get_static_cov(), {'s': 1})
        self.assertTrue((tsdataset.get_target().time_index == \
            pd.RangeIndex(start=0, stop=200, step=1)).all())
        #case3
        tsdataset = TSDataset.load_from_csv(
            "/tmp/sample1.csv",
            target_cols='a', 
            observed_cov_cols=['c', 'd'], 
            static_cov_cols='s', 
            time_col='f',
            dtype=np.int64,
        )
        self.assertEqual(tsdataset.dtypes['a'], 'int64')


    def test_get_property(self):
        """
        unittest function
        """
        tsdataset = TSDataset(self.target, self.observed_cov, self.known_cov, self.static_cov)
        observed_cov = tsdataset.get_observed_cov()
        observed_cov.data['s_a'] = pd.Series(
            [1 for i in range(200)], 
            index=pd.date_range('2022-01-01', periods=200, freq='1D')
        )
        self.assertEqual(tsdataset.get_observed_cov().data.shape, (200, 3))
        all_cov = tsdataset.get_all_cov()
        self.assertEqual(all_cov.data.shape, (200, 5))

        df = tsdataset.to_dataframe()
        self.assertTrue(isinstance(df, pd.DataFrame))
        self.assertEqual(df.shape, (200, 6))
        ndarray = tsdataset.to_numpy()
        self.assertTrue(isinstance(ndarray, np.ndarray))
        self.assertEqual(ndarray.shape, (200, 6))
        self.assertEqual(tsdataset.label.data.shape, (200, 1))
        self.assertEqual(tsdataset.feature.data.shape, (200, 3))

    def test_set_property(self):
        """
        unittest function
        """
        tsdataset = TSDataset(self.target)
        tsdataset.set_known_cov(self.known_cov)
        self.assertEqual(tsdataset.get_known_cov().data.shape, (200, 2))
        tsdataset.set_observed_cov(self.observed_cov)
        self.assertEqual(tsdataset.get_observed_cov().data.shape, (200, 2))
        tsdataset.set_static_cov(self.static_cov)
        self.assertEqual(tsdataset.get_static_cov(), {'f': 1, 'g': 2})
        tsdataset.set_static_cov({'f': 2, 'e': 4})
        self.assertEqual(tsdataset.get_static_cov(), {'f': 2, 'g': 2, 'e': 4})
        tsdataset.set_static_cov({'f': 2, 'e': 4}, append=False)
        self.assertEqual(tsdataset.get_static_cov(), {'f': 2, 'e': 4})
    
        #test property and setter
        tsdataset.target = None
        self.assertEqual(tsdataset.target, None)
        tsdataset.target = self.target
        self.assertEqual(tsdataset.target.data.shape, (200, 1))
        tsdataset.known_cov = None
        self.assertEqual(tsdataset.known_cov, None)
        tsdataset.known_cov = self.known_cov
        self.assertEqual(tsdataset.known_cov.data.shape, (200, 2))
        tsdataset.observed_cov = None
        self.assertEqual(tsdataset.observed_cov, None)
        tsdataset.observed_cov = self.observed_cov
        self.assertEqual(tsdataset.observed_cov.data.shape, (200, 2))
        tsdataset.static_cov = None
        self.assertEqual(tsdataset.static_cov, None)
        tsdataset.static_cov = {'f': 2, 'e': 4}
        self.assertEqual(tsdataset.static_cov, {'f': 2, 'e': 4})

        tsdataset.label = self.target
        self.assertEqual(tsdataset.label.data.shape, (200, 1))
        tsdataset.feature = self.observed_cov
        self.assertEqual(tsdataset.feature.data.shape, (200, 2))

    def test_split(self):
        """
        unittest function
        """
        tsdataset = TSDataset(self.target, self.observed_cov, self.known_cov, self.static_cov)
        train, test = tsdataset.split('2022-01-08')
        self.assertEqual(train.get_target().data.shape, (8, 1))
        self.assertEqual(train.get_observed_cov().data.shape, (8, 2))
        self.assertEqual(train.get_known_cov().data.shape, (200, 2))
        self.assertEqual(train.get_static_cov(), {'f': 1, 'g': 2})
        self.assertEqual(test.get_target().data.shape, (192, 1))
        self.assertEqual(test.get_observed_cov().data.shape, (192, 2))
        self.assertEqual(test.get_known_cov().data.shape, (192, 2))
        self.assertEqual(test.get_static_cov(), {'f': 1, 'g': 2})
    
        start_num = 120
        train, test = tsdataset.split(start_num)
        self.assertEqual(train.get_target().data.shape, (start_num, 1))
        self.assertEqual(train.get_observed_cov().data.shape, (start_num, 2))
        self.assertEqual(train.get_known_cov().data.shape, (200, 2))
        self.assertEqual(train.get_static_cov(), {'f': 1, 'g': 2})
        self.assertEqual(test.get_target().data.shape, (200 - start_num, 1))
        self.assertEqual(test.get_observed_cov().data.shape, (200 - start_num, 2))
        self.assertEqual(test.get_known_cov().data.shape, (200 - start_num, 2))
        self.assertEqual(test.get_static_cov(), {'f': 1, 'g': 2})

        start_ratio = 0.8
        train, test = tsdataset.split(start_ratio)
        self.assertEqual(train.get_target().data.shape, (200 * start_ratio, 1))
        self.assertEqual(train.get_observed_cov().data.shape, (200 * start_ratio, 2))
        self.assertEqual(train.get_known_cov().data.shape, (200, 2))
        self.assertEqual(train.get_static_cov(), {'f': 1, 'g': 2})
        self.assertEqual(test.get_target().data.shape, (40, 1))
        self.assertEqual(test.get_observed_cov().data.shape, (40, 2))
        self.assertEqual(test.get_known_cov().data.shape, (40, 2))
        self.assertEqual(test.get_static_cov(), {'f': 1, 'g': 2})
    
        sample1 = pd.DataFrame(
            np.random.randn(200, 3), 
            columns=['a', 'b', 'c']
        )
        tsdataset = TSDataset.load_from_dataframe(sample1, target_cols='a', observed_cov_cols='b', known_cov_cols='c')
        start_num = 120
        train, test = tsdataset.split(start_num)
        self.assertEqual(train.get_target().data.shape, (start_num, 1))
        self.assertEqual(train.get_observed_cov().data.shape, (start_num, 1))
        self.assertEqual(train.get_known_cov().data.shape, (200, 1))
        self.assertEqual(test.get_target().data.shape, (200 - start_num, 1))
        self.assertEqual(test.get_observed_cov().data.shape, (200 - start_num, 1))
        self.assertEqual(test.get_known_cov().data.shape, (200 - start_num, 1))

        tsdataset = TSDataset(None, self.observed_cov, None, self.static_cov)
        start_num = 120
        train, test = tsdataset.split(start_num)
        self.assertEqual(train.get_target(), None)
        self.assertEqual(train.get_observed_cov().data.shape, (start_num, 2))
        self.assertEqual(train.get_known_cov(), None)
        self.assertEqual(test.get_observed_cov().data.shape, (200 - start_num, 2))

        start_ratio = 0.8
        train, test = tsdataset.split(start_ratio)
        self.assertEqual(train.get_observed_cov().data.shape, (200 * start_ratio, 2))
        self.assertEqual(test.get_observed_cov().data.shape, (40, 2))
        self.assertEqual(test.get_static_cov(), {'f': 1, 'g': 2})

        train, test = tsdataset.split('2022-01-08')
        self.assertEqual(train.get_observed_cov().data.shape, (8, 2))
        self.assertEqual(test.get_observed_cov().data.shape, (192, 2))
        self.assertEqual(test.get_static_cov(), {'f': 1, 'g': 2})

        tsdataset = TSDataset(None, self.observed_cov, self.known_cov, self.static_cov)
        start_num = 120
        with self.assertRaises(ValueError):
            train, test = tsdataset.split(start_num)

    def test_copy(self):
        """
        unittest function
        """
        ts1 = TSDataset(self.target, self.observed_cov, self.known_cov, self.static_cov)
        ts2 = ts1
        self.assertTrue(id(ts1) == id(ts2))
        self.assertTrue(id(ts1.get_target()) == id(ts2.get_target()))        
        ts2 = ts1.copy()
        self.assertTrue(id(ts1) != id(ts2))
        self.assertTrue(id(ts1.get_target()) != id(ts2.get_target()))

    def test_get_item(self):
        """
        unittest function
        """
        ts1 = TSDataset(self.target, self.observed_cov, self.known_cov, self.static_cov)
        self.assertEqual(ts1.get_item_from_column('a'), ts1.get_target())
        self.assertEqual(ts1.get_item_from_column('c'), ts1.get_observed_cov())
        self.assertEqual(ts1.get_item_from_column('c1'), ts1.get_known_cov())
        self.assertEqual(ts1.get_item_from_column('g'), ts1.get_static_cov())

        self.assertEqual(ts1['c'].shape, ts1.get_observed_cov().data['c'].shape)
        self.assertEqual(list(ts1[['a', 'c', 'c1', 'g']].columns), ['a', 'c', 'c1', 'g'])
        self.assertEqual(list(ts1[['a', 'c1', 'c', 'g']].columns), ['a', 'c1', 'c', 'g'])
        self.assertEqual(list(ts1[['g', 'c1', 'c', 'a']].columns), ['g', 'c1', 'c', 'a'])
        self.assertEqual(ts1[['a', 'c', 'c1', 'g']].shape, (200, 4))
        error = None
        try:
            ts1['g1_not_exists'] 
        except ValueError as e:
            error = str(e)
        self.assertNotEqual(error, None)

    def test_drop(self):
        """
        unittest function
        """
        ts1 = TSDataset(self.target, self.observed_cov, self.known_cov, self.static_cov)
        ts1.drop('a')
        self.assertEqual(ts1.get_target(), None)
        ts1.drop('b')
        self.assertEqual(ts1.get_observed_cov().data.shape, (200, 1))
        ts1.drop(['b1', 'c1'])
        self.assertEqual(ts1.get_known_cov(), None)
        ts1.drop('g')
        self.assertEqual(ts1.get_static_cov(), {'f': 1})

    def test_set_column(self):
        """
        unittest function
        """
        sample1 = pd.Series(
            np.random.randn(200), 
            index=pd.date_range('2022-01-01', periods=200, freq='1D'),
            name='a'
        )    
        ts1 = TSDataset(target=TimeSeries.load_from_dataframe(sample1))
        sample2 = sample1 + 1
        ts1.set_column('b', sample2, 'target')
        self.assertEqual(ts1.get_target().data.shape, (200, 2))
        ts1.set_column('b', sample1)
        self.assertEqual(ts1.get_target().data.shape, (200, 2))

        ts1.set_column('c', sample1, 'known_cov')
        self.assertEqual(ts1.get_known_cov().data.shape, (200, 1))
        ts1.set_column('c1', sample1, 'known_cov')
        self.assertEqual(ts1.get_known_cov().data.shape, (200, 2))
        ts1.set_column('c1', sample2, 'known_cov')
        self.assertEqual(ts1.get_known_cov().data.shape, (200, 2))

        ts1.set_column('e', sample1, 'observed_cov')
        self.assertEqual(ts1.get_observed_cov().data.shape, (200, 1))
        ts1.set_column('e1', sample1, 'observed_cov')
        self.assertEqual(ts1.get_observed_cov().data.shape, (200, 2))
        ts1.set_column('e1', sample2, 'observed_cov')
        self.assertEqual(ts1.get_observed_cov().data.shape, (200, 2))

        ts1.set_column('c3', 1, 'static_cov')
        self.assertEqual(ts1.get_static_cov(), {'c3': 1})

        ts1['c3'] = 2
        ts1['b'] = sample2
        self.assertEqual(ts1.get_target().data.shape, (200, 2))
        ts1['c5'] = sample2
        self.assertEqual(ts1.get_known_cov().data.shape, (200, 3))

        ts1 = TSDataset(target=TimeSeries.load_from_dataframe(sample1))
        ts1.set_column('c6', pd.Series(
            [1 for i in range(200)], 
            index=pd.date_range('2022-01-01', periods=200, freq='1D'), 
            name='a')
        )
        self.assertEqual(ts1.get_known_cov().data.shape, (200, 1))

    def test_mock_transfrom(self):
        """
        unittest function
        """
        ts1 = TSDataset(self.target, self.observed_cov, self.known_cov, self.static_cov)
        # case1: Simulate feature transformation, such as modifying a column (eg: normalization, encode)
        ts2 = ts1.copy()
        column_name = 'c'
        column_new = ts2['c'] + 10
        ts2.set_column(column_name, column_new)

        # case2: Simulate feature generation
        ts2 = ts1.copy()
        column_name = ['c', 'c1']
        column_new = ts2['c'] + ts2['c1']
        # The newly generated columns_name is defined as `c_c1`, and is set to the observed feature
        ts2.set_column("c_c1", column_new, 'observed_cov')
    
    def test_save_and_load(self):
        """
        unittest function
        """
        ts1 = TSDataset(self.target, self.observed_cov, self.known_cov, self.static_cov)
        ts1.save("/tmp/ts.tmp")
        ts2 = TSDataset.load("/tmp/ts.tmp")
        self.assertEqual(ts1.to_dataframe().shape, ts2.to_dataframe().shape)

        json_data = ts1.to_json()
        ts2 = TSDataset.load_from_json(json_data)
        self.assertEqual(ts1.to_dataframe().shape, ts2.to_dataframe().shape)
    
    def test_property(self):
        """
        unittest function
        """
        ts1 = TSDataset(self.target, self.observed_cov, self.known_cov, self.static_cov)
        res = {'a':'target', 'b': 'observed_cov', 'c': 'observed_cov', 'b1': 'known_cov', 'c1': 'known_cov'}
        self.assertEqual(res, ts1.columns)

        #test dtypes
        self.assertEqual(ts1.dtypes.shape, (7, ))

        self.assertEqual(repr(ts1.to_dataframe()), repr(ts1))
        self.assertEqual(str(ts1.to_dataframe()), str(ts1))

    def test_concat(self):
        """
        unittest function
        """
        #case1 test axis=0
        sample1 = pd.DataFrame(
            np.random.randn(200, 5), 
            index=pd.date_range('2022-01-01', periods=200, freq='1D'),
            columns=['a', 'b', 'c', 'd', 'e']
        )
        sample1['s'] = pd.Series([1 for i in range(200)], index=pd.date_range('2022-01-01', periods=200, freq='1D'))
        tsdataset1 = TSDataset.load_from_dataframe(
            df=sample1, 
            target_cols='a', 
            observed_cov_cols=['b', 'c', 'd'], 
            known_cov_cols='e', 
            static_cov_cols='s'
        )
        sample2 = pd.DataFrame(
            np.random.randn(200, 5), 
            index=pd.date_range('2022-07-20', periods=200, freq='1D'),
            columns=['a', 'b', 'c', 'd', 'e']
        )
        sample2['s'] = pd.Series([1 for i in range(200)], index=pd.date_range('2022-07-20', periods=200, freq='1D'))
        tsdataset2 = TSDataset.load_from_dataframe(
            df=sample2, 
            target_cols='a', 
            observed_cov_cols=['b', 'c', 'd'], 
            known_cov_cols='e', 
            static_cov_cols='s'
        )
        tsdataset3 = TSDataset.concat([tsdataset1, tsdataset2])
        self.assertEqual(tsdataset3.get_all_cov().data.shape, (400, 4))
        sample2 = pd.DataFrame(
            np.random.randn(200, 5), 
            index=pd.date_range('2022-06-20', periods=200, freq='1D'),
            columns=['a', 'b', 'c', 'd', 'e']
        )
        sample2['s'] = pd.Series([1 for i in range(200)], index=pd.date_range('2022-06-20', periods=200, freq='1D'))
        tsdataset2 = TSDataset.load_from_dataframe(
            df=sample2, 
            target_cols='a', 
            observed_cov_cols=['b', 'c', 'd'], 
            known_cov_cols='e', 
            static_cov_cols='s'
        )
        tsdataset3 = TSDataset.concat([tsdataset1, tsdataset2])
        #2022-01-01~2022-06-20 + 200days = 370days
        self.assertEqual(tsdataset3.get_all_cov().data.shape, (370, 4))

        #case1 test axis=1
        sample1 = pd.DataFrame(
            np.random.randn(200, 5), 
            index=pd.date_range('2022-01-01', periods=200, freq='1D'),
            columns=['a', 'b', 'c', 'd', 'e']
        )
        sample1['s'] = pd.Series([1 for i in range(200)], index=pd.date_range('2022-01-01', periods=200, freq='1D'))
        tsdataset1 = TSDataset.load_from_dataframe(
            df=sample1, 
            target_cols='a', 
            observed_cov_cols=['b', 'c', 'd'], 
            known_cov_cols='e', 
            static_cov_cols='s'
        )
        sample2 = pd.DataFrame(
            np.random.randn(200, 5), 
            index=pd.date_range('2022-01-01', periods=200, freq='1D'),
            columns=['a1', 'b1', 'c1', 'd1', 'e1']
        )
        sample2['s1'] = pd.Series([1 for i in range(200)], index=pd.date_range('2022-01-01', periods=200, freq='1D'))
        tsdataset2 = TSDataset.load_from_dataframe(
            df=sample2, 
            target_cols='a1', 
            observed_cov_cols=['b1', 'c1', 'd1'], 
            known_cov_cols='e1', 
            static_cov_cols='s1'
        )
        tsdataset3 = TSDataset.concat([tsdataset1, tsdataset2], axis=1)
        self.assertEqual(tsdataset3.get_all_cov().data.shape, (200, 8))

        sample2 = pd.DataFrame(
            np.random.randn(200, 5), 
            index=pd.date_range('2022-01-01', periods=200, freq='1D'),
            columns=['a', 'b', 'c', 'd', 'e']
        )
        sample2['s'] = pd.Series([1 for i in range(200)], index=pd.date_range('2022-01-01', periods=200, freq='1D'))
        tsdataset2 = TSDataset.load_from_dataframe(
            df=sample2, 
            target_cols='a', 
            observed_cov_cols=['b', 'c', 'd'], 
            known_cov_cols='e', 
            static_cov_cols='s'
        )   
        tsdataset3 = TSDataset.concat([tsdataset1, tsdataset2], axis=1)
        self.assertEqual(tsdataset3.get_all_cov().data.shape, (200, 4))

    def test_plot(self):
        """
        unittest function
        """
        ts = TSDataset(self.target, self.observed_cov, self.known_cov, self.static_cov)
        #case1, plot by default, only display target_cols
        ts.plot()

        #case2, Specify column to plot 
        ts.plot(columns="a")
        ts.plot(columns=["a","c"])

        train, test = ts.split(0.75)

        #case3, Joint plot, by add_data, plot target_cols by default
        train.plot(add_data=test)

        #case4, Joint plot, by add_data, Specify column to plot
        train.plot(columns=["a","b"],add_data=test)

        #case5, Joint plot, by add_data, Specify column to plot
        train.plot(columns=["a","b"],add_data=test)
        
        #case6, Joint plot, by add_data, custom labels
        train.plot(columns=["a","b"], add_data=test ,labels=["pred1"])

        #case7 badcase, labels lens not match origin labels
        with self.assertRaises(ValueError):
            train.plot(columns=["a","b"], add_data=test, labels=["a","b"])
        #badcase
        with self.assertRaises(ValueError):
            ts.plot(columns="h")

        ###quantile plot
        target = TimeSeries.load_from_dataframe(
            pd.DataFrame(np.random.randn(400,2).astype(np.float32),
                    index=pd.date_range("2022-01-01", periods=400, freq="15T"),
                        columns=["a1", "a2"]
                    ))

        observed_cov = TimeSeries.load_from_dataframe(
            pd.DataFrame(
                np.random.randn(400, 2).astype(np.float32),
                index=pd.date_range("2022-01-01", periods=400, freq="15T"),
                columns=["index", "c"]
            ))
        known_cov = TimeSeries.load_from_dataframe(
            pd.DataFrame(
                np.random.randn(500, 2).astype(np.float32),
                index=pd.date_range("2022-01-01", periods=500, freq="15T"),
                columns=["b1", "c1"]
            ))
        static_cov = {"f": 1, "g": 2}
        ts = TSDataset(target, observed_cov, known_cov, None)
        from paddlets.models.forecasting import DeepARModel
        reg = DeepARModel(
            in_chunk_len=10,
            out_chunk_len=5,
            skip_chunk_len=4 * 4,
            eval_metrics=["mse", "mae"],
            batch_size=512,
            num_samples = 101,
            regression_mode="sampling",
            output_mode="quantiles",
            max_epochs=5
        )

        reg.fit(ts, ts)
        res = reg.predict(ts)

        res.plot()
        res.plot(["a1"])
        res.plot(["a1"],add_data = [res], labels = ["pred1"])
        res.plot(["a1","a2"],add_data = [res], labels = ["pred1"])
        res.plot(["a1","a2"],add_data = [res,res], labels = ["pred1","pred2"])

    def test_astype(self):
        """
        unittest function
        """
        sample1 = pd.DataFrame(
            np.random.randn(200, 5), 
            index=pd.date_range('2022-01-01', periods=200, freq='1D'),
            columns=['a', 'b', 'c', 'd', 'e']
        )
        sample1['s'] = pd.Series([1 for i in range(200)], index=pd.date_range('2022-01-01', periods=200, freq='1D'))
        tsdataset1 = TSDataset.load_from_dataframe(
            df=sample1, 
            target_cols='a', 
            observed_cov_cols=['b', 'c', 'd'], 
            known_cov_cols='e', 
            static_cov_cols='s'
        )
        tsdataset2 = tsdataset1.copy()
        tsdataset2.astype('float32')
        self.assertEqual(tsdataset2.get_target().data.dtypes['a'], 'float32')
        self.assertEqual(tsdataset2.get_all_cov().data.dtypes['e'], 'float32')
        self.assertEqual(type(tsdataset2.static_cov['s']), np.float32)
        tsdataset2 = tsdataset1.copy()
        tsdataset2.astype({'a': 'float32', 's': 'float32'})
        self.assertEqual(tsdataset2.get_target().data.dtypes['a'], 'float32')
        self.assertEqual(tsdataset2.get_all_cov().data.dtypes['e'], 'float64')
        self.assertEqual(type(tsdataset2.static_cov['s']), np.float32)

        tsdataset2.astype({'s': 'float32'})
        self.assertEqual(type(tsdataset2.static_cov['s']), np.float32)

    def test_reindex(self):
        """
        unittest function
        """
        sample1 = pd.DataFrame(
            np.random.randn(200, 5), 
            index=pd.date_range('2022-01-01', periods=200, freq='1D'),
            columns=['a', 'b', 'c', 'd', 'e']
        )
        sample1['s'] = pd.Series([1 for i in range(200)], index=pd.date_range('2022-01-01', periods=200, freq='1D'))
        tsdataset1 = TSDataset.load_from_dataframe(
            df=sample1, 
            target_cols='a', 
            observed_cov_cols=['b', 'c', 'd'], 
            known_cov_cols='e', 
            static_cov_cols='s'
        )
        fill_value= 1.1024
        len_before_reindex = len(tsdataset1.get_known_cov().to_dataframe())
        reindex_added_row_length = 5
        tsdataset1.get_known_cov().reindex(
                pd.date_range(start=tsdataset1.get_known_cov().start_time, 
                    end=tsdataset1.get_known_cov().end_time +\
                         reindex_added_row_length * tsdataset1.get_target().time_index.freq,
                    freq=tsdataset1.get_known_cov().time_index.freq),
                    fill_value=fill_value
            )
        len(tsdataset1.get_known_cov().to_dataframe())
        self.assertEqual(len(tsdataset1.get_known_cov().to_dataframe()), \
            len_before_reindex + reindex_added_row_length)
        self.assertEqual(tsdataset1.get_known_cov().to_dataframe()['e'][-1], fill_value)

    def test_sort_columns(self):
        """
        unittest function
        """
        sample1 = pd.DataFrame(
            np.random.randn(200, 5), 
            index=pd.date_range('2022-01-01', periods=200, freq='1D'),
            columns=['a', 'b', 'c', 'd', 'e']
        )
        sample1['s'] = pd.Series([1 for i in range(200)], index=pd.date_range('2022-01-01', periods=200, freq='1D'))
        tsdataset1 = TSDataset.load_from_dataframe(
            df=sample1, 
            target_cols='a', 
            observed_cov_cols=['d', 'c', 'b'], 
            known_cov_cols='e', 
            static_cov_cols='s'
        )
        self.assertEqual(list(tsdataset1.columns.keys()), ['a', 'e', 'd', 'c', 'b'])
        tsdataset1.sort_columns()
        self.assertEqual(list(tsdataset1.columns.keys()), ['a', 'e', 'b', 'c', 'd'])
        tsdataset1.sort_columns(ascending=False)
        self.assertEqual(list(tsdataset1.columns.keys()), ['a', 'e', 'd', 'c', 'b'])
    
    def test_drop_tail_nan(self):
        """
        unittest function
        """
        sample1 = pd.DataFrame(
            np.random.randn(200, 5), 
            index=pd.date_range('2022-01-01', periods=200, freq='1D'),
            columns=['a', 'b', 'c', 'd', 'e']
        )
        sample1['s'] = pd.Series([1 for i in range(200)], index=pd.date_range('2022-01-01', periods=200, freq='1D'))
        sample1.iloc[100:, 0] = np.nan
        tsdataset1 = TSDataset.load_from_dataframe(
            df=sample1, 
            target_cols='a', 
            observed_cov_cols=['d', 'c', 'b'], 
            known_cov_cols='e', 
            static_cov_cols='s'
        )
        self.assertEqual(tsdataset1.target.data.shape, (200, 1))
        self.assertEqual(tsdataset1.observed_cov.data.shape, (200, 3))
        tsdataset1 = TSDataset.load_from_dataframe(
            df=sample1, 
            target_cols='a', 
            observed_cov_cols=['d', 'c', 'b'], 
            known_cov_cols='e', 
            static_cov_cols='s',
            drop_tail_nan=True
        )
        self.assertEqual(tsdataset1.target.data.shape, (100, 1))
        self.assertEqual(tsdataset1.observed_cov.data.shape, (200, 3))

if __name__ == "__main__":
    unittest.main()
