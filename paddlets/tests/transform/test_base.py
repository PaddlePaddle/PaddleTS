# !/usr/bin/env python3
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import os
import sys
import time
import unittest
from unittest import TestCase
from paddlets.transform.base import UdBaseTransform
from paddlets.datasets.tsdataset import TimeSeries
from paddlets.datasets.tsdataset import TSDataset
from paddlets.datasets.repository import get_dataset
from sklearn.preprocessing import MinMaxScaler

class _UdMockTransform(UdBaseTransform):
    def _fit(self, input):
        pass
    def _transform(self, input):
        return input.to_numpy()
    def _fit_transform(self, input):
        return input.to_numpy()
    def _inverse_transform(self, input):
        return input.to_numpy()

class TestUdBaseTransform(TestCase):
    def setUp(self):
        """
        unittest function
        """
        super().setUp()
        self.dataset, _ = get_dataset('WTH').split(1000)
        
    def test__gen_input(self):
        """
        unittest function
        """
        ud_transform = _UdMockTransform(
            MinMaxScaler(),
            drop_origin_columns=True
        )
        #case1
        input = ud_transform._gen_input(self.dataset, self.dataset.target.columns[0])
        self.assertEqual(input.shape, (len(self.dataset.target.data), 1))
        #case2
        input = ud_transform._gen_input(self.dataset, [self.dataset.target.columns[0], *self.dataset.observed_cov.columns[:5]])
        self.assertEqual(input.shape, (len(self.dataset.target.data), 6))

    def test__gen_output(self):
        """
        unittest function
        """
        ud_transform = _UdMockTransform(
            MinMaxScaler(),
            drop_origin_columns=True
        )
        #case1
        raw_output = np.array([[1, 2, 3], [4, 5, 6]])
        output = ud_transform._gen_output(raw_output)
        self.assertEqual(raw_output.shape, output.shape)

    def test__check_output(self):
        """
        unittest function
        """
        #case1
        ud_transform = _UdMockTransform(
            MinMaxScaler(),
            drop_origin_columns=True
        )
        input = ud_transform._gen_input(self.dataset, self.dataset.target.columns[0])
        output = MinMaxScaler().fit_transform(input)
        ud_transform._check_output(self.dataset, input, output)
        #case2
        input_2 = ud_transform._gen_input(self.dataset, [self.dataset.target.columns[0], *self.dataset.observed_cov.columns[:5]])
        output_2 = MinMaxScaler().fit_transform(input_2)
        ud_transform._check_output(self.dataset, input_2, output_2)
        #case3
        ud_transform._check_output(self.dataset, input, output_2)
        #case4
        with self.assertRaises(ValueError):
            ud_transform._check_output(self.dataset, input_2, output)
        #case5
        ud_transform = _UdMockTransform(
            MinMaxScaler(),
            drop_origin_columns=False
        )        
        with self.assertRaises(ValueError):
            ud_transform._check_output(self.dataset, input_2, output_2)
        #case6
        output_3 = output[:-10, :]
        with self.assertRaises(ValueError):
            ud_transform._check_output(self.dataset, input, output_3)
        #case7
        ud_transform = _UdMockTransform(
            MinMaxScaler(),
            drop_origin_columns=True,
            out_col_names=['test1']
        )
        ud_transform._check_output(self.dataset, input, output)
        #case8
        ud_transform = _UdMockTransform(
            MinMaxScaler(),
            drop_origin_columns=True,
            out_col_names=['test1', 'test2']
        )
        with self.assertRaises(ValueError):
            ud_transform._check_output(self.dataset, input, output)
        #case9
        ud_transform = _UdMockTransform(
            MinMaxScaler(),
            drop_origin_columns=False,
            out_col_types='target'
        )
        ud_transform._check_output(self.dataset, input_2, output_2)
        ud_transform._check_output(self.dataset, input_2, output)
        #case10
        self.dataset.target.reindex(self.dataset.target.time_index[100:])
        with self.assertRaises(ValueError):
            ud_transform._check_output(self.dataset, input_2, output_2)

    def test__infer_output_column_types(self):
        """
        unittest function
        """
        #case1
        ud_transform = _UdMockTransform(
            MinMaxScaler(),
            drop_origin_columns=False,
            out_col_types='target'
        )
        input = ud_transform._gen_input(self.dataset, self.dataset.target.columns[0])
        output = MinMaxScaler().fit_transform(input)
        self.assertEqual(ud_transform._infer_output_column_types(self.dataset, input), 'target')
        ud_transform = _UdMockTransform(
            MinMaxScaler(),
            drop_origin_columns=False,
            out_col_types=['target', 'observed_cov', 'observed_cov']
        )
        input_2 = ud_transform._gen_input(self.dataset, [self.dataset.target.columns[0], *self.dataset.observed_cov.columns[:2]])
        output_2 = MinMaxScaler().fit_transform(input_2)
        self.assertEqual(ud_transform._infer_output_column_types(self.dataset, input_2), ['target', 'observed_cov', 'observed_cov'])
        #case2
        ud_transform = _UdMockTransform(
            MinMaxScaler(),
            drop_origin_columns=False
        )
        self.assertEqual(ud_transform._infer_output_column_types(self.dataset, input), 'target')
        #case3
        ud_transform = _UdMockTransform(
            MinMaxScaler(),
            drop_origin_columns=True
        )
        self.assertEqual(ud_transform._infer_output_column_types(self.dataset, input_2), ['target', 'observed_cov', 'observed_cov'])

    def test__get_output_column_names(self):
        """
        unittest function
        """
        #case1
        ud_transform = _UdMockTransform(
            MinMaxScaler(),
            drop_origin_columns=False,
            out_col_names=['name']
        )
        input = ud_transform._gen_input(self.dataset, self.dataset.target.columns[0])
        output = MinMaxScaler().fit_transform(input)
        self.assertEqual(ud_transform._get_output_column_names(input, output), ['name'])
        #case2
        ud_transform = _UdMockTransform(
            MinMaxScaler(),
            drop_origin_columns=True
        )
        input_2 = ud_transform._gen_input(self.dataset, [self.dataset.target.columns[0], *self.dataset.observed_cov.columns[:2]])
        output_2 = MinMaxScaler().fit_transform(input_2)
        self.assertEqual(ud_transform._get_output_column_names(input_2, output_2), ['WetBulbCelsius', 'Visibility', 'DryBulbFarenheit'])
        #case3
        ud_transform = _UdMockTransform(
            MinMaxScaler(),
            drop_origin_columns=False
        )
        self.assertEqual(
            ud_transform._get_output_column_names(input_2, output_2), 
            [
                'MinMaxScaler_WetBulbCelsius-Visibility-DryBulbFarenheit_0', 
                'MinMaxScaler_WetBulbCelsius-Visibility-DryBulbFarenheit_1', 
                'MinMaxScaler_WetBulbCelsius-Visibility-DryBulbFarenheit_2'
            ]
        )

    def test_fit(self):
        """
        unittest function
        """
        #case1
        ud_transform = _UdMockTransform(
            MinMaxScaler(),
            drop_origin_columns=True
        )
        ud_transform.fit(self.dataset)
        self.assertFalse(hasattr(ud_transform, '_ud_transformer_col_list'))
        #case2
        ud_transform = _UdMockTransform(
            MinMaxScaler(),
            per_col_transform=True,
            drop_origin_columns=True
        )
        ud_transform.fit(self.dataset)
        self.assertEqual(len(ud_transform._ud_transformer_col_list), len(self.dataset.columns))
        #case3
        ud_transform = _UdMockTransform(
            MinMaxScaler(),
            in_col_names=['WetBulbCelsius', 'Visibility', 'DryBulbFarenheit'],
            per_col_transform=True,
            drop_origin_columns=True
        )
        ud_transform.fit(self.dataset)
        self.assertEqual(len(ud_transform._ud_transformer_col_list), 3)
    
    def test_transform(self):
        #case1
        ud_transform = _UdMockTransform(
            MinMaxScaler(),
            drop_origin_columns=True
        )
        ud_transform.fit(self.dataset)
        dataset = ud_transform.transform(self.dataset)
        self.assertEqual(dataset.columns, self.dataset.columns)
        #case2
        ud_transform = _UdMockTransform(
            MinMaxScaler(),
            per_col_transform=True,
            drop_origin_columns=True
        )
        ud_transform.fit(self.dataset)
        dataset = ud_transform.transform(self.dataset)
        self.assertEqual(dataset.columns, self.dataset.columns)
        #case3
        ud_transform = _UdMockTransform(
            MinMaxScaler(),
            in_col_names=['WetBulbCelsius', 'Visibility', 'DryBulbFarenheit'],
            per_col_transform=True,
            drop_origin_columns=True
        )
        ud_transform.fit(self.dataset)
        dataset = ud_transform.transform(self.dataset)
        self.assertEqual(dataset.columns, self.dataset.columns)
        #case4
        ud_transform = _UdMockTransform(
            MinMaxScaler(),
            in_col_names=['WetBulbCelsius', 'Visibility', 'DryBulbFarenheit'],
            per_col_transform=True,
            drop_origin_columns=False,
            out_col_types='known_cov'
        )
        ud_transform.fit(self.dataset)
        dataset = ud_transform.transform(self.dataset)
        self.assertEqual(dataset.known_cov.data.shape, (len(self.dataset.target), 3))
        #case5
        ud_transform = _UdMockTransform(
            MinMaxScaler(),
            in_col_names=['Visibility', 'DryBulbFarenheit'],
            per_col_transform=True,
            drop_origin_columns=False,
        )
        ud_transform.fit(self.dataset)
        dataset = ud_transform.transform(self.dataset)
        self.assertEqual(dataset.observed_cov.data.shape, (len(self.dataset.target), 13))
        #case6
        ud_transform = _UdMockTransform(
            MinMaxScaler(),
            in_col_names=['Visibility', 'DryBulbFarenheit'],
            per_col_transform=True,
            drop_origin_columns=True,
        )
        ud_transform.fit(self.dataset)
        dataset = ud_transform.transform(self.dataset)
        self.assertEqual(dataset.observed_cov.data.shape, (len(self.dataset.target), 11))


if __name__ == "__main__":
    unittest.main()
        