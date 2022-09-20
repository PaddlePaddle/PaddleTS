# !/usr/bin/env python3
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import os
import sys
import time
import unittest
from unittest import TestCase
from paddlets.transform.sklearn_transforms_base import SklearnTransformWrapper
from paddlets.datasets.tsdataset import TimeSeries
from paddlets.datasets.tsdataset import TSDataset
from paddlets.datasets.repository import get_dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA

class TestSklearnTransformWrapper(TestCase):
    def setUp(self):
        """
        unittest function
        """
        super().setUp()
        self.dataset, _ = get_dataset('WTH').split(1000)
    
    def test_transform_case_0(self):
        """
        unittest function
        """
        #case1
        min_max_sacler = SklearnTransformWrapper(
            MinMaxScaler,
            drop_origin_columns=True
        )
        dataset = min_max_sacler.fit_transform(self.dataset)
        self.assertEqual(dataset.columns, self.dataset.columns)
        #case2
        dataset = min_max_sacler.fit(self.dataset)
        new_ts = self.dataset.copy()
        new_ts.observed_cov = None
        with self.assertRaises(ValueError):
            dataset = min_max_sacler.transform(new_ts)
        #case3
        min_max_sacler = SklearnTransformWrapper(
            MinMaxScaler,
            drop_origin_columns=True,
            per_col_transform=True
        )
        dataset = min_max_sacler.fit(self.dataset)
        new_ts = self.dataset.copy()
        new_ts.observed_cov = None
        dataset = min_max_sacler.transform(new_ts)
        self.assertEqual(dataset.columns, {"WetBulbCelsius": "target"})
        #case4
        min_max_sacler = SklearnTransformWrapper(
            MinMaxScaler,
            in_col_names=['WetBulbCelsius'],
            drop_origin_columns=True,
            per_col_transform=True
        )
        dataset = min_max_sacler.fit(self.dataset)
        new_ts = self.dataset.copy()
        new_ts.observed_cov = None
        dataset = min_max_sacler.transform(new_ts)
        self.assertEqual(dataset.columns, {"WetBulbCelsius": "target"})
        #case5
        min_max_sacler = SklearnTransformWrapper(
            MinMaxScaler,
            in_col_names=['WetBulbCelsius', 'Visibility', 'DryBulbFarenheit'],
            drop_origin_columns=True,
            per_col_transform=True
        )
        dataset = min_max_sacler.fit(self.dataset)
        new_ts = self.dataset.copy()
        new_ts.observed_cov = None
        dataset = min_max_sacler.transform(new_ts)
        self.assertEqual(dataset.columns, {"WetBulbCelsius": "target"})
    
    def test_transform_case_1(self):
        """
        unittest function
        """
        #case1
        onehot = SklearnTransformWrapper(
            OneHotEncoder,
            in_col_names=["WetBulbCelsius"],
            drop_origin_columns=True
        )
        dataset = onehot.fit_transform(self.dataset)
        onehot_nums = len(set(self.dataset['WetBulbCelsius']))
        self.assertTrue(len(dataset.target.columns) == onehot_nums)
        #case2
        onehot = SklearnTransformWrapper(
            OneHotEncoder,
            in_col_names=['WetBulbCelsius', 'Visibility', 'DryBulbFarenheit'],
            drop_origin_columns=True
        )
        with self.assertRaises(ValueError):
             dataset = onehot.fit_transform(self.dataset)
        #case3
        onehot = SklearnTransformWrapper(
            OneHotEncoder,
            in_col_names=['WetBulbCelsius', 'Visibility', 'DryBulbFarenheit'],
            out_col_types='known_cov',
            drop_origin_columns=False
        )
        dataset = onehot.fit_transform(self.dataset)
        self.assertTrue(dataset.known_cov is not None)

    def test_transform_case_2(self):
        """
        unittest function
        """
        #case1
        pca = SklearnTransformWrapper(
            PCA,
            in_col_names=list(self.dataset.observed_cov.columns),
            drop_origin_columns=True,
            n_components=2
        )
        dataset = pca.fit_transform(self.dataset)
        self.assertEqual(len(dataset.observed_cov.columns), 2)
    

if __name__ == "__main__":
    unittest.main()
        
