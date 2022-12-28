# !/usr/bin/env python3
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np

import os
import sys
import time
import unittest
from unittest import TestCase

from paddlets.datasets.repository import dataset_list, get_dataset, DATASETS


class TestDatasetRepository(TestCase):
    def setUp(self):
        """
        unittest function
        """
        super().setUp()

    def test_dataset_list(self):
        """
        unittest function
        """
        #case1
        tmp = dataset_list()
        self.assertEqual(tmp, list(DATASETS.keys()))
    
    def test_get_dataset(self):
        """
        unittest function
        """
        #case1 get origin demo file
        ts = get_dataset("UNI_WTH")
        self.assertEqual(ts.get_target().data.shape, (35064, 1))
        ts = get_dataset("WTH")
        self.assertEqual(ts.get_target().data.shape, (35064, 1))
        self.assertEqual(ts.get_all_cov().data.shape, (35064, 11))

        #case3 get public data
        dataset_name = "ETTh1"
        ts = get_dataset(dataset_name)
        self.assertEqual(ts.get_target().data.shape, (17420, 1))
        self.assertEqual(ts.get_all_cov().data.shape, (17420, 6))

        dataset_name = "ETTm1"
        ts = get_dataset(dataset_name)
        self.assertEqual(ts.get_target().data.shape, (69680, 1))
        self.assertEqual(ts.get_all_cov().data.shape, (69680, 6))

        dataset_name = "ECL"
        ts = get_dataset(dataset_name)
        self.assertEqual(ts.get_target().data.shape, (26304, 1))
        self.assertEqual(ts.get_all_cov().data.shape, (26304, 320))

        dataset_name = "WTH"
        ts = get_dataset(dataset_name)
        self.assertEqual(ts.get_target().data.shape, (35064, 1))
        self.assertEqual(ts.get_all_cov().data.shape, (35064, 11))
        
        dataset_name = "NAB_TEMP"
        ts = get_dataset(dataset_name)
        self.assertEqual(ts.get_target().data.shape, (22683, 1))
        self.assertEqual(ts.get_all_cov().data.shape, (22683, 1))
        self.assertEqual(ts.get_label().data.shape, (22683, 1))
        self.assertEqual(ts.get_feature().data.shape, (22683, 1))
        
        dataset_name = "psm_train"
        ts = get_dataset(dataset_name)
        self.assertEqual(ts.get_all_cov().data.shape, (132481, 25))
        self.assertEqual(ts.get_feature().data.shape, (132481, 25))
        
        dataset_name = "psm_test"
        ts = get_dataset(dataset_name)
        self.assertEqual(ts.get_target().data.shape, (87841, 1))
        self.assertEqual(ts.get_all_cov().data.shape, (87841, 25))
        self.assertEqual(ts.get_label().data.shape, (87841, 1))
        self.assertEqual(ts.get_feature().data.shape, (87841, 25))

        dataset_name = "BasicMotions_Train"
        tss, y_labels = get_dataset(dataset_name)
        self.assertEqual(len(tss), 40)
        self.assertEqual(len(tss), len(y_labels))
        for dataset in tss:
            self.assertEqual(dataset.get_target().data.shape, (100, 6))
        
        dataset_name = "BasicMotions_Test"
        ts, y_labels = get_dataset(dataset_name)
        self.assertEqual(len(tss), 40)
        self.assertEqual(len(tss), len(y_labels))
        for dataset in tss:
            self.assertEqual(dataset.get_target().data.shape, (100, 6))


if __name__ == "__main__":
    unittest.main()
