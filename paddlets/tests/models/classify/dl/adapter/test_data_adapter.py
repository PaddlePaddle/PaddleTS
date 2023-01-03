# !/usr/bin/env python3
# -*- coding:utf-8 -*-

from paddlets.models.classify.dl.adapter import ClassifyDataAdapter, ClassifyPaddleDatasetImpl
from paddlets import TSDataset, TimeSeries

import paddle
from paddle.io import Dataset, DataLoader
import unittest
import datetime
import pandas as pd
import numpy as np
import math
from typing import List, Dict, Set, Union, Optional


class TestClassifyDataAdapter(unittest.TestCase):
    def setUp(self):
        """
        unittest setup
        """
        self._adapter = ClassifyDataAdapter()
        np.random.seed(2022)
        paddlets_ds, labels = self._build_mock_data_and_label(range_index=True)
        self._paddlets_ds = paddlets_ds
        self._labels = labels
        super().setUp()

    def test_to_paddle_dataset_good_case(self):
        """
        Test good cases for to_paddle_dataset()
        """
        # training dataset
        #########################################################################
        # case 1                                                                #
        # 1) train scenario. Built paddle ds with X and Y. #
        #########################################################################
        train_paddle_ds = self._adapter.to_paddle_dataset(self._paddlets_ds, self._labels)
        # self.assertEqual(len(train_paddle_ds), self._paddlets_ds.shape[0], len(self._labels))

        #########################################################################
        # case 2                                                                #
        # 1) Predict scenario. Built paddles only contain X, but not contain Y. #
        #########################################################################
        paddle_ds = self._adapter.to_paddle_dataset(self._paddlets_ds, None)
        # self.assertEqual(len(paddle_ds), self._paddlets_ds.shape[0])

    def test_to_paddle_dataset_bad_case(self):
        """to_paddle_dataset bad cases."""
        ######################################################
        # case 1                                             #
        # target is None. #
        ######################################################
        with self.assertRaises(ValueError):
            _ = self._adapter.to_paddle_dataset(None, self._labels)

    def test_to_paddle_dataloader_good_case(self):
        """
        Test to_paddle_dataloader() good cases.
        """
        ############################################################
        # case 1 (Typical usage for forcasting and representation) #
        # 1) target, known_cov, observed_cov, static_cov NOT None. #
        ############################################################
        paddle_ds = self._adapter.to_paddle_dataset(self._paddlets_ds, self._labels)
        batch_size = 2
        _ = self._adapter.to_paddle_dataloader(paddle_ds, batch_size, shuffle=False)

    @staticmethod
    def _build_mock_data_and_label(
            target_periods: int = 200,
            target_dims: int = 5,
            n_classes: int = 4,
            instance_cnt: int = 100,
            random_data: bool = True,
            range_index: bool = False,
            seed: bool = False
    ):
        """
        build train datasets and labels.
        todo:not equal target_periods?
        """
        if seed:
            np.random.seed(2022)

        target_cols = [f"dim_{k}" for k in range(target_dims)]
        labels = [f"class" + str(item) for item in np.random.randint(0, n_classes, instance_cnt)]

        ts_datasets = []
        for i in range(instance_cnt):
            if random_data:
                target_data = np.random.randint(0, 10, (target_periods, target_dims))
            else:
                target_data = target_periods * [target_dims * [0]]
            if range_index:
                target_df = pd.DataFrame(
                    target_data,
                    index=pd.RangeIndex(0, target_periods, 1),
                    columns=target_cols
                )
            else:
                target_df = pd.DataFrame(
                    target_data,
                    index=pd.date_range("2022-01-01", periods=target_periods, freq="1D"),
                    columns=target_cols
                )
            ts_datasets.append(
                TSDataset(target=TimeSeries.load_from_dataframe(data=target_df).astype(np.float32))
            )

        return ts_datasets, labels
