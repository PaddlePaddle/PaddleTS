# !/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import sys
import time
import unittest
from unittest import TestCase

import pandas as pd
import numpy as np

from paddlets.xai.post_hoc.data_wrapper import DatasetWrapper
from paddlets.datasets.tsdataset import TSDataset


class TestDatasetWrapper(TestCase):
    def setUp(self):
        """
        unittest function
        """
        super().setUp()

    def test_dataset_to_dataframe(self):
        """
        unittest function
        """
        dw = DatasetWrapper(
            in_chunk_len=10,
            out_chunk_len=3,
            skip_chunk_len=0,
            sampling_stride=10)
        ts = TSDataset.load_from_dataframe(
            pd.DataFrame(
                np.random.randn(250, 2).astype(np.float32),
                index=pd.date_range(
                    "2022-01-01", periods=250, freq="15T"),
                columns=["c1", "c2"]),
            target_cols=['c1'],
            known_cov_cols=['c2'])
        df = dw.dataset_to_dataframe(ts)
        self.assertEqual(df.shape, (25, 23))

    def test_dataframe_to_paddledsfromdf(self):
        """
        unittest function
        """
        dw = DatasetWrapper(
            in_chunk_len=10,
            out_chunk_len=3,
            skip_chunk_len=0,
            sampling_stride=10)
        ts = TSDataset.load_from_dataframe(
            pd.DataFrame(
                np.random.randn(250, 2).astype(np.float32),
                index=pd.date_range(
                    "2022-01-01", periods=250, freq="15T"),
                columns=["c1", "c2"]),
            target_cols=['c1'],
            known_cov_cols=['c2'])
        df = dw.dataset_to_dataframe(ts)
        tss = dw.dataframe_to_paddledsfromdf(df)
        self.assertEqual(len(tss), 25)

    def test_dataframe_to_ts(self):
        """
        unittest function
        """
        dw = DatasetWrapper(
            in_chunk_len=10,
            out_chunk_len=3,
            skip_chunk_len=0,
            sampling_stride=10)
        ts = TSDataset.load_from_dataframe(
            pd.DataFrame(
                np.random.randn(250, 2).astype(np.float32),
                index=pd.date_range(
                    "2022-01-01", periods=250, freq="15T"),
                columns=["c1", "c2"]),
            target_cols=['c1'],
            known_cov_cols=['c2'])
        df = dw.dataset_to_dataframe(ts)
        tss = dw.dataframe_to_ts(df)
        self.assertEqual(len(tss), 25)


if __name__ == "__main__":
    unittest.main()
