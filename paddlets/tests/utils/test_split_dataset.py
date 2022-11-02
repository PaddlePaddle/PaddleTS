# !/usr/bin/env python3
# -*- coding:utf-8 -*-

import unittest

import pandas as pd
import numpy as np
from unittest import TestCase

from paddlets.datasets.tsdataset import TimeSeries, TSDataset
from paddlets.utils.utils import split_dataset


class TestUtils(TestCase):
    def setUp(self):
        """
        unittest function
        """
        super().setUp()

    def test_split_dataset(self):
        """
        unittest function
        """

        #DateTimeIndex
        np.random.seed(2022)

        target = TimeSeries.load_from_dataframe(
            pd.Series(np.random.randn(2000).astype(np.float32),
                      index=pd.date_range("2022-01-01", periods=2000, freq="15T"),
                      name="a"
                      ))

        observed_cov = TimeSeries.load_from_dataframe(
            pd.DataFrame(
                np.random.randn(2000, 2).astype(np.float32),
                index=pd.date_range("2022-01-01", periods=2000, freq="15T"),
                columns=["b", "c"]
            ))
        known_cov = TimeSeries.load_from_dataframe(
            pd.DataFrame(
                np.random.randn(2500, 2).astype(np.float32),
                index=pd.date_range("2022-01-01", periods=2500, freq="15T"),
                columns=["b1", "c1"]
            ))
        static_cov = {"f": 1.0, "g": 2.0}
        tsdataset = TSDataset(target, observed_cov, known_cov, static_cov)
        # case1
        pre, after = split_dataset(tsdataset, 1900)
        assert len(pre.target) == 1900
        assert len(after.target) == 100
        assert len(after.observed_cov) == 100
        assert len(after.known_cov) == 600

        # case2
        pre, after = split_dataset(tsdataset, 2000)
        assert len(pre.target) == 2000
        assert after.target is None
        assert len(after.known_cov) == 500
        assert after.observed_cov is None

        # case3
        pre, after = split_dataset(tsdataset, 2100)
        assert len(pre.target) == 2000
        assert after.target is None
        assert len(after.known_cov) == 400
        assert after.observed_cov is None

        # case4
        with self.assertRaises(Exception):
            pre, after = split_dataset(tsdataset, 2600)

        # case5
        with self.assertRaises(Exception):
            pre, after = split_dataset(tsdataset, -1)

        # case6
        with self.assertRaises(Exception):
            pre, after = split_dataset(tsdataset, "2000")

        #RangeIndex
        np.random.seed(2022)
        target = TimeSeries.load_from_dataframe(
            pd.Series(np.random.randn(2000).astype(np.float32),
                    #index=pd.date_range("2022-01-01", periods=2000, freq="15T"),
                    index=pd.RangeIndex(start=10, stop=4010, step=2),
                    name="a"
                    ), freq = 2)

        observed_cov = TimeSeries.load_from_dataframe(
            pd.DataFrame(
                np.random.randn(2000, 2).astype(np.float32),
                #index=pd.date_range("2022-01-01", periods=2000, freq="15T"),
                index=pd.RangeIndex(start=10, stop=4010, step=2),
                columns=["b", "c"]
            ), freq = 2)
        known_cov = TimeSeries.load_from_dataframe(
            pd.DataFrame(
                np.random.randn(2500, 2).astype(np.float32),
                #index=pd.date_range("2022-01-01", periods=2500, freq="15T"),
                index=pd.RangeIndex(start=10, stop=5010, step=2),
                columns=["b1", "c1"]
            ), freq = 2)
        static_cov = {"f": 1.0, "g": 2.0}
        tsdataset = TSDataset(target, observed_cov, known_cov, static_cov)

        # case1
        pre, after = split_dataset(tsdataset, 1900)
        assert len(pre.target) == 1900
        assert len(after.target) == 100
        assert len(after.observed_cov) == 100
        assert len(after.known_cov) == 600

        # case2
        pre, after = split_dataset(tsdataset, 2000)
        assert len(pre.target) == 2000
        assert after.target is None
        assert len(after.known_cov) == 500
        assert after.observed_cov is None

        # case3
        pre, after = split_dataset(tsdataset, 2100)
        assert len(pre.target) == 2000
        assert after.target is None
        assert len(after.known_cov) == 400
        assert after.observed_cov is None

        # case4
        with self.assertRaises(Exception):
            pre, after = split_dataset(tsdataset, 2600)

        # case5
        with self.assertRaises(Exception):
            pre, after = split_dataset(tsdataset, -1)

        # case6
        with self.assertRaises(Exception):
            pre, after = split_dataset(tsdataset, "2000")


        #RangeIndex (different start point)
        np.random.seed(2022)
        target = TimeSeries.load_from_dataframe(
            pd.Series(np.random.randn(2000).astype(np.float32),
                    #index=pd.date_range("2022-01-01", periods=2000, freq="15T"),
                    index=pd.RangeIndex(start=1010, stop=5010, step=2),
                    name="a"
                    ), freq = 2)

        observed_cov = TimeSeries.load_from_dataframe(
            pd.DataFrame(
                np.random.randn(2000, 2).astype(np.float32),
                #index=pd.date_range("2022-01-01", periods=2000, freq="15T"),
                index=pd.RangeIndex(start=10, stop=4010, step=2),
                columns=["b", "c"]
            ), freq = 2)
        known_cov = TimeSeries.load_from_dataframe(
            pd.DataFrame(
                np.random.randn(2500, 2).astype(np.float32),
                #index=pd.date_range("2022-01-01", periods=2500, freq="15T"),
                index=pd.RangeIndex(start=10, stop=5010, step=2),
                columns=["b1", "c1"]
            ), freq = 2)
        static_cov = {"f": 1.0, "g": 2.0}
        tsdataset = TSDataset(target, observed_cov, known_cov, static_cov)

        # case1
        pre, after = split_dataset(tsdataset, 1900)
        assert len(pre.target) == 1400
        assert len(after.target) == 600
        assert len(after.observed_cov) == 100
        assert len(after.known_cov) == 600

        # case2
        pre, after = split_dataset(tsdataset, 2000)
        assert len(pre.target) == 1500
        assert len(after.target) == 500
        assert len(after.known_cov) == 500
        assert after.observed_cov is None

        # case3
        pre, after = split_dataset(tsdataset, 2100)
        assert len(pre.target) == 1600
        assert len(after.target) == 400
        assert len(after.known_cov) == 400
        assert after.observed_cov is None

        # case4
        with self.assertRaises(Exception):
            pre, after = split_dataset(tsdataset, 2600)

        # case5
        with self.assertRaises(Exception):
            pre, after = split_dataset(tsdataset, -1)

        # case6
        with self.assertRaises(Exception):
            pre, after = split_dataset(tsdataset, "2000")
if __name__ == "__main__":
    unittest.main()
