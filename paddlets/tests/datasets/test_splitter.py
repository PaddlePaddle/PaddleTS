# !/usr/bin/env python3
# -*- coding:utf-8 -*-
import sys

sys.path.append(".")
from typing import List
from unittest import TestCase, skip
import unittest

import pandas as pd
import numpy as np

from paddlets.datasets import TimeSeries, TSDataset
from paddlets.datasets.splitter import ExpandingWindowSplitter, HoldoutSplitter, SlideWindowSplitter


class TestSplitter(TestCase):
    def test_holdout_splitter(self):
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
        static_cov = {"f": 1, "g": 2}
        dataset = TSDataset(target, observed_cov, known_cov, static_cov)

        # case 1
        splitter = HoldoutSplitter(test_size=200)
        splits = splitter.split(dataset)
        for train, test in splits:
            assert isinstance(train, TSDataset)

        # case 2
        splitter = HoldoutSplitter(test_size=200)
        splits = splitter.split(dataset, return_index=True)
        for train, test in splits:
            assert isinstance(train, pd.DatetimeIndex)

        # case 3
        splitter = HoldoutSplitter(test_size=200)
        splits = splitter.split(dataset)
        for train, test in splits:
            tmp = train.target
            break
        assert tmp.__len__() == 1800

        # case 4
        splitter = HoldoutSplitter(test_size=0.2)
        splits = splitter.split(dataset)
        for train, test in splits:
            tmp = train.target
            break
        assert tmp.__len__() == 1600

        # case 5
        splitter = HoldoutSplitter(test_size=0.2, verbose=False)
        splits = splitter.split(dataset)
        for train, test in splits:
            tmp = train.target
            break
        assert tmp.__len__() == 1600

        # case 6
        with self.assertRaises(ValueError):
            splitter = HoldoutSplitter(test_size=3000)
            splits = splitter.split(dataset)
            for train, test in splits:
                tmp = train.target
                break

    def test_expanding_window_splitter(self):
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
        static_cov = {"f": 1, "g": 2}
        dataset = TSDataset(target, observed_cov, known_cov, static_cov)

        # case 0
        splitter = ExpandingWindowSplitter()
        splits = splitter.split(dataset)
        for train, test in splits:
            assert isinstance(train, TSDataset)

        # case 1
        splitter = ExpandingWindowSplitter()
        splits = splitter.split(dataset, return_index=True)
        for train, test in splits:
            assert isinstance(train, pd.DatetimeIndex)

        # case 2
        splitter = ExpandingWindowSplitter(n_splits=9)
        splits = splitter.split(dataset)
        for train, test in splits:
            tmp = train.target
            break
        assert tmp.__len__() == 200

        # case 3
        splitter = ExpandingWindowSplitter(n_splits=10, test_size=100)
        splits = splitter.split(dataset)
        for train, test in splits:
            tmp = train.target
            break
        assert tmp.__len__() == 1000

        # case 4
        splitter = ExpandingWindowSplitter(n_splits=10, test_size=100, max_train_size=50)
        splits = splitter.split(dataset)
        for train, test in splits:
            tmp = train.target
            break
        assert tmp.__len__() == 50

        # case 5
        splitter = ExpandingWindowSplitter(n_splits=10, test_size=100, skip_size=100)
        splits = splitter.split(dataset)
        for train, test in splits:
            tmp = train.target
            break
        assert tmp.__len__() == 900

        # case 6
        splitter = ExpandingWindowSplitter(n_splits=9, verbose=False)
        splits = splitter.split(dataset)
        for train, test in splits:
            tmp = train.target
            break
        assert tmp.__len__() == 200

        # case 7
        with self.assertRaises(ValueError):
            splitter = ExpandingWindowSplitter(n_splits=9, test_size=300)
            splits = splitter.split(dataset)
            for train, test in splits:
                tmp = train.target
                break

        # case 8
        with self.assertRaises(ValueError):
            splitter = ExpandingWindowSplitter(test_size=300, skip_size=2000)
            splits = splitter.split(dataset)
            for train, test in splits:
                tmp = train.target
                break

    def test_sliding_window_splitter(self):
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
        static_cov = {"f": 1, "g": 2}
        dataset = TSDataset(target, observed_cov, known_cov, static_cov)

        # case 0
        splitter = SlideWindowSplitter(train_size=100, test_size=100)
        splits = splitter.split(dataset)
        for train, test in splits:
            assert isinstance(train, TSDataset)

        # case 1
        splitter = SlideWindowSplitter(train_size=100, test_size=100)
        splits = splitter.split(dataset, return_index=True)
        for train, test in splits:
            assert isinstance(train, pd.DatetimeIndex)

        # case 2
        splitter = SlideWindowSplitter(train_size=200, test_size=100)
        splits = splitter.split(dataset)
        for train, test in splits:
            tmp = train.target
            break
        assert tmp.__len__() == 200

        # case 3
        splitter = SlideWindowSplitter(train_size=200, test_size=100, step_size=400)
        splits = splitter.split(dataset)
        for train, test in splits:
            tmp = train.target
            break
        assert tmp.__len__() == 200

        # case 4
        splitter = SlideWindowSplitter(train_size=200, test_size=100, skip_size=100)
        splits = splitter.split(dataset)
        for train, test in splits:
            tmp = train.target
            break
        assert tmp.__len__() == 200

        # case 5
        splitter = SlideWindowSplitter(train_size=200, test_size=100, skip_size=100)
        splits = splitter.split(dataset)
        for train, test in splits:
            tmp = train.target
            break
        assert tmp.__len__() == 200

        # case 6
        with self.assertRaises(ValueError):
            splitter = SlideWindowSplitter(train_size=20000, test_size=100)
            splits = splitter.split(dataset)
            for train, test in splits:
                tmp = train.target
                break

        # case 7
        with self.assertRaises(ValueError):
            splitter = SlideWindowSplitter(train_size=100, test_size=100, skip_size=20000)
            splits = splitter.split(dataset)
            for train, test in splits:
                tmp = train.target
                break

    def test_range_index(self):
        np.random.seed(2022)
        target = TimeSeries.load_from_dataframe(
            pd.Series(np.random.randn(1000).astype(np.float32),
                    #index=pd.date_range("2022-01-01", periods=2000, freq="15T"),
                    index=pd.RangeIndex(start=10, stop=2010, step=2),
                    name="a"
                    ), freq = 2)

        observed_cov = TimeSeries.load_from_dataframe(
            pd.DataFrame(
                np.random.randn(1000, 2).astype(np.float32),
                #index=pd.date_range("2022-01-01", periods=2000, freq="15T"),
                index=pd.RangeIndex(start=10, stop=2010, step=2),
                columns=["b", "c"]
            ), freq = 2)
        known_cov = TimeSeries.load_from_dataframe(
            pd.DataFrame(
                np.random.randn(1250, 2).astype(np.float32),
                #index=pd.date_range("2022-01-01", periods=2500, freq="15T"),
                index=pd.RangeIndex(start=10, stop=2510, step=2),
                columns=["b1", "c1"]
            ), freq = 2)
        static_cov = {"f": 1, "g": 2}
        dataset = TSDataset(target, observed_cov, known_cov, static_cov)
        # case 0
        splitter = SlideWindowSplitter(train_size=100, test_size=100)
        splits = splitter.split(dataset)
        for train, test in splits:
            assert isinstance(train, TSDataset)

        # case 1
        splitter = SlideWindowSplitter(train_size=100, test_size=100)
        splits = splitter.split(dataset, return_index=True)
        for train, test in splits:
            assert isinstance(train, pd.RangeIndex)
        # case 2
        splitter = SlideWindowSplitter(train_size=200, test_size=100)
        splits = splitter.split(dataset)
        for train, test in splits:
            train = train.target
            test = test.target
            break
        assert train.__len__() == 200
        assert test.__len__() == 100
        
        # case 3
        splitter = SlideWindowSplitter(train_size=200, test_size=100, step_size=400)
        splits = splitter.split(dataset)
        for train, test in splits:
            train = train.target
            test = test.target
            break
        assert train.__len__() == 200
        assert test.__len__() == 100

        # case 4
        splitter = SlideWindowSplitter(train_size=200, test_size=100, skip_size=100)
        splits = splitter.split(dataset)
        for train, test in splits:
            train = train.target
            test = test.target
            break
        assert train.__len__() == 200
        assert test.__len__() == 100

        # case 5
        splitter = SlideWindowSplitter(train_size=200, test_size=100, skip_size=100)
        splits = splitter.split(dataset)
        for train, test in splits:
            train = train.target
            test = test.target
            break
        assert train.__len__() == 200
        assert test.__len__() == 100
        # case 6
        with self.assertRaises(ValueError):
            splitter = SlideWindowSplitter(train_size=20000, test_size=100)
            splits = splitter.split(dataset)
            for train, test in splits:
                tmp = train.target
                break
        # case 7
        with self.assertRaises(ValueError):
            splitter = SlideWindowSplitter(train_size=100, test_size=100, skip_size=20000)
            splits = splitter.split(dataset)
            for train, test in splits:
                tmp = train.target
                break

if __name__ == "__main__":
    unittest.main()

