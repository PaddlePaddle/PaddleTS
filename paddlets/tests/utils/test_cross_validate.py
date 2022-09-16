# !/usr/bin/env python3
# -*- coding:utf-8 -*-
import sys
sys.path.append(".")
from typing import List
from unittest import TestCase
import unittest

import pandas as pd
import numpy as np

from paddlets.datasets.tsdataset import TimeSeries, TSDataset
from paddlets.utils.validation import cross_validate
from paddlets.datasets.splitter import ExpandingWindowSplitter, SlideWindowSplitter
from paddlets.models.forecasting import LSTNetRegressor
from paddlets.utils import check_train_valid_continuity


class TestCV(TestCase):
    def test_cv(self):
        np.random.seed(2022)

        target = TimeSeries.load_from_dataframe(
            pd.Series(np.random.randn(400).astype(np.float32),
                      index=pd.date_range("2022-01-01", periods=400, freq="15T"),
                      name="a"
                      ))
        target2 = TimeSeries.load_from_dataframe(
            pd.DataFrame(
                np.random.randn(400, 2).astype(np.float32),
                index=pd.date_range("2022-01-01", periods=400, freq="15T"),
                columns=["a1", "a2"]
            ))
        observed_cov = TimeSeries.load_from_dataframe(
            pd.DataFrame(
                np.random.randn(400, 2).astype(np.float32),
                index=pd.date_range("2022-01-01", periods=400, freq="15T"),
                columns=["b", "c"]
            ))
        known_cov = TimeSeries.load_from_dataframe(
            pd.DataFrame(
                np.random.randn(500, 2).astype(np.float32),
                index=pd.date_range("2022-01-01", periods=500, freq="15T"),
                columns=["b1", "c1"]
            ))
        static_cov = {"f": 1, "g": 2}
        dataset = TSDataset(target, observed_cov, known_cov, static_cov)
        self.tsdataset2 = TSDataset(target2, observed_cov, known_cov, static_cov)


        #case1
        param = {
            "skip_size": 1,
            "channels": 1,
            "kernel_size": 3,
            "rnn_cell_type": "GRU",
            "skip_rnn_cell_type": "GRU",
            "output_activation": None

        }
        lstnet = LSTNetRegressor(
            in_chunk_len=8,
            out_chunk_len=4,
            skip_chunk_len=0,
            max_epochs=1,
            **param
        )

        score = cross_validate(dataset, lstnet)
        assert score > 0

        #case2
        param = {
            "skip_size": 1,
            "channels": 1,
            "kernel_size": 3,
            "rnn_cell_type": "GRU",
            "skip_rnn_cell_type": "GRU",
            "output_activation": None

        }
        lstnet = LSTNetRegressor(
            in_chunk_len=8,
            out_chunk_len=4,
            skip_chunk_len=0,
            max_epochs=1,
            **param
        )

        score = cross_validate(dataset, lstnet, use_backtest=True)
        assert score > 0

        #case3
        param = {
            "skip_size": 1,
            "channels": 1,
            "kernel_size": 3,
            "rnn_cell_type": "GRU",
            "skip_rnn_cell_type": "GRU",
            "output_activation": None

        }
        lstnet = LSTNetRegressor(
            in_chunk_len=8,
            out_chunk_len=4,
            skip_chunk_len=0,
            max_epochs=1,
            **param
        )
        splitter = SlideWindowSplitter(train_size=100, test_size=100)
        score = cross_validate(dataset, lstnet,splitter=splitter, use_backtest=True)
        assert score > 0

        #case4
        param = {
            "skip_size": 1,
            "channels": 1,
            "kernel_size": 3,
            "rnn_cell_type": "GRU",
            "skip_rnn_cell_type": "GRU",
            "output_activation": None

        }
        lstnet = LSTNetRegressor(
            in_chunk_len=8,
            out_chunk_len=4,
            skip_chunk_len=0,
            max_epochs=1,
            **param
        )

        score = cross_validate(dataset, lstnet,splitter=splitter, use_backtest=True)
        assert score > 0

        #case5
        param = {
            "skip_size": 1,
            "channels": 1,
            "kernel_size": 3,
            "rnn_cell_type": "GRU",
            "skip_rnn_cell_type": "GRU",
            "output_activation": None

        }
        lstnet = LSTNetRegressor(
            in_chunk_len=8,
            out_chunk_len=4,
            skip_chunk_len=0,
            max_epochs=1,
            **param
        )

        score = cross_validate(dataset, lstnet,splitter=splitter, use_backtest=True, predict_window=24)
        assert score > 0

        #case5
        param = {
            "skip_size": 1,
            "channels": 1,
            "kernel_size": 3,
            "rnn_cell_type": "GRU",
            "skip_rnn_cell_type": "GRU",
            "output_activation": None

        }
        lstnet = LSTNetRegressor(
            in_chunk_len=8,
            out_chunk_len=4,
            skip_chunk_len=0,
            max_epochs=1,
            **param
        ) 

        res = cross_validate(dataset, lstnet,splitter=splitter, use_backtest=True, predict_window=24, return_score=False)
        assert res[1]["score"] > 0

    def test_check_train_valid_continuity(self):
        #DateTimeIndex
        target = TimeSeries.load_from_dataframe(
            pd.Series(np.random.randn(400).astype(np.float32),
                      index=pd.date_range("2022-01-01", periods=400, freq="15T"),
                      name="a"
                      ))
        observed_cov = TimeSeries.load_from_dataframe(
            pd.DataFrame(
                np.random.randn(400, 2).astype(np.float32),
                index=pd.date_range("2022-01-01", periods=400, freq="15T"),
                columns=["b", "c"]
            ))
        known_cov = TimeSeries.load_from_dataframe(
            pd.DataFrame(
                np.random.randn(500, 2).astype(np.float32),
                index=pd.date_range("2022-01-01", periods=500, freq="15T"),
                columns=["b1", "c1"]
            ))
        static_cov = {"f": 1, "g": 2}
        dataset = TSDataset(target, observed_cov, known_cov, static_cov)

        ts1,ts2 = dataset.split(0.5)
        ts3,ts4 = dataset.split(0.6)
        assert check_train_valid_continuity(ts1, ts2) == True
        assert check_train_valid_continuity(ts1, ts4) == False

        #RangeIndex
        index = pd.RangeIndex(0, 2000, 2)
        index2 = pd.RangeIndex(0, 2500, 2)
        target2 = pd.DataFrame(
                np.random.randn(2000, 2).astype(np.float32),
                index=pd.date_range("2022-01-01", periods=2000, freq="15T"),
                columns=["a1", "a2"])
        observed_cov = pd.DataFrame(
                np.random.randn(2000, 2).astype(np.float32),
                index=pd.date_range("2022-01-01", periods=2000, freq="15T"),
                columns=["b", "c"])
        known_cov = pd.DataFrame(
                np.random.randn(2500, 2).astype(np.float32),
                index=pd.date_range("2022-01-01", periods=2500, freq="15T"),
                columns=["b1", "c1"])
        static_cov = {"f": 1, "g": 2}
        target2 = target2.reset_index(drop=True).reindex(index)
        observed_cov = observed_cov.reset_index(drop=True).reindex(index)
        known_cov = known_cov.reset_index(drop=True).reindex(index2)
        dataset = TSDataset(
            TimeSeries.load_from_dataframe(target2, freq=index.step),
            TimeSeries.load_from_dataframe(observed_cov, freq=index.step),
            TimeSeries.load_from_dataframe(known_cov, freq=index2.step),
            static_cov)

        ts1,ts2 = dataset.split(0.2)
        ts3,ts4 = dataset.split(0.3)
        assert check_train_valid_continuity(ts1, ts2) == True
        assert check_train_valid_continuity(ts1, ts4) == False

if __name__ == "__main__":
    unittest.main()

