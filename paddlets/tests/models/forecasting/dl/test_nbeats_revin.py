# !/usr/bin/env python3
# -*- coding:utf-8 -*-

from unittest import TestCase, mock
import unittest
from enum import Enum
from typing import List, NewType, Tuple, Union

import paddle
import pandas as pd
import numpy as np

from paddlets.datasets import TimeSeries, TSDataset
from paddlets.models.common.callbacks import Callback
from paddlets.models.forecasting.dl.nbeats import (
    NBEATSModel,
    _TrendGenerator,
    _SeasonalityGenerator,
    _GType,
    _Block,
    _Stack
)

np.random.seed(2023)
paddle.seed(0)

        
class TestTrendBlock(TestCase):
    
    def setUp(self):        
        self.module = _Block
        self.params = {"num_layers": 4,
                       "layer_width": 64,
                       "expansion_coefficient_dim": 3,
                       "backcast_length": 96 * 7 + 9 * 4,
                       "in_chunk_len": (96 *7 + 9*4) * 4 + 96 * 2,
                       "target_length": 96,
                       "target_dim": 2,
                       "g_type": _GType.TREND
                      }
        self.data_input = [paddle.randn([64, 96 * 7 + 9 * 4, 2]),
                          paddle.randn([64, 96 * 7 + 9 * 4, 2]),
                          paddle.randn([64, 96, 2])]
    def test(self):
        model = self.module(**self.params)
        ret1, ret2 = model(*self.data_input)
        self.assertEqual(ret1.shape, [64, 708, 2])
        self.assertEqual(ret2.shape, [64, 96, 2])
        
        
class TestSeasonBlock(TestCase):
    
    def setUp(self):        
        self.module = _Block
        self.params = {"num_layers": 4,
                       "layer_width": 64,
                       "expansion_coefficient_dim": 3,
                       "backcast_length": 96 * 7 + 9 * 4,
                       "in_chunk_len": (96 *7 + 9*4) * 4 + 96 * 2,
                       "target_length": 96,
                       "target_dim": 2,
                       "g_type": _GType.SEASONALITY
                      }
        self.data_input = [paddle.randn([64, 96 * 7 + 9 * 4, 2]),
                          paddle.randn([64, 96 * 7 + 9 * 4, 2]),
                          paddle.randn([64, 96, 2])]
    def test(self):
        model = self.module(**self.params)
        ret1, ret2 = model(*self.data_input)
        self.assertEqual(ret1.shape, [64, 708, 2])
        self.assertEqual(ret2.shape, [64, 96, 2])   
        
        
class TestTrendStack(TestCase):
    
    def setUp(self):        
        self.module = _Stack
        self.params = {"num_blocks": 3,
                       "num_layers": 4,
                       "layer_width": 64,
                       "expansion_coefficient_dim": 3,
                       "backcast_length": 96 * 7 + 9 * 4,
                       "in_chunk_len": (96 *7 + 9*4) * 4 + 96 * 2,
                       "target_length": 96,
                       "target_dim": 2,
                       "g_type": _GType.TREND,
                      }
        self.data_input = [paddle.randn([64, 96 * 7 + 9 * 4, 2]),
                          paddle.randn([64, 96 * 7 + 9 * 4, 2]),
                          paddle.randn([64, 96, 2])]
    def test(self):
        model = self.module(**self.params)
        ret1, ret2 = model(*self.data_input)
        self.assertEqual(ret1.shape, [64, 708, 2])
        self.assertEqual(ret2.shape, [64, 96, 2])   
        
        
class TestSeasonStack(TestCase):
    
    def setUp(self):        
        self.module = _Stack
        self.params = {"num_blocks": 3,
                       "num_layers": 4,
                       "layer_width": 64,
                       "expansion_coefficient_dim": 3,
                       "backcast_length": 96 * 7 + 9 * 4,
                       "in_chunk_len": (96 *7 + 9*4) * 4 + 96 * 2,
                       "target_length": 96,
                       "target_dim": 2,
                       "g_type": _GType.SEASONALITY,
                      }
        self.data_input = [paddle.randn([64, 96 * 7 + 9 * 4, 2]),
                          paddle.randn([64, 96 * 7 + 9 * 4, 2]),
                          paddle.randn([64, 96, 2])]
    def test(self):
        model = self.module(**self.params)
        ret1, ret2 = model(*self.data_input)
        self.assertEqual(ret1.shape, [64, 708, 2])
        self.assertEqual(ret2.shape, [64, 96, 2])  
        

class TestNBeatsModel(TestCase):
    def setUp(self):
        target_single = TimeSeries.load_from_dataframe(
            pd.Series(
                np.random.randn(2000).astype(np.float32),
                index=pd.date_range("2022-01-01", periods=2000, freq="15T"),
                name="a"
            ))
        target_multi = TimeSeries.load_from_dataframe(
            pd.DataFrame(
                np.random.randn(2000, 2).astype(np.float32),
                index=pd.date_range("2022-01-01", periods=2000, freq="15T"),
                columns=["a1", "a2"]
            ))
        
        observed_cov = TimeSeries.load_from_dataframe(
            pd.DataFrame(
                np.random.randn(2000, 3).astype(np.float32),
                index=pd.date_range("2022-01-01", periods=2000, freq="15T"),
                columns=["b1", "b2", "b3"]
            ))
        known_cov = TimeSeries.load_from_dataframe(
            pd.DataFrame(
                np.random.randn(2500, 2).astype(np.float32),
                index=pd.date_range("2022-01-01", periods=2500, freq="15T"),
                columns=["c1", "c2"]
            ))
        static_cov = {"f": 1.0, "g": 2.0}

        int_target = TimeSeries.load_from_dataframe(
            pd.Series(
                np.random.randint(0, 10, 2000).astype(np.int32),
                index=pd.date_range("2022-01-01", periods=2000, freq="15T"),
                name="a"
            ))
        category_known_cov = TimeSeries.load_from_dataframe(
            pd.DataFrame(
                np.random.choice(["a", "b", "c"], [2500, 2]),
                index=pd.date_range("2022-01-01", periods=2500, freq="15T"),
                columns=["c1", "c2"]
            ))


        self.tsdataset1 = TSDataset(target_single, observed_cov, known_cov, static_cov)
        self.tsdataset2 = TSDataset(target_multi, observed_cov, known_cov, static_cov)
        self.tsdataset3 = TSDataset(target_single, None, None, None)
        self.tsdataset4 = TSDataset(int_target, None, category_known_cov, None)

        super().setUp()
    
    def test_fit(self):
        # case1: trend & seasonality
        reg = NBEATSModel(
            in_chunk_len = 7 * 96 + 9 * 4,
            out_chunk_len = 96,
            generic_architecture = False,
            num_blocks = [2, 4],
            skip_chunk_len = 15 * 4,
            eval_metrics = ["mse", "mae"],
            layer_widths = [64, 64],
            use_revin=True
        )
        reg.fit(self.tsdataset1)
        reg.fit(self.tsdataset1, self.tsdataset1)

        # case2: general architecture
        reg = NBEATSModel(
                in_chunk_len = 7 * 96 + 9 * 4,
                out_chunk_len = 96,
                generic_architecture = True,
                skip_chunk_len = 15 * 4,
                eval_metrics = ["mse", "mae"],
                layer_widths = 64,
                use_revin=True
                )
        reg.fit(self.tsdataset2, self.tsdataset2)

        # case3: no observed covariates
        reg = NBEATSModel(
                in_chunk_len = 7 * 96 + 9 * 4,
                out_chunk_len = 96,
                generic_architecture = True,
                skip_chunk_len = 15 * 4,
                eval_metrics = ["mse", "mae"],
                use_revin=True
                )
        reg.fit(self.tsdataset3, self.tsdataset3)

        #case4: bad case, invalid dtypes
        with self.assertRaises(ValueError):
            reg = NBEATSModel(
                in_chunk_len= 7 * 96 + 9 * 4,
                out_chunk_len = 96,
                generic_architecture = True,
                skip_chunk_len=4 * 4,
                eval_metrics=["mse", "mae"],
                use_revin=True
            )
            reg.fit(self.tsdataset4, self.tsdataset4)

        # case5: bad case, layer_widths invalid
        with self.assertRaises(ValueError):
            reg = NBEATSModel(
                    in_chunk_len = 7 * 96 + 9 * 4,
                    out_chunk_len = 96,
                    generic_architecture = True,
                    skip_chunk_len = 15 * 4,
                    eval_metrics = ["mse", "mae"],
                    layer_widths = [64, 64, 64]
                    )
            reg.fit(self.tsdataset3, self.tsdataset3)

    def test_predict(self):
        reg = NBEATSModel(
            in_chunk_len = 7 * 96 + 9 * 4,
            out_chunk_len = 96,
            skip_chunk_len = 15 * 4,
            eval_metrics = ["mse", "mae"],
            use_revin=True
        )
        # case1: single target
        reg.fit(self.tsdataset1, self.tsdataset1)
        res = reg.predict(self.tsdataset1)
        self.assertIsInstance(res, TSDataset)
        self.assertEqual(res.get_target().data.shape, (96, 1))

        reg = NBEATSModel(
            in_chunk_len = 7 * 96 + 9 * 4,
            out_chunk_len = 96,
            skip_chunk_len = 15 * 4,
            eval_metrics = ["mse", "mae"],
            use_revin=True
        )

        # case2: multi-target
        reg.fit(self.tsdataset2, self.tsdataset2)
        res = reg.predict(self.tsdataset2)
        self.assertIsInstance(res, TSDataset)
        self.assertEqual(res.get_target().data.shape, (96, 2))

if __name__ == "__main__":
    unittest.main()
