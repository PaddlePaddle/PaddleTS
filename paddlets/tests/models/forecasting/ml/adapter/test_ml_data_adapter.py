# !/usr/bin/env python3
# -*- coding:utf-8 -*-

import unittest

import numpy as np
import pandas as pd
from typing import Dict, Set
import math

from paddlets import TSDataset, TimeSeries
from paddlets.models.forecasting.ml.adapter.data_adapter import DataAdapter
from paddlets.models.forecasting.ml.adapter.ml_dataset import MLDataset
from paddlets.models.forecasting.ml.adapter.ml_dataloader import MLDataLoader


class TestDataAdapter(unittest.TestCase):
    def setup(self):
        """
        unittest setup
        """
        super().setUp()

    def test_to_ml_dataset(self):
        """
        测试 DataAdapter.to_ml_dataset()
        """
        ###############################################################
        # case 0 (good case) 非lag场景 + 给定一个数据集, 返回其中所有的样本 #
        ###############################################################
        # 这种场景是最简单的构建样本的调用方式, 常用于构建训练样本.
        # 构造paddlets tsdataset
        target_periods = 10
        observed_periods = target_periods
        known_periods = target_periods + 10

        param = {
            "in_chunk_len": 1,
            "skip_chunk_len": 0,
            "out_chunk_len": 1,
            "sampling_stride": 1,
            "time_window": (1, 9)
        }
        
        # 0.1 Both numeric and categorical cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_contain_numeric=True,
            cov_dtypes_contain_categorical=True
        )
        adapter = DataAdapter()
        
        # 调用接口, 构建样本(除了数据集之外, 不传入任何其他参数)
        sample_ds = adapter.to_ml_dataset(tsdataset)
        
        # 默认作为 X 的 target 长度为与 in_chunk 长度相等, 默认值为 1
        self.assertEqual(param["in_chunk_len"], sample_ds._target_in_chunk_len)
        # 默认不跳过任何时间点 (skip = 0)
        self.assertEqual(param["skip_chunk_len"], sample_ds._target_skip_chunk_len)
        # 默认输出的Y长度为1
        self.assertEqual(param["out_chunk_len"], sample_ds._target_out_chunk_len)
        # 默认样本步长 = 1
        self.assertEqual(param["sampling_stride"], sample_ds._sampling_stride)
        # 因为非lag场景, 因此目前不会计算默认的window值, 默认为None(什么时候计算取决于 MLDataset 的TODO完成情况)
        self.assertEqual(param["time_window"], sample_ds._time_window)
        # 根据默认参数可以计算出样本数量为 8000 - 1 条, 和 target_periods 的长度 - 1 相同
        self.assertEqual(target_periods - 1, len(sample_ds.samples))

        self._compare_tsdataset_and_sample_dataset(tsdataset=tsdataset, sample_ds=sample_ds, param=param, lag=False)

        ##############################################################################
        # case 1 (good case) 非lag场景, 给定一个数据集, 从中拆分用于训练, 验证, 测试的样本集 #
        ##############################################################################
        target_periods = 12
        observed_periods = target_periods
        known_periods = target_periods + 10
        tsdataset = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)

        # 初始化 adapter
        adapter = DataAdapter()
        common_param = {
            "in_chunk_len": 2,
            "skip_chunk_len": 1,
            "out_chunk_len": 2,
            "sampling_stride": 1
        }
        ratio = (0.5, 0.25, 0.25)
        target_len = len(tsdataset.get_target().data)
        window_bias = common_param["in_chunk_len"] + \
            common_param["skip_chunk_len"] + \
            common_param["out_chunk_len"] - \
            1

        # 构建train样本
        train_window_min = 0 + window_bias
        train_window_max = math.ceil(target_len * sum(ratio[:1]))
        train_param = {**common_param, "time_window": (train_window_min, train_window_max)}
        train_sample_ds = adapter.to_ml_dataset(tsdataset, **train_param)
        # 验证train的参数
        self.assertEqual(train_param["time_window"], train_sample_ds._time_window)

        # 验证train的数据
        self._compare_tsdataset_and_sample_dataset(
            tsdataset=tsdataset,
            sample_ds=train_sample_ds,
            param=train_param,
            lag=False
        )

        # 构建valid样本
        valid_window_min = train_window_max + 1
        valid_window_max = math.ceil(target_len * sum(ratio[:2]))
        valid_param = {**common_param, "time_window": (valid_window_min, valid_window_max)}
        valid_sample_ds = adapter.to_ml_dataset(tsdataset, **valid_param)

        # 验证valid的参数
        self.assertEqual(valid_param["time_window"], valid_sample_ds._time_window)

        # 验证 valid 数据
        self._compare_tsdataset_and_sample_dataset(
            tsdataset=tsdataset,
            sample_ds=valid_sample_ds,
            param=valid_param,
            lag=False
        )

        # 构建test样本
        test_window_min = valid_window_max + 1
        test_window_max = min(math.ceil(target_len * sum(ratio[:3])), target_len - 1)
        test_param = {**common_param, "time_window": (test_window_min, test_window_max)}
        test_sample_ds = adapter.to_ml_dataset(tsdataset, **test_param)

        # 验证test参数
        self.assertEqual(test_param["time_window"], test_sample_ds._time_window)

        # 验证 test 数据
        self._compare_tsdataset_and_sample_dataset(
            tsdataset=tsdataset,
            sample_ds=test_sample_ds,
            param=test_param,
            lag=False
        )

        ##############################################################################
        # case 2 (good case) 非lag场景, 给定一个数据集, 只从中构建一条只有X, 没有Y的预测样本 #
        ##############################################################################
        # 常用于在生产环境中，给真实的时序数据构建一条待预测的样本
        # 构造paddlets tsdataset
        target_periods = 10
        observed_periods = target_periods
        known_periods = target_periods + 10
        tsdataset = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)

        # 初始化 adapter
        adapter = DataAdapter()

        max_target_idx = len(tsdataset.get_target().data) - 1
        param = {
            "in_chunk_len": 2,
            "skip_chunk_len": 1,
            "out_chunk_len": 2,
            "sampling_stride": 1,
            "time_window": (
                # max_target_idx + skip + chunk
                max_target_idx + 1 + 2,
                max_target_idx + 1 + 2
            )
        }
        # 构建样本
        sample_ds = adapter.to_ml_dataset(tsdataset, **param)

        # 验证参数
        self.assertEqual(param["time_window"], sample_ds._time_window)

        # 验证数据
        self._compare_tsdataset_and_sample_dataset(
            tsdataset=tsdataset,
            sample_ds=sample_ds,
            param=param,
            future_target_is_nan=True,
            lag=False
        )

        ###################################################
        # case 3 (good case)                              #
        # 1) NOT lag.                                     #
        # 2) known_cov is None.                           #
        # 3) observed_cov is None.                        #
        # 4) static_cov is None.                          #
        # 5) built only one sample contains both X and Y. #
        ###################################################
        # 这种场景是在TSDataset被lag处理的前提下, 最简单的构建样本的调用方式, 常用于构建训练样本.
        # 构造paddlets tsdataset
        target_periods = 10
        observed_periods = target_periods
        known_periods = target_periods + 10
        tsdataset = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)

        # Explicitly set to None.
        tsdataset._known_cov = None
        tsdataset._observed_cov = None
        tsdataset._static_cov = None

        adapter = DataAdapter()

        param = {
            "in_chunk_len": 2,
            "skip_chunk_len": 0,
            "out_chunk_len": 1,
            "sampling_stride": 1,
            "time_window": (2, 9)
        }
        sample_ds = adapter.to_ml_dataset(tsdataset, **param)

        self.assertEqual(param["in_chunk_len"], sample_ds._target_in_chunk_len)
        self.assertEqual(param["skip_chunk_len"], sample_ds._target_skip_chunk_len)
        self.assertEqual(param["out_chunk_len"], sample_ds._target_out_chunk_len)
        # in chunk len is always equal to observed chunk len.
        self.assertEqual(param["in_chunk_len"], sample_ds._observed_cov_chunk_len)
        self.assertEqual(param["sampling_stride"], sample_ds._sampling_stride)
        self.assertEqual(param["time_window"], sample_ds._time_window)
        self.assertEqual(8, len(sample_ds.samples))

        self._compare_tsdataset_and_sample_dataset(tsdataset=tsdataset, sample_ds=sample_ds, param=param, lag=False)

        #############################################################
        # case 4 (good case) lag场景 + 给定一个数据集, 返回其中所有的样本 #
        #############################################################
        # 这种场景是在TSDataset被lag处理的前提下, 最简单的构建样本的调用方式, 常用于构建训练样本.
        # 构造paddlets tsdataset
        target_periods = 10
        observed_periods = target_periods
        known_periods = target_periods + 10
        tsdataset = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)

        # 初始化 adapter
        adapter = DataAdapter()

        # 调用接口, 构建lag场景的训练样本
        param = {
            # in = 0 代表 TSDataset被lag transform处理过
            "in_chunk_len": 0,
            "skip_chunk_len": 0,
            "out_chunk_len": 1,
            "sampling_stride": 1,
            "time_window": (1, 9)
        }
        sample_ds = adapter.to_ml_dataset(tsdataset, **param)

        # 默认作为 X 的 target 长度为与 in_chunk 长度相等, 默认值为 1
        self.assertEqual(param["in_chunk_len"], sample_ds._target_in_chunk_len)
        # 默认不跳过任何时间点 (skip = 0)
        self.assertEqual(param["skip_chunk_len"], sample_ds._target_skip_chunk_len)
        # 默认输出的Y长度为1
        self.assertEqual(param["out_chunk_len"], sample_ds._target_out_chunk_len)
        # lag场景中， observed_cov 长度固定为1
        self.assertEqual(1, sample_ds._observed_cov_chunk_len)
        # 默认样本步长 = 1
        self.assertEqual(param["sampling_stride"], sample_ds._sampling_stride)
        # 因为非lag场景, 因此目前不会计算默认的window值, 默认为None(什么时候计算取决于 MLDataset 的TODO完成情况)
        self.assertEqual(param["time_window"], sample_ds._time_window)
        # 根据默认参数可以计算出样本数量为 8000 - 1 条, 和 target_periods 的长度 - 1 相同
        self.assertEqual(target_periods - 1, len(sample_ds.samples))

        self._compare_tsdataset_and_sample_dataset(tsdataset=tsdataset, sample_ds=sample_ds, param=param, lag=True)

        ############################################################################
        # case 5 (good case) lag场景, 给定一个数据集, 从中拆分用于训练, 验证, 测试的样本集 #
        ############################################################################
        target_periods = 12
        observed_periods = target_periods
        known_periods = target_periods + 10
        tsdataset = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)

        # 初始化 adapter
        adapter = DataAdapter()
        common_param = {
            # in = 0 代表TSDataset被 lag transform处理过.
            "in_chunk_len": 0,
            "skip_chunk_len": 1,
            "out_chunk_len": 2,
            "sampling_stride": 1
        }
        ratio = (0.5, 0.25, 0.25)
        target_len = len(tsdataset.get_target().data)
        window_bias = max(1, common_param["in_chunk_len"]) + \
            common_param["skip_chunk_len"] + \
            common_param["out_chunk_len"] - \
            1

        # 构建train样本
        train_window_min = 0 + window_bias
        train_window_max = math.ceil(target_len * sum(ratio[:1]))
        train_param = {**common_param, "time_window": (train_window_min, train_window_max)}
        train_sample_ds = adapter.to_ml_dataset(tsdataset, **train_param)
        # 验证train的参数
        self.assertEqual(train_param["time_window"], train_sample_ds._time_window)

        # 验证train的数据
        self._compare_tsdataset_and_sample_dataset(
            tsdataset=tsdataset,
            sample_ds=train_sample_ds,
            param=train_param,
            lag=True
        )

        # 构建valid样本
        valid_window_min = train_window_max + 1
        valid_window_max = math.ceil(target_len * sum(ratio[:2]))
        valid_param = {**common_param, "time_window": (valid_window_min, valid_window_max)}
        valid_sample_ds = adapter.to_ml_dataset(tsdataset, **valid_param)

        # 验证valid的参数
        self.assertEqual(valid_param["time_window"], valid_sample_ds._time_window)

        # 验证 valid 数据
        self._compare_tsdataset_and_sample_dataset(
            tsdataset=tsdataset,
            sample_ds=valid_sample_ds,
            param=valid_param,
            lag=True
        )

        # 构建test样本
        test_window_min = valid_window_max + 1
        test_window_max = min(math.ceil(target_len * sum(ratio[:3])), target_len - 1)
        test_param = {**common_param, "time_window": (test_window_min, test_window_max)}
        test_sample_ds = adapter.to_ml_dataset(tsdataset, **test_param)

        # 验证test参数
        self.assertEqual(test_param["time_window"], test_sample_ds._time_window)

        # 验证 test 数据
        self._compare_tsdataset_and_sample_dataset(
            tsdataset=tsdataset,
            sample_ds=test_sample_ds,
            param=test_param,
            lag=True
        )

        ############################################################################
        # case 6 (good case) lag场景, 给定一个数据集, 只从中构建一条只有X, 没有Y的预测样本 #
        ############################################################################
        # 常用于在生产环境中，给真实的时序数据构建一条待预测的样本
        # 构造paddlets tsdataset
        target_periods = 10
        observed_periods = target_periods
        known_periods = target_periods + 10
        tsdataset = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)

        # 初始化 adapter
        adapter = DataAdapter()

        max_target_idx = len(tsdataset.get_target().data) - 1
        param = {
            # in = 0 代表TSDataset被 lag transform处理过.
            "in_chunk_len": 0,
            "skip_chunk_len": 1,
            "out_chunk_len": 2,
            "sampling_stride": 1,
            "time_window": (
                # max_target_idx + skip + chunk
                max_target_idx + 1 + 2,
                max_target_idx + 1 + 2
            )
        }
        # 构建样本
        sample_ds = adapter.to_ml_dataset(tsdataset, **param)

        # 验证参数
        self.assertEqual(param["time_window"], sample_ds._time_window)

        # 验证数据
        self._compare_tsdataset_and_sample_dataset(
            tsdataset=tsdataset,
            sample_ds=sample_ds,
            param=param,
            future_target_is_nan=True,
            lag=True
        )

        ###################################################
        # case 7 (good case)                              #
        # 1) Lag.                                         #
        # 2) known_cov is None.                           #
        # 3) observed_cov is None.                        #
        # 4) static_cov is None.                          #
        # 5) built only one sample contains both X and Y. #
        ###################################################
        # 常用于在生产环境中，在真实的时序数据中所有协变量均为None的情况下，构建训练样本。
        # 构造paddlets tsdataset
        target_periods = 10
        observed_periods = target_periods
        known_periods = target_periods + 10
        tsdataset = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)

        # Explicitly set all covariates to None.
        tsdataset._known_cov = None
        tsdataset._observed_cov = None
        tsdataset._static_cov = None

        # 初始化 adapter
        adapter = DataAdapter()

        param = {
            # in = 0 代表TSDataset被 lag transform处理过.
            "in_chunk_len": 0,
            "skip_chunk_len": 1,
            "out_chunk_len": 2,
            "sampling_stride": 1,
            "time_window": (3, 9)
        }
        # 构建样本
        sample_ds = adapter.to_ml_dataset(tsdataset, **param)

        # 验证参数
        self.assertEqual(param["time_window"], sample_ds._time_window)

        # 验证数据
        self._compare_tsdataset_and_sample_dataset(
            tsdataset=tsdataset,
            sample_ds=sample_ds,
            param=param,
            future_target_is_nan=False,
            lag=True
        )

        ##############################################################
        # case 8 (bad case) 非 lag + 给定一个数据集, 且time_window越下界 #
        ##############################################################
        # 构造paddlets tsdataset
        target_periods = 12
        observed_periods = target_periods
        known_periods = target_periods + 10
        tsdataset = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)

        # Given the following common params, there will be totally 4 invalid window lower bound (0, 1, 2, 3).
        # 基于给定的这组公共参数, 不符合预期的window下界共有 0, 1, 2, 3 这四个值.
        common_param = {
            # in > 0 means that TSDataset has NOT been processed by lag-transform.
            # in > 0 代表TSDataset没有被 lag transform处理过.
            "in_chunk_len": 2,
            "skip_chunk_len": 1,
            "out_chunk_len": 2,
            "sampling_stride": 1
        }

        # The following window[0] (i.e. 3) < min allowed w[0] (i.e. in + skip + out - 1), window[1] is valid.
        # 这里给定的window[0] (即 3) 小于允许的下限 (in + skip + out - 1 = 4). 上界没问题.
        param = {**common_param, "time_window": (3, len(tsdataset.get_target().data) - 1)}
        succeed = True
        try:
            _ = adapter.to_ml_dataset(tsdataset, **param)
        except ValueError:
            succeed = False
        self.assertFalse(succeed)

        # The following window[0] (i.e. 2) < min allowed w[0] (i.e. in + skip + out - 1), window[1] is valid.
        # 这里给定的window[0] (即 2) 小于允许的下限 (in + skip + out - 1 = 4). 上界没问题.
        param = {**common_param, "time_window": (2, len(tsdataset.get_target().data) - 1)}
        succeed = True
        try:
            _ = adapter.to_ml_dataset(tsdataset, **param)
        except ValueError:
            succeed = False
        self.assertFalse(succeed)

        # The following window[0] (i.e. 1) < min allowed w[0] (i.e. in + skip + out - 1), window[1] is valid.
        # 这里给定的window[0] (即 1) 小于允许的下限 (in + skip + out - 1 = 4). 上界没问题.
        param = {**common_param, "time_window": (1, len(tsdataset.get_target().data) - 1)}
        succeed = True
        try:
            _ = adapter.to_ml_dataset(tsdataset, **param)
        except ValueError:
            succeed = False
        self.assertFalse(succeed)

        # The following window[0] (i.e. 0) < min allowed w[0] (i.e. in + skip + out - 1), window[1] is valid.
        # 这里给定的window[0] (即 0) 小于允许的下限 (in + skip + out - 1 = 4). 上界没问题.
        param = {**common_param, "time_window": (0, len(tsdataset.get_target().data) - 1)}
        succeed = True
        try:
            _ = adapter.to_ml_dataset(tsdataset, **param)
        except ValueError:
            succeed = False
        self.assertFalse(succeed)

        ############################################################
        # case 9 (bad case) lag + 给定一个数据集, 且time_window越下界 #
        ############################################################
        # 构造paddlets tsdataset
        target_periods = 12
        observed_periods = target_periods
        known_periods = target_periods + 10
        tsdataset = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)

        # 初始化 adapter
        adapter = DataAdapter()

        # Given the following common params, there will be totally 3 invalid window lower bound (0, 1, 2).
        # 基于给定的这组公共参数, 不考虑负数的情况下, 不符合预期的window下界共有 0, 1, 2 这三个值.
        common_param = {
            # in = 0 means that TSDataset has been processed by lag-transform.
            # in = 0 代表TSDataset被 lag transform处理过.
            "in_chunk_len": 0,
            "skip_chunk_len": 1,
            "out_chunk_len": 2,
            "sampling_stride": 1
        }

        # The following window[0] (i.e. 2) < min allowed w[0] (i.e. in + skip + out - 1), window[1] is valid.
        # 这里给定的window[0] (即 2) 小于允许的下限 (in + skip + out - 1 = 4). 上界没问题.
        param = {**common_param, "time_window": (2, len(tsdataset.get_target().data) - 1)}
        succeed = True
        try:
            _ = adapter.to_ml_dataset(tsdataset, **param)
        except ValueError:
            succeed = False
        self.assertFalse(succeed)

        # The following window[0] (i.e. 1) < min allowed w[0] (i.e. in + skip + out - 1), window[1] is valid.
        # 这里给定的window[0] (即 1) 小于允许的下限 (in + skip + out - 1 = 4). 上界没问题.
        param = {**common_param, "time_window": (1, len(tsdataset.get_target().data) - 1)}
        succeed = True
        try:
            _ = adapter.to_ml_dataset(tsdataset, **param)
        except ValueError:
            succeed = False
        self.assertFalse(succeed)

        # The following window[0] (i.e. 0) < min allowed w[0] (i.e. in + skip + out - 1), window[1] is valid.
        # 这里给定的window[0] (即 0) 小于允许的下限 (in + skip + out - 1 = 4), 上界没问题.
        param = {**common_param, "time_window": (0, len(tsdataset.get_target().data) - 1)}
        succeed = True
        try:
            _ = adapter.to_ml_dataset(tsdataset, **param)
        except ValueError:
            succeed = False
        self.assertFalse(succeed)

        ##############################################################
        # case 10 (bad case) 非lag场景 + 给定数据集, 且time_window越上界 #
        ##############################################################
        # 构造paddlets tsdataset
        target_periods = 10
        observed_periods = target_periods
        known_periods = target_periods + 10
        tsdataset = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)

        # 初始化 adapter
        adapter = DataAdapter()
        param = {
            # in > 0, 因此是非lag场景
            "in_chunk_len": 2,
            "skip_chunk_len": 1,
            "out_chunk_len": 2,
            "sampling_stride": 1,
            # 这里给定的window[1] 超过了允许的上限 (target_ts长度 + skip + out - 1), 因此不符合预期, 会报错. 下界没问题
            # 这里给定的上界在尾部多加了一个 1, 超过了一个时间点 (预期是 10 + 1 + 2 - 1 = 12, 实际给定的是 13)
            "time_window": (
                2,
                len(tsdataset.get_target().data) + 1 + 2 - 1 + 1
            )
        }
        succeed = True
        try:
            _ = adapter.to_ml_dataset(tsdataset, **param)
        except ValueError:
            succeed = False
        self.assertFalse(succeed)

        #############################################################
        # case 11 (bad case) lag场景 + 给定数据集, 且time_window越上界 #
        #############################################################
        # 构造paddlets tsdataset
        target_periods = 10
        observed_periods = target_periods
        known_periods = target_periods + 10
        tsdataset = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)

        # 初始化 adapter
        adapter = DataAdapter()
        param = {
            # in = 0, 因此是lag场景
            "in_chunk_len": 0,
            "skip_chunk_len": 1,
            "out_chunk_len": 2,
            "sampling_stride": 1,
            # 这里给定的window[1] 超过了允许的上限 (target_ts长度 + skip + out - 1), 因此不符合预期, 会报错. 下界没问题
            # 这里给定的上界在尾部多加了一个 1, 超过了一个时间点 (预期是 10 + 1 + 2 - 1 = 12, 实际给定的是 13)
            "time_window": (
                2,
                len(tsdataset.get_target().data) + 1 + 2 - 1 + 1
            )
        }
        succeed = True
        try:
            _ = adapter.to_ml_dataset(tsdataset, **param)
        except ValueError:
            succeed = False
        self.assertFalse(succeed)

        #####################################################################################
        # case 12 (bad case) 非lag场景 + 给定数据集 + 给定 time_window, 且known_cov长度不符合预期 #
        #####################################################################################
        # 这个 case 中 time_window 上界没有超过 max_target_idx, 构建的样本即包含X, 也包含skip_chunk + Y.
        # 构造paddlets tsdataset
        target_periods = 10
        observed_periods = target_periods

        # 因为 window 上界没有超过 max_target, 所以不需要将 max_known_timestamp 和 max_target_timestamp 相比, 而只需要比较
        # window[1] 对应的时间戳  window_upper_bound_timestamp 和 max_known_timestamp 的时间戳大小即可.
        # 另外, 这里所说的超过, 是指在 TimeSeries 中的TimeStamp时间戳的大小比较, 而不是这里的periods / idx 的数字大小.
        # 示例:
        # 给定 known_periods 长度比 target_periods 长度小 1 个时间步. 求解过程如下:
        # 已知 target_periods = 10, 则
        # target_timeindex = [0:00, 1:00, 2:00, 3:00, 4:00, 5:00, 6:00, 7:00, 8:00, 9:00]
        # 可知 max_target_idx = 9, max_target_timestamp = target_timeindex[-1] = 9:00
        # 同时, 已知 known_periods = 10 - 1 = 9, 所以
        # known_timeindex = [0:00, 1:00, 2:00, 3:00, 4:00, 5:00, 6:00, 7:00, 8:00]
        # max_known_timeindex = known_timeindex[-1] = 8:00, 并且
        # max_target_timestamp (即9:00) 在 known_timeindex 中的位置 idx = known_timeindex.index_of(9:00) = 9,
        # 同时, 因为 time_window = (3, 9), 所以
        # window_upper_bound_timestamp = target_timeindex[9] = 9:00, 所以
        # max_known_timestamp (即 8:00) < window_upper_bound_timestamp (即 9:00),  所以无法构建样本.
        known_periods = target_periods - 1
        tsdataset = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)

        # 初始化 adapter
        adapter = DataAdapter()
        param = {
            # in > 0, 因此是非lag场景
            "in_chunk_len": 1,
            "skip_chunk_len": 1,
            "out_chunk_len": 2,
            "sampling_stride": 1,
            # known_cov长度=8 小于window[1]
            "time_window": (3, 9)
        }
        succeed = True
        try:
            _ = adapter.to_ml_dataset(tsdataset, **param)
        except ValueError:
            succeed = False
        self.assertFalse(succeed)

        #####################################################################################
        # case 13 (bad case) 非lag场景 + 给定数据集 + 给定 time_window, 且known_cov长度不符合预期 #
        #####################################################################################
        # 这个 case 中 time_window 上界超过了 max_target_idx, 构建的样本只包含X, 不包含 skip_chunk 或 Y.
        # 构造paddlets tsdataset
        target_periods = 10
        observed_periods = target_periods

        # 这里所说的超过, 是指在 TimeSeries 中的TimeStamp时间戳的大小比较, 而不是这里的periods / idx 的数字大小.
        # 示例:
        # 给定一个超过了 max_target_idx (9) 的 known_cov, 但没超过 time_window[1] (即12), 所以仍然不符合预期.
        # 已知target_periods = 10, 则
        # target_timeindex = [0:00, 1:00, 2:00, 3:00, 4:00, 5:00, 6:00, 7:00, 8:00, 9:00]
        # 可知 max_target_idx = 9, max_target_timestamp = target_timeindex[-1] = 9:00
        # 同时, 已知 known_periods = 11, 所以
        # known_timeindex = [0:00, 1:00, 2:00, 3:00, 4:00, 5:00, 6:00, 7:00, 8:00, 9:00, 10:00]
        # max_known_timeindex = known_timeindex[-1] = 10:00, 并且
        # max_target_timestamp (即9:00) 在 known_timeindex 中的位置 idx = known_timeindex.index_of(9:00) = 9,
        # 同时, 因为 time_window = (12, 12), 所以window的上界超出target的时间步数量是 exceed_time_steps = 12 - 9 = 3
        # 所以要求 known_cov 在 idx 之后, 还需要有额外的 3 个时间点, 即 known_timeindex[9:] 的长度必须 > 3. 但是实际上
        # len(known_timeindex[9:]) = len([9:00, 10:00]) = 2, 小于 exceed_time_steps (即 3), 所以无法构建样本.
        known_periods = 11
        tsdataset = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)

        # 初始化 adapter
        adapter = DataAdapter()
        param = {
            # in > 0, 因此是非lag场景
            "in_chunk_len": 1,
            "skip_chunk_len": 1,
            "out_chunk_len": 2,
            "sampling_stride": 1,
            # known_cov长度=11 小于window[1]
            "time_window": (12, 12)
        }
        succeed = True
        try:
            _ = adapter.to_ml_dataset(tsdataset, **param)
        except ValueError:
            succeed = False
        self.assertFalse(succeed)

        ####################################################################################
        # case 14 (bad case) lag场景 + 给定数据集 + 给定 time_window, 且known_cov长度不符合预期 #
        ####################################################################################
        # 这个 case 中 time_window 上界没有超过 max_target_idx, 构建的样本即包含X, 也包含skip_chunk + Y.
        # 构造paddlets tsdataset
        target_periods = 10
        observed_periods = target_periods

        # 因为 window 上界没有超过 max_target, 所以不需要将 max_known_timestamp 和 max_target_timestamp 相比, 而只需要比较
        # window[1] 对应的时间戳  window_upper_bound_timestamp 和 max_known_timestamp 的时间戳大小即可.
        # 另外, 这里所说的超过, 是指在 TimeSeries 中的TimeStamp时间戳的大小比较, 而不是这里的periods / idx 的数字大小.
        # 示例:
        # 给定 known_periods 长度比 target_periods 长度小 1 个时间步. 求解过程如下:
        # 已知 target_periods = 10, 则 max_target_idx = 9, 又因为 window[1] = 7 < 9, 所以
        # 这个case不用考虑 max_target, 只需要考虑 window[1] 和 max_known 即可.
        # 首先, 可知 target_timeindex = [0:00, 1:00, 2:00, 3:00, 4:00, 5:00, 6:00, 7:00, 8:00, 9:00]
        # 同时, 因为 window[1] = 7, 则 window_upper_bound_timestamp = target_timeindex[7] = 7:00
        # 同时, 已知 known_periods = 7, 所以
        # known_timeindex = [0:00, 1:00, 2:00, 3:00, 4:00, 5:00, 6:00]
        # 所以 max_known_timeindex = known_timeindex[-1] = 6:00
        # 所以, 因为 max_known_timeindex (即 6:00) < window_upper_bound_timestamp (即 7:00), 所以无法构建样本.
        known_periods = 7
        tsdataset = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)

        # 初始化 adapter
        adapter = DataAdapter()
        param = {
            # in = 0, 因此是lag场景
            "in_chunk_len": 0,
            "skip_chunk_len": 1,
            "out_chunk_len": 2,
            "sampling_stride": 1,
            "time_window": (3, 7)
        }
        succeed = True
        try:
            _ = adapter.to_ml_dataset(tsdataset, **param)
        except ValueError:
            succeed = False
        self.assertFalse(succeed)

        ####################################################################################
        # case 15 (bad case) lag场景 + 给定数据集 + 给定 time_window, 且known_cov长度不符合预期 #
        ####################################################################################
        # 这个 case 中 time_window 上界超过了 max_target_idx, 构建的样本只包含X, 不包含 skip_chunk 或 Y.
        # 构造paddlets tsdataset
        target_periods = 10
        observed_periods = target_periods

        # 这里所说的超过, 是指在 TimeSeries 中的TimeStamp时间戳的大小比较, 而不是这里的periods / idx 的数字大小.
        # 示例:
        # 给定一个超过了 max_target_idx (9) 的 known_cov, 但没超过 time_window[1] (即12), 所以仍然不符合预期.
        # 已知target_periods = 10, 则
        # target_timeindex = [0:00, 1:00, 2:00, 3:00, 4:00, 5:00, 6:00, 7:00, 8:00, 9:00]
        # 可知 max_target_idx = 9, max_target_timestamp target_timeindex[-1] = 9:00
        # 同时, 已知 known_periods = 11, 所以
        # known_timeindex = [0:00, 1:00, 2:00, 3:00, 4:00, 5:00, 6:00, 7:00, 8:00, 9:00, 10:00]
        # max_known_timeindex = known_timeindex[-1] = 10:00, 并且
        # max_target_timestamp 在 known_timeindex 中的位置是 idx = known_timeindex.index_of(9:00) = 9.
        # 同时, 因为 time_window = (12, 12), 所以window的上界超出target的时间步数量是 exceed_time_steps = 12 - 9 = 3
        # 所以要求 known_cov 在 idx 之后, 还需要有额外的 3 个时间点, 即 known_timeindex[9:] 的长度必须 > 3. 但是实际上
        # len(known_timeindex[9:]) = len([9:00, 10:00]) = 2, 小于 exceed_time_steps (即 3), 所以无法构建样本.
        known_periods = 11
        tsdataset = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)

        # 初始化 adapter
        adapter = DataAdapter()
        param = {
            # in = 0, 因此是lag场景
            "in_chunk_len": 0,
            "skip_chunk_len": 1,
            "out_chunk_len": 2,
            "sampling_stride": 1,
            # known_cov长度11没超过 time_window[1] (即12), 所以仍然不符合预期.
            "time_window": (12, 12)
        }
        succeed = True
        try:
            _ = adapter.to_ml_dataset(tsdataset, **param)
        except ValueError:
            succeed = False
        self.assertFalse(succeed)

        ###############################################################################################################
        # case 16 (bad case) 非lag场景 + 给定 time_window + time_window上界没有超过max_target, 但observed_cov时间戳上界太小 #
        ###############################################################################################################
        target_periods = 10

        # 这里所说的超过, 是比较 timestamp 时间戳的大小比较, 而不是比较 periods / idx 的数字大小.
        # 示例:
        # 已知 target_periods = 10, 则
        # target_timeindex = [0:00, 1:00, 2:00, 3:00, 4:00, 5:00, 6:00, 7:00, 8:00, 9:00]
        # 同时, 因为 observed_periods = 5, 则
        # observed_timeindex = [0:00, 1:00, 2:00, 3:00, 4:00]
        # 所以 max_observed_timestamp = observed_timeindex[-1] = 4:00
        # 同时, 因为 skip_chunk_len = 1, out_chunk_len = 2, time_window = (3, 8), 则
        # 最后一条样本的 past_target (即X部分) 的末尾索引 = window[1] - out_chunk_len - skip_chunk_len = 8 - 2 - 1 = 5
        # 所以最后一条样本的X部分末尾的时间戳 last_sample_past_target_tail_timestamp = target_timeindex[5] = 5:00.
        # 因为 max_observed_timestamp (即 4:00) < last_sample_past_target_tail_timestamp (即 5:00), 所以无法构建样本.
        observed_periods = 5
        known_periods = target_periods + 10
        tsdataset = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)

        # 初始化 adapter
        adapter = DataAdapter()
        param = {
            # in > 0, 因此是非lag场景
            "in_chunk_len": 1,
            "skip_chunk_len": 1,
            "out_chunk_len": 2,
            "sampling_stride": 1,
            # time_window[1] 没有超过 max_target_idx
            "time_window": (3, 8)
        }
        succeed = True
        try:
            _ = adapter.to_ml_dataset(tsdataset, **param)
        except ValueError:
            succeed = False
        self.assertFalse(succeed)

        #############################################################################################################
        # case 17 (bad case) lag场景 + 给定 time_window + time_window上界没有超过max_target, 但observed_cov时间戳上界太小 #
        #############################################################################################################
        # 因为 observed 的校验逻辑中, 只需要考虑 skip_chunk_len 和 out_chunk_len, 和 in_chunk_len 无关，所以其实
        # case 13 和 case 14 (当前case) 逻辑上没有区别。但是仍然区分了 lag / 非lag 两个场景测试两次，没有坏处。
        # 构造paddlets tsdataset
        target_periods = 10

        # 这里所说的超过, 是比较 timestamp 时间戳的大小比较, 而不是比较 periods / idx 的数字大小.
        # 示例:
        # 已知 target_periods = 10, 则
        # target_timeindex = [0:00, 1:00, 2:00, 3:00, 4:00, 5:00, 6:00, 7:00, 8:00, 9:00]
        # 同时, 因为 observed_periods = 5, 则
        # observed_timeindex = [0:00, 1:00, 2:00, 3:00, 4:00]
        # 所以 max_observed_timestamp = observed_timeindex[-1] = 4:00
        # 同时, 因为 skip_chunk_len = 1, out_chunk_len = 2, time_window = (3, 8), 则
        # 最后一条样本的 past_target (即X部分) 的末尾索引 = window[1] - out_chunk_len - skip_chunk_len = 8 - 2 - 1 = 5
        # 所以最后一条样本的X部分末尾的时间戳 last_sample_past_target_tail_timestamp = target_timeindex[5] = 5:00.
        # 因为 max_observed_timestamp (即 4:00) < last_sample_past_target_tail_timestamp (即 5:00), 所以无法构建样本.
        observed_periods = 5
        known_periods = target_periods + 10
        tsdataset = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)

        # 初始化 adapter
        adapter = DataAdapter()
        param = {
            # in = 0, 因此是lag场景
            "in_chunk_len": 0,
            "skip_chunk_len": 1,
            "out_chunk_len": 2,
            "sampling_stride": 1,
            # observed_cov_timestamp 没有超过 min(max_target_timestamp, time_window[1]对应的time_stamp), 仍然不符合预期.
            "time_window": (3, 8)
        }
        succeed = True
        try:
            _ = adapter.to_ml_dataset(tsdataset, **param)
        except ValueError:
            succeed = False
        self.assertFalse(succeed)

        #############################################################################################################
        # case 18 (bad case) 非lag场景 + 给定 time_window + time_window上界超过了max_target, 但observed_cov时间戳上界太小 #
        #############################################################################################################
        target_periods = 10

        # 这里所说的超过, 是比较 timestamp 时间戳的大小比较, 而不是比较 periods / idx 的数字大小.
        # 示例:
        # 已知 target_periods = 10, 则
        # target_timeindex = [0:00, 1:00, 2:00, 3:00, 4:00, 5:00, 6:00, 7:00, 8:00, 9:00]
        # 所以 max_target_timestamp = target_timeindex[-1] = 9:00
        # 同时, 因为 observed_periods = 9, 则
        # observed_timeindex = [0:00, 1:00, 2:00, 3:00, 4:00, 5:00, 6:00, 7:00, 8:00]
        # 所以 max_observed_timestamp = observed_timeindex[-1] = 8:00
        # 同时, time_window = (3, 8), 则只需要和 max_target_timestamp 比较即可.
        # 因为 max_observed_timestamp (即 8:00) < max_target_timestamp (即 9:00), 所以无法构建样本.
        observed_periods = target_periods - 1
        known_periods = target_periods + 10
        tsdataset = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)

        # 初始化 adapter
        adapter = DataAdapter()
        param = {
            # in > 0, 因此是非lag场景
            "in_chunk_len": 1,
            "skip_chunk_len": 1,
            "out_chunk_len": 2,
            "sampling_stride": 1,
            # time_window[1] 超过了 max_target_idx
            "time_window": (12, 12)
        }
        succeed = True
        try:
            _ = adapter.to_ml_dataset(tsdataset, **param)
        except ValueError:
            succeed = False
        self.assertFalse(succeed)

        ############################################################################################################
        # case 19 (bad case) lag场景 + 给定 time_window + time_window上界超过了max_target, 但observed_cov时间戳上界太小 #
        ############################################################################################################
        # 因为 observed 的校验逻辑中, 只需要考虑 skip_chunk_len 和 out_chunk_len, 和 in_chunk_len 无关，所以其实
        # case 16 和 case 17 (当前case) 逻辑上没有区别。但是仍然区分了 lag / 非lag 两个场景测试两次，没有坏处。
        # 构造paddlets tsdataset
        target_periods = 10

        # 这里所说的超过, 是比较 timestamp 时间戳的大小比较, 而不是比较 periods / idx 的数字大小.
        # 示例:
        # 已知 target_periods = 10, 则
        # target_timeindex = [0:00, 1:00, 2:00, 3:00, 4:00, 5:00, 6:00, 7:00, 8:00, 9:00]
        # 所以 max_target_timestamp = target_timeindex[-1] = 9:00
        # 同时, 因为 observed_periods = 9, 则
        # observed_timeindex = [0:00, 1:00, 2:00, 3:00, 4:00, 5:00, 6:00, 7:00, 8:00]
        # 所以 max_observed_timestamp = observed_timeindex[-1] = 8:00
        # 同时, time_window = (3, 8), 则只需要和 max_target_timestamp 比较即可.
        # 因为 max_observed_timestamp (即 8:00) < max_target_timestamp (即 9:00), 所以无法构建样本.
        observed_periods = target_periods - 1
        known_periods = target_periods + 10
        tsdataset = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)

        # 初始化 adapter
        adapter = DataAdapter()
        param = {
            # in = 0, 因此是lag场景
            "in_chunk_len": 0,
            "skip_chunk_len": 1,
            "out_chunk_len": 2,
            "sampling_stride": 1,
            #  time_window[1] 超过了 max_target_idx
            "time_window": (12, 12)
        }
        succeed = True
        try:
            _ = adapter.to_ml_dataset(tsdataset, **param)
        except ValueError:
            succeed = False
        self.assertFalse(succeed)

    def test_to_ml_dataloader(self):
        """
        测试 DataAdapter.to_ml_dataloader()
        """
        lag_param = {
            "in_chunk_len": 0,
            "skip_chunk_len": 0,
            "out_chunk_len": 1,
            "sampling_stride": 1,
            "time_window": (1, 19)
        }

        not_lag_param = {
            # in > 0, 因此不是lag场景
            "in_chunk_len": 3,
            "skip_chunk_len": 0,
            "out_chunk_len": 1,
            "sampling_stride": 1,
            "time_window": (3, 19)
        }

        all_target_keys = {"past_target", "future_target"}
        all_known_keys = {"known_cov_numeric", "known_cov_categorical"}
        all_observed_keys = {"observed_cov_numeric", "observed_cov_categorical"}
        all_static_keys = {"static_cov_numeric", "static_cov_categorical"}

        all_numeric_keys = {
            "known_cov_numeric",
            "observed_cov_numeric",
            "static_cov_numeric"
        }

        all_categorical_keys = {
            "known_cov_categorical",
            "observed_cov_categorical",
            "static_cov_categorical"
        }
        ################################
        # case 0 (good case)           #
        # 1) Not Lag.                  #
        # 2) known_cov is NOT None.    #
        # 3) observed_cov is NOT None. #
        # 2) static_cov is NOT None.   #
        ################################
        # 构造paddlets tsdataset
        target_periods = 20
        observed_periods = target_periods
        known_periods = target_periods + 10

        # 0.1 Both numeric and categorical cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_contain_numeric=True,
            cov_dtypes_contain_categorical=True
        )
        adapter = DataAdapter()
        sample_ds = adapter.to_ml_dataset(tsdataset, **not_lag_param)
        batch_size = 2
        sample_dataloader = adapter.to_ml_dataloader(sample_ds, batch_size=batch_size)

        self._compare_sample_dataset_and_sample_dataloader(
            sample_ds=sample_ds,
            sample_dataloader=sample_dataloader,
            batch_size=batch_size,
            good_keys=all_target_keys.union(all_numeric_keys).union(all_categorical_keys),
            lag=False
        )

        # 0.2 ONLY numeric cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_contain_numeric=True,
            cov_dtypes_contain_categorical=False
        )

        adapter = DataAdapter()
        sample_ds = adapter.to_ml_dataset(tsdataset, **not_lag_param)
        batch_size = 2
        sample_dataloader = adapter.to_ml_dataloader(sample_ds, batch_size=batch_size)

        self._compare_sample_dataset_and_sample_dataloader(
            sample_ds=sample_ds,
            sample_dataloader=sample_dataloader,
            batch_size=batch_size,
            good_keys=all_target_keys.union(all_numeric_keys),
            lag=False
        )

        # 0.3 ONLY categorical cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_contain_numeric=False,
            cov_dtypes_contain_categorical=True
        )

        adapter = DataAdapter()
        sample_ds = adapter.to_ml_dataset(tsdataset, **not_lag_param)
        batch_size = 2
        sample_dataloader = adapter.to_ml_dataloader(sample_ds, batch_size=batch_size)

        self._compare_sample_dataset_and_sample_dataloader(
            sample_ds=sample_ds,
            sample_dataloader=sample_dataloader,
            batch_size=batch_size,
            good_keys=all_target_keys.union(all_categorical_keys),
            lag=False
        )

        ################################
        # case 1 (good case)           #
        # 1) Lag.                      #
        # 2) known_cov is NOT None.    #
        # 3) observed_cov is NOT None. #
        # 2) static_cov is NOT None.   #
        ################################
        # 1.1 Both numeric and categorical cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_contain_numeric=True,
            cov_dtypes_contain_categorical=True
        )

        adapter = DataAdapter()
        sample_ds = adapter.to_ml_dataset(tsdataset, **lag_param)
        batch_size = 2
        sample_dataloader = adapter.to_ml_dataloader(sample_ds, batch_size=batch_size)

        self._compare_sample_dataset_and_sample_dataloader(
            sample_ds=sample_ds,
            sample_dataloader=sample_dataloader,
            batch_size=batch_size,
            good_keys=all_target_keys.union(all_numeric_keys).union(all_categorical_keys),
            lag=True
        )

        # 1.2 ONLY numeric cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_contain_numeric=True,
            cov_dtypes_contain_categorical=False
        )

        adapter = DataAdapter()
        sample_ds = adapter.to_ml_dataset(tsdataset, **lag_param)
        batch_size = 2
        sample_dataloader = adapter.to_ml_dataloader(sample_ds, batch_size=batch_size)

        self._compare_sample_dataset_and_sample_dataloader(
            sample_ds=sample_ds,
            sample_dataloader=sample_dataloader,
            batch_size=batch_size,
            good_keys=all_target_keys.union(all_numeric_keys),
            lag=True
        )

        # 1.3 ONLY categorical cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_contain_numeric=False,
            cov_dtypes_contain_categorical=True
        )

        adapter = DataAdapter()
        sample_ds = adapter.to_ml_dataset(tsdataset, **lag_param)
        batch_size = 2
        sample_dataloader = adapter.to_ml_dataloader(sample_ds, batch_size=batch_size)

        self._compare_sample_dataset_and_sample_dataloader(
            sample_ds=sample_ds,
            sample_dataloader=sample_dataloader,
            batch_size=batch_size,
            good_keys=all_target_keys.union(all_categorical_keys),
            lag=True
        )

        ################################
        # case 2 (good case)           #
        # 1) Not Lag.                  #
        # 2) known_cov is None.        #
        # 3) observed_cov is NOT None. #
        # 2) static_cov is NOT None.   #
        ################################
        # 2.1 Both numeric and categorical cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_contain_numeric=True,
            cov_dtypes_contain_categorical=True
        )
        tsdataset._known_cov = None

        adapter = DataAdapter()
        sample_ds = adapter.to_ml_dataset(tsdataset, **not_lag_param)
        batch_size = 2
        sample_dataloader = adapter.to_ml_dataloader(sample_ds, batch_size=batch_size)

        self._compare_sample_dataset_and_sample_dataloader(
            sample_ds=sample_ds,
            sample_dataloader=sample_dataloader,
            batch_size=batch_size,
            good_keys=all_target_keys.union(all_numeric_keys).union(all_categorical_keys) - all_known_keys,
            lag=False
        )

        # 2.2 ONLY numeric cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_contain_numeric=True,
            cov_dtypes_contain_categorical=False
        )
        tsdataset._known_cov = None

        adapter = DataAdapter()
        sample_ds = adapter.to_ml_dataset(tsdataset, **not_lag_param)
        batch_size = 2
        sample_dataloader = adapter.to_ml_dataloader(sample_ds, batch_size=batch_size)

        self._compare_sample_dataset_and_sample_dataloader(
            sample_ds=sample_ds,
            sample_dataloader=sample_dataloader,
            batch_size=batch_size,
            good_keys=all_target_keys.union(all_numeric_keys) - all_known_keys,
            lag=False
        )

        # 2.3 ONLY categorical cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_contain_numeric=False,
            cov_dtypes_contain_categorical=True
        )
        tsdataset._known_cov = None

        adapter = DataAdapter()
        sample_ds = adapter.to_ml_dataset(tsdataset, **not_lag_param)
        batch_size = 2
        sample_dataloader = adapter.to_ml_dataloader(sample_ds, batch_size=batch_size)

        self._compare_sample_dataset_and_sample_dataloader(
            sample_ds=sample_ds,
            sample_dataloader=sample_dataloader,
            batch_size=batch_size,
            good_keys=all_target_keys.union(all_categorical_keys) - all_known_keys,
            lag=False
        )

        ################################
        # case 3 (good case)           #
        # 1) Lag.                      #
        # 2) known_cov is None.        #
        # 3) observed_cov is NOT None. #
        # 2) static_cov is NOT None.   #
        ################################
        # 2.1 Both numeric and categorical cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_contain_numeric=True,
            cov_dtypes_contain_categorical=True
        )
        tsdataset._known_cov = None

        adapter = DataAdapter()
        sample_ds = adapter.to_ml_dataset(tsdataset, **lag_param)
        batch_size = 2
        sample_dataloader = adapter.to_ml_dataloader(sample_ds, batch_size=batch_size)

        self._compare_sample_dataset_and_sample_dataloader(
            sample_ds=sample_ds,
            sample_dataloader=sample_dataloader,
            batch_size=batch_size,
            good_keys=all_target_keys.union(all_numeric_keys).union(all_categorical_keys) - all_known_keys,
            lag=True
        )

        # 2.2 ONLY numeric cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_contain_numeric=True,
            cov_dtypes_contain_categorical=False
        )
        tsdataset._known_cov = None

        adapter = DataAdapter()
        sample_ds = adapter.to_ml_dataset(tsdataset, **lag_param)
        batch_size = 2
        sample_dataloader = adapter.to_ml_dataloader(sample_ds, batch_size=batch_size)

        self._compare_sample_dataset_and_sample_dataloader(
            sample_ds=sample_ds,
            sample_dataloader=sample_dataloader,
            batch_size=batch_size,
            good_keys=all_target_keys.union(all_numeric_keys) - all_known_keys,
            lag=True
        )

        # 2.3 ONLY categorical cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_contain_numeric=False,
            cov_dtypes_contain_categorical=True
        )
        tsdataset._known_cov = None

        adapter = DataAdapter()
        sample_ds = adapter.to_ml_dataset(tsdataset, **lag_param)
        batch_size = 2
        sample_dataloader = adapter.to_ml_dataloader(sample_ds, batch_size=batch_size)

        self._compare_sample_dataset_and_sample_dataloader(
            sample_ds=sample_ds,
            sample_dataloader=sample_dataloader,
            batch_size=batch_size,
            good_keys=all_target_keys.union(all_categorical_keys) - all_known_keys,
            lag=True
        )

        ##############################
        # case 4 (good case)         #
        # 1) Not Lag.                #
        # 2) known_cov is NOT None.  #
        # 3) observed_cov is None.   #
        # 2) static_cov is NOT None. #
        ##############################
        # 2.1 Both numeric and categorical cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_contain_numeric=True,
            cov_dtypes_contain_categorical=True
        )
        tsdataset._observed_cov = None

        adapter = DataAdapter()
        sample_ds = adapter.to_ml_dataset(tsdataset, **not_lag_param)
        batch_size = 2
        sample_dataloader = adapter.to_ml_dataloader(sample_ds, batch_size=batch_size)

        self._compare_sample_dataset_and_sample_dataloader(
            sample_ds=sample_ds,
            sample_dataloader=sample_dataloader,
            batch_size=batch_size,
            good_keys=all_target_keys.union(all_numeric_keys).union(all_categorical_keys) - all_observed_keys,
            lag=False
        )

        # 2.2 ONLY numeric cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_contain_numeric=True,
            cov_dtypes_contain_categorical=False
        )
        tsdataset._observed_cov = None

        adapter = DataAdapter()
        sample_ds = adapter.to_ml_dataset(tsdataset, **not_lag_param)
        batch_size = 2
        sample_dataloader = adapter.to_ml_dataloader(sample_ds, batch_size=batch_size)

        self._compare_sample_dataset_and_sample_dataloader(
            sample_ds=sample_ds,
            sample_dataloader=sample_dataloader,
            batch_size=batch_size,
            good_keys=all_target_keys.union(all_numeric_keys) - all_observed_keys,
            lag=False
        )

        # 2.3 ONLY categorical cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_contain_numeric=False,
            cov_dtypes_contain_categorical=True
        )
        tsdataset._observed_cov = None

        adapter = DataAdapter()
        sample_ds = adapter.to_ml_dataset(tsdataset, **not_lag_param)
        batch_size = 2
        sample_dataloader = adapter.to_ml_dataloader(sample_ds, batch_size=batch_size)

        self._compare_sample_dataset_and_sample_dataloader(
            sample_ds=sample_ds,
            sample_dataloader=sample_dataloader,
            batch_size=batch_size,
            good_keys=all_target_keys.union(all_categorical_keys) - all_observed_keys,
            lag=False
        )

        ##############################
        # case 5 (good case)         #
        # 1) Lag.                    #
        # 2) known_cov is NOT None.  #
        # 3) observed_cov is None.   #
        # 2) static_cov is NOT None. #
        ##############################
        # 2.1 Both numeric and categorical cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_contain_numeric=True,
            cov_dtypes_contain_categorical=True
        )
        tsdataset._observed_cov = None

        adapter = DataAdapter()
        sample_ds = adapter.to_ml_dataset(tsdataset, **lag_param)
        batch_size = 2
        sample_dataloader = adapter.to_ml_dataloader(sample_ds, batch_size=batch_size)

        self._compare_sample_dataset_and_sample_dataloader(
            sample_ds=sample_ds,
            sample_dataloader=sample_dataloader,
            batch_size=batch_size,
            good_keys=all_target_keys.union(all_numeric_keys).union(all_categorical_keys) - all_observed_keys,
            lag=True
        )

        # 2.2 ONLY numeric cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_contain_numeric=True,
            cov_dtypes_contain_categorical=False
        )
        tsdataset._observed_cov = None

        adapter = DataAdapter()
        sample_ds = adapter.to_ml_dataset(tsdataset, **lag_param)
        batch_size = 2
        sample_dataloader = adapter.to_ml_dataloader(sample_ds, batch_size=batch_size)

        self._compare_sample_dataset_and_sample_dataloader(
            sample_ds=sample_ds,
            sample_dataloader=sample_dataloader,
            batch_size=batch_size,
            good_keys=all_target_keys.union(all_numeric_keys) - all_observed_keys,
            lag=True
        )

        # 2.3 ONLY categorical cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_contain_numeric=False,
            cov_dtypes_contain_categorical=True
        )
        tsdataset._observed_cov = None

        adapter = DataAdapter()
        sample_ds = adapter.to_ml_dataset(tsdataset, **lag_param)
        batch_size = 2
        sample_dataloader = adapter.to_ml_dataloader(sample_ds, batch_size=batch_size)

        self._compare_sample_dataset_and_sample_dataloader(
            sample_ds=sample_ds,
            sample_dataloader=sample_dataloader,
            batch_size=batch_size,
            good_keys=all_target_keys.union(all_categorical_keys) - all_observed_keys,
            lag=True
        )

        ################################
        # case 6 (good case)           #
        # 1) Not Lag.                  #
        # 2) known_cov is NOT None.    #
        # 3) observed_cov is NOT None. #
        # 2) static_cov is None.       #
        ################################
        # 2.1 Both numeric and categorical cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_contain_numeric=True,
            cov_dtypes_contain_categorical=True
        )
        tsdataset._static_cov = None

        adapter = DataAdapter()
        sample_ds = adapter.to_ml_dataset(tsdataset, **not_lag_param)
        batch_size = 2
        sample_dataloader = adapter.to_ml_dataloader(sample_ds, batch_size=batch_size)

        self._compare_sample_dataset_and_sample_dataloader(
            sample_ds=sample_ds,
            sample_dataloader=sample_dataloader,
            batch_size=batch_size,
            good_keys=all_target_keys.union(all_numeric_keys).union(all_categorical_keys) - all_static_keys,
            lag=False
        )

        # 2.2 ONLY numeric cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_contain_numeric=True,
            cov_dtypes_contain_categorical=False
        )
        tsdataset._static_cov = None

        adapter = DataAdapter()
        sample_ds = adapter.to_ml_dataset(tsdataset, **not_lag_param)
        batch_size = 2
        sample_dataloader = adapter.to_ml_dataloader(sample_ds, batch_size=batch_size)

        self._compare_sample_dataset_and_sample_dataloader(
            sample_ds=sample_ds,
            sample_dataloader=sample_dataloader,
            batch_size=batch_size,
            good_keys=all_target_keys.union(all_numeric_keys) - all_static_keys,
            lag=False
        )

        # 2.3 ONLY categorical cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_contain_numeric=False,
            cov_dtypes_contain_categorical=True
        )
        tsdataset._static_cov = None

        adapter = DataAdapter()
        sample_ds = adapter.to_ml_dataset(tsdataset, **not_lag_param)
        batch_size = 2
        sample_dataloader = adapter.to_ml_dataloader(sample_ds, batch_size=batch_size)

        self._compare_sample_dataset_and_sample_dataloader(
            sample_ds=sample_ds,
            sample_dataloader=sample_dataloader,
            batch_size=batch_size,
            good_keys=all_target_keys.union(all_categorical_keys) - all_static_keys,
            lag=False
        )

        ################################
        # case 7 (good case)           #
        # 1) Lag.                      #
        # 2) known_cov is NOT None.    #
        # 3) observed_cov is NOT None. #
        # 2) static_cov is None.       #
        ################################
        # 2.1 Both numeric and categorical cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_contain_numeric=True,
            cov_dtypes_contain_categorical=True
        )
        tsdataset._static_cov = None

        adapter = DataAdapter()
        sample_ds = adapter.to_ml_dataset(tsdataset, **lag_param)
        batch_size = 2
        sample_dataloader = adapter.to_ml_dataloader(sample_ds, batch_size=batch_size)

        self._compare_sample_dataset_and_sample_dataloader(
            sample_ds=sample_ds,
            sample_dataloader=sample_dataloader,
            batch_size=batch_size,
            good_keys=all_target_keys.union(all_numeric_keys).union(all_categorical_keys) - all_static_keys,
            lag=True
        )

        # 2.2 ONLY numeric cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_contain_numeric=True,
            cov_dtypes_contain_categorical=False
        )
        tsdataset._static_cov = None

        adapter = DataAdapter()
        sample_ds = adapter.to_ml_dataset(tsdataset, **lag_param)
        batch_size = 2
        sample_dataloader = adapter.to_ml_dataloader(sample_ds, batch_size=batch_size)

        self._compare_sample_dataset_and_sample_dataloader(
            sample_ds=sample_ds,
            sample_dataloader=sample_dataloader,
            batch_size=batch_size,
            good_keys=all_target_keys.union(all_numeric_keys) - all_static_keys,
            lag=True
        )

        # 2.3 ONLY categorical cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_contain_numeric=False,
            cov_dtypes_contain_categorical=True
        )
        tsdataset._static_cov = None

        adapter = DataAdapter()
        sample_ds = adapter.to_ml_dataset(tsdataset, **lag_param)
        batch_size = 2
        sample_dataloader = adapter.to_ml_dataloader(sample_ds, batch_size=batch_size)

        self._compare_sample_dataset_and_sample_dataloader(
            sample_ds=sample_ds,
            sample_dataloader=sample_dataloader,
            batch_size=batch_size,
            good_keys=all_target_keys.union(all_categorical_keys) - all_static_keys,
            lag=True
        )

        ############################
        # case 8 (good case)       #
        # 1) Lag.                  #
        # 2) known_cov is None.    #
        # 3) observed_cov is None. #
        # 2) static_cov is None.   #
        ############################
        # 2.1 Both numeric and categorical cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_contain_numeric=True,
            cov_dtypes_contain_categorical=True
        )
        tsdataset._known_cov = None
        tsdataset._observed_cov = None
        tsdataset._static_cov = None

        adapter = DataAdapter()
        sample_ds = adapter.to_ml_dataset(tsdataset, **lag_param)
        batch_size = 2
        sample_dataloader = adapter.to_ml_dataloader(sample_ds, batch_size=batch_size)

        good_keys = all_target_keys.union(all_numeric_keys).union(all_categorical_keys) \
            - all_known_keys \
            - all_observed_keys \
            - all_static_keys
        self._compare_sample_dataset_and_sample_dataloader(
            sample_ds=sample_ds,
            sample_dataloader=sample_dataloader,
            batch_size=batch_size,
            good_keys=good_keys,
            lag=True
        )

        # 2.2 ONLY numeric cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_contain_numeric=True,
            cov_dtypes_contain_categorical=False
        )
        tsdataset._known_cov = None
        tsdataset._observed_cov = None
        tsdataset._static_cov = None

        adapter = DataAdapter()
        sample_ds = adapter.to_ml_dataset(tsdataset, **lag_param)
        batch_size = 2
        sample_dataloader = adapter.to_ml_dataloader(sample_ds, batch_size=batch_size)

        self._compare_sample_dataset_and_sample_dataloader(
            sample_ds=sample_ds,
            sample_dataloader=sample_dataloader,
            batch_size=batch_size,
            good_keys=all_target_keys.union(all_numeric_keys) - all_known_keys - all_observed_keys - all_static_keys,
            lag=True
        )

        # 2.3 ONLY categorical cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_contain_numeric=False,
            cov_dtypes_contain_categorical=True
        )
        tsdataset._known_cov = None
        tsdataset._observed_cov = None
        tsdataset._static_cov = None

        adapter = DataAdapter()
        sample_ds = adapter.to_ml_dataset(tsdataset, **lag_param)
        batch_size = 2
        sample_dataloader = adapter.to_ml_dataloader(sample_ds, batch_size=batch_size)

        good_keys = all_target_keys.union(all_categorical_keys) - all_known_keys - all_observed_keys - all_static_keys
        self._compare_sample_dataset_and_sample_dataloader(
            sample_ds=sample_ds,
            sample_dataloader=sample_dataloader,
            batch_size=batch_size,
            good_keys=good_keys,
            lag=True
        )

    def _build_mock_ts_dataset(
        self,
        target_periods: int,
        known_periods: int,
        observed_periods: int,
        cov_dtypes_contain_numeric: bool = True,
        cov_dtypes_contain_categorical: bool = True
    ):
        """Build mock dataset"""
        numeric_dtype = np.float32
        categorical_dtype = np.int64
        freq: str = "1D"

        # target
        target_df = pd.DataFrame(
            data=np.array([i for i in range(target_periods)], dtype=numeric_dtype),
            index=pd.date_range("2022-01-01", periods=target_periods, freq=freq),
            columns=["target_numeric_0"]
        )
        
        # known
        known_raw_data = [(i * 10, i * 100) for i in range(known_periods)]
        known_numeric_df = None
        if cov_dtypes_contain_numeric:
            known_numeric_data = np.array(known_raw_data, dtype=numeric_dtype)
            known_numeric_df = pd.DataFrame(
                data=known_numeric_data,
                index=pd.date_range("2022-01-01", periods=known_periods, freq=freq),
                columns=["known_numeric_0", "known_numeric_1"]
            )

        known_categorical_df = None
        if cov_dtypes_contain_categorical:
            known_categorical_data = np.array(known_raw_data, dtype=categorical_dtype)
            known_categorical_df = pd.DataFrame(
                data=known_categorical_data,
                index=pd.date_range("2022-01-01", periods=known_periods, freq=freq),
                columns=["known_categorical_0", "known_categorical_1"]
            )
        if (known_numeric_df is None) and (known_categorical_df is None):
            raise Exception(f"failed to build known cov data, both numeric df and categorical df are all None.")
        if (known_numeric_df is not None) and (known_categorical_df is not None):
            # both are NOT None.
            known_cov_df = pd.concat([known_numeric_df, known_categorical_df], axis=1)
        else:
            known_cov_df = [known_numeric_df, known_categorical_df][1 if known_numeric_df is None else 0]

        # observed
        observed_raw_data = [(i * -1, i * -10) for i in range(observed_periods)]
        observed_numeric_df = None
        if cov_dtypes_contain_numeric:
            observed_numeric_data = np.array(observed_raw_data, dtype=numeric_dtype)
            observed_numeric_df = pd.DataFrame(
                data=observed_numeric_data,
                index=pd.date_range("2022-01-01", periods=observed_periods, freq=freq),
                columns=["observed_numeric_0", "observed_numeric_1"]
            )

        observed_categorical_df = None
        if cov_dtypes_contain_categorical:
            observed_categorical_data = np.array(observed_raw_data, dtype=categorical_dtype)
            observed_categorical_df = pd.DataFrame(
                data=observed_categorical_data,
                index=pd.date_range("2022-01-01", periods=observed_periods, freq=freq),
                columns=["observed_categorical_0", "observed_categorical_1"]
            )

        if (observed_numeric_df is None) and (observed_categorical_df is None):
            raise Exception(f"failed to build observed cov data, both numeric df and categorical df are all None.")
        if (observed_numeric_df is not None) and (observed_categorical_df is not None):
            # both are NOT None.
            observed_cov_df = pd.concat([observed_numeric_df, observed_categorical_df], axis=1)
        else:
            observed_cov_df = [observed_numeric_df, observed_categorical_df][1 if observed_numeric_df is None else 0]

        # static
        static = dict()
        if cov_dtypes_contain_numeric:
            static["static_numeric"] = np.float32(1)
        if cov_dtypes_contain_categorical:
            static["static_categorical"] = np.int64(2)

        return TSDataset(
            target=TimeSeries.load_from_dataframe(data=target_df),
            known_cov=TimeSeries.load_from_dataframe(data=known_cov_df),
            observed_cov=TimeSeries.load_from_dataframe(data=observed_cov_df),
            static_cov=static
        )

    def _compare_tsdataset_and_sample_dataset(
        self,
        tsdataset: TSDataset,
        sample_ds: MLDataset,
        param: Dict,
        lag: bool = False,
        future_target_is_nan: bool = False
    ) -> None:
        """
        功能函数, 给定一个tsdataset数据集, 以及从中构建出来的样本集, 比较其中数据是否匹配

        Args:
            tsdataset(TSDataset): 构建sample_dataset使用的原数据集.
            sample_ds(MLDataset): 构建完成得到的样本集.
            param(Dict): adapter相关参数
            lag(bool, optional): 传入的tsdataset是否经历过lag transform处理.
            future_target_is_nan(bool, optional) True说明构建的样本Y是全NaN的空矩阵; False说明不为NaN, 正常检查即可.
        """
        numeric_dtype = np.float32
        categorical_dtype = np.int64

        in_chunk_len = param["in_chunk_len"]
        skip_chunk_len = param["skip_chunk_len"]
        out_chunk_len = param["out_chunk_len"]
        sampling_stride = param["sampling_stride"]
        time_window = param["time_window"]
        target_ts = tsdataset.get_target()
        known_ts = tsdataset.get_known_cov()
        observed_ts = tsdataset.get_observed_cov()
        static_cov = tsdataset.get_static_cov()

        # 验证: 验证每一条样本在paddlets dataset 和 ml dataset 中的数据是否可以匹配
        for sidx in range(len(sample_ds.samples)):
            curr_sample = sample_ds[sidx]

            ###############
            # past_target #
            ###############
            target_df = target_ts.to_dataframe(copy=False)
            if lag is True:
                self.assertEqual((0, 0), curr_sample["past_target"].shape)
            else:
                past_target_tail = time_window[0] + sidx * sampling_stride - skip_chunk_len - out_chunk_len
                past_target_ndarray = \
                    target_df.to_numpy(copy=False)[past_target_tail - in_chunk_len + 1:past_target_tail + 1]
                # data ok.
                self.assertTrue(np.alltrue(past_target_ndarray == curr_sample["past_target"]))
                # dtype ok.
                self.assertEqual(past_target_ndarray.dtype, curr_sample["past_target"].dtype)

            #################
            # future_target #
            #################
            if future_target_is_nan is True:
                # do NOT need Y, all NaN.
                self.assertEqual(0, curr_sample["future_target"].shape[0])
                self.assertTrue(np.alltrue(np.isnan(curr_sample["future_target"])))
            else:
                # need Y
                future_target_tail = time_window[0] + (sidx * sampling_stride) + 1
                future_target_head = future_target_tail - out_chunk_len
                future_target_ndarray = target_df.to_numpy(copy=False)[future_target_head:future_target_tail]
                # data ok.
                self.assertTrue(np.alltrue(future_target_ndarray == curr_sample["future_target"]))
                # dtype ok.
                self.assertEqual(future_target_ndarray.dtype, curr_sample["future_target"].dtype)

            #############
            # known_cov #
            #############
            if known_ts is not None:
                known_df = known_ts.to_dataframe(copy=False)

                known_right_tail = time_window[0] + (sidx * sampling_stride) + 1
                known_right_head = known_right_tail - out_chunk_len

                known_left_tail = known_right_head - 1 - skip_chunk_len + 1
                known_left_head = known_left_tail - in_chunk_len
                # numeric
                if "known_cov_numeric" in curr_sample.keys():
                    numeric_df = known_df.select_dtypes(include=numeric_dtype)
                    numeric_ndarray = numeric_df.to_numpy(copy=False)
                    numeric_right_ndarray = numeric_ndarray[known_right_head:known_right_tail]
                    numeric_left_ndarray = numeric_ndarray[known_left_head:known_left_tail]
                    known_numeric_ndarray = np.vstack((numeric_left_ndarray, numeric_right_ndarray))
                    # data ok.
                    self.assertTrue(np.alltrue(known_numeric_ndarray == curr_sample["known_cov_numeric"]))
                    # dtype ok.
                    self.assertEqual(known_numeric_ndarray.dtype, curr_sample["known_cov_numeric"].dtype)

                # categorical
                if "known_cov_categorical" in curr_sample.keys():
                    categorical_df = known_df.select_dtypes(include=categorical_dtype)
                    categorical_ndarray = categorical_df.to_numpy(copy=False)
                    categorical_right_ndarray = categorical_ndarray[known_right_head:known_right_tail]
                    categorical_left_ndarray = categorical_ndarray[known_left_head:known_left_tail]
                    known_categorical_ndarray = np.vstack((categorical_left_ndarray, categorical_right_ndarray))
                    # data ok.
                    self.assertTrue(np.alltrue(known_categorical_ndarray == curr_sample["known_cov_categorical"]))
                    # dtype ok.
                    self.assertEqual(known_categorical_ndarray.dtype, curr_sample["known_cov_categorical"].dtype)
            # known_cov is None.
            else:
                self.assertTrue("known_cov_numeric" not in curr_sample.keys())
                self.assertTrue("known_cov_categorical" not in curr_sample.keys())

            ################
            # observed_cov #
            ################
            if observed_ts is not None:
                observed_df = observed_ts.to_dataframe(copy=False)
                observed_tail = time_window[0] + sidx * sampling_stride - skip_chunk_len - out_chunk_len
                # numeric
                if "observed_cov_numeric" in curr_sample.keys():
                    numeric_df = observed_df.select_dtypes(include=numeric_dtype)
                    numeric_ndarray = numeric_df.to_numpy(copy=False)
                    observed_numeric_ndarray = numeric_ndarray[observed_tail - in_chunk_len + 1:observed_tail + 1]
                    # data ok.
                    self.assertTrue(np.alltrue(observed_numeric_ndarray == curr_sample["observed_cov_numeric"]))
                    # dtype ok.
                    self.assertEqual(observed_numeric_ndarray.dtype, curr_sample["observed_cov_numeric"].dtype)
                # categorical
                if "observed_cov_categorical" in curr_sample.keys():
                    categorical_df = observed_df.select_dtypes(include=categorical_dtype)
                    categorical_ndarray = categorical_df.to_numpy(copy=False)
                    observed_categorical_ndarray = \
                        categorical_ndarray[observed_tail - in_chunk_len + 1:observed_tail + 1]
                    # data ok.
                    self.assertTrue(np.alltrue(observed_categorical_ndarray == curr_sample["observed_cov_categorical"]))
                    # dtype ok.
                    self.assertEqual(observed_categorical_ndarray.dtype, curr_sample["observed_cov_categorical"].dtype)
            # observed_cov is None.
            else:
                self.assertTrue("observed_cov_numeric" not in curr_sample.keys())
                self.assertTrue("observed_cov_categorical" not in curr_sample.keys())

            ##############
            # static_cov #
            ##############
            if static_cov is not None:
                # unsorted dict -> sorted list
                sorted_static_cov = sorted(static_cov.items(), key=lambda t: t[0])
                # numeric
                if "static_cov_numeric" in curr_sample.keys():
                    sorted_static_cov_numeric = \
                        [t[1] for t in sorted_static_cov if isinstance(t[1], numeric_dtype)]
                    # data ok.
                    self.assertTrue(np.alltrue(sorted_static_cov_numeric == curr_sample["static_cov_numeric"][0]))
                    # dtype ok.
                    self.assertEqual(sorted_static_cov_numeric[0].dtype, curr_sample["static_cov_numeric"][0].dtype)
                # categorical
                if "static_cov_categorical" in curr_sample.keys():
                    sorted_static_cov_categorical = \
                        [t[1] for t in sorted_static_cov if isinstance(t[1], categorical_dtype)]
                    # data ok.
                    self.assertTrue(np.alltrue(
                        sorted_static_cov_categorical == curr_sample["static_cov_categorical"][0])
                    )
                    # dtype ok.
                    self.assertEqual(
                        sorted_static_cov_categorical[0].dtype, curr_sample["static_cov_categorical"][0].dtype
                    )
            # static_cov is None
            else:
                self.assertTrue("static_cov_numeric" not in curr_sample.keys())
                self.assertTrue("static_cov_categorical" not in curr_sample.keys())

    def _compare_sample_dataset_and_sample_dataloader(
        self,
        sample_ds: MLDataset,
        sample_dataloader: MLDataLoader,
        batch_size: int,
        good_keys: Set[str],
        lag: bool = False
    ):
        """Check if sample dataset matches batched sample dataloader."""
        all_keys = {
            "past_target",
            "future_target",
            "known_cov_numeric",
            "known_cov_categorical",
            "observed_cov_numeric",
            "observed_cov_categorical",
            "static_cov_numeric",
            "static_cov_categorical"
        }
        none_keys = all_keys - good_keys
        for batch_idx, batch_dict in enumerate(sample_dataloader):
            curr_batch_size = list(batch_dict.values())[0].shape[0]
            for sample_idx in range(curr_batch_size):
                dataset_sample = sample_ds[batch_idx * batch_size + sample_idx]
                for key in all_keys:
                    if key in none_keys:
                        # self.assertEqual((0, 0), dataset_sample[key].shape)
                        # self.assertEqual((0, 0), batch_dict[key][sample_idx].shape)
                        self.assertTrue(key not in dataset_sample.keys())
                        continue

                    # good keys
                    dataloader_ndarray_sample = batch_dict[key][sample_idx]
                    dataset_ndarray_sample = dataset_sample[key]
                    self.assertTrue(np.alltrue(dataloader_ndarray_sample == dataset_ndarray_sample))
                    self.assertEqual(dataset_ndarray_sample.dtype, dataloader_ndarray_sample.dtype)

                    if not lag:
                        continue
                    if key == "past_target":
                        # lag 场景 past_target.shape 永远为 (0, 0)
                        self.assertEqual((0, 0), dataloader_ndarray_sample.shape)
