# !/usr/bin/env python3
# -*- coding:utf-8 -*-

import unittest

import numpy as np
import pandas as pd
from typing import Dict
import math

from paddlets import TSDataset, TimeSeries
from paddlets.models.forecasting.ml.adapter.data_adapter import DataAdapter
from paddlets.models.forecasting.ml.adapter.ml_dataset import MLDataset


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
        paddlets_ds = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)

        # 初始化 adapter
        adapter = DataAdapter()

        # 调用接口, 构建样本(除了数据集之外, 不传入任何其他参数)
        ml_ds = adapter.to_ml_dataset(paddlets_ds)

        # 首先验证默认参数
        expect_param = {
            "in_chunk_len": 1,
            "skip_chunk_len": 0,
            "out_chunk_len": 1,
            "sampling_stride": 1,
            "time_window": (1, 9)
        }
        # 默认作为 X 的 target 长度为与 in_chunk 长度相等, 默认值为 1
        self.assertEqual(expect_param["in_chunk_len"], ml_ds._target_in_chunk_len)
        # 默认不跳过任何时间点 (skip = 0)
        self.assertEqual(expect_param["skip_chunk_len"], ml_ds._target_skip_chunk_len)
        # 默认输出的Y长度为1
        self.assertEqual(expect_param["out_chunk_len"], ml_ds._target_out_chunk_len)
        # 默认样本步长 = 1
        self.assertEqual(expect_param["sampling_stride"], ml_ds._sampling_stride)
        # 因为非lag场景, 因此目前不会计算默认的window值, 默认为None(什么时候计算取决于 MLDataset 的TODO完成情况)
        self.assertEqual(expect_param["time_window"], ml_ds._time_window)
        # 根据默认参数可以计算出样本数量为 8000 - 1 条, 和 target_periods 的长度 - 1 相同
        self.assertEqual(target_periods - 1, len(ml_ds.samples))

        self._compare_if_paddlets_sample_match_ml_sample(paddlets_ds=paddlets_ds, ml_ds=ml_ds, param=expect_param, lag=False)

        ##############################################################################
        # case 1 (good case) 非lag场景, 给定一个数据集, 从中拆分用于训练, 验证, 测试的样本集 #
        ##############################################################################
        target_periods = 12
        observed_periods = target_periods
        known_periods = target_periods + 10
        paddlets_ds = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)

        # 初始化 adapter
        adapter = DataAdapter()
        common_param = {
            "in_chunk_len": 2,
            "skip_chunk_len": 1,
            "out_chunk_len": 2,
            "sampling_stride": 1
        }
        ratio = (0.5, 0.25, 0.25)
        target_len = len(paddlets_ds.get_target().data)
        window_bias = common_param["in_chunk_len"] + \
            common_param["skip_chunk_len"] + \
            common_param["out_chunk_len"] - \
            1

        # 构建train样本
        train_window_min = 0 + window_bias
        train_window_max = math.ceil(target_len * sum(ratio[:1]))
        train_param = {**common_param, "time_window": (train_window_min, train_window_max)}
        train_ml_ds = adapter.to_ml_dataset(paddlets_ds, **train_param)
        # 验证train的参数
        self.assertEqual(train_param["time_window"], train_ml_ds._time_window)

        # 验证train的数据
        self._compare_if_paddlets_sample_match_ml_sample(paddlets_ds=paddlets_ds, ml_ds=train_ml_ds, param=train_param, lag=False)

        # 构建valid样本
        valid_window_min = train_window_max + 1
        valid_window_max = math.ceil(target_len * sum(ratio[:2]))
        valid_param = {**common_param, "time_window": (valid_window_min, valid_window_max)}
        valid_ml_ds = adapter.to_ml_dataset(paddlets_ds, **valid_param)

        # 验证valid的参数
        self.assertEqual(valid_param["time_window"], valid_ml_ds._time_window)

        # 验证 valid 数据
        self._compare_if_paddlets_sample_match_ml_sample(paddlets_ds=paddlets_ds, ml_ds=valid_ml_ds, param=valid_param, lag=False)

        # 构建test样本
        test_window_min = valid_window_max + 1
        test_window_max = min(math.ceil(target_len * sum(ratio[:3])), target_len - 1)
        test_param = {**common_param, "time_window": (test_window_min, test_window_max)}
        test_ml_ds = adapter.to_ml_dataset(paddlets_ds, **test_param)

        # 验证test参数
        self.assertEqual(test_param["time_window"], test_ml_ds._time_window)

        # 验证 test 数据
        self._compare_if_paddlets_sample_match_ml_sample(paddlets_ds=paddlets_ds, ml_ds=test_ml_ds, param=test_param, lag=False)

        ##############################################################################
        # case 2 (good case) 非lag场景, 给定一个数据集, 只从中构建一条只有X, 没有Y的预测样本 #
        ##############################################################################
        # 常用于在生产环境中，给真实的时序数据构建一条待预测的样本
        # 构造paddlets tsdataset
        target_periods = 10
        observed_periods = target_periods
        known_periods = target_periods + 10
        paddlets_ds = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)

        # 初始化 adapter
        adapter = DataAdapter()

        max_target_idx = len(paddlets_ds.get_target().data) - 1
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
        ml_ds = adapter.to_ml_dataset(paddlets_ds, **param)

        # 验证参数
        self.assertEqual(param["time_window"], ml_ds._time_window)

        # 验证数据
        self._compare_if_paddlets_sample_match_ml_sample(
            paddlets_ds=paddlets_ds,
            ml_ds=ml_ds,
            param=param,
            future_target_is_nan=True,
            lag=False
        )

        #############################################################
        # case 3 (good case) lag场景 + 给定一个数据集, 返回其中所有的样本 #
        #############################################################
        # 这种场景是在TSDataset被lag处理的前提下, 最简单的构建样本的调用方式, 常用于构建训练样本.
        # 构造paddlets tsdataset
        target_periods = 10
        observed_periods = target_periods
        known_periods = target_periods + 10
        paddlets_ds = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)

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
        ml_ds = adapter.to_ml_dataset(paddlets_ds, **param)

        # 默认作为 X 的 target 长度为与 in_chunk 长度相等, 默认值为 1
        self.assertEqual(param["in_chunk_len"], ml_ds._target_in_chunk_len)
        # 默认不跳过任何时间点 (skip = 0)
        self.assertEqual(param["skip_chunk_len"], ml_ds._target_skip_chunk_len)
        # 默认输出的Y长度为1
        self.assertEqual(param["out_chunk_len"], ml_ds._target_out_chunk_len)
        # lag场景中， observed_cov 长度固定为1
        self.assertEqual(1, ml_ds._observed_cov_chunk_len)
        # 默认样本步长 = 1
        self.assertEqual(param["sampling_stride"], ml_ds._sampling_stride)
        # 因为非lag场景, 因此目前不会计算默认的window值, 默认为None(什么时候计算取决于 MLDataset 的TODO完成情况)
        self.assertEqual(param["time_window"], ml_ds._time_window)
        # 根据默认参数可以计算出样本数量为 8000 - 1 条, 和 target_periods 的长度 - 1 相同
        self.assertEqual(target_periods - 1, len(ml_ds.samples))

        self._compare_if_paddlets_sample_match_ml_sample(paddlets_ds=paddlets_ds, ml_ds=ml_ds, param=param, lag=True)

        ############################################################################
        # case 4 (good case) lag场景, 给定一个数据集, 从中拆分用于训练, 验证, 测试的样本集 #
        ############################################################################
        target_periods = 12
        observed_periods = target_periods
        known_periods = target_periods + 10
        paddlets_ds = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)

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
        target_len = len(paddlets_ds.get_target().data)
        window_bias = max(1, common_param["in_chunk_len"]) + \
            common_param["skip_chunk_len"] + \
            common_param["out_chunk_len"] - \
            1

        # 构建train样本
        train_window_min = 0 + window_bias
        train_window_max = math.ceil(target_len * sum(ratio[:1]))
        train_param = {**common_param, "time_window": (train_window_min, train_window_max)}
        train_ml_ds = adapter.to_ml_dataset(paddlets_ds, **train_param)
        # 验证train的参数
        self.assertEqual(train_param["time_window"], train_ml_ds._time_window)

        # 验证train的数据
        self._compare_if_paddlets_sample_match_ml_sample(paddlets_ds=paddlets_ds, ml_ds=train_ml_ds, param=train_param, lag=True)

        # 构建valid样本
        valid_window_min = train_window_max + 1
        valid_window_max = math.ceil(target_len * sum(ratio[:2]))
        valid_param = {**common_param, "time_window": (valid_window_min, valid_window_max)}
        valid_ml_ds = adapter.to_ml_dataset(paddlets_ds, **valid_param)

        # 验证valid的参数
        self.assertEqual(valid_param["time_window"], valid_ml_ds._time_window)

        # 验证 valid 数据
        self._compare_if_paddlets_sample_match_ml_sample(paddlets_ds=paddlets_ds, ml_ds=valid_ml_ds, param=valid_param, lag=True)

        # 构建test样本
        test_window_min = valid_window_max + 1
        test_window_max = min(math.ceil(target_len * sum(ratio[:3])), target_len - 1)
        test_param = {**common_param, "time_window": (test_window_min, test_window_max)}
        test_ml_ds = adapter.to_ml_dataset(paddlets_ds, **test_param)

        # 验证test参数
        self.assertEqual(test_param["time_window"], test_ml_ds._time_window)

        # 验证 test 数据
        self._compare_if_paddlets_sample_match_ml_sample(paddlets_ds=paddlets_ds, ml_ds=test_ml_ds, param=test_param, lag=True)

        ############################################################################
        # case 5 (good case) lag场景, 给定一个数据集, 只从中构建一条只有X, 没有Y的预测样本 #
        ############################################################################
        # 常用于在生产环境中，给真实的时序数据构建一条待预测的样本
        # 构造paddlets tsdataset
        target_periods = 10
        observed_periods = target_periods
        known_periods = target_periods + 10
        paddlets_ds = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)

        # 初始化 adapter
        adapter = DataAdapter()

        max_target_idx = len(paddlets_ds.get_target().data) - 1
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
        ml_ds = adapter.to_ml_dataset(paddlets_ds, **param)

        # 验证参数
        self.assertEqual(param["time_window"], ml_ds._time_window)

        # 验证数据
        self._compare_if_paddlets_sample_match_ml_sample(
            paddlets_ds=paddlets_ds,
            ml_ds=ml_ds,
            param=param,
            future_target_is_nan=True,
            lag=False
        )

        ##############################################################
        # case 6 (bad case) 非 lag + 给定一个数据集, 且time_window越下界 #
        ##############################################################
        # 构造paddlets tsdataset
        target_periods = 12
        observed_periods = target_periods
        known_periods = target_periods + 10
        paddlets_ds = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)

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
        param = {**common_param, "time_window": (3, len(paddlets_ds.get_target().data) - 1)}
        succeed = True
        try:
            ml_ds = adapter.to_ml_dataset(paddlets_ds, **param)
        except Exception as e:
            succeed = False
        self.assertFalse(succeed)

        # The following window[0] (i.e. 2) < min allowed w[0] (i.e. in + skip + out - 1), window[1] is valid.
        # 这里给定的window[0] (即 2) 小于允许的下限 (in + skip + out - 1 = 4). 上界没问题.
        param = {**common_param, "time_window": (2, len(paddlets_ds.get_target().data) - 1)}
        succeed = True
        try:
            ml_ds = adapter.to_ml_dataset(paddlets_ds, **param)
        except Exception as e:
            succeed = False
        self.assertFalse(succeed)

        # The following window[0] (i.e. 1) < min allowed w[0] (i.e. in + skip + out - 1), window[1] is valid.
        # 这里给定的window[0] (即 1) 小于允许的下限 (in + skip + out - 1 = 4). 上界没问题.
        param = {**common_param, "time_window": (1, len(paddlets_ds.get_target().data) - 1)}
        succeed = True
        try:
            ml_ds = adapter.to_ml_dataset(paddlets_ds, **param)
        except Exception as e:
            succeed = False
        self.assertFalse(succeed)

        # The following window[0] (i.e. 0) < min allowed w[0] (i.e. in + skip + out - 1), window[1] is valid.
        # 这里给定的window[0] (即 0) 小于允许的下限 (in + skip + out - 1 = 4). 上界没问题.
        param = {**common_param, "time_window": (0, len(paddlets_ds.get_target().data) - 1)}
        succeed = True
        try:
            ml_ds = adapter.to_ml_dataset(paddlets_ds, **param)
        except Exception as e:
            succeed = False
        self.assertFalse(succeed)

        ############################################################
        # case 7 (bad case) lag + 给定一个数据集, 且time_window越下界 #
        ############################################################
        # 构造paddlets tsdataset
        target_periods = 12
        observed_periods = target_periods
        known_periods = target_periods + 10
        paddlets_ds = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)

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
        param = {**common_param, "time_window": (2, len(paddlets_ds.get_target().data) - 1)}
        succeed = True
        try:
            ml_ds = adapter.to_ml_dataset(paddlets_ds, **param)
        except Exception as e:
            succeed = False
        self.assertFalse(succeed)

        # The following window[0] (i.e. 1) < min allowed w[0] (i.e. in + skip + out - 1), window[1] is valid.
        # 这里给定的window[0] (即 1) 小于允许的下限 (in + skip + out - 1 = 4). 上界没问题.
        param = {**common_param, "time_window": (1, len(paddlets_ds.get_target().data) - 1)}
        succeed = True
        try:
            ml_ds = adapter.to_ml_dataset(paddlets_ds, **param)
        except Exception as e:
            succeed = False
        self.assertFalse(succeed)

        # The following window[0] (i.e. 0) < min allowed w[0] (i.e. in + skip + out - 1), window[1] is valid.
        # 这里给定的window[0] (即 0) 小于允许的下限 (in + skip + out - 1 = 4), 上界没问题.
        param = {**common_param, "time_window": (0, len(paddlets_ds.get_target().data) - 1)}
        succeed = True
        try:
            ml_ds = adapter.to_ml_dataset(paddlets_ds, **param)
        except Exception as e:
            succeed = False
        self.assertFalse(succeed)

        #############################################################
        # case 8 (bad case) 非lag场景 + 给定数据集, 且time_window越上界 #
        #############################################################
        # 构造paddlets tsdataset
        target_periods = 10
        observed_periods = target_periods
        known_periods = target_periods + 10
        paddlets_ds = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)

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
                len(paddlets_ds.get_target().data) + 1 + 2 - 1 + 1
            )
        }
        succeed = True
        try:
            ml_ds = adapter.to_ml_dataset(paddlets_ds, **param)
        except Exception as e:
            succeed = False
        self.assertFalse(succeed)

        ############################################################
        # case 9 (bad case) lag场景 + 给定数据集, 且time_window越上界 #
        ############################################################
        # 构造paddlets tsdataset
        target_periods = 10
        observed_periods = target_periods
        known_periods = target_periods + 10
        paddlets_ds = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)

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
                len(paddlets_ds.get_target().data) + 1 + 2 - 1 + 1
            )
        }
        succeed = True
        try:
            ml_ds = adapter.to_ml_dataset(paddlets_ds, **param)
        except Exception as e:
            succeed = False
        self.assertFalse(succeed)

        #####################################################################################
        # case 10 (bad case) 非lag场景 + 给定数据集 + 给定 time_window, 且known_cov长度不符合预期 #
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
        paddlets_ds = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)

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
            ml_ds = adapter.to_ml_dataset(paddlets_ds, **param)
        except Exception as e:
            succeed = False
        self.assertFalse(succeed)

        #####################################################################################
        # case 11 (bad case) 非lag场景 + 给定数据集 + 给定 time_window, 且known_cov长度不符合预期 #
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
        paddlets_ds = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)

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
            ml_ds = adapter.to_ml_dataset(paddlets_ds, **param)
        except Exception as e:
            succeed = False
        self.assertFalse(succeed)

        ####################################################################################
        # case 12 (bad case) lag场景 + 给定数据集 + 给定 time_window, 且known_cov长度不符合预期 #
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
        paddlets_ds = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)

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
            ml_ds = adapter.to_ml_dataset(paddlets_ds, **param)
        except Exception as e:
            succeed = False
        self.assertFalse(succeed)

        ####################################################################################
        # case 13 (bad case) lag场景 + 给定数据集 + 给定 time_window, 且known_cov长度不符合预期 #
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
        paddlets_ds = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)

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
            ml_ds = adapter.to_ml_dataset(paddlets_ds, **param)
        except Exception as e:
            succeed = False
        self.assertFalse(succeed)

        ###############################################################################################################
        # case 14 (bad case) 非lag场景 + 给定 time_window + time_window上界没有超过max_target, 但observed_cov时间戳上界太小 #
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
        paddlets_ds = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)

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
            ml_ds = adapter.to_ml_dataset(paddlets_ds, **param)
        except Exception as e:
            succeed = False
        self.assertFalse(succeed)

        #############################################################################################################
        # case 15 (bad case) lag场景 + 给定 time_window + time_window上界没有超过max_target, 但observed_cov时间戳上界太小 #
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
        paddlets_ds = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)

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
            ml_ds = adapter.to_ml_dataset(paddlets_ds, **param)
        except Exception as e:
            succeed = False
        self.assertFalse(succeed)

        #############################################################################################################
        # case 16 (bad case) 非lag场景 + 给定 time_window + time_window上界超过了max_target, 但observed_cov时间戳上界太小 #
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
        paddlets_ds = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)

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
            ml_ds = adapter.to_ml_dataset(paddlets_ds, **param)
        except Exception as e:
            succeed = False
        self.assertFalse(succeed)

        ############################################################################################################
        # case 17 (bad case) lag场景 + 给定 time_window + time_window上界超过了max_target, 但observed_cov时间戳上界太小 #
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
        paddlets_ds = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)

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
            ml_ds = adapter.to_ml_dataset(paddlets_ds, **param)
        except Exception as e:
            succeed = False
        self.assertFalse(succeed)

    def test_to_ml_dataloader(self):
        """
        测试 DataAdapter.to_ml_dataloader()
        """
        ###################################################################################
        # case 0 (good case) 非lag场景, 且给定数据集的 known / observed TimeSeries 均不为None #
        ###################################################################################
        # 构造paddlets tsdataset
        target_periods = 20
        observed_periods = target_periods
        known_periods = target_periods + 10
        paddlets_ds = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)

        # 初始化 adapter
        param = {
            # in > 0, 因此是非lag场景
            "in_chunk_len": 3,
            "skip_chunk_len": 0,
            "out_chunk_len": 1,
            "sampling_stride": 1,
            "time_window": (3, 19)
        }
        adapter = DataAdapter()

        # 构造 ml dataset
        ml_ds = adapter.to_ml_dataset(paddlets_ds, **param)

        # 验证
        batch_size = 2
        ml_dataloader = adapter.to_ml_dataloader(ml_ds, batch_size=batch_size)

        test_keys = ["past_target", "future_target", "known_cov", "observed_cov"]
        for i, d in enumerate(ml_dataloader):
            # dataloader 是 np.ndarray((batch_size, M, N)
            for key in test_keys:
                # d = {"past_target": ndarray, "future_target": ndarray, "known_cov": ndarray, "observed_cov": ndarray}
                for element_idx in range(d[key].shape[0]):
                    # 为了避免数据类型不同导致返回False, 这里统一将dtype设置为 float64
                    dataloader_ndarray_raw_element = d[key][element_idx]
                    dataloader_ndarray_float_element = np.array(dataloader_ndarray_raw_element, dtype="float64")

                    dataset_ndarray_raw_element = ml_ds[i * batch_size + element_idx][key]
                    dataset_ndarray_float_element = np.array(dataset_ndarray_raw_element, dtype="float64")
                    self.assertTrue(np.alltrue(dataloader_ndarray_float_element == dataset_ndarray_float_element))

        #################################################################################
        # case 1 (good case) lag场景, 且给定数据集的 known / observed TimeSeries 均不为None #
        #################################################################################
        # 构造paddlets tsdataset
        target_periods = 20
        observed_periods = target_periods
        known_periods = target_periods + 10
        paddlets_ds = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)

        # 初始化 adapter
        param = {
            # in = 0, 说明是lag场景
            "in_chunk_len": 0,
            "skip_chunk_len": 0,
            "out_chunk_len": 1,
            "sampling_stride": 1,
            "time_window": (1, 19)
        }
        adapter = DataAdapter()

        # 构造 ml dataset
        ml_ds = adapter.to_ml_dataset(paddlets_ds, **param)

        # 验证
        batch_size = 2
        ml_dataloader = adapter.to_ml_dataloader(ml_ds, batch_size=batch_size)

        all_keys = ["past_target", "future_target", "known_cov", "observed_cov"]
        for i, batch in enumerate(ml_dataloader):
            # dataloader 是 np.ndarray((batch_size, M, N)
            for key in all_keys:
                for element_idx in range(batch[key].shape[0]):
                    # 为避免2个ndarray 的数据类型不同导致False, 这里统一转为float64
                    dataloader_ndarray_raw_element = batch[key][element_idx]
                    dataloader_ndarray_float_element = np.array(dataloader_ndarray_raw_element, dtype="float64")
                    # lag场景的past/future target shape 有点区别, 需测试, 但observed / known shape 没有特殊性, 因此不再测试.
                    # lag 场景 past_target.shape 永远为 (0, 0)
                    if key == "past_target":
                        self.assertEqual(0, dataloader_ndarray_float_element.shape[1])
                    # lag 场景 + window[1] 未超过 max_target_idx, 因此 future_target.shape 正常
                    if key == "future_target":
                        self.assertEqual(
                            len(paddlets_ds.get_target().data.columns),
                            dataloader_ndarray_float_element.shape[1]
                        )
                    dataset_ndarray_raw_element = ml_ds[i * batch_size + element_idx][key]
                    dataset_ndarray_float_element = np.array(dataset_ndarray_raw_element, dtype="float64")
                    self.assertTrue(np.alltrue(dataloader_ndarray_float_element == dataset_ndarray_float_element))

        #####################################################################
        # case 2 (good case) 非lag场景, 且给定数据集的 known TimeSeries 为None #
        #####################################################################
        target_periods = 20
        observed_periods = target_periods
        known_periods = target_periods + 10
        paddlets_ds = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)
        # 手动设置一个 (非target) 的TimeSeries 为None
        paddlets_ds._known_cov = None

        # 初始化 adapter
        param = {
            # in > 0, 因此是非lag场景
            "in_chunk_len": 3,
            "skip_chunk_len": 0,
            "out_chunk_len": 1,
            "sampling_stride": 1,
            "time_window": (3, 19)
        }
        adapter = DataAdapter()

        # 构造 ml dataset
        ml_ds = adapter.to_ml_dataset(paddlets_ds, **param)

        # 验证
        batch_size = 2
        ml_dataloader = adapter.to_ml_dataloader(ml_ds, batch_size=batch_size)

        good_keys = ["past_target", "future_target", "observed_cov"]
        none_keys = ["known_cov"]
        all_keys = good_keys + none_keys
        for i, d in enumerate(ml_dataloader):
            # d = {"past_target": ndarray, "future_target": ndarray, "known_cov": ndarray, "observed_cov": ndarray}
            for key in all_keys:
                for element_idx in range(d[key].shape[0]):
                    dataloader_ndarray_element = d[key][element_idx]
                    dataset_ndarray_element = ml_ds[i * batch_size + element_idx][key]
                    if key in good_keys:
                        self.assertTrue(np.alltrue(dataloader_ndarray_element == dataset_ndarray_element))
                        continue
                    # 特别的, 对于 TimeSeries 本身就为None的情况，额外检查以下 shape 是否为 (0, 0)
                    if key in none_keys:
                        self.assertEqual((0, 0), dataloader_ndarray_element.shape)
                        self.assertEqual((0, 0), dataset_ndarray_element.shape)

        ######################################################################
        # case 3 (good case) lag场景, 且给定数据集的 observed TimeSeries 为None #
        ######################################################################
        target_periods = 20
        observed_periods = target_periods
        known_periods = target_periods + 10
        paddlets_ds = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)
        # 手动设置一个 (非target) 的TimeSeries 为None
        paddlets_ds._observed_cov = None

        # 初始化 adapter
        param = {
            # in > 0, 因此是非lag场景
            "in_chunk_len": 3,
            "skip_chunk_len": 0,
            "out_chunk_len": 1,
            "sampling_stride": 1,
            "time_window": (3, 19)
        }
        adapter = DataAdapter()

        # 构造 ml dataset
        ml_ds = adapter.to_ml_dataset(paddlets_ds, **param)

        # 验证
        batch_size = 2
        ml_dataloader = adapter.to_ml_dataloader(ml_ds, batch_size=batch_size)

        good_keys = ["past_target", "future_target", "known_cov"]
        none_keys = ["observed_cov"]
        all_keys = good_keys + none_keys
        for i, d in enumerate(ml_dataloader):
            # d = {"past_target": ndarray, "future_target": ndarray, "known_cov": ndarray, "observed_cov": ndarray}
            for key in all_keys:
                for element_idx in range(d[key].shape[0]):
                    dataloader_ndarray_element = d[key][element_idx]
                    dataset_ndarray_element = ml_ds[i * batch_size + element_idx][key]
                    if key in good_keys:
                        self.assertTrue(np.alltrue(dataloader_ndarray_element == dataset_ndarray_element))
                        continue
                    # 特别的, 对于 TimeSeries 本身就为None的情况，额外检查以下 shape 是否为 (0, 0)
                    if key in none_keys:
                        self.assertEqual((0, 0), dataloader_ndarray_element.shape)
                        self.assertEqual((0, 0), dataset_ndarray_element.shape)

    def _build_mock_ts_dataset(self, target_periods, known_periods, observed_periods):
        """构建用于测试的 paddlets dataset"""
        target_df = pd.Series(
            [i for i in range(target_periods)],
            index=pd.date_range("2022-01-01", periods=target_periods, freq="1D"),
            name="target0"
        )

        known_cov_df = pd.DataFrame(
            [(i * 10, i * 100) for i in range(known_periods)],
            index=pd.date_range("2022-01-01", periods=known_periods, freq="1D"),
            columns=["known0", "known1"]
        )

        observed_cov_df = pd.DataFrame(
            [(i * -1, i * -10) for i in range(observed_periods)],
            index=pd.date_range("2022-01-01", periods=observed_periods, freq="1D"),
            columns=["past0", "past1"]
        )

        return TSDataset(
            target=TimeSeries.load_from_dataframe(data=target_df),
            known_cov=TimeSeries.load_from_dataframe(data=known_cov_df),
            observed_cov=TimeSeries.load_from_dataframe(data=observed_cov_df),
            static_cov={"static0": 1, "static1": 2}
        )

    def _compare_if_paddlets_sample_match_ml_sample(
        self,
        paddlets_ds: TSDataset,
        ml_ds: MLDataset,
        param: Dict,
        lag: bool = False,
        future_target_is_nan: bool = False
    ) -> None:
        """
        功能函数, 给定一个paddlets数据集, 以及从中构建出来的 ml dataset 样本集, 比较其中数据是否匹配

        Args:
            paddlets_ds(TSDataset): 构建ml_ds使用的原数据集.
            ml_ds(MLDataset): 构建完成得到的样本集.
            param(Dict): adapter相关参数
            lag(bool, optional): 传入的paddlets_ds是否经历过lag transform处理.
            future_target_is_nan(bool, optional) False说明构建的样本Y是全NaN的空矩阵; True说明不为NaN, 正常检查即可.
        """
        # 验证: 验证每一条样本在paddlets dataset 和 ml dataset 中的数据是否可以匹配
        for sidx in range(len(ml_ds.samples)):
            # past_target
            if lag is True:
                paddlets_past_target = np.zeros(shape=(0, 0), dtype="float64")
            else:
                paddlets_past_target_tail = param["time_window"][0] + \
                    sidx * param["sampling_stride"] - \
                    param["skip_chunk_len"] - \
                    param["out_chunk_len"]
                paddlets_past_target = paddlets_ds \
                    .get_target() \
                    .to_numpy(False)[paddlets_past_target_tail - param["in_chunk_len"] + 1:paddlets_past_target_tail + 1]
            ml_past_target = ml_ds.samples[sidx]["past_target"]
            self.assertTrue(np.alltrue(paddlets_past_target == ml_past_target))

            # future_target
            ml_future_target = ml_ds.samples[sidx]["future_target"]
            if future_target_is_nan is True:
                # 不需要Y, 全为NaN
                self.assertTrue(np.alltrue(np.isnan(ml_future_target)))
            else:
                # 需要Y
                paddlets_future_target_tail = param["time_window"][0] + (sidx * param["sampling_stride"]) + 1
                paddlets_future_target_head = paddlets_future_target_tail - param["out_chunk_len"]
                paddlets_future_target = paddlets_ds \
                    .get_target() \
                    .to_numpy(False)[paddlets_future_target_head:paddlets_future_target_tail]
                self.assertTrue(np.alltrue(paddlets_future_target == ml_future_target))

            # known_cov
            paddlets_known_cov_right_tail = param["time_window"][0] + (sidx * param["sampling_stride"]) + 1
            paddlets_known_cov_right_head = paddlets_known_cov_right_tail - param["out_chunk_len"]
            paddlets_known_cov_right = paddlets_ds \
                .get_known_cov() \
                .to_numpy(False)[paddlets_known_cov_right_head:paddlets_known_cov_right_tail]

            paddlets_known_cov_left_tail = paddlets_known_cov_right_head - 1 - param["skip_chunk_len"] + 1
            paddlets_known_cov_left_head = paddlets_known_cov_left_tail - param["in_chunk_len"]
            paddlets_known_cov_left = paddlets_ds \
                .get_known_cov() \
                .to_numpy(False)[paddlets_known_cov_left_head:paddlets_known_cov_left_tail]

            paddlets_known_cov = np.vstack((paddlets_known_cov_left, paddlets_known_cov_right))
            ml_known_cov = ml_ds[sidx]["known_cov"]
            self.assertTrue(np.alltrue(paddlets_known_cov == ml_known_cov))

            # observed_cov
            paddlets_observed_cov_tail = param["time_window"][0] + \
                sidx * param["sampling_stride"] - \
                param["skip_chunk_len"] - \
                param["out_chunk_len"]
            paddlets_observed_cov = paddlets_ds \
                .get_observed_cov() \
                .to_numpy(False)[paddlets_observed_cov_tail - param["in_chunk_len"] + 1:paddlets_observed_cov_tail + 1]
            ml_observed_cov = ml_ds.samples[sidx]["observed_cov"]
            self.assertTrue(np.alltrue(paddlets_observed_cov == ml_observed_cov))
