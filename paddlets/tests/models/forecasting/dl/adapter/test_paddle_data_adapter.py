# !/usr/bin/env python3
# -*- coding:utf-8 -*-

from paddlets.models.forecasting.dl.adapter import DataAdapter
from paddlets.models.forecasting.dl.adapter.paddle_dataset_impl import PaddleDatasetImpl
from paddlets import TSDataset, TimeSeries

import unittest
import pandas as pd
import numpy as np
import math
from typing import Dict, Set
import paddle.io


class TestDataAdapter(unittest.TestCase):
    def setUp(self):
        """
        unittest setup
        """
        super().setUp()

    def test_to_paddle_dataset(self):
        """
        Test DataAdapter.to_paddle_dataset()
        """
        ######################################
        # case 0 (good case)                 #
        # 1) TSDataset is valid.             #
        # 2) Use default adapter parameters. #
        ######################################
        # This is the typical scenario for illustrating the basic usage.
        target_periods = 10
        known_periods = target_periods + 10
        observed_periods = target_periods

        expect_param = {
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

        sample_ds = adapter.to_paddle_dataset(tsdataset)
        self.assertEqual(expect_param["in_chunk_len"], sample_ds._target_in_chunk_len)
        self.assertEqual(expect_param["skip_chunk_len"], sample_ds._target_skip_chunk_len)
        self.assertEqual(expect_param["out_chunk_len"], sample_ds._target_out_chunk_len)
        self.assertEqual(expect_param["sampling_stride"], sample_ds._sampling_stride)
        self.assertEqual(expect_param["time_window"], sample_ds._time_window)
        self._compare_tsdataset_and_sample_dataset(
            tsdataset=tsdataset,
            sample_ds=sample_ds,
            param=expect_param,
            future_target_is_nan=False
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

        sample_ds = adapter.to_paddle_dataset(tsdataset)
        self.assertEqual(expect_param["in_chunk_len"], sample_ds._target_in_chunk_len)
        self.assertEqual(expect_param["skip_chunk_len"], sample_ds._target_skip_chunk_len)
        self.assertEqual(expect_param["out_chunk_len"], sample_ds._target_out_chunk_len)
        self.assertEqual(expect_param["sampling_stride"], sample_ds._sampling_stride)
        self.assertEqual(expect_param["time_window"], sample_ds._time_window)
        self._compare_tsdataset_and_sample_dataset(
            tsdataset=tsdataset,
            sample_ds=sample_ds,
            param=expect_param,
            future_target_is_nan=False
        )

        # 0.3 ONLY categorical cov features.
        # 0.2 ONLY numeric cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_contain_numeric=False,
            cov_dtypes_contain_categorical=True
        )

        adapter = DataAdapter()

        sample_ds = adapter.to_paddle_dataset(tsdataset)
        self.assertEqual(expect_param["in_chunk_len"], sample_ds._target_in_chunk_len)
        self.assertEqual(expect_param["skip_chunk_len"], sample_ds._target_skip_chunk_len)
        self.assertEqual(expect_param["out_chunk_len"], sample_ds._target_out_chunk_len)
        self.assertEqual(expect_param["sampling_stride"], sample_ds._sampling_stride)
        self.assertEqual(expect_param["time_window"], sample_ds._time_window)
        self._compare_tsdataset_and_sample_dataset(
            tsdataset=tsdataset,
            sample_ds=sample_ds,
            param=expect_param,
            future_target_is_nan=False
        )

        #######################################################
        # case 1 (good case)                                  #
        # 1) TSDataset is valid.                              #
        # 2) Split TSDataset to train / valid / test dataset. #
        # 3) Do NOT use default adapter parameters.           #
        #######################################################
        target_periods = 12
        known_periods = target_periods + 10
        observed_periods = target_periods

        common_param = {
            "in_chunk_len": 2,
            "skip_chunk_len": 1,
            "out_chunk_len": 2,
            "sampling_stride": 1
        }
        ratio = (0.5, 0.25, 0.25)
        window_bias = common_param["in_chunk_len"] + \
                      common_param["skip_chunk_len"] + \
                      common_param["out_chunk_len"] - \
                      1

        # 1.1 Both numeric and categorical cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_contain_numeric=True,
            cov_dtypes_contain_categorical=True
        )
        target_len = len(tsdataset.get_target().data)

        adapter = DataAdapter()

        # training dataset
        train_window_min = 0 + window_bias
        train_window_max = math.ceil(target_len * sum(ratio[:1]))
        train_param = {**common_param, "time_window": (train_window_min, train_window_max)}
        train_sample_ds = adapter.to_paddle_dataset(tsdataset, **train_param)
        self.assertEqual(train_param["time_window"], train_sample_ds._time_window)
        self._compare_tsdataset_and_sample_dataset(
            tsdataset=tsdataset,
            sample_ds=train_sample_ds,
            param=train_param,
            future_target_is_nan=False
        )

        # validation dataset
        valid_window_min = train_window_max + 1
        valid_window_max = math.ceil(target_len * sum(ratio[:2]))
        valid_param = {**common_param, "time_window": (valid_window_min, valid_window_max)}
        valid_sample_ds = adapter.to_paddle_dataset(tsdataset, **valid_param)

        self.assertEqual(valid_param["time_window"], valid_sample_ds._time_window)
        self._compare_tsdataset_and_sample_dataset(
            tsdataset=tsdataset,
            sample_ds=valid_sample_ds,
            param=valid_param,
            future_target_is_nan=False
        )

        # test dataset
        test_window_min = valid_window_max + 1
        test_window_max = min(math.ceil(target_len * sum(ratio[:3])), target_len - 1)
        test_param = {**common_param, "time_window": (test_window_min, test_window_max)}
        test_sample_ds = adapter.to_paddle_dataset(tsdataset, **test_param)

        self.assertEqual(test_param["time_window"], test_sample_ds._time_window)
        self._compare_tsdataset_and_sample_dataset(
            tsdataset=tsdataset,
            sample_ds=test_sample_ds,
            param=test_param,
            future_target_is_nan=False
        )

        # 1.2 ONLY numeric cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_contain_numeric=True,
            cov_dtypes_contain_categorical=False
        )
        target_len = len(tsdataset.get_target().data)

        adapter = DataAdapter()

        # training dataset
        train_window_min = 0 + window_bias
        train_window_max = math.ceil(target_len * sum(ratio[:1]))
        train_param = {**common_param, "time_window": (train_window_min, train_window_max)}
        train_sample_ds = adapter.to_paddle_dataset(tsdataset, **train_param)
        self.assertEqual(train_param["time_window"], train_sample_ds._time_window)
        self._compare_tsdataset_and_sample_dataset(
            tsdataset=tsdataset,
            sample_ds=train_sample_ds,
            param=train_param,
            future_target_is_nan=False
        )

        # validation dataset
        valid_window_min = train_window_max + 1
        valid_window_max = math.ceil(target_len * sum(ratio[:2]))
        valid_param = {**common_param, "time_window": (valid_window_min, valid_window_max)}
        valid_sample_ds = adapter.to_paddle_dataset(tsdataset, **valid_param)

        self.assertEqual(valid_param["time_window"], valid_sample_ds._time_window)
        self._compare_tsdataset_and_sample_dataset(
            tsdataset=tsdataset,
            sample_ds=valid_sample_ds,
            param=valid_param,
            future_target_is_nan=False
        )

        # test dataset
        test_window_min = valid_window_max + 1
        test_window_max = min(math.ceil(target_len * sum(ratio[:3])), target_len - 1)
        test_param = {**common_param, "time_window": (test_window_min, test_window_max)}
        test_sample_ds = adapter.to_paddle_dataset(tsdataset, **test_param)

        self.assertEqual(test_param["time_window"], test_sample_ds._time_window)
        self._compare_tsdataset_and_sample_dataset(
            tsdataset=tsdataset,
            sample_ds=test_sample_ds,
            param=test_param,
            future_target_is_nan=False
        )

        # 1.3 ONLY categorical cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_contain_numeric=False,
            cov_dtypes_contain_categorical=True
        )
        target_len = len(tsdataset.get_target().data)

        adapter = DataAdapter()

        # training dataset
        train_window_min = 0 + window_bias
        train_window_max = math.ceil(target_len * sum(ratio[:1]))
        train_param = {**common_param, "time_window": (train_window_min, train_window_max)}
        train_sample_ds = adapter.to_paddle_dataset(tsdataset, **train_param)
        self.assertEqual(train_param["time_window"], train_sample_ds._time_window)
        self._compare_tsdataset_and_sample_dataset(
            tsdataset=tsdataset,
            sample_ds=train_sample_ds,
            param=train_param,
            future_target_is_nan=False
        )

        # validation dataset
        valid_window_min = train_window_max + 1
        valid_window_max = math.ceil(target_len * sum(ratio[:2]))
        valid_param = {**common_param, "time_window": (valid_window_min, valid_window_max)}
        valid_sample_ds = adapter.to_paddle_dataset(tsdataset, **valid_param)

        self.assertEqual(valid_param["time_window"], valid_sample_ds._time_window)
        self._compare_tsdataset_and_sample_dataset(
            tsdataset=tsdataset,
            sample_ds=valid_sample_ds,
            param=valid_param,
            future_target_is_nan=False
        )

        # test dataset
        test_window_min = valid_window_max + 1
        test_window_max = min(math.ceil(target_len * sum(ratio[:3])), target_len - 1)
        test_param = {**common_param, "time_window": (test_window_min, test_window_max)}
        test_sample_ds = adapter.to_paddle_dataset(tsdataset, **test_param)

        self.assertEqual(test_param["time_window"], test_sample_ds._time_window)
        self._compare_tsdataset_and_sample_dataset(
            tsdataset=tsdataset,
            sample_ds=test_sample_ds,
            param=test_param,
            future_target_is_nan=False
        )

        ############################################################################################
        # case 2 (good case)                                                                       #
        # 1) TSDataset is valid.                                                                   #
        # 2) Predict scenario. The built sample only contains X, but not contains skip_chunk or Y. #
        # 3) Do NOT use default adapter parameters.                                                #
        ############################################################################################
        target_periods = 10
        known_periods = target_periods + 10
        observed_periods = target_periods

        # 2.1 Both numeric and categorical cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_contain_numeric=True,
            cov_dtypes_contain_categorical=True
        )

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
        # Build sample.
        sample_ds = adapter.to_paddle_dataset(tsdataset, **param)

        self.assertEqual(param["time_window"], sample_ds._time_window)
        self._compare_tsdataset_and_sample_dataset(
            tsdataset=tsdataset,
            sample_ds=sample_ds,
            param=param,
            future_target_is_nan=True
        )

        # 2.2 ONLY numeric cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_contain_numeric=True,
            cov_dtypes_contain_categorical=False
        )

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
        sample_ds = adapter.to_paddle_dataset(tsdataset, **param)

        self.assertEqual(param["time_window"], sample_ds._time_window)
        self._compare_tsdataset_and_sample_dataset(
            tsdataset=tsdataset,
            sample_ds=sample_ds,
            param=param,
            future_target_is_nan=True
        )

        # 2.3 ONLY categorical features
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_contain_numeric=False,
            cov_dtypes_contain_categorical=True
        )

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
        sample_ds = adapter.to_paddle_dataset(tsdataset, **param)

        self.assertEqual(param["time_window"], sample_ds._time_window)
        self._compare_tsdataset_and_sample_dataset(
            tsdataset=tsdataset,
            sample_ds=sample_ds,
            param=param,
            future_target_is_nan=True
        )

        ############################
        # case 3 (good case)       #
        # 1) known_cov is None.    #
        # 2) observed_cov is None. #
        # 3) static_cov is None.   #
        ############################
        target_periods = 10
        known_periods = target_periods + 10
        observed_periods = target_periods

        # 3.1 Both numeric and categorical cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_contain_numeric=True,
            cov_dtypes_contain_categorical=True
        )
        # Explicitly set to None
        tsdataset._known_cov = None
        tsdataset._observed_cov = None
        tsdataset._static_cov = None

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
        # Build sample.
        sample_ds = adapter.to_paddle_dataset(tsdataset, **param)

        self.assertEqual(param["time_window"], sample_ds._time_window)
        self._compare_tsdataset_and_sample_dataset(
            tsdataset=tsdataset,
            sample_ds=sample_ds,
            param=param,
            future_target_is_nan=True
        )

        # 3.2 ONLY numeric cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_contain_numeric=True,
            cov_dtypes_contain_categorical=False
        )
        # Explicitly set to None
        tsdataset._known_cov = None
        tsdataset._observed_cov = None
        tsdataset._static_cov = None

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
        # Build sample.
        sample_ds = adapter.to_paddle_dataset(tsdataset, **param)

        self.assertEqual(param["time_window"], sample_ds._time_window)
        self._compare_tsdataset_and_sample_dataset(
            tsdataset=tsdataset,
            sample_ds=sample_ds,
            param=param,
            future_target_is_nan=True
        )

        # 3.3 ONLY categorical cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_contain_numeric=False,
            cov_dtypes_contain_categorical=True
        )
        # Explicitly set to None
        tsdataset._known_cov = None
        tsdataset._observed_cov = None
        tsdataset._static_cov = None

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
        # Build sample.
        sample_ds = adapter.to_paddle_dataset(tsdataset, **param)

        self.assertEqual(param["time_window"], sample_ds._time_window)
        self._compare_tsdataset_and_sample_dataset(
            tsdataset=tsdataset,
            sample_ds=sample_ds,
            param=param,
            future_target_is_nan=True
        )

        ###############################################
        # case 4 (bad case) TSDataset.target is None. #
        ###############################################
        target_periods = 10
        known_periods = target_periods + 10
        observed_periods = target_periods
        tsdataset = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)
        # target is None.
        tsdataset._target = None

        adapter = DataAdapter()
        succeed = True
        try:
            _ = adapter.to_paddle_dataset(tsdataset)
        except ValueError:
            succeed = False
        self.assertFalse(succeed)

        #######################################
        # case 5 (bad case) in_chunk_len < 1. #
        #######################################
        target_periods = 10
        known_periods = target_periods + 10
        observed_periods = target_periods
        tsdataset = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)

        adapter = DataAdapter()
        succeed = True
        try:
            _ = adapter.to_paddle_dataset(tsdataset, in_chunk_len=0)
        except ValueError:
            succeed = False
        self.assertFalse(succeed)

        #########################################
        # case 6 (bad case) skip_chunk_len < 0. #
        #########################################
        target_periods = 10
        known_periods = target_periods + 10
        observed_periods = target_periods
        tsdataset = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)

        adapter = DataAdapter()
        succeed = True
        try:
            _ = adapter.to_paddle_dataset(tsdataset, skip_chunk_len=-1)
        except ValueError:
            succeed = False
        self.assertFalse(succeed)

        ########################################
        # case 7 (bad case) out_chunk_len < 1. #
        ########################################
        target_periods = 10
        known_periods = target_periods + 10
        observed_periods = target_periods
        tsdataset = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)

        adapter = DataAdapter()
        succeed = True
        try:
            _ = adapter.to_paddle_dataset(tsdataset, out_chunk_len=0)
        except ValueError:
            succeed = False
        self.assertFalse(succeed)

        ##########################################
        # case 8 (bad case) sampling_stride < 1. #
        ##########################################
        target_periods = 10
        known_periods = target_periods + 10
        observed_periods = target_periods
        tsdataset = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)

        adapter = DataAdapter()
        succeed = True
        try:
            _ = adapter.to_paddle_dataset(tsdataset, sampling_stride=0)
        except ValueError:
            succeed = False
        self.assertFalse(succeed)

        ###########################################################
        # case 9 (bad case) time_window lower bound is too small. #
        ###########################################################
        target_periods = 12
        known_periods = target_periods + 10
        observed_periods = target_periods
        tsdataset = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)

        adapter = DataAdapter()
        # Given the following params, the min allowed time_window lower bound can be calculated as follows:
        # min_allowed_time_window_lower_bound = (in + skip + out - 1) = 5 - 1 = 4.
        # Thus, the invalid (too small) time_window lower bound values are 0, 1, 2, 3.
        common_param = {
            "in_chunk_len": 2,
            "skip_chunk_len": 1,
            "out_chunk_len": 2,
            "sampling_stride": 1
        }

        # time_window[0] = 3, too small.
        param = {**common_param, "time_window": (3, len(tsdataset.get_target().data) - 1)}
        succeed = True
        try:
            _ = adapter.to_paddle_dataset(tsdataset, **param)
        except ValueError:
            succeed = False
        self.assertFalse(succeed)

        # time_window[0] = 2, too small.
        param = {**common_param, "time_window": (2, len(tsdataset.get_target().data) - 1)}
        succeed = True
        try:
            _ = adapter.to_paddle_dataset(tsdataset, **param)
        except ValueError:
            succeed = False
        self.assertFalse(succeed)

        # time_window[0] = 1, too small.
        param = {**common_param, "time_window": (1, len(tsdataset.get_target().data) - 1)}
        succeed = True
        try:
            _ = adapter.to_paddle_dataset(tsdataset, **param)
        except ValueError:
            succeed = False
        self.assertFalse(succeed)

        # time_window[0] = 0, too small.
        param = {**common_param, "time_window": (0, len(tsdataset.get_target().data) - 1)}
        succeed = True
        try:
            _ = adapter.to_paddle_dataset(tsdataset, **param)
        except ValueError:
            succeed = False
        self.assertFalse(succeed)

        ############################################################
        # case 10 (bad case) time_window upper bound is too large. #
        ############################################################
        target_periods = 10
        known_periods = target_periods + 10
        observed_periods = target_periods
        tsdataset = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)

        adapter = DataAdapter()

        # Given the following params, the max allowed time_window upper bound can be calculated as follows:
        # max_allowed_time_window_lower_bound = (target_ts_len + skip + out - 1) = 10 + 1 + 2 - 1 = 12
        # Thus, any values which are larger than 12 will be invalid.
        param = {
            "in_chunk_len": 2,
            "skip_chunk_len": 1,
            "out_chunk_len": 2,
            "sampling_stride": 1,
            "time_window": (
                2,
                # 13 > 12, too large.
                len(tsdataset.get_target().data) + 1 + 2 - 1 + 1
            )
        }
        succeed = True
        try:
            _ = adapter.to_paddle_dataset(tsdataset, **param)
        except ValueError:
            succeed = False
        self.assertFalse(succeed)

        #############################################################
        # case 11 (bad case)                                        #
        # 1) Given time_window[1] <= max_target_idx                 #
        # 2) (bad) TSDataset.known_cov.time_index[-1] is too small. #
        #############################################################
        target_periods = 10
        observed_periods = target_periods

        # Let max_known_timestamp be the max timestamp in the existing known_cov.time_index,
        # window_upper_bound_timestamp be the timestamp in the target.time_index which time_window[1] pointed to.
        # The following inequation MUST be held:
        # max_known_timestamp >= window_upper_bound_timestamp
        #
        # Given the following:
        # target_periods = 10
        # target_timeindex = [0:00, 1:00, 2:00, 3:00, 4:00, 5:00, 6:00, 7:00, 8:00, 9:00]
        # known_periods = target_periods - 1 = 9
        # known_timeindex = [0:00, 1:00, 2:00, 3:00, 4:00, 5:00, 6:00, 7:00, 8:00]
        # time_window = (3, 9)
        #
        # Firstly, compute window_upper_bound_timestamp:
        # window_upper_bound_timestamp = target_timeindex[time_window[1]] = target_timeindex[9] = 9:00
        #
        # Secondly, compute max_known_timestamp:
        # max_known_timestamp = known_timeindex[-1] = 8:00
        #
        # Finally, compare:
        # max_known_timestamp (i.e. 8:00) < window_upper_bound_timestamp (i.e. 9:00)
        #
        # According to the compare result, the max timestamp of the given known_ts is too small to build samples.
        # End.
        known_periods = target_periods - 1
        tsdataset = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)

        adapter = DataAdapter()
        param = {
            "in_chunk_len": 1,
            "skip_chunk_len": 1,
            "out_chunk_len": 2,
            "sampling_stride": 1,
            "time_window": (3, 9)
        }
        succeed = True
        try:
            _ = adapter.to_paddle_dataset(tsdataset, **param)
        except ValueError:
            succeed = False
        self.assertFalse(succeed)

        #############################################################
        # case 12 (bad case)                                        #
        # 1) Given time_window[1] > max_target_idx.                 #
        # 2) (bad) TSDataset.known_cov.time_index[-1] is too small. #
        #############################################################
        target_periods = 10
        observed_periods = target_periods

        # Let max_known_timestamp be the max timestamp in the existing known_cov.time_index,
        # window_upper_bound_timestamp be the timestamp in the target.time_index which time_window[1] pointed to,
        # max_target_timestamp be the timestamp of the target.time_index.
        #
        # Before start, we should be aware of the following:
        # To test a "too small known cov" scenario, there are 2 sub-scenarios:
        # Sub-scenario 1:
        # max_known_timestamp < max_target_timestamp < window_upper_bound_timestamp
        # In this scenario, we do not need to compare the max_known_timestamp with window_upper_bound_timestamp,
        # but instead just need to compare max_known_timestamp with max_target_timestamp.
        #
        # Sub-scenario 2:
        # max_target_timestamp < max_known_timestamp < window_upper_bound_timestamp
        # In this scenario, because max_target_timestamp is smaller than max_known_timestamp, so we must compare
        # max_known_timestamp with window_upper_bound_timestamp.
        #
        # This case focus on the Sub-scenario 1, the latter case (i.e. case 12) will focus on the sub-scenario 2.
        #
        # Now start to test. In this case, the following inequation MUST be held:
        # max_known_timestamp >= max_target_timestamp
        #
        # Given the following:
        # target_periods = 10
        # target_timeindex = [0:00, 1:00, 2:00, 3:00, 4:00, 5:00, 6:00, 7:00, 8:00, 9:00]
        # known_periods = target_periods - 1 = 9
        # known_timeindex = [0:00, 1:00, 2:00, 3:00, 4:00, 5:00, 6:00, 7:00, 8:00]
        # time_window = (12, 12)
        #
        # Firstly, compute max_target_timestamp:
        # max_target_timestamp = target_timeindex[-1] = 9:00
        #
        # Secondly, compute max_known_timestamp:
        # max_known_timestamp = known_timeindex[-1] = 8:00
        #
        # Finally), compare:
        # max_known_timestamp (i.e. 8:00) < max_target_timestamp (i.e. 9:00)
        #
        # According to the compare result, the max timestamp of the given known_ts is too small to build samples.
        # End.
        known_periods = target_periods - 1
        tsdataset = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)

        adapter = DataAdapter()
        param = {
            "in_chunk_len": 1,
            "skip_chunk_len": 1,
            "out_chunk_len": 2,
            "sampling_stride": 1,
            "time_window": (12, 12)
        }
        succeed = True
        try:
            _ = adapter.to_paddle_dataset(tsdataset, **param)
        except ValueError:
            succeed = False
        self.assertFalse(succeed)

        #############################################################
        # case 13 (bad case)                                        #
        # 1) Given time_window[1] > max_target_idx.                 #
        # 2) (bad) TSDataset.known_cov.time_index[-1] is too small. #
        #############################################################
        target_periods = 10
        observed_periods = target_periods

        # Let max_known_timestamp be the max timestamp in the existing known_cov.time_index,
        # window_upper_bound_timestamp be the timestamp in the target.time_index which time_window[1] pointed to,
        # max_target_timestamp be the timestamp of the target.time_index.
        #
        # Before start, we should be aware of the following:
        # To test a "too small known cov" scenario, there are 2 sub-scenarios:
        # Sub-scenario 1:
        # max_known_timestamp < max_target_timestamp < window_upper_bound_timestamp
        # In this scenario, we do not need to compare the max_known_timestamp with window_upper_bound_timestamp,
        # but instead just need to compare max_known_timestamp with max_target_timestamp.
        #
        # Sub-scenario 2:
        # max_target_timestamp < max_known_timestamp < window_upper_bound_timestamp
        # In this scenario, because max_target_timestamp is smaller than max_known_timestamp, so we must compare
        # max_known_timestamp with window_upper_bound_timestamp.
        #
        # This case focus on the Sub-scenario 2, the former case (i.e. case 11) already handled the sub-scenario 1.
        #
        # Now start to test. In this case, the following inequation MUST be held:
        # max_known_timestamp >= window_upper_bound_timestamp
        #
        # Given the following:
        # target_periods = 10
        # target_timeindex = [0:00, 1:00, 2:00, 3:00, 4:00, 5:00, 6:00, 7:00, 8:00, 9:00]
        # known_periods = target_periods + 1 = 11
        # known_timeindex = [0:00, 1:00, 2:00, 3:00, 4:00, 5:00, 6:00, 7:00, 8:00, 9:00, 10:00]
        # time_window = (12, 12)
        #
        # Firstly, compute window_upper_bound_timestamp:
        # max_target_timestamp = target_timeindex[-1] = 9:00
        # extra_timestamp_num = time_window[1] - len(target_timeindex) = 12 - 10 = 2
        # window_upper_bound_timestamp = max_target_timestamp + extra_timestamp_num = 9:00 + 2 = 11:00
        #
        # Secondly, compute max_known_timestamp:
        # max_known_timestamp = known_cov_timeindex[-1] = 10:00
        #
        # Finally), compare:
        # max_known_timestamp (i.e. 10:00) < window_upper_bound_timestamp (i.e. 11:00)
        #
        # According to the compare result, the max timestamp of the given known_ts is too small to build samples.
        # End.
        known_periods = 11
        tsdataset = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)

        adapter = DataAdapter()
        param = {
            "in_chunk_len": 1,
            "skip_chunk_len": 1,
            "out_chunk_len": 2,
            "sampling_stride": 1,
            # known_cov len = 11, smaller than window[1].
            "time_window": (12, 12)
        }
        succeed = True
        try:
            _ = adapter.to_paddle_dataset(tsdataset, **param)
        except ValueError:
            succeed = False
        self.assertFalse(succeed)

        ################################################################
        # case 14 (bad case)                                           #
        # 1) time_window[1] > max_target_idx.                          #
        # 2) (bad) TSDataset.observed_cov.time_index[-1] is too small. #
        ################################################################
        target_periods = 10
        known_periods = target_periods + 10

        # Let max_observed_timestamp be the max timestamp in the existing observed_cov.time_index,
        # max_target_timestamp be the timestamp of the target.time_index.
        # As time_window[1] > max_target_idx, thus the following inequation MUST be held:
        # max_observed_timestamp >= max_target_timestamp
        #
        # Given the following:
        # target_periods = 10
        # target_timeindex = [0:00, 1:00, 2:00, 3:00, 4:00, 5:00, 6:00, 7:00, 8:00, 9:00]
        # observed_periods = target_periods - 1 = 9
        # observed_timeindex = [0:00, 1:00, 2:00, 3:00, 4:00, 5:00, 6:00, 7:00, 8:00]
        # time_window = (12, 12)
        #
        # Firstly, compute max_target_timestamp:
        # max_target_timestamp = target_timeindex[-1] = 9:00
        #
        # Secondly, compute max_observed_timestamp:
        # max_observed_timestamp = target_timeindex[-1] = 8:00
        #
        # Finally, compare:
        # max_observed_timestamp (i.e. 8:00) < max_target_timestamp (i.e. 9:00)
        #
        # According to the compare result, the max timestamp of the given observed_ts is too small to build samples.
        # End.
        observed_periods = target_periods - 1
        tsdataset = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)

        adapter = DataAdapter()
        param = {
            "in_chunk_len": 1,
            "skip_chunk_len": 1,
            "out_chunk_len": 2,
            "sampling_stride": 1,
            # validates observed_cov does NOT need to check if time_window[1] > max_target_idx, because the check
            # logic is same.
            "time_window": (12, 12)
        }
        succeed = True
        try:
            _ = adapter.to_paddle_dataset(tsdataset, **param)
        except ValueError:
            succeed = False
        self.assertFalse(succeed)

        ################################################################
        # case 15 (bad case)                                           #
        # 1) time_window[1] <= max_target_idx.                         #
        # 2) (bad) TSDataset.observed_cov.time_index[-1] is too small. #
        ################################################################
        target_periods = 10

        # Let max_observed_timestamp be the max timestamp in the existing observed_cov.time_index,
        # window_upper_bound_timestamp be the timestamp in the target.time_index which time_window[1] pointed to,
        # last_sample_past_target_tail_timestamp be the timestamp in the target.time_index which the tail index of the
        # past_target chunk pointed to.
        # As time_window[1] <= max_target_idx, thus the following inequation MUST be held:
        # max_observed_timestamp >= last_sample_past_target_tail_timestamp
        #
        # Given the following:
        # target_periods = 10
        # target_timeindex = [0:00, 1:00, 2:00, 3:00, 4:00, 5:00, 6:00, 7:00, 8:00, 9:00]
        # observed_periods = 5
        # observed_timeindex = [0:00, 1:00, 2:00, 3:00, 4:00]
        # time_window = (3, 8)
        #
        # Firstly, compute window_upper_bound_timestamp:
        # window_upper_bound_timestamp = target_timeindex[time_window[1]] = target_timeindex[8] = 8:00
        #
        # Secondly, compute last_sample_past_target_tail_timestamp:
        # last_sample_past_target_tail = window[1] - out_chunk_len - skip_chunk_len = 8 - 2 - 1 = 5
        # last_sample_past_target_tail_timestamp = target_timeindex[last_sample_past_target_tail]
        #                                        = target_timeindex[5]
        #                                        = 5:00
        #
        # Thirdly, compute max_observed_timestamp:
        # max_observed_timestamp = observed_time_index[-1] = 4:00
        #
        # Finally, compare:
        # max_observed_timestamp (i.e. 4:00) < last_sample_past_target_tail_timestamp (i.e. 5:00)
        #
        # According to the compare result, the max timestamp of the given observed_ts is too small to build samples.
        # End.
        observed_periods = 5
        known_periods = target_periods + 10
        tsdataset = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)

        adapter = DataAdapter()
        param = {
            "in_chunk_len": 1,
            "skip_chunk_len": 1,
            "out_chunk_len": 2,
            "sampling_stride": 1,
            "time_window": (3, 8)
        }
        succeed = True
        try:
            _ = adapter.to_paddle_dataset(tsdataset, **param)
        except ValueError:
            succeed = False
        self.assertFalse(succeed)

    def test_to_paddle_dataloader(self):
        """
        Test DataAdapter.to_paddle_dataloader()
        """
        ################################
        # case 0 (good case)           #
        # 1) known_cov is NOT None.    #
        # 2) observed_cov is NOT None. #
        # 3) static_cov is NOT None.   #
        ################################
        target_periods = 10
        known_periods = target_periods + 10
        observed_periods = target_periods

        # 0.1 Both numeric and categorical cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_contain_numeric=True,
            cov_dtypes_contain_categorical=True
        )

        adapter = DataAdapter()
        sample_ds = adapter.to_paddle_dataset(tsdataset)
        batch_size = 2
        sample_dataloader = adapter.to_paddle_dataloader(sample_ds, batch_size, shuffle=False)

        good_keys = {
            "past_target",
            "future_target",
            "known_cov_numeric",
            "known_cov_categorical",
            "observed_cov_numeric",
            "observed_cov_categorical",
            "static_cov_numeric",
            "static_cov_categorical"
        }
        self._compare_sample_dataset_and_sample_dataloader(sample_ds, sample_dataloader, batch_size, good_keys)

        # 0.2 ONLY numeric cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_contain_numeric=True,
            cov_dtypes_contain_categorical=False
        )

        adapter = DataAdapter()
        sample_ds = adapter.to_paddle_dataset(tsdataset)
        batch_size = 2
        sample_dataloader = adapter.to_paddle_dataloader(sample_ds, batch_size, shuffle=False)

        good_keys = {
            "past_target",
            "future_target",
            "known_cov_numeric",
            "observed_cov_numeric",
            "static_cov_numeric"
        }
        self._compare_sample_dataset_and_sample_dataloader(sample_ds, sample_dataloader, batch_size, good_keys)

        # 0.3 ONLY categorical cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_contain_numeric=False,
            cov_dtypes_contain_categorical=True
        )

        adapter = DataAdapter()
        sample_ds = adapter.to_paddle_dataset(tsdataset)
        batch_size = 2
        sample_dataloader = adapter.to_paddle_dataloader(sample_ds, batch_size, shuffle=False)

        good_keys = {
            "past_target",
            "future_target",
            "known_cov_categorical",
            "observed_cov_categorical",
            "static_cov_categorical"
        }
        self._compare_sample_dataset_and_sample_dataloader(sample_ds, sample_dataloader, batch_size, good_keys)

        ################################
        # case 1 (good case)           #
        # 1) known_cov is None.        #
        # 2) observed_cov is NOT None. #
        # 3) static_cov is NOT None.   #
        ################################
        target_periods = 10
        known_periods = target_periods + 10
        observed_periods = target_periods

        # 1.1 Both numeric and categorical cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_contain_numeric=True,
            cov_dtypes_contain_categorical=True
        )
        # Explicitly set to None.
        tsdataset._known_cov = None

        adapter = DataAdapter()
        sample_ds = adapter.to_paddle_dataset(tsdataset)
        batch_size = 2
        sample_dataloader = adapter.to_paddle_dataloader(sample_ds, batch_size, shuffle=False)

        good_keys = {
            "past_target",
            "future_target",
            "observed_cov_numeric",
            "observed_cov_categorical",
            "static_cov_numeric",
            "static_cov_categorical"
        }
        self._compare_sample_dataset_and_sample_dataloader(sample_ds, sample_dataloader, batch_size, good_keys)

        # 1.2 ONLY numeric cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_contain_numeric=True,
            cov_dtypes_contain_categorical=False
        )
        # Explicitly set to None.
        tsdataset._known_cov = None

        adapter = DataAdapter()
        sample_ds = adapter.to_paddle_dataset(tsdataset)
        batch_size = 2
        sample_dataloader = adapter.to_paddle_dataloader(sample_ds, batch_size, shuffle=False)

        good_keys = {
            "past_target",
            "future_target",
            "observed_cov_numeric",
            "static_cov_numeric"
        }
        self._compare_sample_dataset_and_sample_dataloader(sample_ds, sample_dataloader, batch_size, good_keys)

        # 1.3 ONLY categorical cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_contain_numeric=False,
            cov_dtypes_contain_categorical=True
        )
        # Explicitly set to None.
        tsdataset._known_cov = None

        adapter = DataAdapter()
        sample_ds = adapter.to_paddle_dataset(tsdataset)
        batch_size = 2
        sample_dataloader = adapter.to_paddle_dataloader(sample_ds, batch_size, shuffle=False)

        good_keys = {
            "past_target",
            "future_target",
            "observed_cov_categorical",
            "static_cov_categorical"
        }
        self._compare_sample_dataset_and_sample_dataloader(sample_ds, sample_dataloader, batch_size, good_keys)

        ##############################
        # case 2 (good case)         #
        # 1) known_cov is NOT None.  #
        # 2) observed_cov is None.   #
        # 3) static_cov is NOT None. #
        ##############################
        target_periods = 10
        known_periods = target_periods + 10
        observed_periods = target_periods

        # 2.1 Both numeric and categorical cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_contain_numeric=True,
            cov_dtypes_contain_categorical=True
        )
        # Explicitly set to None.
        tsdataset._observed_cov = None

        adapter = DataAdapter()
        sample_ds = adapter.to_paddle_dataset(tsdataset)
        batch_size = 2
        sample_dataloader = adapter.to_paddle_dataloader(sample_ds, batch_size, shuffle=False)

        good_keys = {
            "past_target",
            "future_target",
            "known_cov_numeric",
            "known_cov_categorical",
            "static_cov_numeric",
            "static_cov_categorical"
        }
        self._compare_sample_dataset_and_sample_dataloader(sample_ds, sample_dataloader, batch_size, good_keys)

        # 2.2 ONLY numeric cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_contain_numeric=True,
            cov_dtypes_contain_categorical=False
        )
        # Explicitly set to None.
        tsdataset._observed_cov = None

        adapter = DataAdapter()
        sample_ds = adapter.to_paddle_dataset(tsdataset)
        batch_size = 2
        sample_dataloader = adapter.to_paddle_dataloader(sample_ds, batch_size, shuffle=False)

        good_keys = {
            "past_target",
            "future_target",
            "known_cov_numeric",
            "static_cov_numeric"
        }
        self._compare_sample_dataset_and_sample_dataloader(sample_ds, sample_dataloader, batch_size, good_keys)

        # 2.3 ONLY categorical cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_contain_numeric=False,
            cov_dtypes_contain_categorical=True
        )
        # Explicitly set to None.
        tsdataset._observed_cov = None

        adapter = DataAdapter()
        sample_ds = adapter.to_paddle_dataset(tsdataset)
        batch_size = 2
        sample_dataloader = adapter.to_paddle_dataloader(sample_ds, batch_size, shuffle=False)

        good_keys = {
            "past_target",
            "future_target",
            "known_cov_categorical",
            "static_cov_categorical"
        }
        self._compare_sample_dataset_and_sample_dataloader(sample_ds, sample_dataloader, batch_size, good_keys)

        ################################
        # case 3 (good case)           #
        # 1) known_cov is NOT None.    #
        # 2) observed_cov is NOT None. #
        # 3) static_cov is None.       #
        ################################
        target_periods = 10
        known_periods = target_periods + 10
        observed_periods = target_periods

        # 3.1 Both numeric and categorical cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_contain_numeric=True,
            cov_dtypes_contain_categorical=True
        )
        # Explicitly set to None.
        tsdataset._static_cov = None

        adapter = DataAdapter()
        sample_ds = adapter.to_paddle_dataset(tsdataset)
        batch_size = 2
        sample_dataloader = adapter.to_paddle_dataloader(sample_ds, batch_size, shuffle=False)

        good_keys = {
            "past_target",
            "future_target",
            "known_cov_numeric",
            "known_cov_categorical",
            "observed_cov_numeric",
            "observed_cov_categorical"
        }
        self._compare_sample_dataset_and_sample_dataloader(sample_ds, sample_dataloader, batch_size, good_keys)

        # 3.2 ONLY numeric cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_contain_numeric=True,
            cov_dtypes_contain_categorical=False
        )
        # Explicitly set to None.
        tsdataset._static_cov = None

        adapter = DataAdapter()
        sample_ds = adapter.to_paddle_dataset(tsdataset)
        batch_size = 2
        sample_dataloader = adapter.to_paddle_dataloader(sample_ds, batch_size, shuffle=False)

        good_keys = {
            "past_target",
            "future_target",
            "known_cov_numeric",
            "observed_cov_numeric"
        }
        self._compare_sample_dataset_and_sample_dataloader(sample_ds, sample_dataloader, batch_size, good_keys)

        # 3.3 ONLY categorical cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_contain_numeric=False,
            cov_dtypes_contain_categorical=True
        )
        # Explicitly set to None.
        tsdataset._static_cov = None

        adapter = DataAdapter()
        sample_ds = adapter.to_paddle_dataset(tsdataset)
        batch_size = 2
        sample_dataloader = adapter.to_paddle_dataloader(sample_ds, batch_size, shuffle=False)

        good_keys = {
            "past_target",
            "future_target",
            "known_cov_categorical",
            "observed_cov_categorical"
        }
        self._compare_sample_dataset_and_sample_dataloader(sample_ds, sample_dataloader, batch_size, good_keys)

        ############################
        # case 4 (good case)       #
        # 1) known_cov is None.    #
        # 2) observed_cov is None. #
        # 3) static_cov is None.   #
        ############################
        target_periods = 10
        known_periods = target_periods + 10
        observed_periods = target_periods

        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            # does not matter because it will be then set to None.
            cov_dtypes_contain_numeric=True,
            # does not matter because it will be then set to None.
            cov_dtypes_contain_categorical=True
        )
        # Explicitly set known and observed timeseries to None.
        tsdataset._known_cov = None
        tsdataset._observed_cov = None
        tsdataset._static_cov = None

        adapter = DataAdapter()
        sample_ds = adapter.to_paddle_dataset(tsdataset)
        batch_size = 2
        sample_dataloader = adapter.to_paddle_dataloader(sample_ds, batch_size, shuffle=False)

        good_keys = {
            "past_target",
            "future_target"
        }
        self._compare_sample_dataset_and_sample_dataloader(sample_ds, sample_dataloader, batch_size, good_keys)

    def _build_mock_ts_dataset(
            self,
            target_periods: int,
            known_periods: int,
            observed_periods: int,
            cov_dtypes_contain_numeric: bool = True,
            cov_dtypes_contain_categorical: bool = True
    ):
        """Build mock bts dataset"""
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
            sample_ds: PaddleDatasetImpl,
            param: Dict,
            future_target_is_nan: bool = False
    ) -> None:
        """
        Given a TSDataset and a built sample Dataset, compare if these data are matched.

        Args:
            tsdataset(TSDataset): Raw TSDataset.
            sample_ds(PaddleDatasetImpl): Built sample Dataset.
            param(Dict): param for building samples.
            future_target_is_nan(bool, optional): Set to True to indicates that the label (i.e. Y) of the built
                sample(s) are np.NaN. Default is False.
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

        for sidx in range(len(sample_ds.samples)):
            curr_sample = sample_ds[sidx]
            ###############
            # past_target #
            ###############
            target_df = target_ts.to_dataframe(copy=False)
            past_target_tail = time_window[0] + sidx * sampling_stride - skip_chunk_len - out_chunk_len
            past_target_ndarray = \
                target_df.to_numpy(copy=False)[past_target_tail - in_chunk_len + 1:past_target_tail + 1]
            # data ok.
            self.assertTrue(np.alltrue(past_target_ndarray == curr_sample["past_target"]))
            # dtype ok.
            self.assertEqual(past_target_ndarray.dtype, curr_sample["past_target"].dtype)
            # # numeric
            # if "past_target_numeric" in curr_sample.keys():
            #     numeric_df = target_df.select_dtypes(include=numeric_dtype)
            #     numeric_ndarray = numeric_df.to_numpy(copy=False)
            #     past_target_numeric_ndarray = numeric_ndarray[past_target_tail - in_chunk_len + 1:past_target_tail + 1]
            #     # data ok.
            #     self.assertTrue(np.alltrue(past_target_numeric_ndarray == curr_sample["past_target_numeric"]))
            #     # dtype ok.
            #     self.assertEqual(past_target_numeric_ndarray.dtype, curr_sample["past_target_numeric"].dtype)
            #
            # # categorical
            # if "past_target_categorical" in curr_sample.keys():
            #     categorical_df = target_df.select_dtypes(include=categorical_dtype)
            #     categorical_ndarray = categorical_df.to_numpy(copy=False)
            #     past_target_categorical_ndarray = \
            #         categorical_ndarray[past_target_tail - in_chunk_len + 1:past_target_tail + 1]
            #     # data ok.
            #     self.assertTrue(np.alltrue(past_target_categorical_ndarray == curr_sample["past_target_categorical"]))
            #     # dtype ok.
            #     self.assertEqual(past_target_categorical_ndarray.dtype, curr_sample["past_target_categorical"].dtype)

            #################
            # future_target #
            #################
            # Built sample does NOT contain Y, i.e. the chunk is filled with np.NaN.
            if future_target_is_nan is True:
                self.assertEqual(out_chunk_len, curr_sample["future_target"].shape[0])
                self.assertTrue(np.alltrue(np.isnan(curr_sample["future_target"])))

            # Built samples contain Y.
            else:
                future_target_tail = time_window[0] + (sidx * sampling_stride) + 1
                future_target_head = future_target_tail - out_chunk_len
                future_target_ndarray = target_df.to_numpy(copy=False)[future_target_head:future_target_tail]
                # data ok.
                self.assertTrue(np.alltrue(future_target_ndarray == curr_sample["future_target"]))
                # dtype ok.
                self.assertEqual(future_target_ndarray.dtype, curr_sample["future_target"].dtype)
                # # numeric
                # if "future_target_numeric" in curr_sample.keys():
                #     numeric_df = target_df.select_dtypes(include=numeric_dtype)
                #     numeric_ndarray = numeric_df.to_numpy(copy=False)
                #     future_target_numeric_ndarray = numeric_ndarray[future_target_head:future_target_tail]
                #     # data ok.
                #     self.assertTrue(np.alltrue(future_target_numeric_ndarray == curr_sample["future_target_numeric"]))
                #     # dtype ok.
                #     self.assertEqual(future_target_numeric_ndarray.dtype, curr_sample["future_target_numeric"].dtype)
                # # categorical
                # if "future_target_categorical" in curr_sample.keys():
                #     categorical_df = target_df.select_dtypes(include=categorical_dtype)
                #     categorical_ndarray = categorical_df.to_numpy(copy=False)
                #     future_target_categorical_ndarray = categorical_ndarray[future_target_head:future_target_tail]
                #     # data ok.
                #     self.assertTrue(np.alltrue(
                #         future_target_categorical_ndarray == curr_sample["future_target_categorical"]
                #     ))
                #     # dtype ok.
                #     self.assertEqual(
                #         future_target_categorical_ndarray.dtype, curr_sample["future_target_categorical"].dtype
                #     )

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
                    sorted_static_cov_numeric = [t[1] for t in sorted_static_cov if isinstance(t[1], numeric_dtype)]
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
            sample_ds: paddle.io.Dataset,
            sample_dataloader: paddle.io.DataLoader,
            batch_size: int,
            good_keys: Set[str]
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
                        self.assertTrue(key not in dataset_sample.keys())
                        continue

                    # good keys
                    dataloader_ndarray_sample = batch_dict[key][sample_idx].numpy()
                    dataset_ndarray_sample = dataset_sample[key]
                    self.assertTrue(np.alltrue(dataloader_ndarray_sample == dataset_ndarray_sample))
                    self.assertEqual(dataloader_ndarray_sample.dtype, dataset_ndarray_sample.dtype)
