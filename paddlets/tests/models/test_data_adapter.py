# !/usr/bin/env python3
# -*- coding:utf-8 -*-

from paddlets.models.data_adapter import DataAdapter, SampleDataset, MLDataLoader
from paddlets import TSDataset, TimeSeries

import paddle
import unittest
import datetime
import pandas as pd
import numpy as np
import math
from typing import Dict, Set, Union, Optional


class TestDataAdapter(unittest.TestCase):
    def setUp(self):
        """
        unittest setup
        """
        self._adapter = DataAdapter()
        self._both_numeric_and_categorical_dtype_bits = 2**6 - 1
        self._only_numeric_dtype_bits = 2**3 - 1
        self._only_categorical_dtype_bits = 2**3 + 2**4 + 2**5

        self._numeric_dtype = np.float32
        self._categorical_dtype = np.int64
        super().setUp()

    def test_to_sample_dataset_good_case(self):
        """
        Test good cases for to_sample_dataset()
        """
        #######################################################
        # case 1                                              #
        # 1) Split TSDataset to train / valid / test dataset. #
        #######################################################
        target_periods = 12
        known_periods = target_periods + 10
        observed_periods = target_periods

        common_param = {
            "in_chunk_len": 2,
            "skip_chunk_len": 1,
            "out_chunk_len": 2,
            "sampling_stride": 1,
            "fill_last_value": None
        }
        ratio = (0.5, 0.25, 0.25)
        window_bias = common_param["in_chunk_len"] + \
            common_param["skip_chunk_len"] + \
            common_param["out_chunk_len"] - \
            1

        # (a) Both numeric and categorical cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_bits=self._both_numeric_and_categorical_dtype_bits
        )
        target_len = len(tsdataset.get_target().data)

        # training dataset
        train_window_min = 0 + window_bias
        train_window_max = math.ceil(target_len * sum(ratio[:1]))
        train_param = {**common_param, "time_window": (train_window_min, train_window_max)}
        train_sample_ds = self._adapter.to_sample_dataset(tsdataset, **train_param)
        self.assertEqual(train_param["time_window"], train_sample_ds._time_window)
        self._compare_tsdataset_and_sample_dataset(
            tsdataset=tsdataset,
            sample_ds=train_sample_ds,
            has_future_target_key=True
        )

        # validation dataset
        valid_window_min = train_window_max + 1
        valid_window_max = math.ceil(target_len * sum(ratio[:2]))
        valid_param = {**common_param, "time_window": (valid_window_min, valid_window_max)}
        valid_sample_ds = self._adapter.to_sample_dataset(tsdataset, **valid_param)

        self.assertEqual(valid_param["time_window"], valid_sample_ds._time_window)
        self._compare_tsdataset_and_sample_dataset(
            tsdataset=tsdataset,
            sample_ds=valid_sample_ds,
            has_future_target_key=True
        )

        # test dataset
        test_window_min = valid_window_max + 1
        test_window_max = min(math.ceil(target_len * sum(ratio[:3])), target_len - 1)
        test_param = {**common_param, "time_window": (test_window_min, test_window_max)}
        test_sample_ds = self._adapter.to_sample_dataset(tsdataset, **test_param)

        self.assertEqual(test_param["time_window"], test_sample_ds._time_window)
        self._compare_tsdataset_and_sample_dataset(
            tsdataset=tsdataset,
            sample_ds=test_sample_ds,
            has_future_target_key=True
        )

        # (b) ONLY numeric cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_bits=self._only_numeric_dtype_bits
        )
        target_len = len(tsdataset.get_target().data)

        # training dataset
        train_window_min = 0 + window_bias
        train_window_max = math.ceil(target_len * sum(ratio[:1]))
        train_param = {**common_param, "time_window": (train_window_min, train_window_max)}
        train_sample_ds = self._adapter.to_sample_dataset(tsdataset, **train_param)
        self.assertEqual(train_param["time_window"], train_sample_ds._time_window)
        self._compare_tsdataset_and_sample_dataset(
            tsdataset=tsdataset,
            sample_ds=train_sample_ds,
            has_future_target_key=True
        )

        # validation dataset
        valid_window_min = train_window_max + 1
        valid_window_max = math.ceil(target_len * sum(ratio[:2]))
        valid_param = {**common_param, "time_window": (valid_window_min, valid_window_max)}
        valid_sample_ds = self._adapter.to_sample_dataset(tsdataset, **valid_param)

        self.assertEqual(valid_param["time_window"], valid_sample_ds._time_window)
        self._compare_tsdataset_and_sample_dataset(
            tsdataset=tsdataset,
            sample_ds=valid_sample_ds,
            has_future_target_key=True
        )

        # test dataset
        test_window_min = valid_window_max + 1
        test_window_max = min(math.ceil(target_len * sum(ratio[:3])), target_len - 1)
        test_param = {**common_param, "time_window": (test_window_min, test_window_max)}
        test_sample_ds = self._adapter.to_sample_dataset(tsdataset, **test_param)

        self.assertEqual(test_param["time_window"], test_sample_ds._time_window)
        self._compare_tsdataset_and_sample_dataset(
            tsdataset=tsdataset,
            sample_ds=test_sample_ds,
            has_future_target_key=True
        )

        # (c) ONLY categorical cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_bits=self._only_categorical_dtype_bits
        )
        target_len = len(tsdataset.get_target().data)

        # training dataset
        train_window_min = 0 + window_bias
        train_window_max = math.ceil(target_len * sum(ratio[:1]))
        train_param = {**common_param, "time_window": (train_window_min, train_window_max)}
        train_sample_ds = self._adapter.to_sample_dataset(tsdataset, **train_param)
        self.assertEqual(train_param["time_window"], train_sample_ds._time_window)
        self._compare_tsdataset_and_sample_dataset(
            tsdataset=tsdataset,
            sample_ds=train_sample_ds,
            has_future_target_key=True
        )

        # validation dataset
        valid_window_min = train_window_max + 1
        valid_window_max = math.ceil(target_len * sum(ratio[:2]))
        valid_param = {**common_param, "time_window": (valid_window_min, valid_window_max)}
        valid_sample_ds = self._adapter.to_sample_dataset(tsdataset, **valid_param)

        self.assertEqual(valid_param["time_window"], valid_sample_ds._time_window)
        self._compare_tsdataset_and_sample_dataset(
            tsdataset=tsdataset,
            sample_ds=valid_sample_ds,
            has_future_target_key=True
        )

        # test dataset
        test_window_min = valid_window_max + 1
        test_window_max = min(math.ceil(target_len * sum(ratio[:3])), target_len - 1)
        test_param = {**common_param, "time_window": (test_window_min, test_window_max)}
        test_sample_ds = self._adapter.to_sample_dataset(tsdataset, **test_param)

        self.assertEqual(test_param["time_window"], test_sample_ds._time_window)
        self._compare_tsdataset_and_sample_dataset(
            tsdataset=tsdataset,
            sample_ds=test_sample_ds,
            has_future_target_key=True
        )

        #########################################################################
        # case 2                                                                #
        # 1) Predict scenario. Built samples only contain X, but not contain Y. #
        #########################################################################
        target_periods = 10
        known_periods = target_periods + 10
        observed_periods = target_periods

        # (a) Both numeric and categorical cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_bits=self._both_numeric_and_categorical_dtype_bits
        )

        max_target_idx = len(tsdataset.get_target().data) - 1
        param = {
            "in_chunk_len": 2,
            "skip_chunk_len": 1,
            "out_chunk_len": 2,
            "sampling_stride": 1,
            "fill_last_value": None,
            "time_window": (
                # max_target_idx + skip + chunk
                max_target_idx + 1 + 2,
                max_target_idx + 1 + 2
            )
        }
        # Build sample.
        sample_ds = self._adapter.to_sample_dataset(tsdataset, **param)

        self.assertEqual(param["time_window"], sample_ds._time_window)
        self._compare_tsdataset_and_sample_dataset(
            tsdataset=tsdataset,
            sample_ds=sample_ds,
            has_future_target_key=False
        )

        # (b) ONLY numeric cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_bits=self._only_numeric_dtype_bits
        )

        max_target_idx = len(tsdataset.get_target().data) - 1
        param = {
            "in_chunk_len": 2,
            "skip_chunk_len": 1,
            "out_chunk_len": 2,
            "sampling_stride": 1,
            "fill_last_value": None,
            "time_window": (
                # max_target_idx + skip + chunk
                max_target_idx + 1 + 2,
                max_target_idx + 1 + 2
            )
        }
        sample_ds = self._adapter.to_sample_dataset(tsdataset, **param)

        self.assertEqual(param["time_window"], sample_ds._time_window)
        self._compare_tsdataset_and_sample_dataset(
            tsdataset=tsdataset,
            sample_ds=sample_ds,
            has_future_target_key=False
        )

        # (c) ONLY categorical features
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_bits=self._only_categorical_dtype_bits
        )

        max_target_idx = len(tsdataset.get_target().data) - 1
        param = {
            "in_chunk_len": 2,
            "skip_chunk_len": 1,
            "out_chunk_len": 2,
            "sampling_stride": 1,
            "fill_last_value": None,
            "time_window": (
                # max_target_idx + skip + chunk
                max_target_idx + 1 + 2,
                max_target_idx + 1 + 2
            )
        }
        sample_ds = self._adapter.to_sample_dataset(tsdataset, **param)

        self.assertEqual(param["time_window"], sample_ds._time_window)
        self._compare_tsdataset_and_sample_dataset(
            tsdataset=tsdataset,
            sample_ds=sample_ds,
            has_future_target_key=False
        )

        ####################################################
        # case 3                                           #
        # 1) target is NOT None.                           #
        # 2) known_cov = observed_cov = static_cov = None. #
        ####################################################
        target_periods = 10
        known_periods = target_periods + 10
        observed_periods = target_periods

        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            # because all cov is None in this case, so this cov_dtypes_bits param does not matter.
            cov_dtypes_bits=self._both_numeric_and_categorical_dtype_bits
        )
        # Explicitly set to None
        tsdataset.known_cov = None
        tsdataset.observed_cov = None
        tsdataset.static_cov = None

        # Build sample.
        param = {"out_chunk_len": 1, "fill_last_value": None}
        sample_ds = self._adapter.to_sample_dataset(tsdataset, **param)

        self._compare_tsdataset_and_sample_dataset(
            tsdataset=tsdataset,
            sample_ds=sample_ds,
            has_future_target_key=True
        )

        ##############################################
        # case 4                                     #
        # 1) observed_cov is NOT None.               #
        # 2) target = known_cov = static_cov = None. #
        ##############################################
        # Tips: Typical usage for anomaly detection models.
        target_periods = 10
        known_periods = target_periods + 10
        observed_periods = target_periods

        # (a) Both numeric and categorical cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_bits=self._both_numeric_and_categorical_dtype_bits
        )
        # Explicitly set to None
        tsdataset.target = None
        tsdataset.known_cov = None
        tsdataset.static_cov = None

        # Build sample.
        param = {"out_chunk_len": 0}
        sample_ds = self._adapter.to_sample_dataset(tsdataset, **param)

        self._compare_tsdataset_and_sample_dataset(
            tsdataset=tsdataset,
            sample_ds=sample_ds,
            has_future_target_key=False
        )

        # (b) ONLY numeric cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_bits=self._only_numeric_dtype_bits
        )
        # Explicitly set to None
        tsdataset.target = None
        tsdataset.known_cov = None
        tsdataset.static_cov = None

        # Build samples.
        param = {"out_chunk_len": 0}
        sample_ds = self._adapter.to_sample_dataset(tsdataset, **param)

        self._compare_tsdataset_and_sample_dataset(
            tsdataset=tsdataset,
            sample_ds=sample_ds,
            has_future_target_key=False
        )

        # (c) ONLY categorical cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_bits=self._only_categorical_dtype_bits
        )
        # Explicitly set to None
        tsdataset.target = None
        tsdataset.known_cov = None
        tsdataset.static_cov = None

        # Build samples.
        param = {"out_chunk_len": 0}
        sample_ds = self._adapter.to_sample_dataset(tsdataset, **param)

        self._compare_tsdataset_and_sample_dataset(
            tsdataset=tsdataset,
            sample_ds=sample_ds,
            has_future_target_key=False
        )

        ##################################################
        # case 5                                         #
        # 1) target is NOT None.                         #
        # 2) known_cov is NOT None.                      #
        # 3) observed_cov is NOT None.                   #
        # 4) static_cov is NOT None.                     #
        # 5) out_chunk_len = 0. (i.e., no future_target) #
        ##################################################
        # Tips: Typical usage for representation models.
        target_periods = 10
        known_periods = target_periods + 10
        observed_periods = target_periods

        param = {
            "in_chunk_len": 3,
            "out_chunk_len": 0,
            "skip_chunk_len": 0,
            "sampling_stride": 3
        }

        # (5.1) do not fill last sample.
        not_fill_param = {**param, "fill_last_value": None}
        # (a) Both numeric and categorical cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_bits=self._both_numeric_and_categorical_dtype_bits
        )

        # Build sample.
        sample_ds = self._adapter.to_sample_dataset(tsdataset, **not_fill_param)

        self._compare_tsdataset_and_sample_dataset(
            tsdataset=tsdataset,
            sample_ds=sample_ds,
            # because out_chunk_len = 0, so no future_target key.
            has_future_target_key=False
        )

        # (b) ONLY numeric cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_bits=self._only_numeric_dtype_bits
        )

        # Build sample.
        sample_ds = self._adapter.to_sample_dataset(tsdataset, **not_fill_param)

        self._compare_tsdataset_and_sample_dataset(
            tsdataset=tsdataset,
            sample_ds=sample_ds,
            # because out_chunk_len = 0, so no future_target key.
            has_future_target_key=False
        )

        # (c) ONLY categorical cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_bits=self._only_categorical_dtype_bits
        )

        # Build sample.
        sample_ds = self._adapter.to_sample_dataset(tsdataset, **not_fill_param)

        self._compare_tsdataset_and_sample_dataset(
            tsdataset=tsdataset,
            sample_ds=sample_ds,
            # because out_chunk_len = 0, so no future_target key.
            has_future_target_key=False
        )

        # (5.2) fill last sample.
        target_periods = 10
        known_periods = target_periods + 10
        observed_periods = target_periods

        param = {
            "in_chunk_len": 3,
            "out_chunk_len": 0,
            "skip_chunk_len": 0,
            "sampling_stride": 3
        }

        for fill_value in [t(99.9) for t in [float, np.float, np.float16, np.float32, np.float64]]:
            fill_param = {**param, "fill_last_value": fill_value}
            # (a) fill value is numeric, samples also only contain numeric too.
            # currently fill categorical data is not supported, so no need to test, so ignore (b) and (c).
            tsdataset = self._build_mock_ts_dataset(
                target_periods=target_periods,
                known_periods=known_periods,
                observed_periods=observed_periods,
                cov_dtypes_bits=self._only_numeric_dtype_bits
            )

            # Build sample.
            sample_ds = self._adapter.to_sample_dataset(tsdataset, **fill_param)

            self._compare_tsdataset_and_sample_dataset(
                tsdataset=tsdataset,
                sample_ds=sample_ds,
                # because out_chunk_len = 0, so no future_target key.
                has_future_target_key=False
            )

    def test_to_sample_dataset_bad_case(self):
        """to_sample_dataset bad cases."""
        ######################################################
        # case 1                                             #
        # target and observed_cov are all None at same time. #
        ######################################################
        target_periods = 10
        known_periods = target_periods + 10
        observed_periods = target_periods
        tsdataset = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)
        # Explicitly set target and observed_cov to None.
        tsdataset.target = None
        tsdataset.observed_cov = None

        with self.assertRaises(ValueError):
            _ = self._adapter.to_sample_dataset(tsdataset)

        #####################
        # case 2            #
        # in_chunk_len < 1. #
        #####################
        target_periods = 10
        known_periods = target_periods + 10
        observed_periods = target_periods
        tsdataset = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)

        with self.assertRaises(ValueError):
            _ = self._adapter.to_sample_dataset(tsdataset, in_chunk_len=0)

        #######################
        # case 3              #
        # skip_chunk_len < 0. #
        #######################
        target_periods = 10
        known_periods = target_periods + 10
        observed_periods = target_periods
        tsdataset = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)

        with self.assertRaises(ValueError):
            _ = self._adapter.to_sample_dataset(tsdataset, skip_chunk_len=-1)

        ######################
        # case 4             #
        # out_chunk_len < 0. #
        ######################
        target_periods = 10
        known_periods = target_periods + 10
        observed_periods = target_periods
        tsdataset = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)

        with self.assertRaises(ValueError):
            _ = self._adapter.to_sample_dataset(tsdataset, out_chunk_len=-1)

        ########################
        # case 5               #
        # sampling_stride < 1. #
        ########################
        target_periods = 10
        known_periods = target_periods + 10
        observed_periods = target_periods
        tsdataset = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)

        with self.assertRaises(ValueError):
            _ = self._adapter.to_sample_dataset(tsdataset, sampling_stride=0)

        #################################################################
        # case 6                                                        #
        # (fill_last_value is not None) and (time_window[1] too large). #
        #################################################################
        # This case is to guarantee that if fill_last_value is not None, time_window[1] must <= max_target_idx
        target_periods = 10
        known_periods = target_periods + 10
        observed_periods = target_periods
        tsdataset = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)

        with self.assertRaises(ValueError):
            param = {
                "in_chunk_len": 2,
                "skip_chunk_len": 3,
                "out_chunk_len": 4,
                "fill_last_value": 10.1,
                # time windows > max_target_idx, too large.
                "time_window": (
                    (target_periods - 1) + (3 + 4),
                    (target_periods - 1) + (3 + 4)
                )
            }
            _ = self._adapter.to_sample_dataset(rawdataset=tsdataset, **param)

        #############################
        # case 7                    #
        # time_window[0] too small. #
        #############################
        target_periods = 12
        known_periods = target_periods + 10
        observed_periods = target_periods
        tsdataset = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)

        # Given the following params, the min allowed time_window[0] can be calculated as follows:
        # min_allowed_time_window_lower_bound = (in + skip + out - 1) = 5 - 1 = 4.
        # Thus, the invalid (too small) time_window[0] values are 0, 1, 2, 3.
        common_param = {
            "in_chunk_len": 2,
            "skip_chunk_len": 1,
            "out_chunk_len": 2,
            "sampling_stride": 1
        }

        # time_window[0] = 3, too small.
        param = {**common_param, "time_window": (3, len(tsdataset.get_target().data) - 1)}
        with self.assertRaises(ValueError):
            _ = self._adapter.to_sample_dataset(tsdataset, **param)

        # time_window[0] = 2, too small.
        param = {**common_param, "time_window": (2, len(tsdataset.get_target().data) - 1)}
        with self.assertRaises(ValueError):
            _ = self._adapter.to_sample_dataset(tsdataset, **param)

        # time_window[0] = 1, too small.
        param = {**common_param, "time_window": (1, len(tsdataset.get_target().data) - 1)}
        with self.assertRaises(ValueError):
            _ = self._adapter.to_sample_dataset(tsdataset, **param)

        # time_window[0] = 0, too small.
        param = {**common_param, "time_window": (0, len(tsdataset.get_target().data) - 1)}
        with self.assertRaises(ValueError):
            _ = self._adapter.to_sample_dataset(tsdataset, **param)

        #############################
        # case 8                    #
        # time_window[1] too large. #
        #############################
        target_periods = 10
        known_periods = target_periods + 10
        observed_periods = target_periods
        tsdataset = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)

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
        with self.assertRaises(ValueError):
            _ = self._adapter.to_sample_dataset(tsdataset, **param)

        #################################################
        # case 9                                        #
        # TSDataset.known_cov.time_index[-1] too small. #
        #################################################
        target_periods = 10
        observed_periods = target_periods

        # (9.a) Given:
        # time_window[1] <= max_target_idx
        #
        # Below is case analysis:
        # Let max_known_timestamp be the max timestamp in the existing known_cov.time_index,
        # window_upper_bound_timestamp be the timestamp in the target.time_index which time_window[1] pointed to.
        #
        # The following inequation MUST be held:
        # max_known_timestamp >= window_upper_bound_timestamp
        #
        # Assume the following:
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

        param = {
            "in_chunk_len": 1,
            "skip_chunk_len": 1,
            "out_chunk_len": 2,
            "sampling_stride": 1,
            "time_window": (3, 9)
        }
        with self.assertRaises(ValueError):
            _ = self._adapter.to_sample_dataset(tsdataset, **param)

        # (9.b) Given:
        # time_window[1] > max_target_idx
        # and
        # max_known_timestamp < max_target_timestamp < window_upper_bound_timestamp
        #
        # Below is case analysis:
        # Let max_known_timestamp be the max timestamp in the existing known_cov.time_index,
        # window_upper_bound_timestamp be the timestamp in the target.time_index which time_window[1] pointed to,
        # max_target_timestamp be the timestamp of the target.time_index.
        #
        # Because we already know that max_known_timestamp < max_target_timestamp, so we do not need to compare
        # max_known_timestamp with window_upper_bound_timestamp. Instead, we just need to compare
        # max_known_timestamp with max_target_timestamp.
        #
        # In this case, the following inequation MUST be held:
        # max_known_timestamp >= max_target_timestamp
        #
        # Assume the following:
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

        param = {
            "in_chunk_len": 1,
            "skip_chunk_len": 1,
            "out_chunk_len": 2,
            "sampling_stride": 1,
            "time_window": (12, 12)
        }
        with self.assertRaises(ValueError):
            _ = self._adapter.to_sample_dataset(tsdataset, **param)

        # (9.c) Given:
        # time_window[1] > max_target_idx
        # and
        # max_target_timestamp < max_known_timestamp < window_upper_bound_timestamp
        #
        # Below is case analysis:
        # Let max_known_timestamp be the max timestamp in the existing known_cov.time_index,
        # window_upper_bound_timestamp be the timestamp in the target.time_index which time_window[1] pointed to,
        # max_target_timestamp be the timestamp of the target.time_index.
        #
        # Because we already know that max_target_timestamp < max_known_timestamp, so we must compare
        # max_known_timestamp with window_upper_bound_timestamp.
        #
        # In this case, the following inequation MUST be held:
        # max_known_timestamp >= window_upper_bound_timestamp
        #
        # Assume the following:
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
        known_periods = target_periods + 1
        tsdataset = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)

        param = {
            "in_chunk_len": 1,
            "skip_chunk_len": 1,
            "out_chunk_len": 2,
            "sampling_stride": 1,
            # known_cov len = 11, smaller than window[1].
            "time_window": (12, 12)
        }
        with self.assertRaises(ValueError):
            _ = self._adapter.to_sample_dataset(tsdataset, **param)

        #####################################################
        # case 10                                           #
        # TSDataset.observed_cov.time_index[-1] too small.  #
        #####################################################
        target_periods = 10
        known_periods = target_periods + 10

        # (10.a) Given:
        # time_window[1] > max_target_idx
        #
        # Below is case analysis:
        # Let max_observed_timestamp be the max timestamp in the existing observed_cov.time_index,
        # max_target_timestamp be the timestamp of the target.time_index.
        # As time_window[1] > max_target_idx, thus the following inequation MUST be held:
        # max_observed_timestamp >= max_target_timestamp
        #
        # Assume the following:
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

        param = {
            "in_chunk_len": 1,
            "skip_chunk_len": 1,
            "out_chunk_len": 2,
            "sampling_stride": 1,
            # validates observed_cov does NOT need to check if time_window[1] > max_target_idx, because the check
            # logic is same.
            "time_window": (12, 12)
        }
        with self.assertRaises(ValueError):
            _ = self._adapter.to_sample_dataset(tsdataset, **param)

        # (10.b) Given:
        # time_window[1] <= max_target_idx
        #
        # Below is case analysis:
        # Let max_observed_timestamp be the max timestamp in the existing observed_cov.time_index,
        # window_upper_bound_timestamp be the timestamp in the target.time_index which time_window[1] pointed to,
        # last_sample_past_target_tail_timestamp be the timestamp in the target.time_index which the tail index of the
        # past_target chunk pointed to.
        # As we already know that time_window[1] <= max_target_idx, so the following inequation MUST be held:
        # max_observed_timestamp >= last_sample_past_target_tail_timestamp
        #
        # Assume the following:
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

        param = {
            "in_chunk_len": 1,
            "skip_chunk_len": 1,
            "out_chunk_len": 2,
            "sampling_stride": 1,
            "time_window": (3, 8)
        }
        with self.assertRaises(ValueError):
            _ = self._adapter.to_sample_dataset(tsdataset, **param)

        #############################################################################################################
        # case 11                                                                                                   #
        # fill_last_value is categorical/numeric, but TSDataset contains numeric/categorical columns, respectively. #
        #############################################################################################################
        # This case is to guarantee:
        # (1) If fill_last_value is int-like, then TSDataset.target/known_cov/observed_cov must not contain float-like
        # data.
        # (2) Similarly, if fill_last_value is float-like, then TSDataset.target/known_cov/observed_cov must not contain
        # int-like data.
        target_periods = known_periods = observed_periods = 10
        bad_contain_data_map = {
            "float": [t(1.1) for t in [float, np.float, np.float16, np.float32, np.float64]],
            "int": [t(1) for t in [int, np.int, np.int8, np.int16, np.int32, np.int64]]
        }
        for bad_contain_type in bad_contain_data_map.keys():
            for bad_ts_name in ["target", "known_cov", "observed_cov"]:
                if bad_ts_name == "target":
                    if bad_contain_type == "float":
                        # make known_cov and observed_cov (good) int type.
                        cov_dtypes_bits = self._only_categorical_dtype_bits
                    else:
                        # make known_cov and observed_cov (good) float type.
                        cov_dtypes_bits = self._only_numeric_dtype_bits
                    target_none = False
                elif bad_ts_name == "known_cov":
                    if bad_contain_type == "float":
                        # make known bad (float, 2**0), make observed_cov good (int, 2**4)
                        cov_dtypes_bits = 2**0 + 2**4
                    else:
                        # make known bad (int, 2**3), make observed_cov good (float, 2**1)
                        cov_dtypes_bits = 2**1 + 2**3
                    target_none = True
                # bad_ts_name == "observed_cov"
                else:
                    if bad_contain_type == "float":
                        # make known good (int, 2**3), make observed_cov bad (float, 2**1)
                        # 2**3 = categorical known (good)
                        cov_dtypes_bits = 2**1 + 2**3
                    else:
                        # make known good (float, 2**0), make observed_cov bad (int, 2**1)
                        cov_dtypes_bits = 2**1 + 2**3
                    target_none = True

                tsdataset = self._build_mock_ts_dataset(
                    target_periods=target_periods,
                    known_periods=known_periods,
                    observed_periods=observed_periods,
                    cov_dtypes_bits=cov_dtypes_bits
                )
                if target_none:
                    tsdataset.target = None

                for bad_fill_value in bad_contain_data_map[bad_contain_type]:
                    param = {"fill_last_value": bad_fill_value, "time_window": None}
                    with self.assertRaises(ValueError):
                        _ = self._adapter.to_sample_dataset(tsdataset, **param)

    def test_to_paddle_dataloader_good_case(self):
        """
        Test to_paddle_dataloader() good cases.
        """
        param = {"out_chunk_len": 1}

        target_keys = {"past_target", "future_target"}
        known_keys = {"known_cov_numeric", "known_cov_categorical"}
        observed_keys = {"observed_cov_numeric", "observed_cov_categorical"}
        static_keys = {"static_cov_numeric", "static_cov_categorical"}

        numeric_keys = {"known_cov_numeric", "observed_cov_numeric", "static_cov_numeric"}
        categorical_keys = {"known_cov_categorical", "observed_cov_categorical", "static_cov_categorical"}
        ############################################################
        # case 1 (Typical usage for forcasting and representation) #
        # 1) target, known_cov, observed_cov, static_cov NOT None. #
        ############################################################
        target_periods = 10
        known_periods = target_periods + 10
        observed_periods = target_periods

        # (a) Both numeric and categorical cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_bits=self._both_numeric_and_categorical_dtype_bits
        )

        sample_ds = self._adapter.to_sample_dataset(tsdataset, **param)
        batch_size = 2
        sample_dataloader = self._adapter.to_paddle_dataloader(sample_ds, batch_size, shuffle=False)

        self._compare_sample_dataset_and_sample_dataloader(
            sample_ds=sample_ds,
            sample_dataloader=sample_dataloader,
            batch_size=batch_size,
            good_keys=target_keys.union(numeric_keys).union(categorical_keys)
        )

        # (b) ONLY numeric cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_bits=self._only_numeric_dtype_bits
        )

        sample_ds = self._adapter.to_sample_dataset(tsdataset, **param)
        batch_size = 2
        sample_dataloader = self._adapter.to_paddle_dataloader(sample_ds, batch_size, shuffle=False)

        self._compare_sample_dataset_and_sample_dataloader(
            sample_ds=sample_ds,
            sample_dataloader=sample_dataloader,
            batch_size=batch_size,
            good_keys=target_keys.union(numeric_keys)
        )

        # (c) ONLY categorical cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_bits=self._only_categorical_dtype_bits
        )

        sample_ds = self._adapter.to_sample_dataset(tsdataset, out_chunk_len=1)
        batch_size = 2
        sample_dataloader = self._adapter.to_paddle_dataloader(sample_ds, batch_size, shuffle=False)

        self._compare_sample_dataset_and_sample_dataloader(
            sample_ds=sample_ds,
            sample_dataloader=sample_dataloader,
            batch_size=batch_size,
            good_keys=target_keys.union(categorical_keys)
        )

        ############################################################
        # case 2 (Typical usage for forcasting and representation) #
        # 2) known_cov is None.                                    #
        # 3) target, observed_cov, static_cov NOT None.            #
        ############################################################
        target_periods = 10
        known_periods = target_periods + 10
        observed_periods = target_periods

        # (a) Both numeric and categorical cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_bits=self._both_numeric_and_categorical_dtype_bits
        )
        # Explicitly set to None.
        tsdataset.known_cov = None

        sample_ds = self._adapter.to_sample_dataset(tsdataset, **param)
        batch_size = 2
        sample_dataloader = self._adapter.to_paddle_dataloader(sample_ds, batch_size, shuffle=False)

        self._compare_sample_dataset_and_sample_dataloader(
            sample_ds=sample_ds,
            sample_dataloader=sample_dataloader,
            batch_size=batch_size,
            good_keys=target_keys.union(numeric_keys).union(categorical_keys) - known_keys
        )

        # (b) ONLY numeric cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_bits=self._only_numeric_dtype_bits
        )
        # Explicitly set to None.
        tsdataset.known_cov = None

        sample_ds = self._adapter.to_sample_dataset(tsdataset, out_chunk_len=1)
        batch_size = 2
        sample_dataloader = self._adapter.to_paddle_dataloader(sample_ds, batch_size, shuffle=False)

        self._compare_sample_dataset_and_sample_dataloader(
            sample_ds=sample_ds,
            sample_dataloader=sample_dataloader,
            batch_size=batch_size,
            good_keys=target_keys.union(numeric_keys) - known_keys
        )

        # (c) ONLY categorical cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_bits=self._only_categorical_dtype_bits
        )
        # Explicitly set to None.
        tsdataset.known_cov = None

        sample_ds = self._adapter.to_sample_dataset(tsdataset, **param)
        batch_size = 2
        sample_dataloader = self._adapter.to_paddle_dataloader(sample_ds, batch_size, shuffle=False)

        self._compare_sample_dataset_and_sample_dataloader(
            sample_ds=sample_ds,
            sample_dataloader=sample_dataloader,
            batch_size=batch_size,
            good_keys=target_keys.union(categorical_keys) - known_keys
        )

        ############################################################
        # case 3 (Typical usage for forcasting and representation) #
        # 3) observed_cov is None.                                 #
        # 4) target, known_cov, static_cov NOT None.               #
        ############################################################
        target_periods = 10
        known_periods = target_periods + 10
        observed_periods = target_periods

        # (a) Both numeric and categorical cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_bits=self._both_numeric_and_categorical_dtype_bits
        )
        # Explicitly set to None.
        tsdataset.observed_cov = None

        sample_ds = self._adapter.to_sample_dataset(tsdataset, **param)
        batch_size = 2
        sample_dataloader = self._adapter.to_paddle_dataloader(sample_ds, batch_size, shuffle=False)

        self._compare_sample_dataset_and_sample_dataloader(
            sample_ds=sample_ds,
            sample_dataloader=sample_dataloader,
            batch_size=batch_size,
            good_keys=target_keys.union(numeric_keys).union(categorical_keys) - observed_keys
        )

        # (b) ONLY numeric cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_bits=self._only_numeric_dtype_bits
        )
        # Explicitly set to None.
        tsdataset.observed_cov = None

        sample_ds = self._adapter.to_sample_dataset(tsdataset, **param)
        batch_size = 2
        sample_dataloader = self._adapter.to_paddle_dataloader(sample_ds, batch_size, shuffle=False)

        self._compare_sample_dataset_and_sample_dataloader(
            sample_ds=sample_ds,
            sample_dataloader=sample_dataloader,
            batch_size=batch_size,
            good_keys=target_keys.union(numeric_keys) - observed_keys
        )

        # 2.3 ONLY categorical cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_bits=self._only_categorical_dtype_bits
        )
        # Explicitly set to None.
        tsdataset.observed_cov = None

        sample_ds = self._adapter.to_sample_dataset(tsdataset, **param)
        batch_size = 2
        sample_dataloader = self._adapter.to_paddle_dataloader(sample_ds, batch_size, shuffle=False)

        self._compare_sample_dataset_and_sample_dataloader(
            sample_ds=sample_ds,
            sample_dataloader=sample_dataloader,
            batch_size=batch_size,
            good_keys=target_keys.union(categorical_keys) - observed_keys
        )

        ############################################################
        # case 4 (Typical usage for forcasting and representation) #
        # 1) static_cov is None.                                   #
        # 2) target, known_cov, observed_cov NOT None.             #
        ############################################################
        target_periods = 10
        known_periods = target_periods + 10
        observed_periods = target_periods

        # (a) Both numeric and categorical cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_bits=self._both_numeric_and_categorical_dtype_bits
        )
        # Explicitly set to None.
        tsdataset.static_cov = None

        sample_ds = self._adapter.to_sample_dataset(tsdataset, **param)
        batch_size = 2
        sample_dataloader = self._adapter.to_paddle_dataloader(sample_ds, batch_size, shuffle=False)

        self._compare_sample_dataset_and_sample_dataloader(
            sample_ds=sample_ds,
            sample_dataloader=sample_dataloader,
            batch_size=batch_size,
            good_keys=target_keys.union(numeric_keys).union(categorical_keys) - static_keys
        )

        # (b) ONLY numeric cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_bits=self._only_numeric_dtype_bits
        )
        # Explicitly set to None.
        tsdataset.static_cov = None

        sample_ds = self._adapter.to_sample_dataset(tsdataset, **param)
        batch_size = 2
        sample_dataloader = self._adapter.to_paddle_dataloader(sample_ds, batch_size, shuffle=False)

        self._compare_sample_dataset_and_sample_dataloader(
            sample_ds=sample_ds,
            sample_dataloader=sample_dataloader,
            batch_size=batch_size,
            good_keys=target_keys.union(numeric_keys) - static_keys
        )

        # (c) ONLY categorical cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_bits=self._only_categorical_dtype_bits
        )
        # Explicitly set to None.
        tsdataset.static_cov = None

        sample_ds = self._adapter.to_sample_dataset(tsdataset, **param)
        batch_size = 2
        sample_dataloader = self._adapter.to_paddle_dataloader(sample_ds, batch_size, shuffle=False)

        self._compare_sample_dataset_and_sample_dataloader(
            sample_ds=sample_ds,
            sample_dataloader=sample_dataloader,
            batch_size=batch_size,
            good_keys=target_keys.union(categorical_keys) - static_keys
        )

        ############################################################
        # case 5 (Typical usage for forcasting and representation) #
        # 1) target is NOT None.                                   #
        # 2) known_cov = observed_cov = static_cov = None.         #
        ############################################################
        target_periods = 10
        known_periods = target_periods + 10
        observed_periods = target_periods

        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            # cov_dtypes_bits does not matter because all covariates will be set to None.
            cov_dtypes_bits=self._both_numeric_and_categorical_dtype_bits
        )
        # Explicitly set known and observed timeseries to None.
        tsdataset.known_cov = None
        tsdataset.observed_cov = None
        tsdataset.static_cov = None

        sample_ds = self._adapter.to_sample_dataset(tsdataset, **param)
        batch_size = 2
        sample_dataloader = self._adapter.to_paddle_dataloader(sample_ds, batch_size, shuffle=False)

        self._compare_sample_dataset_and_sample_dataloader(
            sample_ds=sample_ds,
            sample_dataloader=sample_dataloader,
            batch_size=batch_size,
            good_keys=target_keys - known_keys - observed_keys - static_keys
        )

        ##############################################
        # case 6 (Typical usage for anomaly)         #
        # 1) observed_cov is NOT None.               #
        # 2) target = known_cov = static_cov = None. #
        ##############################################
        target_periods = 10
        known_periods = target_periods + 10
        observed_periods = target_periods

        # (a) Both numeric and categorical cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_bits=self._both_numeric_and_categorical_dtype_bits
        )
        # Explicitly set to None.
        tsdataset.target = None
        tsdataset.known_cov = None
        tsdataset.static_cov = None

        sample_ds = self._adapter.to_sample_dataset(tsdataset, **param)
        batch_size = 2
        sample_dataloader = self._adapter.to_paddle_dataloader(sample_ds, batch_size, shuffle=False)

        self._compare_sample_dataset_and_sample_dataloader(
            sample_ds=sample_ds,
            sample_dataloader=sample_dataloader,
            batch_size=batch_size,
            good_keys=observed_keys - target_keys - known_keys - static_keys
        )

        # (b) ONLY numeric cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_bits=self._only_numeric_dtype_bits
        )
        # Explicitly set target, known_cov and static_cov to None.
        tsdataset.target = None
        tsdataset.known_cov = None
        tsdataset.static_cov = None

        sample_ds = self._adapter.to_sample_dataset(tsdataset, **param)
        batch_size = 2
        sample_dataloader = self._adapter.to_paddle_dataloader(sample_ds, batch_size, shuffle=False)

        self._compare_sample_dataset_and_sample_dataloader(
            sample_ds=sample_ds,
            sample_dataloader=sample_dataloader,
            batch_size=batch_size,
            good_keys=observed_keys - categorical_keys - target_keys - known_keys - static_keys
        )

        # (c) ONLY categorical cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_bits=self._only_categorical_dtype_bits
        )
        # Explicitly set to None.
        tsdataset.target = None
        tsdataset.known_cov = None
        tsdataset.static_cov = None

        sample_ds = self._adapter.to_sample_dataset(tsdataset, **param)
        batch_size = 2
        sample_dataloader = self._adapter.to_paddle_dataloader(sample_ds, batch_size, shuffle=False)

        self._compare_sample_dataset_and_sample_dataloader(
            sample_ds=sample_ds,
            sample_dataloader=sample_dataloader,
            batch_size=batch_size,
            good_keys=observed_keys - numeric_keys - target_keys - known_keys - static_keys
        )

    def test_to_ml_dataloader_good_case(self):
        """
        test to_ml_dataloader() good cases.
        """
        param = {
            "in_chunk_len": 1,
            "skip_chunk_len": 0,
            "out_chunk_len": 1,
            "sampling_stride": 1,
            "time_window": (1, 19)
        }

        target_keys = {"past_target", "future_target"}
        known_keys = {"known_cov_numeric", "known_cov_categorical"}
        observed_keys = {"observed_cov_numeric", "observed_cov_categorical"}
        static_keys = {"static_cov_numeric", "static_cov_categorical"}

        numeric_keys = {"known_cov_numeric", "observed_cov_numeric", "static_cov_numeric"}
        categorical_keys = {"known_cov_categorical", "observed_cov_categorical", "static_cov_categorical"}
        ############################################################
        # case 1 (Typical usage for forcasting and representation) #
        # 1) target, known_cov, observed_cov, static_cov NOT None. #
        ############################################################
        target_periods = 20
        observed_periods = target_periods
        known_periods = target_periods + 10

        # (a) Both numeric and categorical cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_bits=self._both_numeric_and_categorical_dtype_bits
        )
        sample_ds = self._adapter.to_sample_dataset(tsdataset, **param)
        batch_size = 2
        sample_dataloader = self._adapter.to_ml_dataloader(sample_dataset=sample_ds, batch_size=batch_size)

        self._compare_sample_dataset_and_sample_dataloader(
            sample_ds=sample_ds,
            sample_dataloader=sample_dataloader,
            batch_size=batch_size,
            good_keys=target_keys.union(numeric_keys).union(categorical_keys)
        )

        # (b) ONLY numeric cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_bits=self._only_numeric_dtype_bits
        )

        sample_ds = self._adapter.to_sample_dataset(tsdataset, **param)
        batch_size = 2
        sample_dataloader = self._adapter.to_ml_dataloader(sample_dataset=sample_ds, batch_size=batch_size)

        self._compare_sample_dataset_and_sample_dataloader(
            sample_ds=sample_ds,
            sample_dataloader=sample_dataloader,
            batch_size=batch_size,
            good_keys=target_keys.union(numeric_keys)
        )

        # (c) ONLY categorical cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_bits=self._only_categorical_dtype_bits
        )

        sample_ds = self._adapter.to_sample_dataset(tsdataset, **param)
        batch_size = 2
        sample_dataloader = self._adapter.to_ml_dataloader(sample_dataset=sample_ds, batch_size=batch_size)

        self._compare_sample_dataset_and_sample_dataloader(
            sample_ds=sample_ds,
            sample_dataloader=sample_dataloader,
            batch_size=batch_size,
            good_keys=target_keys.union(categorical_keys)
        )

        ############################################################
        # case 2 (Typical usage for forcasting and representation) #
        # 1) known_cov is None.                                    #
        # 2) target, observed_cov, static_cov all NOT None.        #
        ############################################################
        # (a) Both numeric and categorical cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_bits=self._both_numeric_and_categorical_dtype_bits
        )
        tsdataset.known_cov = None

        sample_ds = self._adapter.to_sample_dataset(tsdataset, **param)
        batch_size = 2
        sample_dataloader = self._adapter.to_ml_dataloader(sample_dataset=sample_ds, batch_size=batch_size)

        self._compare_sample_dataset_and_sample_dataloader(
            sample_ds=sample_ds,
            sample_dataloader=sample_dataloader,
            batch_size=batch_size,
            good_keys=target_keys.union(numeric_keys).union(categorical_keys) - known_keys
        )

        # (b) ONLY numeric cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_bits=self._only_numeric_dtype_bits
        )
        tsdataset.known_cov = None

        sample_ds = self._adapter.to_sample_dataset(tsdataset, **param)
        batch_size = 2
        sample_dataloader = self._adapter.to_ml_dataloader(sample_dataset=sample_ds, batch_size=batch_size)

        self._compare_sample_dataset_and_sample_dataloader(
            sample_ds=sample_ds,
            sample_dataloader=sample_dataloader,
            batch_size=batch_size,
            good_keys=target_keys.union(numeric_keys) - known_keys
        )

        # (c) ONLY categorical cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_bits=self._only_categorical_dtype_bits
        )
        tsdataset.known_cov = None

        sample_ds = self._adapter.to_sample_dataset(tsdataset, **param)
        batch_size = 2
        sample_dataloader = self._adapter.to_ml_dataloader(sample_dataset=sample_ds, batch_size=batch_size)

        self._compare_sample_dataset_and_sample_dataloader(
            sample_ds=sample_ds,
            sample_dataloader=sample_dataloader,
            batch_size=batch_size,
            good_keys=target_keys.union(categorical_keys) - known_keys
        )

        ############################################################
        # case 3 (Typical usage for forcasting and representation) #
        # 1) observed_cov is None.                                 #
        # 2) target, static_cov, known_cov NOT None.               #
        ############################################################
        # (a) Both numeric and categorical cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_bits=self._both_numeric_and_categorical_dtype_bits
        )
        tsdataset.observed_cov = None

        sample_ds = self._adapter.to_sample_dataset(tsdataset, **param)
        batch_size = 2
        sample_dataloader = self._adapter.to_ml_dataloader(sample_dataset=sample_ds, batch_size=batch_size)

        self._compare_sample_dataset_and_sample_dataloader(
            sample_ds=sample_ds,
            sample_dataloader=sample_dataloader,
            batch_size=batch_size,
            good_keys=target_keys.union(numeric_keys).union(categorical_keys) - observed_keys
        )

        # (b) ONLY numeric cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_bits=self._only_numeric_dtype_bits
        )
        tsdataset.observed_cov = None

        sample_ds = self._adapter.to_sample_dataset(tsdataset, **param)
        batch_size = 2
        sample_dataloader = self._adapter.to_ml_dataloader(sample_dataset=sample_ds, batch_size=batch_size)

        self._compare_sample_dataset_and_sample_dataloader(
            sample_ds=sample_ds,
            sample_dataloader=sample_dataloader,
            batch_size=batch_size,
            good_keys=target_keys.union(numeric_keys) - observed_keys
        )

        # (c) ONLY categorical cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_bits=self._only_categorical_dtype_bits
        )
        tsdataset.observed_cov = None

        sample_ds = self._adapter.to_sample_dataset(tsdataset, **param)
        batch_size = 2
        sample_dataloader = self._adapter.to_ml_dataloader(sample_dataset=sample_ds, batch_size=batch_size)

        self._compare_sample_dataset_and_sample_dataloader(
            sample_ds=sample_ds,
            sample_dataloader=sample_dataloader,
            batch_size=batch_size,
            good_keys=target_keys.union(categorical_keys) - observed_keys
        )

        ##############################################
        # case 4 (Typical usage for anomaly)         #
        # 2) target, known_cov, static_cov are None. #
        # 1) observed_cov NOT None.                  #
        ##############################################
        # (a) Both numeric and categorical cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_bits=self._both_numeric_and_categorical_dtype_bits
        )
        tsdataset.target = None
        tsdataset.known_cov = None
        tsdataset.static_cov = None

        sample_ds = self._adapter.to_sample_dataset(tsdataset, **param)
        batch_size = 2
        sample_dataloader = self._adapter.to_ml_dataloader(sample_dataset=sample_ds, batch_size=batch_size)

        self._compare_sample_dataset_and_sample_dataloader(
            sample_ds=sample_ds,
            sample_dataloader=sample_dataloader,
            batch_size=batch_size,
            good_keys=observed_keys - target_keys - known_keys - static_keys
        )

        # (b) ONLY numeric cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_bits=self._only_numeric_dtype_bits
        )
        tsdataset.target = None
        tsdataset.known_cov = None
        tsdataset.static_cov = None

        sample_ds = self._adapter.to_sample_dataset(tsdataset, **param)
        batch_size = 2
        sample_dataloader = self._adapter.to_ml_dataloader(sample_dataset=sample_ds, batch_size=batch_size)

        self._compare_sample_dataset_and_sample_dataloader(
            sample_ds=sample_ds,
            sample_dataloader=sample_dataloader,
            batch_size=batch_size,
            good_keys=observed_keys - categorical_keys - target_keys - known_keys - static_keys
        )

        # (c) ONLY categorical cov features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_bits=self._only_categorical_dtype_bits
        )
        tsdataset.target = None
        tsdataset.known_cov = None
        tsdataset.static_cov = None

        sample_ds = self._adapter.to_sample_dataset(tsdataset, **param)
        batch_size = 2
        sample_dataloader = self._adapter.to_ml_dataloader(sample_dataset=sample_ds, batch_size=batch_size)

        self._compare_sample_dataset_and_sample_dataloader(
            sample_ds=sample_ds,
            sample_dataloader=sample_dataloader,
            batch_size=batch_size,
            good_keys=observed_keys - numeric_keys - target_keys - known_keys - static_keys
        )

    def _build_mock_ts_dataset(
        self,
        target_periods: int,
        known_periods: int,
        observed_periods: int,
        target_start_timestamp: pd.Timestamp = None,
        known_start_timestamp: pd.Timestamp = None,
        observed_start_timestamp: pd.Timestamp = None,
        target_timeindex_type: str = pd.DatetimeIndex.__name__,
        known_timeindex_type: str = pd.DatetimeIndex.__name__,
        observed_timeindex_type: str = pd.DatetimeIndex.__name__,
        cov_dtypes_bits: int = 2**0 + 2**2 + 2**4
    ):
        """
        Build mock dataset.

        cov_dtypes_bits totally control 6 bits:
        known_numeric_bit = 2**0 (default True)
        observed_numeric_bit = 2**1 (default True)
        static_numeric_bit = 2**2 (default True)
        known_categorical_bit = 2**3 (default False)
        observed_categorical_bit = 2**4 (default False)
        static_categorical_bit = 2**5 (default False)

        Usage Example:
        both numeric and categorical = 2**6 - 1
        only numeric = 2**3 - 1
        only categorical = sum([2**n for n in range(3, 6)])
        """
        # Only need to control 6 bits, so cannot be larger than 2**6 - 1.
        max_control_bit_num = 6
        self.assertTrue(0 <= cov_dtypes_bits <= 2**max_control_bit_num - 1)
        known_contain_numeric_bit = 2**0
        observed_contain_numeric_bit = 2**1
        static_contain_numeric_bit = 2**2
        known_contain_categorical_bit = 2**3
        observed_contain_categorical_bit = 2**4
        static_contain_categorical_bit = 2**5

        numeric_dtype = np.float32
        categorical_dtype = np.int64
        freq: str = "1D"

        if (target_start_timestamp and known_start_timestamp and observed_start_timestamp) is not True:
            # default_timestamp = pd.Timestamp(datetime.datetime.now().date())
            default_timestamp = pd.Timestamp("2022-01-01")
            target_start_timestamp = default_timestamp
            known_start_timestamp = default_timestamp
            observed_start_timestamp = default_timestamp

        # target
        self.assertTrue(target_timeindex_type in {pd.DatetimeIndex.__name__, pd.RangeIndex.__name__})
        if target_timeindex_type == pd.DatetimeIndex.__name__:
            target_index = pd.date_range(start=target_start_timestamp, periods=target_periods, freq=freq)
        else:
            target_index = pd.RangeIndex(start=0, stop=target_periods, step=1)
        target_df = pd.DataFrame(
            data=np.array([i for i in range(target_periods)], dtype=numeric_dtype),
            index=target_index,
            columns=["target_numeric_0"]
        )

        # known
        self.assertTrue(known_timeindex_type in {pd.DatetimeIndex.__name__, pd.RangeIndex.__name__})
        if known_timeindex_type == pd.DatetimeIndex.__name__:
            known_index = pd.date_range(start=known_start_timestamp, periods=known_periods, freq=freq)
        else:
            known_index = pd.RangeIndex(start=0, stop=known_periods, step=1)
        known_raw_data = [(i * 10, i * 100) for i in range(known_periods)]
        known_numeric_df = None
        if (cov_dtypes_bits & known_contain_numeric_bit) == known_contain_numeric_bit:
            known_numeric_data = np.array(known_raw_data, dtype=numeric_dtype)
            known_numeric_df = pd.DataFrame(
                data=known_numeric_data,
                index=known_index,
                columns=["known_numeric_0", "known_numeric_1"]
            )

        known_categorical_df = None
        if (cov_dtypes_bits & known_contain_categorical_bit) == known_contain_categorical_bit:
            known_categorical_data = np.array(known_raw_data, dtype=categorical_dtype)
            known_categorical_df = pd.DataFrame(
                data=known_categorical_data,
                index=known_index,
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
        self.assertTrue(observed_timeindex_type in {pd.DatetimeIndex.__name__, pd.RangeIndex.__name__})
        if observed_timeindex_type == pd.DatetimeIndex.__name__:
            observed_index = pd.date_range(start=observed_start_timestamp, periods=observed_periods, freq=freq)
        else:
            observed_index = pd.RangeIndex(start=0, stop=known_periods, step=1)
        observed_raw_data = [(i * -1, i * -10) for i in range(observed_periods)]
        observed_numeric_df = None
        if (cov_dtypes_bits & observed_contain_numeric_bit) == observed_contain_numeric_bit:
            observed_numeric_data = np.array(observed_raw_data, dtype=numeric_dtype)
            observed_numeric_df = pd.DataFrame(
                data=observed_numeric_data,
                index=observed_index,
                columns=["observed_numeric_0", "observed_numeric_1"]
            )

        observed_categorical_df = None
        if (cov_dtypes_bits & observed_contain_categorical_bit) == observed_contain_categorical_bit:
            observed_categorical_data = np.array(observed_raw_data, dtype=categorical_dtype)
            observed_categorical_df = pd.DataFrame(
                data=observed_categorical_data,
                index=observed_index,
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
        if (cov_dtypes_bits & static_contain_numeric_bit) == static_contain_numeric_bit:
            static["static_numeric"] = np.float32(1)
        if (cov_dtypes_bits & static_contain_categorical_bit) == static_contain_categorical_bit:
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
        sample_ds: SampleDataset,
        has_future_target_key: bool = True
    ) -> None:
        """
        Given a TSDataset and a built sample Dataset, compare if these data are matched.

        Args:
            tsdataset(TSDataset): Raw TSDataset.
            sample_ds(PaddleDatasetImpl): Built sample Dataset.
            has_future_target_key(bool, optional): Whether the built samples contain "future_target" key. Default True.
        """
        in_chunk_len = sample_ds._in_chunk_len
        skip_chunk_len = sample_ds._skip_chunk_len
        out_chunk_len = sample_ds._out_chunk_len
        sampling_stride = sample_ds._sampling_stride
        time_window = sample_ds._time_window
        fill_last_value = sample_ds._fill_last_value

        target_ts = tsdataset.target
        known_ts = tsdataset.known_cov
        observed_ts = tsdataset.observed_cov
        static_cov = tsdataset.static_cov

        first_sample_tail_idx = time_window[0]
        sidx = 0

        # As target/known cov/observed cov might start with different timestamp, thus needs to compute offset.
        std_start_timestamp = sample_ds._std_timeindex[0]
        target_offset = 0 if target_ts is None else target_ts.time_index.get_loc(std_start_timestamp)
        known_offset = 0 if known_ts is None else known_ts.time_index.get_loc(std_start_timestamp)
        observed_offset = 0 if observed_ts is None else observed_ts.time_index.get_loc(std_start_timestamp)

        # validate samples EXCEPT last sample.
        while sidx < len(sample_ds.samples) - 1:
            curr_sample = sample_ds[sidx]
            curr_sample_tail = first_sample_tail_idx + sidx * sampling_stride
            if target_ts is not None:
                df = target_ts.to_dataframe(copy=False)
                ###############
                # past_target #
                ###############
                ndarray_tail = target_offset + curr_sample_tail - skip_chunk_len - out_chunk_len
                ndarray = df.to_numpy(copy=False)[ndarray_tail - in_chunk_len + 1:ndarray_tail + 1]
                # data ok.
                self.assertTrue(np.alltrue(ndarray == curr_sample["past_target"]))
                # dtype ok.
                self.assertEqual(ndarray.dtype, curr_sample["past_target"].dtype)

                #################
                # future_target #
                #################
                # Built sample does NOT contain Y.
                if has_future_target_key is False:
                    self.assertTrue("future_target" not in curr_sample.keys())
                # Built samples contain Y.
                else:
                    ndarray_tail = target_offset + curr_sample_tail + 1
                    ndarray_head = ndarray_tail - out_chunk_len
                    ndarray = df.to_numpy(copy=False)[ndarray_head:ndarray_tail]
                    # data ok.
                    self.assertTrue(np.alltrue(ndarray == curr_sample["future_target"]))
                    # dtype ok.
                    self.assertEqual(ndarray.dtype, curr_sample["future_target"].dtype)

            #############
            # known_cov #
            #############
            if known_ts is not None:
                df = known_ts.to_dataframe(copy=False)

                ndarray_right_tail = known_offset + curr_sample_tail + 1
                ndarray_right_head = ndarray_right_tail - out_chunk_len

                ndarray_left_tail = ndarray_right_head - 1 - skip_chunk_len + 1
                ndarray_left_head = ndarray_left_tail - in_chunk_len
                # numeric
                if "known_cov_numeric" in curr_sample.keys():
                    numeric_ndarray = df.select_dtypes(include=self._numeric_dtype).to_numpy(copy=False)
                    ndarray = np.vstack(tup=(
                        numeric_ndarray[ndarray_left_head:ndarray_left_tail],
                        numeric_ndarray[ndarray_right_head:ndarray_right_tail]
                    ))
                    # data ok.
                    self.assertTrue(np.alltrue(ndarray == curr_sample["known_cov_numeric"]))
                    # dtype ok.
                    self.assertEqual(ndarray.dtype, curr_sample["known_cov_numeric"].dtype)

                # categorical
                if "known_cov_categorical" in curr_sample.keys():
                    categorical_ndarray = df.select_dtypes(include=self._categorical_dtype).to_numpy(copy=False)
                    ndarray = np.vstack(tup=(
                        categorical_ndarray[ndarray_left_head:ndarray_left_tail],
                        categorical_ndarray[ndarray_right_head:ndarray_right_tail]
                    ))
                    # categorical_df = known_df.select_dtypes(include=categorical_dtype)
                    # categorical_ndarray = categorical_df.to_numpy(copy=False)
                    # categorical_right_ndarray = categorical_ndarray[known_right_head:known_right_tail]
                    # categorical_left_ndarray = categorical_ndarray[known_left_head:known_left_tail]
                    # known_categorical_ndarray = np.vstack((categorical_left_ndarray, categorical_right_ndarray))
                    # data ok.
                    self.assertTrue(np.alltrue(ndarray == curr_sample["known_cov_categorical"]))
                    # dtype ok.
                    self.assertEqual(ndarray.dtype, curr_sample["known_cov_categorical"].dtype)
            # known_cov is None.
            else:
                self.assertTrue("known_cov_numeric" not in curr_sample.keys())
                self.assertTrue("known_cov_categorical" not in curr_sample.keys())

            ################
            # observed_cov #
            ################
            if observed_ts is not None:
                df = observed_ts.to_dataframe(copy=False)
                ndarray_tail = observed_offset + curr_sample_tail - skip_chunk_len - out_chunk_len
                # numeric
                if "observed_cov_numeric" in curr_sample.keys():
                    numeric_ndarray = df.select_dtypes(include=self._numeric_dtype).to_numpy(copy=False)
                    # numeric_ndarray = numeric_df.to_numpy(copy=False)
                    ndarray = numeric_ndarray[ndarray_tail - in_chunk_len + 1:ndarray_tail + 1]
                    # observed_numeric_ndarray = numeric_ndarray[observed_tail - in_chunk_len + 1:observed_tail + 1]
                    # data ok.
                    self.assertTrue(np.alltrue(ndarray == curr_sample["observed_cov_numeric"]))
                    # dtype ok.
                    self.assertEqual(ndarray.dtype, curr_sample["observed_cov_numeric"].dtype)
                # categorical
                if "observed_cov_categorical" in curr_sample.keys():
                    categorical_ndarray = df.select_dtypes(include=self._categorical_dtype).to_numpy(copy=False)
                    # categorical_ndarray = categorical_df.to_numpy(copy=False)
                    ndarray = categorical_ndarray[ndarray_tail - in_chunk_len + 1:ndarray_tail + 1]
                    # data ok.
                    self.assertTrue(np.alltrue(ndarray == curr_sample["observed_cov_categorical"]))
                    # dtype ok.
                    self.assertEqual(ndarray.dtype, curr_sample["observed_cov_categorical"].dtype)
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
                    numeric = [t[1] for t in sorted_static_cov if isinstance(t[1], self._numeric_dtype)]
                    # data ok.
                    self.assertTrue(np.alltrue(numeric == curr_sample["static_cov_numeric"][0]))
                    # dtype ok.
                    self.assertEqual(numeric[0].dtype, curr_sample["static_cov_numeric"][0].dtype)
                # categorical
                if "static_cov_categorical" in curr_sample.keys():
                    categorical = [t[1] for t in sorted_static_cov if isinstance(t[1], self._categorical_dtype)]
                    # data ok.
                    self.assertTrue(np.alltrue(categorical == curr_sample["static_cov_categorical"][0]))
                    # dtype ok.
                    self.assertEqual(categorical[0].dtype, curr_sample["static_cov_categorical"][0].dtype)
            # static_cov is None
            else:
                self.assertTrue("static_cov_numeric" not in curr_sample.keys())
                self.assertTrue("static_cov_categorical" not in curr_sample.keys())

            sidx += 1

        # check last sample, possibly be filled.
        last_sample = sample_ds.samples[sidx]
        last_sample_tail_timestamp = sample_ds._compute_last_sample_tail_timestamp()

        # because fill_last_value has no effect for static_cov, so check this part first.
        ##############
        # static_cov #
        ##############
        if static_cov is not None:
            # unsorted dict -> sorted list
            sorted_static_cov = sorted(static_cov.items(), key=lambda t: t[0])
            # numeric
            if "static_cov_numeric" in last_sample.keys():
                numeric = [t[1] for t in sorted_static_cov if isinstance(t[1], self._numeric_dtype)]
                # data ok.
                self.assertTrue(np.alltrue(numeric == last_sample["static_cov_numeric"][0]))
                # dtype ok.
                self.assertEqual(numeric[0].dtype, last_sample["static_cov_numeric"][0].dtype)
            # categorical
            if "static_cov_categorical" in last_sample.keys():
                categorical = [t[1] for t in sorted_static_cov if isinstance(t[1], self._categorical_dtype)]
                # data ok.
                self.assertTrue(np.alltrue(categorical == last_sample["static_cov_categorical"][0]))
                # dtype ok.
                self.assertEqual(categorical[0].dtype, last_sample["static_cov_categorical"][0].dtype)
        # static_cov is None
        else:
            self.assertTrue("static_cov_numeric" not in last_sample.keys())
            self.assertTrue("static_cov_categorical" not in last_sample.keys())

        # Then check target, known_cov and observed_cov for last sample.
        # last sample not filled
        if fill_last_value is None:
            last_sample_tail = first_sample_tail_idx + sidx * sampling_stride
            if target_ts is not None:
                df = target_ts.to_dataframe(copy=False)
                ###############################
                # past_target for last_sample #
                ###############################
                ndarray_tail = target_offset + last_sample_tail - skip_chunk_len - out_chunk_len
                ndarray = df.to_numpy(copy=False)[ndarray_tail - in_chunk_len + 1:ndarray_tail + 1]
                # data ok.
                self.assertTrue(np.alltrue(ndarray == last_sample["past_target"]))
                # dtype ok.
                self.assertEqual(ndarray.dtype, last_sample["past_target"].dtype)

                #################################
                # future_target for last_sample #
                #################################
                # Built sample does NOT contain Y.
                if has_future_target_key is False:
                    self.assertTrue("future_target" not in last_sample.keys())
                # Built samples contain Y.
                else:
                    ndarray_tail = target_offset + last_sample_tail + 1
                    ndarray_head = ndarray_tail - out_chunk_len
                    ndarray = df.to_numpy(copy=False)[ndarray_head:ndarray_tail]
                    # data ok.
                    self.assertTrue(np.alltrue(ndarray == last_sample["future_target"]))
                    # dtype ok.
                    self.assertEqual(ndarray.dtype, last_sample["future_target"].dtype)

            #############################
            # known_cov for last_sample #
            #############################
            if known_ts is not None:
                df = known_ts.to_dataframe(copy=False)

                ndarray_right_tail = known_offset + last_sample_tail + 1
                ndarray_right_head = ndarray_right_tail - out_chunk_len

                ndarray_left_tail = ndarray_right_head - 1 - skip_chunk_len + 1
                ndarray_left_head = ndarray_left_tail - in_chunk_len
                # numeric
                if "known_cov_numeric" in last_sample.keys():
                    numeric_ndarray = df.select_dtypes(include=self._numeric_dtype).to_numpy(copy=False)
                    ndarray = np.vstack(tup=(
                        numeric_ndarray[ndarray_left_head:ndarray_left_tail],
                        numeric_ndarray[ndarray_right_head:ndarray_right_tail]
                    ))
                    # data ok.
                    self.assertTrue(np.alltrue(ndarray == last_sample["known_cov_numeric"]))
                    # dtype ok.
                    self.assertEqual(ndarray.dtype, last_sample["known_cov_numeric"].dtype)

                # categorical
                if "known_cov_categorical" in last_sample.keys():
                    categorical_ndarray = df.select_dtypes(include=self._categorical_dtype).to_numpy(copy=False)
                    ndarray = np.vstack(tup=(
                        categorical_ndarray[ndarray_left_head:ndarray_left_tail],
                        categorical_ndarray[ndarray_right_head:ndarray_right_tail]
                    ))
                    # categorical_df = known_df.select_dtypes(include=categorical_dtype)
                    # categorical_ndarray = categorical_df.to_numpy(copy=False)
                    # categorical_right_ndarray = categorical_ndarray[known_right_head:known_right_tail]
                    # categorical_left_ndarray = categorical_ndarray[known_left_head:known_left_tail]
                    # known_categorical_ndarray = np.vstack((categorical_left_ndarray, categorical_right_ndarray))
                    # data ok.
                    self.assertTrue(np.alltrue(ndarray == last_sample["known_cov_categorical"]))
                    # dtype ok.
                    self.assertEqual(ndarray.dtype, last_sample["known_cov_categorical"].dtype)
            # known_cov is None.
            else:
                self.assertTrue("known_cov_numeric" not in last_sample.keys())
                self.assertTrue("known_cov_categorical" not in last_sample.keys())

            ################################
            # observed_cov for last_sample #
            ################################
            if observed_ts is not None:
                df = observed_ts.to_dataframe(copy=False)
                ndarray_tail = observed_offset + last_sample_tail - skip_chunk_len - out_chunk_len
                # numeric
                if "observed_cov_numeric" in last_sample.keys():
                    numeric_ndarray = df.select_dtypes(include=self._numeric_dtype).to_numpy(copy=False)
                    # numeric_ndarray = numeric_df.to_numpy(copy=False)
                    ndarray = numeric_ndarray[ndarray_tail - in_chunk_len + 1:ndarray_tail + 1]
                    # observed_numeric_ndarray = numeric_ndarray[observed_tail - in_chunk_len + 1:observed_tail + 1]
                    # data ok.
                    self.assertTrue(np.alltrue(ndarray == last_sample["observed_cov_numeric"]))
                    # dtype ok.
                    self.assertEqual(ndarray.dtype, last_sample["observed_cov_numeric"].dtype)
                # categorical
                if "observed_cov_categorical" in last_sample.keys():
                    categorical_ndarray = df.select_dtypes(include=self._categorical_dtype).to_numpy(copy=False)
                    # categorical_ndarray = categorical_df.to_numpy(copy=False)
                    ndarray = categorical_ndarray[ndarray_tail - in_chunk_len + 1:ndarray_tail + 1]
                    # data ok.
                    self.assertTrue(np.alltrue(ndarray == last_sample["observed_cov_categorical"]))
                    # dtype ok.
                    self.assertEqual(ndarray.dtype, last_sample["observed_cov_categorical"].dtype)
            # observed_cov is None.
            else:
                self.assertTrue("observed_cov_numeric" not in last_sample.keys())
                self.assertTrue("observed_cov_categorical" not in last_sample.keys())
        # last sample filled
        else:
            ###############
            # past_target #
            ###############
            # case 1 - target is too long, no need to fill
            # std_timeindex (filled) =    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
            # target_timeidx (raw)   = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
            # in = 2
            # skip = 0
            # out = 1
            # fill_last_value = 99.99
            # so last_sample["past_target"] (filled) = [10, 11, 12]
            # so extra_timeindex = pd.date_range(start=13, end=12)[1:] = [1:] = []
            # because last_sample_past_target_len = in + skip + out = 3, so
            # so sample_right = last_sample["past_target"][3 - 1 - 0 + 1:] = last_sample["past_target"][3:] = []
            # so in this case, right is empty, expected.
            # case 2 - target is too short, need to fill
            # std_timeindex (filled) =    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
            # target_timeidx (raw)   = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            # in = 2
            # skip = 0
            # out = 1
            # fill_last_value = 99.99
            # so last_sample["past_target"] (filled) = [10, 99.99, 99.99]
            # so extra_timeindex = pd.date_range(start=10, end=12) = [10, 11, 12][1:] = [11, 12]
            # because last_sample_past_target_len = in + skip + out = 3, so
            # sample_right = last_sample["past_target"][3 - 1 - 2 + 1:] = [99.99, 99.99]
            if target_ts is not None:
                self._compare_tsdataset_and_filled_last_sample(
                    timeseries=target_ts,
                    last_sample=last_sample,
                    sample_key="past_target",
                    offset=target_offset,
                    last_sample_tail_timestamp=last_sample_tail_timestamp,
                    std_timeindex=sample_ds._std_timeindex,
                    fill_last_value=fill_last_value,
                    sidx=sidx,
                    one_sample_len=in_chunk_len + skip_chunk_len + out_chunk_len
                )
            else:
                self.assertTrue("past_target" not in last_sample.keys())
            #############
            # known_cov #
            #############
            if known_ts is not None:
                if "known_cov_numeric" in last_sample.keys():
                    self._compare_tsdataset_and_filled_last_sample(
                        timeseries=known_ts,
                        last_sample=last_sample,
                        sample_key="known_cov_numeric",
                        offset=target_offset,
                        last_sample_tail_timestamp=last_sample_tail_timestamp,
                        std_timeindex=sample_ds._std_timeindex,
                        fill_last_value=fill_last_value,
                        sidx=sidx,
                        one_sample_len=in_chunk_len + skip_chunk_len + out_chunk_len
                    )
                if "known_cov_categorical" in last_sample.keys():
                    self._compare_tsdataset_and_filled_last_sample(
                        timeseries=known_ts,
                        last_sample=last_sample,
                        sample_key="known_cov_categorical",
                        offset=target_offset,
                        last_sample_tail_timestamp=last_sample_tail_timestamp,
                        std_timeindex=sample_ds._std_timeindex,
                        fill_last_value=fill_last_value,
                        sidx=sidx,
                        one_sample_len=in_chunk_len + skip_chunk_len + out_chunk_len
                    )
            else:
                self.assertTrue("known_cov_numeric" not in last_sample.keys())
                self.assertTrue("known_cov_categorical" not in last_sample.keys())
            ################
            # observed_cov #
            ################
            if observed_ts is not None:
                if "observed_cov_numeric" in last_sample.keys():
                    self._compare_tsdataset_and_filled_last_sample(
                        timeseries=observed_ts,
                        last_sample=last_sample,
                        sample_key="observed_cov_numeric",
                        offset=target_offset,
                        last_sample_tail_timestamp=last_sample_tail_timestamp,
                        std_timeindex=sample_ds._std_timeindex,
                        fill_last_value=fill_last_value,
                        sidx=sidx,
                        one_sample_len=in_chunk_len + skip_chunk_len + out_chunk_len
                    )
                if "observed_cov_categorical" in last_sample.keys():
                    self._compare_tsdataset_and_filled_last_sample(
                        timeseries=observed_ts,
                        last_sample=last_sample,
                        sample_key="observed_cov_categorical",
                        offset=target_offset,
                        last_sample_tail_timestamp=last_sample_tail_timestamp,
                        std_timeindex=sample_ds._std_timeindex,
                        fill_last_value=fill_last_value,
                        sidx=sidx,
                        one_sample_len=in_chunk_len + skip_chunk_len + out_chunk_len
                    )
            else:
                self.assertTrue("observed_cov_numeric" not in last_sample.keys())
                self.assertTrue("observed_cov_categorical" not in last_sample.keys())

    def _compare_tsdataset_and_filled_last_sample(
        self,
        timeseries: TimeSeries,
        last_sample: Dict[str, np.ndarray],
        sample_key: str,
        offset: int,
        last_sample_tail_timestamp: pd.Timestamp,
        std_timeindex: pd.DatetimeIndex,
        fill_last_value: Optional[Union[np.float32, np.int64]],
        sidx: int,
        one_sample_len: int
    ):
        visited_sample_cnt = sidx
        if np.issubdtype(type(fill_last_value), np.integer):
            std_fill_last_value = self._numeric_dtype(fill_last_value)
        elif np.issubdtype(type(fill_last_value), np.floating) :
            std_fill_last_value = self._numeric_dtype(fill_last_value)
        else:
            raise ValueError(f"fill_last_value type {type(fill_last_value)} not valid.")

        extra_timeindex = pd.date_range(
            start=timeseries.time_index[-1],
            end=last_sample_tail_timestamp,
            freq=pd.infer_freq(std_timeindex)
        )
        extra_timeindex = extra_timeindex[1:]
        # sample = (left, right), where left = raw, right = filled.
        # right
        last_sample_len_for_curr_key = len(last_sample[sample_key])
        sample_right_start = (last_sample_len_for_curr_key - 1) - len(extra_timeindex) + 1
        sample_right = last_sample[sample_key][sample_right_start:]
        self.assertTrue(np.alltrue(sample_right == std_fill_last_value))
        # left
        sample_left = last_sample[sample_key][:sample_right_start]
        ndarray_left_start = offset + (visited_sample_cnt * one_sample_len)
        ndarray_left_end = ndarray_left_start + sample_left.shape[0]
        df = timeseries.to_dataframe(copy=False)
        ndarray_left = df.to_numpy(copy=False)[ndarray_left_start:ndarray_left_end]
        # data ok.
        self.assertTrue(np.alltrue(ndarray_left == sample_left))
        # dtype ok.
        self.assertEqual(ndarray_left.dtype, sample_left.dtype)

    def _compare_sample_dataset_and_sample_dataloader(
        self,
        sample_ds: SampleDataset,
        sample_dataloader: Union[paddle.io.DataLoader, MLDataLoader],
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
                    dataloader_ndarray_sample = batch_dict[key][sample_idx]
                    if isinstance(sample_dataloader, paddle.io.DataLoader):
                        dataloader_ndarray_sample = dataloader_ndarray_sample.numpy()
                    dataset_ndarray_sample = dataset_sample[key]
                    self.assertTrue(np.alltrue(dataloader_ndarray_sample == dataset_ndarray_sample))
                    self.assertEqual(dataloader_ndarray_sample.dtype, dataset_ndarray_sample.dtype)
