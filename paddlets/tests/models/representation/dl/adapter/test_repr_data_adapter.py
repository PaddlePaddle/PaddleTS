# !/usr/bin/env python3
# -*- coding:utf-8 -*-


from paddlets.models.representation.dl.adapter import ReprDataAdapter
from paddlets.models.representation.dl.adapter.data_adapter import ReprPaddleDatasetImpl
from paddlets import TSDataset, TimeSeries

import paddle.io
import unittest
import pandas as pd
import numpy as np
from typing import Dict, Any, Set
import datetime


class TestDataAdapter(unittest.TestCase):
    def setUp(self):
        """
        unittest setup
        """
        self._adapter = ReprDataAdapter()
        super().setUp()

    def test_to_paddle_dataset(self):
        """
        Test DataAdapter.to_paddle_dataset()
        """
        # ##############################################
        # case 0 (good case)                           #
        # 1) TSDataset is valid.                       #
        # 2) Default segment_size and sampling_stride. #
        # 3) Default fill_last_value (np.nan)          #
        # ##############################################
        # Note:
        # 1) Given default segment_size = 1 and sampling_stride = 1, and non-empty tsdataset will be just long enough
        # to build samples and no remaining data exiting in the tail of tsdataset. Thus, no matter fill_last_value is
        # set to None or not, no sample needs be filled. For example:
        # Given:
        # tsdataset = [0, 1, 2, 3]
        # (default) segment_size = 1
        # (default) sampling_stride = 1
        # fill_last_value = None
        # Thus, we can build totally 4 samples = [
        #     [0],
        #     [1],
        #     [2],
        #     [3]
        # ]
        # Thus, all data in the tsdataset are used for building samples, no remaining data in the tail, so
        # nothing needs to be filled.
        target_periods = 10
        known_periods = target_periods
        observed_periods = target_periods
        tsdataset = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)

        sample_ds = self._adapter.to_paddle_dataset(tsdataset)
        expect_param = {
            "segment_size": 1,
            "sampling_stride": 1
        }
        self.assertEqual(expect_param["segment_size"], sample_ds._target_segment_size)
        self.assertEqual(expect_param["sampling_stride"], sample_ds._sampling_stride)
        self._compare_tsdataset_and_sample_dataset(
            tsdataset=tsdataset,
            sample_ds=sample_ds,
            param=expect_param
        )

        ####################################################
        # case 1 (good case)                               #
        # 1) TSDataset is valid.                           #
        # 2) Non-Default segment_size and sampling_stride. #
        # 3) Has remaining data in the tail of TSDataset.  #
        # 3) fill_last_value = None                        #
        ####################################################
        # Note:
        # 1) As fill_last_value is None, no matter there are remaining data in the tail of given tsdataset or not,
        # the adapter will NOT fill any samples. For example:
        # Given:
        # tsdataset = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        # segment_size = 3
        # sampling_stride = 3
        # fill_last_value = None
        # Thus, we can build totally 3 samples = [
        #     [0, 1, 2],
        #     [3, 4, 5],
        #     [6, 7, 8]
        # ]
        # The remaining data (i.e. 9) in the tail of tsdataset is skipped, because fill_last_value is None.

        target_periods = 10
        known_periods = target_periods
        observed_periods = target_periods
        tsdataset = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)

        # start build samples.
        param = {"segment_size": 3, "sampling_stride": 3, "fill_last_value": None}
        sample_ds = self._adapter.to_paddle_dataset(tsdataset, **param)
        # Start to assert
        self.assertEqual(param["segment_size"], sample_ds._target_segment_size)
        self.assertEqual(param["sampling_stride"], sample_ds._sampling_stride)
        self._compare_tsdataset_and_sample_dataset(
            tsdataset=tsdataset,
            sample_ds=sample_ds,
            param=param
        )

        ####################################################
        # case 2 (good case)                               #
        # 1) TSDataset is valid.                           #
        # 2) No remaining data in the tail of TSDataset.   #
        # 3) Non-Default segment_size and sampling_stride. #
        # 4) fill_last_value = np.nan                      #
        ####################################################
        # Note:
        # 1) As fill_last_value is np.nan, if there are remaining data in the tail of the given tsdataset, adapter
        # fill the last sample. Otherwise, if no remaining data, even if fill_last_value is NOT None, adapter will
        # still not fill the last sample. For example:
        # Given:
        # tsdataset = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        # segment_size = 3
        # sampling_stride = 3
        # fill_last_value = None
        # Thus, we can build 3 samples = [
        #     [0, 1, 2],
        #     [3, 4, 5],
        #     [6, 7, 8]
        # ]
        # Even fill_last_value is NOT None, still we do NOT need to fill anything, because all data are used for
        # building samples, no remaining data in the tail of the dataset.

        target_periods = 9
        known_periods = target_periods
        observed_periods = target_periods
        tsdataset = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)

        # start build samples.
        param = {"segment_size": 3, "sampling_stride": 3, "fill_last_value": np.nan}
        sample_ds = self._adapter.to_paddle_dataset(tsdataset, **param)
        # Start to assert
        self.assertEqual(param["segment_size"], sample_ds._target_segment_size)
        self.assertEqual(param["sampling_stride"], sample_ds._sampling_stride)
        self._compare_tsdataset_and_sample_dataset(
            tsdataset=tsdataset,
            sample_ds=sample_ds,
            param=param
        )

        ####################################################
        # case 3 (good case)                               #
        # 1) TSDataset is valid.                           #
        # 2) Has remaining data in the tail of TSDataset.  #
        # 3) Non-Default segment_size and sampling_stride. #
        # 4) fill_last_value = np.nan                      #
        ####################################################
        # Note:
        # 1) As fill_last_value is np.nan, and because has remaining data in the tail of tsdataset, adapter will
        # fill the last sample. For example:
        # Given:
        # tsdataset = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        # segment_size = 3
        # sampling_stride = 3
        # fill_last_value = np.nan
        # Thus, we can build 3 samples = [
        #     [0, 1, 2],
        #     [3, 4, 5],
        #     [6, 7, 8],
        #     [9, nan, nan]
        # ]
        # the remaining data in the tail of tsdataset is 9, meanwhile fill_last_value is np.nan, thus adapter
        # will fill the last sample to [9, nan, nan].
        target_periods = 10
        known_periods = target_periods
        observed_periods = target_periods
        tsdataset = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)

        # start build samples.
        param = {"segment_size": 3, "sampling_stride": 3, "fill_last_value": np.nan}
        sample_ds = self._adapter.to_paddle_dataset(tsdataset, **param)
        # Start assert
        self.assertEqual(param["segment_size"], sample_ds._target_segment_size)
        self.assertEqual(param["sampling_stride"], sample_ds._sampling_stride)
        self._compare_tsdataset_and_sample_dataset(
            tsdataset=tsdataset,
            sample_ds=sample_ds,
            param=param
        )

        #########################################################
        # case 4 (good case)                                    #
        # 1) TSDataset is valid.                                #
        # 2) Has remaining data in the tail of TSDataset.       #
        # 3) known start timestamp < target start timestamp.    #
        # 4) observed start timestamp < target start timestamp. #
        # 5) Non-Default segment_size and sampling_stride.      #
        # 6) fill_last_value = np.nan                           #
        # 7) known_cov is not None                              #
        # 8) observed_cov is not None                           #
        # 9) static_cov is not None                             #
        #########################################################
        # Note:
        # This is a typical case to illustrate that the adapter can successfully build samples for tsdataset where
        # target / known cov / observed cov TimeSeries start with different timestamps.
        target_periods = 10000
        known_periods = target_periods + 1000
        observed_periods = target_periods + 500

        freq_int = 15
        freq = f"{freq_int}Min"
        target_start_timestamp = datetime.datetime(day=1, year=2000, month=1, hour=0, minute=0, second=0)
        # known cov starts 1000 * 5 min earlier than target.
        known_start_timestamp = \
            target_start_timestamp - datetime.timedelta(minutes=(known_periods - target_periods) * freq_int)
        # known cov starts 500 * 5 min earlier than target.
        observed_start_timestamp = \
            target_start_timestamp - datetime.timedelta(minutes=(observed_periods - target_periods) * freq_int)
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            target_start_timestamp=pd.Timestamp(target_start_timestamp),
            known_start_timestamp=pd.Timestamp(known_start_timestamp),
            observed_start_timestamp=pd.Timestamp(observed_start_timestamp),
            freq=freq
        )

        # start build samples.
        param = {"segment_size": 3000, "sampling_stride": 3000, "fill_last_value": np.nan}
        sample_ds = self._adapter.to_paddle_dataset(tsdataset, **param)
        # Start assert
        self.assertEqual(param["segment_size"], sample_ds._target_segment_size)
        self.assertEqual(param["sampling_stride"], sample_ds._sampling_stride)
        self._compare_tsdataset_and_sample_dataset(
            tsdataset=tsdataset,
            sample_ds=sample_ds,
            param=param
        )

        #########################################################
        # case 5 (good case)                                    #
        # 1) TSDataset is valid.                                #
        # 2) Has remaining data in the tail of TSDataset.       #
        # 3) known start timestamp < target start timestamp.    #
        # 4) observed start timestamp < target start timestamp. #
        # 5) Non-Default segment_size and sampling_stride.      #
        # 6) fill_last_value = np.nan                           #
        # 7) known_cov is None                                  #
        # 8) observed_cov is None                               #
        # 8) static_cov is None                                 #
        #########################################################
        # Note:
        # This is a typical case to illustrate that the adapter can successfully build samples for tsdataset where
        # known cov / observed cov / static cov are None.
        target_periods = 10000
        freq_int = 15
        freq = f"{freq_int}Min"
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            freq=freq
        )
        tsdataset.known_cov = None
        tsdataset.observed_cov = None
        tsdataset.static_cov = None

        # start build samples.
        param = {"segment_size": 3000, "sampling_stride": 3000, "fill_last_value": np.nan}
        sample_ds = self._adapter.to_paddle_dataset(tsdataset, **param)
        # Start assert
        self.assertEqual(param["segment_size"], sample_ds._target_segment_size)
        self.assertEqual(param["sampling_stride"], sample_ds._sampling_stride)
        self._compare_tsdataset_and_sample_dataset(
            tsdataset=tsdataset,
            sample_ds=sample_ds,
            param=param
        )

        #########################
        # case 6 (bad case)     #
        # 1) segment_size <= 0. #
        #########################
        target_periods = 10
        known_periods = target_periods
        observed_periods = target_periods
        tsdataset = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)

        succeed = True
        try:
            # segment_size = 0.
            sample_ds = self._adapter.to_paddle_dataset(rawdataset=tsdataset, segment_size=0)
        except ValueError:
            succeed = False
        self.assertFalse(succeed)

        succeed = True
        try:
            # segment_size < 0
            sample_ds = self._adapter.to_paddle_dataset(rawdataset=tsdataset, segment_size=-1)
        except ValueError:
            succeed = False
        self.assertFalse(succeed)

        ############################
        # case 7 (bad case)        #
        # 1) sampling_stride <= 0. #
        ############################
        target_periods = 10
        known_periods = target_periods
        observed_periods = target_periods
        tsdataset = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)

        succeed = True
        try:
            # sampling_stride = 0.
            sample_ds = self._adapter.to_paddle_dataset(rawdataset=tsdataset, sampling_stride=0)
        except ValueError:
            succeed = False
        self.assertFalse(succeed)

        succeed = True
        try:
            # sampling_stride < 0
            sample_ds = self._adapter.to_paddle_dataset(rawdataset=tsdataset, sampling_stride=-1)
        except ValueError:
            succeed = False
        self.assertFalse(succeed)

        ################################
        # case 8 (bad case)            #
        # 1) TSDataset.target is None. #
        ################################
        target_periods = 10
        known_periods = target_periods
        observed_periods = target_periods
        tsdataset = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)
        tsdataset.target = None

        succeed = True
        try:
            _ = self._adapter.to_paddle_dataset(rawdataset=tsdataset)
        except ValueError:
            succeed = False
        self.assertFalse(succeed)

        #################################
        # case 9 (bad case)             #
        # 1) target len < segment_size. #
        #################################
        target_periods = 10
        known_periods = target_periods
        observed_periods = target_periods
        tsdataset = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)

        segment_size = target_periods + 1

        succeed = True
        try:
            # target len < segment_size
            sample_ds = self._adapter.to_paddle_dataset(rawdataset=tsdataset, segment_size=segment_size)
        except ValueError:
            succeed = False
        self.assertFalse(succeed)

        ######################################################
        # case 10 (bad case)                                  #
        # 1) known_cov.time_index[0] > target.time_index[0]. #
        ######################################################
        target_periods = 10
        observed_periods = target_periods
        known_periods = target_periods

        today_date = datetime.datetime.now().date()
        target_start_timestamp = pd.Timestamp(today_date)
        # Below is invalid: known_cov.time_index[0] > target.time_index[0]
        known_start_timestamp = pd.Timestamp(today_date + datetime.timedelta(days=1))
        observed_start_timestamp = target_start_timestamp
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            target_start_timestamp=target_start_timestamp,
            known_start_timestamp=known_start_timestamp,
            observed_start_timestamp=observed_start_timestamp,
            freq="1D"
        )

        succeed = True
        try:
            _ = self._adapter.to_paddle_dataset(rawdataset=tsdataset)
        except ValueError:
            succeed = False
        self.assertFalse(succeed)

        ########################################################
        # case 11 (bad case)                                   #
        # 1) known_cov.time_index[-1] < target.time_index[-1]. #
        ########################################################
        target_periods = 10
        observed_periods = target_periods

        # Below is invalid: known_cov.time_index[-1] < target.time_index[-1]
        known_periods = target_periods - 1

        today_date = datetime.datetime.now().date()
        target_start_timestamp = pd.Timestamp(today_date)
        known_start_timestamp = target_start_timestamp
        observed_start_timestamp = target_start_timestamp
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            target_start_timestamp=target_start_timestamp,
            known_start_timestamp=known_start_timestamp,
            observed_start_timestamp=observed_start_timestamp,
            freq="1D"
        )

        succeed = True
        try:
            _ = self._adapter.to_paddle_dataset(rawdataset=tsdataset)
        except ValueError:
            succeed = False
        self.assertFalse(succeed)

        ########################################################
        # case 12 (bad case)                                   #
        # 1) known_cov.time_index[-1] > target.time_index[-1]. #
        ########################################################
        target_periods = 10
        observed_periods = target_periods

        # Below is invalid: known_cov.time_index[-1] > target.time_index[-1]
        known_periods = target_periods + 1

        today_date = datetime.datetime.now().date()
        target_start_timestamp = pd.Timestamp(today_date)
        known_start_timestamp = target_start_timestamp
        observed_start_timestamp = target_start_timestamp
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            target_start_timestamp=target_start_timestamp,
            known_start_timestamp=known_start_timestamp,
            observed_start_timestamp=observed_start_timestamp,
            freq="1D"
        )

        succeed = True
        try:
            _ = self._adapter.to_paddle_dataset(rawdataset=tsdataset)
        except ValueError:
            succeed = False
        self.assertFalse(succeed)

        #####################################################
        # case 13 (bad case)                                #
        # 1) observed.time_index[0] > target.time_index[0]. #
        #####################################################
        target_periods = 10
        observed_periods = target_periods
        known_periods = target_periods

        today_date = datetime.datetime.now().date()
        target_start_timestamp = pd.Timestamp(today_date)
        known_start_timestamp = target_start_timestamp
        # Below is invalid: known_cov.time_index[0] is one day later than target.time_index[0]
        observed_start_timestamp = pd.Timestamp(today_date + datetime.timedelta(days=1))
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            target_start_timestamp=target_start_timestamp,
            known_start_timestamp=known_start_timestamp,
            observed_start_timestamp=observed_start_timestamp,
            freq="1D"
        )

        succeed = True
        try:
            _ = self._adapter.to_paddle_dataset(rawdataset=tsdataset)
        except ValueError:
            succeed = False
        self.assertFalse(succeed)

        ###########################################################
        # case 14 (bad case)                                      #
        # 1) observed_cov.time_index[-1] < target.time_index[-1]. #
        ###########################################################
        target_periods = 10
        # Below is invalid: observed_cov.time_index[-1] < target.time_index[-1]
        observed_periods = target_periods - 1
        known_periods = target_periods - 1

        today_date = datetime.datetime.now().date()
        target_start_timestamp = pd.Timestamp(today_date)
        known_start_timestamp = target_start_timestamp
        observed_start_timestamp = target_start_timestamp
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            target_start_timestamp=target_start_timestamp,
            known_start_timestamp=known_start_timestamp,
            observed_start_timestamp=observed_start_timestamp,
            freq="1D"
        )

        succeed = True
        try:
            _ = self._adapter.to_paddle_dataset(rawdataset=tsdataset)
        except ValueError:
            succeed = False
        self.assertFalse(succeed)

        ###########################################################
        # case 15 (bad case)                                      #
        # 1) observed_cov.time_index[-1] > target.time_index[-1]. #
        ###########################################################
        target_periods = 10
        # Below is invalid: observed_cov.time_index[-1] < target.time_index[-1]
        observed_periods = target_periods - 1
        known_periods = target_periods - 1

        today_date = datetime.datetime.now().date()
        target_start_timestamp = pd.Timestamp(today_date)
        known_start_timestamp = target_start_timestamp
        observed_start_timestamp = target_start_timestamp
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            target_start_timestamp=target_start_timestamp,
            known_start_timestamp=known_start_timestamp,
            observed_start_timestamp=observed_start_timestamp,
            freq="1D"
        )

        succeed = True
        try:
            _ = self._adapter.to_paddle_dataset(rawdataset=tsdataset)
        except ValueError:
            succeed = False
        self.assertFalse(succeed)

        #######################################################################
        # case 16 (bad case)                                                  #
        # 1) known / observed / static contains np.int64 dtype data. #
        #######################################################################
        # 16.1 known cov contains categorical columns.
        target_periods = 10
        known_periods = target_periods
        observed_periods = target_periods
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            known_numeric=True,
            # set known categorical to True to build int64 dtype known cov timeseries to repro this bad case.
            known_categorical=True
        )

        succeed = True
        try:
            # categorical (int64) data is currently NOT supported.
            _ = self._adapter.to_paddle_dataset(rawdataset=tsdataset)
        except ValueError:
            succeed = False
        self.assertFalse(succeed)

        # 16.2 observed cov contains categorical columns.
        target_periods = 10
        known_periods = target_periods
        observed_periods = target_periods
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            observed_numeric=True,
            # set observed categorical to True to build int64 dtype observed cov timeseries to repro this bad case.
            observed_categorical=True
        )

        succeed = True
        try:
            # categorical (int64) data is currently NOT supported.
            _ = self._adapter.to_paddle_dataset(rawdataset=tsdataset)
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
        # 4) Not Fill.                 #
        ################################
        target_periods = 10
        known_periods = target_periods
        observed_periods = target_periods
        tsdataset = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)

        sample_ds = self._adapter.to_paddle_dataset(tsdataset, fill_last_value=None)
        batch_size = 2
        sample_dataloader = self._adapter.to_paddle_dataloader(sample_ds, batch_size, shuffle=False)

        # categorical feature is currently NOT supported for representation adapter.
        good_keys = {
            "past_target",
            "known_cov_numeric",
            "observed_cov_numeric",
            "static_cov_numeric"
        }
        self._compare_sample_dataset_and_sample_dataloader(
            sample_ds=sample_ds,
            sample_dataloader=sample_dataloader,
            batch_size=batch_size,
            good_keys=good_keys,
            param={"segment_size": sample_ds._target_segment_size},
            target_ts=tsdataset.get_target(),
            fill=False
        )

        ################################
        # case 1 (good case)           #
        # 1) known_cov is NOT None.    #
        # 2) observed_cov is NOT None. #
        # 3) static_cov is NOT None.   #
        # 4) Fill.                     #
        ################################
        target_periods = 10
        known_periods = target_periods
        observed_periods = target_periods
        tsdataset = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)

        param = {
            "segment_size": 3,
            "sampling_stride": 3,
            "fill_last_value": np.nan
        }
        sample_ds = self._adapter.to_paddle_dataset(rawdataset=tsdataset, **param)

        batch_size = 2
        sample_dataloader = self._adapter.to_paddle_dataloader(sample_ds, batch_size, shuffle=False)

        # categorical feature is currently NOT supported for representation adapter.
        good_keys = {
            "past_target",
            "known_cov_numeric",
            "observed_cov_numeric",
            "static_cov_numeric"
        }
        self._compare_sample_dataset_and_sample_dataloader(
            sample_ds=sample_ds,
            sample_dataloader=sample_dataloader,
            batch_size=batch_size,
            good_keys=good_keys,
            param=param,
            target_ts=tsdataset.get_target(),
            fill=True
        )

        ################################
        # case 2 (good case)           #
        # 1) known_cov is None.        #
        # 2) observed_cov is NOT None. #
        # 3) static_cov is NOT None.   #
        # 4) Not Fill.                 #
        ################################
        target_periods = 10
        known_periods = target_periods
        observed_periods = target_periods
        tsdataset = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)
        tsdataset.known_cov = None

        sample_ds = self._adapter.to_paddle_dataset(tsdataset, fill_last_value=None)
        batch_size = 2
        sample_dataloader = self._adapter.to_paddle_dataloader(sample_ds, batch_size, shuffle=False)

        # categorical feature is currently NOT supported for representation adapter.
        good_keys = {
            "past_target",
            "observed_cov_numeric",
            "static_cov_numeric"
        }
        self._compare_sample_dataset_and_sample_dataloader(
            sample_ds=sample_ds,
            sample_dataloader=sample_dataloader,
            batch_size=batch_size,
            good_keys=good_keys,
            param={"segment_size": sample_ds._target_segment_size},
            target_ts=tsdataset.get_target(),
            fill=False
        )

        ################################
        # case 3 (good case)           #
        # 1) known_cov is None.        #
        # 2) observed_cov is NOT None. #
        # 3) static_cov is NOT None.   #
        # 4) Fill.                     #
        ################################
        target_periods = 10
        known_periods = target_periods
        observed_periods = target_periods
        tsdataset = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)
        tsdataset.known_cov = None

        param = {
            "segment_size": 3,
            "sampling_stride": 3,
            "fill_last_value": np.nan
        }
        sample_ds = self._adapter.to_paddle_dataset(rawdataset=tsdataset, **param)
        batch_size = 2
        sample_dataloader = self._adapter.to_paddle_dataloader(sample_ds, batch_size, shuffle=False)

        # categorical feature is currently NOT supported for representation adapter.
        good_keys = {
            "past_target",
            "observed_cov_numeric",
            "static_cov_numeric"
        }
        self._compare_sample_dataset_and_sample_dataloader(
            sample_ds=sample_ds,
            sample_dataloader=sample_dataloader,
            batch_size=batch_size,
            good_keys=good_keys,
            param=param,
            target_ts=tsdataset.get_target(),
            fill=True
        )

        ##############################
        # case 4 (good case)         #
        # 1) known_cov is NOT None.  #
        # 2) observed_cov is None.   #
        # 3) static_cov is NOT None. #
        # 4) Not Fill.               #
        ##############################
        target_periods = 10
        known_periods = target_periods
        observed_periods = target_periods
        tsdataset = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)
        tsdataset.observed_cov = None

        sample_ds = self._adapter.to_paddle_dataset(tsdataset, fill_last_value=None)
        batch_size = 2
        sample_dataloader = self._adapter.to_paddle_dataloader(sample_ds, batch_size, shuffle=False)

        # categorical feature is currently NOT supported for representation adapter.
        good_keys = {
            "past_target",
            "known_cov_numeric",
            "static_cov_numeric"
        }
        self._compare_sample_dataset_and_sample_dataloader(
            sample_ds=sample_ds,
            sample_dataloader=sample_dataloader,
            batch_size=batch_size,
            good_keys=good_keys,
            param={"segment_size": sample_ds._target_segment_size},
            target_ts=tsdataset.get_target(),
            fill=False
        )

        ##############################
        # case 5 (good case)         #
        # 1) known_cov is NOT None.  #
        # 2) observed_cov is None.   #
        # 3) static_cov is NOT None. #
        # 4) Fill.                   #
        ##############################
        target_periods = 10
        known_periods = target_periods
        observed_periods = target_periods
        tsdataset = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)
        tsdataset.observed_cov = None

        param = {
            "segment_size": 3,
            "sampling_stride": 3,
            "fill_last_value": np.nan
        }
        sample_ds = self._adapter.to_paddle_dataset(rawdataset=tsdataset, **param)
        batch_size = 2
        sample_dataloader = self._adapter.to_paddle_dataloader(sample_ds, batch_size, shuffle=False)

        # categorical feature is currently NOT supported for representation adapter.
        good_keys = {
            "past_target",
            "known_cov_numeric",
            "static_cov_numeric"
        }
        self._compare_sample_dataset_and_sample_dataloader(
            sample_ds=sample_ds,
            sample_dataloader=sample_dataloader,
            batch_size=batch_size,
            good_keys=good_keys,
            param=param,
            target_ts=tsdataset.get_target(),
            fill=True
        )

        ############################
        # case 6 (good case)       #
        # 1) known_cov is None.    #
        # 2) observed_cov is None. #
        # 3) static_cov is None.   #
        # 4) Not Fill.             #
        ############################
        target_periods = 10
        known_periods = target_periods
        observed_periods = target_periods
        tsdataset = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)
        tsdataset.known_cov = None
        tsdataset.observed_cov = None
        tsdataset.static_cov = None

        sample_ds = self._adapter.to_paddle_dataset(tsdataset, fill_last_value=None)
        batch_size = 2
        sample_dataloader = self._adapter.to_paddle_dataloader(sample_ds, batch_size, shuffle=False)

        # categorical feature is currently NOT supported for representation adapter.
        good_keys = {"past_target"}
        self._compare_sample_dataset_and_sample_dataloader(
            sample_ds=sample_ds,
            sample_dataloader=sample_dataloader,
            batch_size=batch_size,
            good_keys=good_keys,
            param={"segment_size": sample_ds._target_segment_size},
            target_ts=tsdataset.get_target(),
            fill=False
        )

        ############################
        # case 7 (good case)       #
        # 1) known_cov is None.    #
        # 2) observed_cov is None. #
        # 3) static_cov is None.   #
        # 4) Fill.                 #
        ############################
        target_periods = 10
        known_periods = target_periods
        observed_periods = target_periods
        tsdataset = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)
        tsdataset.known_cov = None
        tsdataset.observed_cov = None
        tsdataset.static_cov = None

        param = {
            "segment_size": 3,
            "sampling_stride": 3,
            "fill_last_value": np.nan
        }
        sample_ds = self._adapter.to_paddle_dataset(rawdataset=tsdataset, **param)
        batch_size = 2
        sample_dataloader = self._adapter.to_paddle_dataloader(sample_ds, batch_size, shuffle=False)

        # categorical feature is currently NOT supported for representation adapter.
        good_keys = {"past_target"}
        self._compare_sample_dataset_and_sample_dataloader(
            sample_ds=sample_ds,
            sample_dataloader=sample_dataloader,
            batch_size=batch_size,
            good_keys=good_keys,
            param=param,
            target_ts=tsdataset.get_target(),
            fill=True
        )

    def _build_mock_ts_dataset(
            self,
            target_periods: int = 10,
            known_periods: int = 10,
            observed_periods: int = 10,
            target_start_timestamp: pd.Timestamp = pd.Timestamp(datetime.datetime.now().date()),
            known_start_timestamp: pd.Timestamp = pd.Timestamp(datetime.datetime.now().date()),
            observed_start_timestamp: pd.Timestamp = pd.Timestamp(datetime.datetime.now().date()),
            known_numeric: bool = True,
            observed_numeric: bool = True,
            static_numeric: bool = True,
            known_categorical: bool = False,
            observed_categorical: bool = False,
            static_categorical: bool = False,
            freq: str = "1D"
    ):
        """
        Build mock bts dataset.

        all timeseries must have same freq.
        """
        numeric_dtype = np.float32
        categorical_dtype = np.int64

        # target
        target_df = pd.DataFrame(
            data=np.array([i for i in range(target_periods)], dtype=numeric_dtype),
            index=pd.date_range(start=target_start_timestamp, periods=target_periods, freq=freq),
            columns=["target_numeric_0"]
        )

        # known
        known_raw_data = [(i * 10, i * 100) for i in range(known_periods)]
        known_numeric_df = None
        if known_numeric:
            known_numeric_data = np.array(known_raw_data, dtype=numeric_dtype)
            known_numeric_df = pd.DataFrame(
                data=known_numeric_data,
                index=pd.date_range(start=known_start_timestamp, periods=known_periods, freq=freq),
                columns=["known_numeric_0", "known_numeric_1"]
            )

        known_categorical_df = None
        if known_categorical:
            known_categorical_data = np.array(known_raw_data, dtype=categorical_dtype)
            known_categorical_df = pd.DataFrame(
                data=known_categorical_data,
                index=pd.date_range(start=known_start_timestamp, periods=known_periods, freq=freq),
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
        if observed_numeric:
            observed_numeric_data = np.array(observed_raw_data, dtype=numeric_dtype)
            observed_numeric_df = pd.DataFrame(
                data=observed_numeric_data,
                index=pd.date_range(start=observed_start_timestamp, periods=observed_periods, freq=freq),
                columns=["observed_numeric_0", "observed_numeric_1"]
            )

        observed_categorical_df = None
        if observed_categorical:
            observed_categorical_data = np.array(observed_raw_data, dtype=categorical_dtype)
            observed_categorical_df = pd.DataFrame(
                data=observed_categorical_data,
                index=pd.date_range(start=observed_start_timestamp, periods=observed_periods, freq=freq),
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
        if static_numeric:
            static["static_numeric"] = np.float32(1)
        if static_categorical:
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
            sample_ds: ReprPaddleDatasetImpl,
            param: Dict[str, Any]
    ) -> None:
        """
        Given a TSDataset and a built paddle.io.Dataset, compare if these data are matched.

        Args:
            tsdataset(TSDataset): Raw TSDataset.
            sample_ds(PaddleDatasetImpl): Built paddle.io.Dataset.
            param(Dict): param for building samples.
        """
        numeric_dtype = np.float32
        categorical_dtype = np.int64

        segment_size = param["segment_size"]
        sampling_stride = param["sampling_stride"]
        target_ts = tsdataset.get_target()
        known_ts = tsdataset.get_known_cov()
        observed_ts = tsdataset.get_observed_cov()
        static_cov = tsdataset.get_static_cov()

        last_sample_idx = len(sample_ds.samples) - 1
        sidx = 0
        first_sample_tail_idx = segment_size - 1

        # As target/known cov/observed cov might start with different timestamp, thus needs to compute offset.
        target_start_timestamp = target_ts.time_index[0]
        known_offset = 0 if known_ts is None else known_ts.time_index.get_loc(target_start_timestamp)
        observed_offset = 0 if observed_ts is None else observed_ts.time_index.get_loc(target_start_timestamp)

        # Start compare.
        while sidx < last_sample_idx:
            curr_sample = sample_ds[sidx]
            tail_idx = first_sample_tail_idx + sidx * sampling_stride
            ###############
            # past_target #
            ###############
            target_df = target_ts.to_dataframe(copy=False)
            past_target_ndarray = target_df.to_numpy(copy=False)[tail_idx - segment_size + 1:tail_idx + 1]
            # data ok.
            self.assertTrue(np.alltrue(past_target_ndarray == curr_sample["past_target"]))
            # dtype ok.
            self.assertEqual(past_target_ndarray.dtype, curr_sample["past_target"].dtype)

            #############
            # known_cov #
            #############
            if known_ts is not None:
                known_df = known_ts.to_dataframe(copy=False)
                known_start = known_offset + tail_idx - segment_size + 1
                known_end = known_offset + tail_idx + 1
                # numeric
                if "known_cov_numeric" in curr_sample.keys():
                    numeric_df = known_df.select_dtypes(include=numeric_dtype)
                    numeric_ndarray = numeric_df.to_numpy(copy=False)
                    known_numeric_ndarray = numeric_ndarray[known_start:known_end]
                    # data ok.
                    self.assertTrue(np.alltrue(known_numeric_ndarray == curr_sample["known_cov_numeric"]))
                    # dtype ok.
                    self.assertEqual(known_numeric_ndarray.dtype, curr_sample["known_cov_numeric"].dtype)

                # categorical (currently not supported)
                self.assertTrue("known_cov_categorical" not in curr_sample.keys())
            # known_cov is None.
            else:
                self.assertTrue("known_cov_numeric" not in curr_sample.keys())
                self.assertTrue("known_cov_categorical" not in curr_sample.keys())

            # observed_cov (possibly be None)
            if tsdataset.get_observed_cov() is not None:
                observed_df = observed_ts.to_dataframe(copy=False)
                observed_start = observed_offset + tail_idx - segment_size + 1
                observed_end = observed_offset + tail_idx + 1
                # numeric
                if "observed_cov_numeric" in curr_sample.keys():
                    numeric_df = observed_df.select_dtypes(include=numeric_dtype)
                    numeric_ndarray = numeric_df.to_numpy(copy=False)
                    observed_numeric_ndarray = numeric_ndarray[observed_start:observed_end]
                    # data ok.
                    self.assertTrue(np.alltrue(observed_numeric_ndarray == curr_sample["observed_cov_numeric"]))
                    # dtype ok.
                    self.assertEqual(observed_numeric_ndarray.dtype, curr_sample["observed_cov_numeric"].dtype)
            else:
                self.assertTrue("observed_cov_categorical" not in curr_sample.keys())

            ################
            # observed_cov #
            ################
            if observed_ts is not None:
                observed_df = observed_ts.to_dataframe(copy=False)
                observed_start = observed_offset + tail_idx - segment_size + 1
                observed_end = observed_offset + tail_idx + 1
                # numeric
                if "observed_cov_numeric" in curr_sample.keys():
                    numeric_df = observed_df.select_dtypes(include=numeric_dtype)
                    numeric_ndarray = numeric_df.to_numpy(copy=False)
                    observed_numeric_ndarray = numeric_ndarray[observed_start:observed_end]
                    # data ok.
                    self.assertTrue(np.alltrue(observed_numeric_ndarray == curr_sample["observed_cov_numeric"]))
                    # dtype ok.
                    self.assertEqual(observed_numeric_ndarray.dtype, curr_sample["observed_cov_numeric"].dtype)
                # categorical (currently not supported)
                self.assertTrue("observed_cov_categorical" not in curr_sample.keys())
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
                # categorical (currently not supported)
                self.assertTrue("static_cov_categorical" not in curr_sample.keys())
            # static_cov is None
            else:
                self.assertTrue("static_cov_numeric" not in curr_sample.keys())
                self.assertTrue("static_cov_categorical" not in curr_sample.keys())

            sidx += 1

        # last sample, possibly be filled.
        last_sample = sample_ds.samples[sidx]
        last_sample_tail_timestamp = sample_ds._compute_last_sample_tail_timestamp()
        if last_sample_tail_timestamp > target_ts.time_index[-1]:
            # last sample is filled.
            extra_timeindex = pd.date_range(
                start=target_ts.time_index[-1],
                end=last_sample_tail_timestamp,
                freq=pd.infer_freq(target_ts.time_index)
            )
            extra_timeindex = extra_timeindex[1:]

            ###############
            # past_target #
            ###############
            target_df = target_ts.to_dataframe(copy=False)
            # past target sample = (left, right), where left = raw data, right = filled all-NaN data.
            # right
            sample_right = last_sample["past_target"][-1 - len(extra_timeindex) + 1:]
            self.assertTrue(np.alltrue(np.isnan(sample_right)))
            # left
            start = (len(target_ts) - 1) - (segment_size - len(extra_timeindex)) + 1
            end = (len(target_ts) - 1) + 1
            past_target_numeric_ndarray_left = target_df.to_numpy(copy=False)[start:end]
            sample_left = last_sample["past_target"][:segment_size - len(extra_timeindex)]
            # data ok.
            self.assertTrue(np.alltrue(past_target_numeric_ndarray_left == sample_left))
            # dtype ok.
            self.assertEqual(past_target_numeric_ndarray_left.dtype, sample_left.dtype)

            #############
            # known_cov #
            #############
            if known_ts is not None:
                known_df = known_ts.to_dataframe(copy=False)
                # numeric
                if "known_cov_numeric" in last_sample.keys():
                    # right (filled part)
                    sample_right = last_sample["known_cov_numeric"][-1 - len(extra_timeindex) + 1:]
                    self.assertTrue(np.alltrue(np.isnan(sample_right)))
                    # left
                    start = len(known_ts) - 1 - (segment_size - len(extra_timeindex)) + 1
                    end = (len(known_ts) - 1) + 1
                    numeric_df = known_df.select_dtypes(include=numeric_dtype)
                    numeric_ndarray = numeric_df.to_numpy(copy=False)
                    known_numeric_ndarray_left = numeric_ndarray[start:end]
                    sample_left = last_sample["known_cov_numeric"][:segment_size - len(extra_timeindex)]
                    # data ok.
                    self.assertTrue(np.alltrue(known_numeric_ndarray_left == sample_left))
                    # dtype ok.
                    self.assertEqual(known_numeric_ndarray_left.dtype, sample_left.dtype)

                # categorical (currently not supported.)
                self.assertTrue("known_cov_categorical" not in last_sample.keys())
            # known_cov is None.
            else:
                self.assertTrue("known_cov_numeric" not in last_sample.keys())
                self.assertTrue("known_cov_categorical" not in last_sample.keys())

            ################
            # observed_cov #
            ################
            if observed_ts is not None:
                observed_df = observed_ts.to_dataframe(copy=False)
                # numeric
                if "observed_cov_numeric" in last_sample.keys():
                    # right (filled)
                    sample_right = last_sample["observed_cov_numeric"][-1 - len(extra_timeindex) + 1:]
                    self.assertTrue(np.alltrue(np.isnan(sample_right)))

                    # left
                    start = len(observed_ts) - 1 - (segment_size - len(extra_timeindex)) + 1
                    end = (len(observed_ts) - 1) + 1
                    numeric_df = observed_df.select_dtypes(include=numeric_dtype)
                    numeric_ndarray = numeric_df.to_numpy(copy=False)
                    observed_numeric_ndarray_left = numeric_ndarray[start:end]
                    sample_left = last_sample["observed_cov_numeric"][:segment_size - len(extra_timeindex)]
                    # data ok.
                    self.assertTrue(np.alltrue(observed_numeric_ndarray_left == sample_left))
                    # dtype ok.
                    self.assertEqual(observed_numeric_ndarray_left.dtype, sample_left.dtype)
                # categorical (currently not supported.)
                self.assertTrue("observed_cov_categorical" not in last_sample.keys())

            ##############
            # static_cov #
            ##############
            if static_cov is not None:
                # unsorted dict -> sorted list
                sorted_static_cov = sorted(static_cov.items(), key=lambda t: t[0])
                # numeric
                if "static_cov_numeric" in last_sample.keys():
                    sorted_static_cov_numeric = \
                        [t[1] for t in sorted_static_cov if isinstance(t[1], numeric_dtype)]
                    # data ok.
                    self.assertTrue(np.alltrue(sorted_static_cov_numeric == last_sample["static_cov_numeric"][0]))
                    # dtype ok.
                    self.assertEqual(sorted_static_cov_numeric[0].dtype, last_sample["static_cov_numeric"][0].dtype)
                # categorical
                if "static_cov_categorical" in last_sample.keys():
                    sorted_static_cov_categorical = \
                        [t[1] for t in sorted_static_cov if isinstance(t[1], categorical_dtype)]
                    # data ok.
                    self.assertTrue(np.alltrue(
                        sorted_static_cov_categorical == last_sample["static_cov_categorical"][0])
                    )
                    # dtype ok.
                    self.assertEqual(
                        sorted_static_cov_categorical[0].dtype, last_sample["static_cov_categorical"][0].dtype
                    )
            # static_cov is None
            else:
                self.assertTrue("static_cov_numeric" not in last_sample.keys())
                self.assertTrue("static_cov_categorical" not in last_sample.keys())

    def _compare_sample_dataset_and_sample_dataloader(
            self,
            sample_ds: ReprPaddleDatasetImpl,
            sample_dataloader: paddle.io.DataLoader,
            batch_size: int,
            good_keys: Set[str],
            param: Dict[str, Any],
            target_ts: TimeSeries,
            fill: bool = False
    ):
        """Check if sample dataset matches batched sample dataloader."""
        segment_size = param["segment_size"]
        target_timeindex = target_ts.time_index

        # categorical feature is currently NOT supported for representation adapter.
        all_keys = {
            "past_target",
            "known_cov_numeric",
            "observed_cov_numeric",
            "static_cov_numeric"
        }
        none_keys = all_keys - good_keys

        sample_cnt = len(sample_ds.samples)
        checked_sample_cnt = 0
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
                    # dtype ok.
                    self.assertEqual(dataloader_ndarray_sample.dtype, dataset_ndarray_sample.dtype)

                    # NOT Fill.
                    if not fill:
                        # data ok.
                        self.assertTrue(np.alltrue(dataloader_ndarray_sample == dataset_ndarray_sample))
                        continue

                    # Fill BUT Not last sample.
                    if checked_sample_cnt < sample_cnt - 1:
                        # data ok.
                        self.assertTrue(np.alltrue(dataloader_ndarray_sample == dataset_ndarray_sample))
                        continue

                    # Fill AND Last sample.
                    # For static cov, as its shape[1] is always 1, so static cov is always NOT filled.
                    # As categorical feature is currently NOT supported, so no need to check static_cov_categorical key.
                    if key == "static_cov_numeric":
                        # data ok.
                        self.assertTrue(np.alltrue(dataloader_ndarray_sample == dataset_ndarray_sample))
                        continue

                    # For target / known cov / observed cov, last_sample = (left, right), where:
                    # left = raw data, use np.alltrue(xxx == left) to compare.
                    # right = np.nan filled data, use np.alltrue(np.isnan(right)) to compare.
                    last_sample_tail_timestamp = sample_ds._compute_last_sample_tail_timestamp()
                    extra_timeindex = pd.date_range(
                        start=target_timeindex[-1],
                        end=last_sample_tail_timestamp,
                        freq=pd.infer_freq(target_timeindex)
                    )
                    extra_timeindex = extra_timeindex[1:]

                    dataloader_right = dataloader_ndarray_sample[-1 - len(extra_timeindex) + 1:]
                    dataset_right = dataset_ndarray_sample[-1 - len(extra_timeindex) + 1:]
                    self.assertTrue(np.alltrue(np.isnan(dataloader_right)))
                    self.assertTrue(np.alltrue(np.isnan(dataset_right)))

                    dataloader_left = dataloader_ndarray_sample[:segment_size - len(extra_timeindex)]
                    dataset_left = dataset_ndarray_sample[:segment_size - len(extra_timeindex)]
                    # data ok.
                    self.assertTrue(np.alltrue(dataloader_left == dataset_left))
                    # dtype ok.
                    self.assertEqual(dataloader_left.dtype, dataset_left.dtype)

                checked_sample_cnt += 1
