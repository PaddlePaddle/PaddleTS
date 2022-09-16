# !/usr/bin/env python3
# -*- coding:utf-8 -*-


from paddlets.models.representation.dl.adapter import ReprDataAdapter
from paddlets.models.representation.dl.adapter.data_adapter import ReprPaddleDatasetImpl
from paddlets import TSDataset, TimeSeries

import unittest
import pandas as pd
import numpy as np
from typing import Dict, Any
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
        # 1) Given default segment_size = 1 and sampling_stride = 1, ant non-empty tsdataset will be just long enough
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
        paddlets_ds = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)

        # Invoke the convert method with default params.
        paddle_ds = self._adapter.to_paddle_dataset(paddlets_ds)
        # Start to assert
        expect_param = {
            "segment_size": 1,
            "sampling_stride": 1
        }
        self.assertEqual(expect_param["segment_size"], paddle_ds._target_segment_size)
        self.assertEqual(expect_param["sampling_stride"], paddle_ds._sampling_stride)
        self._compare_if_paddlets_sample_match_paddle_sample(
            paddlets_ds=paddlets_ds,
            paddle_ds=paddle_ds,
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
        paddlets_ds = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)

        # start build samples.
        param = {"segment_size": 3, "sampling_stride": 3, "fill_last_value": None}
        paddle_ds = self._adapter.to_paddle_dataset(paddlets_ds, **param)
        # Start to assert
        self.assertEqual(param["segment_size"], paddle_ds._target_segment_size)
        self.assertEqual(param["sampling_stride"], paddle_ds._sampling_stride)
        self._compare_if_paddlets_sample_match_paddle_sample(
            paddlets_ds=paddlets_ds,
            paddle_ds=paddle_ds,
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
        paddlets_ds = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)

        # start build samples.
        param = {"segment_size": 3, "sampling_stride": 3, "fill_last_value": np.nan}
        paddle_ds = self._adapter.to_paddle_dataset(paddlets_ds, **param)
        # Start to assert
        self.assertEqual(param["segment_size"], paddle_ds._target_segment_size)
        self.assertEqual(param["sampling_stride"], paddle_ds._sampling_stride)
        self._compare_if_paddlets_sample_match_paddle_sample(
            paddlets_ds=paddlets_ds,
            paddle_ds=paddle_ds,
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
        paddlets_ds = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)

        # start build samples.
        param = {"segment_size": 3, "sampling_stride": 3, "fill_last_value": np.nan}
        paddle_ds = self._adapter.to_paddle_dataset(paddlets_ds, **param)
        # Start assert
        self.assertEqual(param["segment_size"], paddle_ds._target_segment_size)
        self.assertEqual(param["sampling_stride"], paddle_ds._sampling_stride)
        self._compare_if_paddlets_sample_match_paddle_sample(
            paddlets_ds=paddlets_ds,
            paddle_ds=paddle_ds,
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
        paddlets_ds = self._build_mock_ts_dataset(
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
        paddle_ds = self._adapter.to_paddle_dataset(paddlets_ds, **param)
        # Start assert
        self.assertEqual(param["segment_size"], paddle_ds._target_segment_size)
        self.assertEqual(param["sampling_stride"], paddle_ds._sampling_stride)
        self._compare_if_paddlets_sample_match_paddle_sample(
            paddlets_ds=paddlets_ds,
            paddle_ds=paddle_ds,
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
        #########################################################
        # Note:
        # This is a typical case to illustrate that the adapter can successfully build samples for tsdataset where
        # known cov / observed cov TimeSeries are None.
        # More specifically, the following four scenarios are all valid to build samples:
        # -------------------------------
        # | known_cov  |  observed_cov  |
        # | not None   |  not None      |
        # | not None   |  None          |
        # | None       |  not None      |
        # | None       |  None          |
        # -------------------------------
        target_periods = 10000
        freq_int = 15
        freq = f"{freq_int}Min"
        paddlets_ds = self._build_mock_ts_dataset(
            target_periods=target_periods,
            freq=freq
        )
        paddlets_ds.known_cov = None
        paddlets_ds.observed_cov = None

        # start build samples.
        param = {"segment_size": 3000, "sampling_stride": 3000, "fill_last_value": np.nan}
        paddle_ds = self._adapter.to_paddle_dataset(paddlets_ds, **param)
        # Start assert
        self.assertEqual(param["segment_size"], paddle_ds._target_segment_size)
        self.assertEqual(param["sampling_stride"], paddle_ds._sampling_stride)
        self._compare_if_paddlets_sample_match_paddle_sample(
            paddlets_ds=paddlets_ds,
            paddle_ds=paddle_ds,
            param=param
        )

        #########################
        # case 6 (bad case)     #
        # 1) segment_size <= 0. #
        #########################
        target_periods = 10
        known_periods = target_periods
        observed_periods = target_periods
        paddlets_ds = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)

        succeed = True
        try:
            # segment_size = 0.
            paddle_ds = self._adapter.to_paddle_dataset(rawdataset=paddlets_ds, segment_size=0)
        except Exception as e:
            succeed = False
        self.assertFalse(succeed)

        succeed = True
        try:
            # segment_size < 0
            paddle_ds = self._adapter.to_paddle_dataset(rawdataset=paddlets_ds, segment_size=-1)
        except Exception as e:
            succeed = False
        self.assertFalse(succeed)

        ############################
        # case 7 (bad case)        #
        # 1) sampling_stride <= 0. #
        ############################
        target_periods = 10
        known_periods = target_periods
        observed_periods = target_periods
        paddlets_ds = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)

        succeed = True
        try:
            # sampling_stride = 0.
            paddle_ds = self._adapter.to_paddle_dataset(rawdataset=paddlets_ds, sampling_stride=0)
        except Exception as e:
            succeed = False
        self.assertFalse(succeed)

        succeed = True
        try:
            # sampling_stride < 0
            paddle_ds = self._adapter.to_paddle_dataset(rawdataset=paddlets_ds, sampling_stride=-1)
        except Exception as e:
            succeed = False
        self.assertFalse(succeed)

        ################################
        # case 8 (bad case)            #
        # 1) TSDataset.target is None. #
        ################################
        target_periods = 10
        known_periods = target_periods
        observed_periods = target_periods
        paddlets_ds = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)
        paddlets_ds.target = None

        succeed = True
        try:
            paddle_ds = self._adapter.to_paddle_dataset(rawdataset=paddlets_ds)
        except Exception as e:
            succeed = False
        self.assertFalse(succeed)

        #################################
        # case 9 (bad case)             #
        # 1) target len < segment_size. #
        #################################
        target_periods = 10
        known_periods = target_periods
        observed_periods = target_periods
        paddlets_ds = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)

        segment_size = target_periods + 1

        succeed = True
        try:
            # target len < segment_size
            paddle_ds = self._adapter.to_paddle_dataset(rawdataset=paddlets_ds, segment_size=segment_size)
        except Exception as e:
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
        paddlets_ds = self._build_mock_ts_dataset(
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
            paddle_ds = self._adapter.to_paddle_dataset(rawdataset=paddlets_ds)
        except Exception as e:
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
        paddlets_ds = self._build_mock_ts_dataset(
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
            paddle_ds = self._adapter.to_paddle_dataset(rawdataset=paddlets_ds)
        except Exception as e:
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
        paddlets_ds = self._build_mock_ts_dataset(
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
            paddle_ds = self._adapter.to_paddle_dataset(rawdataset=paddlets_ds)
        except Exception as e:
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
        paddlets_ds = self._build_mock_ts_dataset(
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
            paddle_ds = self._adapter.to_paddle_dataset(rawdataset=paddlets_ds)
        except Exception as e:
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
        paddlets_ds = self._build_mock_ts_dataset(
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
            paddle_ds = self._adapter.to_paddle_dataset(rawdataset=paddlets_ds)
        except Exception as e:
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
        paddlets_ds = self._build_mock_ts_dataset(
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
            paddle_ds = self._adapter.to_paddle_dataset(rawdataset=paddlets_ds)
        except Exception as e:
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
        # 3) Not Fill.                 #
        ################################
        target_periods = 10
        known_periods = target_periods
        observed_periods = target_periods
        paddlets_ds = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)

        paddle_ds = self._adapter.to_paddle_dataset(paddlets_ds, fill_last_value=None)
        batch_size = 2
        paddle_dataloader = self._adapter.to_paddle_dataloader(paddle_ds, batch_size, shuffle=False)

        all_keys = {"past_target", "known_cov", "observed_cov"}
        for batch_idx, batch_dict in enumerate(paddle_dataloader):
            curr_batch_size = list(batch_dict.values())[0].shape[0]
            for sample_idx in range(curr_batch_size):
                dataset_sample = paddle_ds[batch_idx * batch_size + sample_idx]
                for key in all_keys:
                    dataloader_ndarray_sample = batch_dict[key][sample_idx].numpy()
                    dataset_ndarray_sample = dataset_sample[key]
                    self.assertTrue(np.alltrue(dataloader_ndarray_sample == dataset_ndarray_sample))

        ################################
        # case 1 (good case)           #
        # 1) known_cov is NOT None.    #
        # 2) observed_cov is NOT None. #
        # 3) Fill.                     #
        ################################
        target_periods = 10
        known_periods = target_periods
        observed_periods = target_periods
        paddlets_ds = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)

        segment_size = 3
        paddle_ds = self._adapter.to_paddle_dataset(
            rawdataset=paddlets_ds,
            segment_size=segment_size,
            sampling_stride=3,
            fill_last_value=np.nan
        )
        sample_cnt = len(paddle_ds.samples)

        batch_size = 2
        paddle_dataloader = self._adapter.to_paddle_dataloader(paddle_ds, batch_size, shuffle=False)

        all_keys = {"past_target", "known_cov", "observed_cov"}
        checked_sample_cnt = 0
        for batch_idx, batch_dict in enumerate(paddle_dataloader):
            curr_batch_size = list(batch_dict.values())[0].shape[0]
            for sample_idx in range(curr_batch_size):
                dataset_sample = paddle_ds[batch_idx * batch_size + sample_idx]
                for key in all_keys:
                    dataloader_ndarray_sample = batch_dict[key][sample_idx].numpy()
                    dataset_ndarray_sample = dataset_sample[key]
                    if checked_sample_cnt < sample_cnt - 1:
                        # not the last sample
                        self.assertTrue(np.alltrue(dataloader_ndarray_sample == dataset_ndarray_sample))
                        continue

                    # Last sample.
                    # last_sample = (left, right), where:
                    # left = raw data, use np.alltrue(xxx == left) to compare.
                    # right = np.nan filled data, use np.alltrue(np.isnan(right)) to compare.
                    last_sample_tail_timestamp = paddle_ds._compute_last_sample_tail_timestamp()
                    extra_timeindex = pd.date_range(
                        start=paddlets_ds.get_target().time_index[-1],
                        end=last_sample_tail_timestamp,
                        freq=pd.infer_freq(paddlets_ds.get_target().time_index)
                    )
                    extra_timeindex = extra_timeindex[1:]

                    dataloader_right = dataloader_ndarray_sample[-1 - len(extra_timeindex) + 1:]
                    dataset_right = dataset_ndarray_sample[-1 - len(extra_timeindex) + 1:]
                    self.assertTrue(np.alltrue(np.isnan(dataloader_right)))
                    self.assertTrue(np.alltrue(np.isnan(dataset_right)))

                    dataloader_left = dataloader_ndarray_sample[:segment_size - len(extra_timeindex)]
                    dataset_left = dataset_ndarray_sample[:segment_size - len(extra_timeindex)]
                    self.assertTrue(np.alltrue(dataloader_left == dataset_left))
                checked_sample_cnt += 1

        ################################
        # case 2 (good case)           #
        # 1) known_cov is None.        #
        # 2) observed_cov is NOT None. #
        # 3) Not Fill.                 #
        ################################
        target_periods = 10
        known_periods = target_periods
        observed_periods = target_periods
        paddlets_ds = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)
        paddlets_ds.known_cov = None

        paddle_ds = self._adapter.to_paddle_dataset(paddlets_ds, fill_last_value=None)
        batch_size = 2
        paddle_dataloader = self._adapter.to_paddle_dataloader(paddle_ds, batch_size, shuffle=False)

        all_keys = {"past_target", "known_cov", "observed_cov"}
        good_keys = {"past_target", "observed_cov"}
        none_keys = all_keys - good_keys
        for batch_idx, batch_dict in enumerate(paddle_dataloader):
            curr_batch_size = list(batch_dict.values())[0].shape[0]
            for sample_idx in range(curr_batch_size):
                dataset_sample = paddle_ds[batch_idx * batch_size + sample_idx]
                for key in all_keys:
                    if key in none_keys:
                        self.assertTrue(key not in dataset_sample.keys())
                        continue

                    # good keys
                    dataloader_ndarray_sample = batch_dict[key][sample_idx].numpy()
                    dataset_ndarray_sample = dataset_sample[key]
                    self.assertTrue(np.alltrue(dataloader_ndarray_sample == dataset_ndarray_sample))

        ################################
        # case 3 (good case)           #
        # 1) known_cov is None.        #
        # 2) observed_cov is NOT None. #
        # 3) Fill.                     #
        ################################
        target_periods = 10
        known_periods = target_periods
        observed_periods = target_periods
        paddlets_ds = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)
        paddlets_ds.known_cov = None

        segment_size = 3
        paddle_ds = self._adapter.to_paddle_dataset(
            rawdataset=paddlets_ds,
            segment_size=segment_size,
            sampling_stride=3,
            fill_last_value=np.nan
        )
        sample_cnt = len(paddle_ds.samples)

        batch_size = 2
        paddle_dataloader = self._adapter.to_paddle_dataloader(paddle_ds, batch_size, shuffle=False)

        all_keys = {"past_target", "known_cov", "observed_cov"}
        good_keys = {"past_target", "observed_cov"}
        none_keys = all_keys - good_keys
        checked_sample_cnt = 0
        for batch_idx, batch_dict in enumerate(paddle_dataloader):
            curr_batch_size = list(batch_dict.values())[0].shape[0]
            for sample_idx in range(curr_batch_size):
                dataset_sample = paddle_ds[batch_idx * batch_size + sample_idx]
                for key in all_keys:
                    if key in none_keys:
                        self.assertTrue(key not in dataset_sample.keys())
                        continue

                    # good keys
                    dataloader_ndarray_sample = batch_dict[key][sample_idx].numpy()
                    dataset_ndarray_sample = dataset_sample[key]

                    # Not last sample
                    if checked_sample_cnt < sample_cnt - 1:
                        self.assertTrue(np.alltrue(dataloader_ndarray_sample == dataset_ndarray_sample))
                        continue

                    # Last sample
                    # last_sample = (left, right), where:
                    # left = raw data, use np.alltrue(xxx == left) to compare.
                    # right = np.nan filled data, use np.alltrue(np.isnan(right)) to compare.
                    last_sample_tail_timestamp = paddle_ds._compute_last_sample_tail_timestamp()
                    extra_timeindex = pd.date_range(
                        start=paddlets_ds.get_target().time_index[-1],
                        end=last_sample_tail_timestamp,
                        freq=pd.infer_freq(paddlets_ds.get_target().time_index)
                    )
                    extra_timeindex = extra_timeindex[1:]

                    dataloader_right = dataloader_ndarray_sample[-1 - len(extra_timeindex) + 1:]
                    dataset_right = dataset_ndarray_sample[-1 - len(extra_timeindex) + 1:]
                    self.assertTrue(np.alltrue(np.isnan(dataloader_right)))
                    self.assertTrue(np.alltrue(np.isnan(dataset_right)))

                    dataloader_left = dataloader_ndarray_sample[:segment_size - len(extra_timeindex)]
                    dataset_left = dataset_ndarray_sample[:segment_size - len(extra_timeindex)]
                    self.assertTrue(np.alltrue(dataloader_left == dataset_left))

                checked_sample_cnt += 1

        #############################
        # case 4 (good case)        #
        # 1) known_cov is NOT None. #
        # 2) observed_cov is None.  #
        # 3) Not Fill.              #
        #############################
        target_periods = 10
        known_periods = target_periods
        observed_periods = target_periods
        paddlets_ds = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)
        paddlets_ds.observed_cov = None

        paddle_ds = self._adapter.to_paddle_dataset(paddlets_ds, fill_last_value=None)
        batch_size = 2
        paddle_dataloader = self._adapter.to_paddle_dataloader(paddle_ds, batch_size, shuffle=False)

        all_keys = {"past_target", "known_cov", "observed_cov"}
        good_keys = {"past_target", "known_cov"}
        none_keys = all_keys - good_keys
        for batch_idx, batch_dict in enumerate(paddle_dataloader):
            curr_batch_size = list(batch_dict.values())[0].shape[0]
            for sample_idx in range(curr_batch_size):
                dataset_sample = paddle_ds[batch_idx * batch_size + sample_idx]
                for key in all_keys:
                    if key in none_keys:
                        self.assertTrue(key not in dataset_sample.keys())
                        continue

                    # good keys
                    dataloader_ndarray_sample = batch_dict[key][sample_idx].numpy()
                    dataset_ndarray_sample = dataset_sample[key]
                    self.assertTrue(np.alltrue(dataloader_ndarray_sample == dataset_ndarray_sample))

        #############################
        # case 5 (good case)        #
        # 1) known_cov is NOT None. #
        # 2) observed_cov is None.  #
        # 3) Fill.                  #
        #############################
        target_periods = 10
        known_periods = target_periods
        observed_periods = target_periods
        paddlets_ds = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)
        paddlets_ds.observed_cov = None

        segment_size = 3
        paddle_ds = self._adapter.to_paddle_dataset(
            rawdataset=paddlets_ds,
            segment_size=segment_size,
            sampling_stride=3,
            fill_last_value=np.nan
        )
        sample_cnt = len(paddle_ds.samples)

        batch_size = 2
        paddle_dataloader = self._adapter.to_paddle_dataloader(paddle_ds, batch_size, shuffle=False)

        all_keys = {"past_target", "known_cov", "observed_cov"}
        good_keys = {"past_target", "known_cov"}
        none_keys = all_keys - good_keys
        checked_sample_cnt = 0
        for batch_idx, batch_dict in enumerate(paddle_dataloader):
            curr_batch_size = list(batch_dict.values())[0].shape[0]
            for sample_idx in range(curr_batch_size):
                dataset_sample = paddle_ds[batch_idx * batch_size + sample_idx]
                for key in all_keys:
                    if key in none_keys:
                        self.assertTrue(key not in dataset_sample.keys())
                        continue

                    # good keys
                    dataloader_ndarray_sample = batch_dict[key][sample_idx].numpy()
                    dataset_ndarray_sample = dataset_sample[key]
                    # Not last sample
                    if checked_sample_cnt < sample_cnt - 1:
                        self.assertTrue(np.alltrue(dataloader_ndarray_sample == dataset_ndarray_sample))
                        continue

                    # Last sample.
                    # last_sample = (left, right), where:
                    # left = raw data, use np.alltrue(xxx == left) to compare.
                    # right = np.nan filled data, use np.alltrue(np.isnan(right)) to compare.
                    last_sample_tail_timestamp = paddle_ds._compute_last_sample_tail_timestamp()
                    extra_timeindex = pd.date_range(
                        start=paddlets_ds.get_target().time_index[-1],
                        end=last_sample_tail_timestamp,
                        freq=pd.infer_freq(paddlets_ds.get_target().time_index)
                    )
                    extra_timeindex = extra_timeindex[1:]

                    dataloader_right = dataloader_ndarray_sample[-1 - len(extra_timeindex) + 1:]
                    dataset_right = dataset_ndarray_sample[-1 - len(extra_timeindex) + 1:]
                    self.assertTrue(np.alltrue(np.isnan(dataloader_right)))
                    self.assertTrue(np.alltrue(np.isnan(dataset_right)))

                    dataloader_left = dataloader_ndarray_sample[:segment_size - len(extra_timeindex)]
                    dataset_left = dataset_ndarray_sample[:segment_size - len(extra_timeindex)]
                    self.assertTrue(np.alltrue(dataloader_left == dataset_left))

                checked_sample_cnt += 1

        ############################
        # case 6 (good case)       #
        # 1) known_cov is None.    #
        # 2) observed_cov is None. #
        # 3) Not Fill.             #
        ############################
        target_periods = 10
        known_periods = target_periods
        observed_periods = target_periods
        paddlets_ds = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)
        paddlets_ds.known_cov = None
        paddlets_ds.observed_cov = None

        paddle_ds = self._adapter.to_paddle_dataset(paddlets_ds, fill_last_value=None)
        batch_size = 2
        paddle_dataloader = self._adapter.to_paddle_dataloader(paddle_ds, batch_size, shuffle=False)

        all_keys = {"past_target", "known_cov", "observed_cov"}
        good_keys = {"past_target"}
        none_keys = all_keys - good_keys
        for batch_idx, batch_dict in enumerate(paddle_dataloader):
            curr_batch_size = list(batch_dict.values())[0].shape[0]
            for sample_idx in range(curr_batch_size):
                dataset_sample = paddle_ds[batch_idx * batch_size + sample_idx]
                for key in all_keys:
                    if key in none_keys:
                        self.assertTrue(key not in dataset_sample.keys())
                        continue
                    # good keys
                    dataloader_ndarray_sample = batch_dict[key][sample_idx].numpy()
                    dataset_ndarray_sample = dataset_sample[key]
                    self.assertTrue(np.alltrue(dataloader_ndarray_sample == dataset_ndarray_sample))

        ############################
        # case 7 (good case)       #
        # 1) known_cov is None.    #
        # 2) observed_cov is None. #
        # 3) Fill.                 #
        ############################
        target_periods = 10
        known_periods = target_periods
        observed_periods = target_periods
        paddlets_ds = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)
        paddlets_ds.known_cov = None
        paddlets_ds.observed_cov = None

        segment_size = 3
        paddle_ds = self._adapter.to_paddle_dataset(
            rawdataset=paddlets_ds,
            segment_size=segment_size,
            sampling_stride=3,
            fill_last_value=np.nan
        )
        sample_cnt = len(paddle_ds.samples)

        batch_size = 2
        paddle_dataloader = self._adapter.to_paddle_dataloader(paddle_ds, batch_size, shuffle=False)

        all_keys = {"past_target", "known_cov", "observed_cov"}
        good_keys = {"past_target"}
        none_keys = all_keys - good_keys
        checked_sample_cnt = 0
        for batch_idx, batch_dict in enumerate(paddle_dataloader):
            curr_batch_size = list(batch_dict.values())[0].shape[0]
            for sample_idx in range(curr_batch_size):
                dataset_sample = paddle_ds[batch_idx * batch_size + sample_idx]
                for key in all_keys:
                    if key in none_keys:
                        self.assertTrue(key not in dataset_sample.keys())
                        continue

                    # good keys
                    dataloader_ndarray_sample = batch_dict[key][sample_idx].numpy()
                    dataset_ndarray_sample = dataset_sample[key]

                    # Not last sample.
                    if checked_sample_cnt < sample_cnt - 1:
                        self.assertTrue(np.alltrue(dataloader_ndarray_sample == dataset_ndarray_sample))
                        continue

                    # Last sample.
                    # last_sample = (left, right), where:
                    # left = raw data, use np.alltrue(xxx == left) to compare.
                    # right = np.nan filled data, use np.alltrue(np.isnan(right)) to compare.
                    last_sample_tail_timestamp = paddle_ds._compute_last_sample_tail_timestamp()
                    extra_timeindex = pd.date_range(
                        start=paddlets_ds.get_target().time_index[-1],
                        end=last_sample_tail_timestamp,
                        freq=pd.infer_freq(paddlets_ds.get_target().time_index)
                    )
                    extra_timeindex = extra_timeindex[1:]

                    dataloader_right = dataloader_ndarray_sample[-1 - len(extra_timeindex) + 1:]
                    dataset_right = dataset_ndarray_sample[-1 - len(extra_timeindex) + 1:]
                    self.assertTrue(np.alltrue(np.isnan(dataloader_right)))
                    self.assertTrue(np.alltrue(np.isnan(dataset_right)))

                    dataloader_left = dataloader_ndarray_sample[:segment_size - len(extra_timeindex)]
                    dataset_left = dataset_ndarray_sample[:segment_size - len(extra_timeindex)]
                    self.assertTrue(np.alltrue(dataloader_left == dataset_left))

                checked_sample_cnt += 1

    def _build_mock_ts_dataset(
        self,
        target_periods: int = 10,
        known_periods: int = 10,
        observed_periods: int = 10,
        target_start_timestamp: pd.Timestamp = pd.Timestamp(datetime.datetime.now().date()),
        known_start_timestamp: pd.Timestamp = pd.Timestamp(datetime.datetime.now().date()),
        observed_start_timestamp: pd.Timestamp = pd.Timestamp(datetime.datetime.now().date()),
        freq: str = "1D"
    ):
        """
        Build mock paddlets dataset.

        all timeseries must have same freq.
        """
        target_df = pd.Series(
            [i for i in range(target_periods)],
            index=pd.date_range(start=target_start_timestamp, periods=target_periods, freq=freq),
            name="target0"
        )

        known_cov_df = pd.DataFrame(
            [(i * 10, i * 100) for i in range(known_periods)],
            index=pd.date_range(start=known_start_timestamp, periods=known_periods, freq=freq),
            columns=["known0", "known1"]
        )

        observed_cov_df = pd.DataFrame(
            [(i * -1, i * -10) for i in range(observed_periods)],
            index=pd.date_range(start=observed_start_timestamp, periods=observed_periods, freq=freq),
            columns=["observed0", "observed1"]
        )

        return TSDataset(
            target=TimeSeries.load_from_dataframe(data=target_df),
            known_cov=TimeSeries.load_from_dataframe(data=known_cov_df),
            observed_cov=TimeSeries.load_from_dataframe(data=observed_cov_df),
            static_cov={"static0": 1, "static1": 2}
        )

    def _compare_if_paddlets_sample_match_paddle_sample(
        self,
        paddlets_ds: TSDataset,
        paddle_ds: ReprPaddleDatasetImpl,
        param: Dict[str, Any]
    ) -> None:
        """
        Given a TSDataset and a built paddle.io.Dataset, compare if these data are matched.

        Args:
            paddlets_ds(TSDataset): Raw TSDataset.
            paddle_ds(PaddleDatasetImpl): Built paddle.io.Dataset.
            param(Dict): param for building samples.
        """
        segment_size = param["segment_size"]
        sampling_stride = param["sampling_stride"]
        target_ts = paddlets_ds.get_target()
        known_ts = paddlets_ds.get_known_cov()
        observed_ts = paddlets_ds.get_observed_cov()

        last_sample_idx = len(paddle_ds.samples) - 1
        sidx = 0
        first_sample_tail_idx = segment_size - 1

        # As target/known cov/observed cov might start with different timestamp, thus needs to compute offset.
        target_start_timestamp = target_ts.time_index[0]
        known_offset = 0 if known_ts is None else known_ts.time_index.get_loc(target_start_timestamp)
        observed_offset = 0 if observed_ts is None else observed_ts.time_index.get_loc(target_start_timestamp)

        # Start compare.
        while sidx < last_sample_idx:
            curr_paddle_sample = paddle_ds[sidx]
            tail_idx = first_sample_tail_idx + sidx * sampling_stride
            # past_target
            paddle_past_target = curr_paddle_sample["past_target"]
            paddlets_past_target = target_ts.to_numpy(False)[tail_idx - segment_size + 1:tail_idx + 1]
            self.assertTrue(np.alltrue(paddlets_past_target == paddle_past_target))

            # known_cov (possibly be None)
            if paddlets_ds.get_known_cov() is not None:
                paddlets_known_start = known_offset + tail_idx - segment_size + 1
                paddlets_known_end = known_offset + tail_idx + 1
                paddlets_known_cov = known_ts.to_numpy(False)[paddlets_known_start:paddlets_known_end]
                self.assertTrue(np.alltrue(paddlets_known_cov == curr_paddle_sample["known_cov"]))
            else:
                self.assertTrue("known_cov" not in curr_paddle_sample.keys())

            # observed_cov (possibly be None)
            if paddlets_ds.get_observed_cov() is not None:
                paddlets_observed_start = observed_offset + tail_idx - segment_size + 1
                paddlets_observed_end = observed_offset + tail_idx + 1
                paddlets_observed_cov = observed_ts.to_numpy(False)[paddlets_observed_start:paddlets_observed_end]
                self.assertTrue(np.alltrue(paddlets_observed_cov == curr_paddle_sample["observed_cov"]))
            else:
                self.assertTrue("observed_cov" not in curr_paddle_sample.keys())

            sidx += 1

        # last sample, possibly be filled.
        last_paddle_sample = paddle_ds.samples[sidx]
        last_sample_tail_timestamp = paddle_ds._compute_last_sample_tail_timestamp()
        if last_sample_tail_timestamp > paddlets_ds.get_target().time_index[-1]:
            # last sample is filled.
            extra_timeindex = pd.date_range(
                start=target_ts.time_index[-1],
                end=last_sample_tail_timestamp,
                freq=pd.infer_freq(target_ts.time_index)
            )
            extra_timeindex = extra_timeindex[1:]
            # First, past target.
            # paddle past target = (left, right), where left = raw data, right = filled all-NaN data.
            paddle_past_target = last_paddle_sample["past_target"]
            paddle_past_target_right = paddle_past_target[-1 - len(extra_timeindex) + 1:]
            self.assertTrue(np.alltrue(np.isnan(paddle_past_target_right)))

            paddle_past_target_left = paddle_past_target[:segment_size - len(extra_timeindex)]
            paddlets_past_target_left_tail = len(target_ts) - 1
            paddlets_past_target_left = target_ts.to_numpy(False)[
                paddlets_past_target_left_tail - (segment_size - len(extra_timeindex)) + 1:paddlets_past_target_left_tail + 1
            ]
            self.assertTrue(np.alltrue(paddlets_past_target_left == paddle_past_target_left))

            # Second, known cov (possibly be None)
            if known_ts is not None:
                paddle_known_cov = last_paddle_sample["known_cov"]
                paddle_known_cov_right = paddle_known_cov[-1 - len(extra_timeindex) + 1:]
                self.assertTrue(np.alltrue(np.isnan(paddle_known_cov_right)))

                paddle_known_cov_left = paddle_known_cov[:segment_size - len(extra_timeindex)]
                paddlets_known_cov_left_tail = len(known_ts) - 1
                paddlets_known_left_start = + paddlets_known_cov_left_tail - (segment_size - len(extra_timeindex)) + 1

                paddlets_known_cov_left = known_ts.to_numpy(False)[paddlets_known_left_start:paddlets_known_cov_left_tail + 1]
                self.assertTrue(np.alltrue(paddlets_known_cov_left == paddle_known_cov_left))
            else:
                self.assertTrue("known_cov" not in last_paddle_sample)

            # Third(Last), observed cov (possibly be None)
            if observed_ts is not None:
                paddle_observed_cov = last_paddle_sample["observed_cov"]
                paddle_observed_cov_right = paddle_observed_cov[-1 - len(extra_timeindex) + 1:]
                self.assertTrue(np.alltrue(np.isnan(paddle_observed_cov_right)))

                paddle_observed_cov_left = paddle_observed_cov[:segment_size - len(extra_timeindex)]
                paddlets_observed_left_tail = len(observed_ts) - 1
                paddlets_observed_left_start = paddlets_observed_left_tail - (segment_size - len(extra_timeindex)) + 1

                paddlets_observed_cov_left = observed_ts.to_numpy(False)[paddlets_observed_left_start:paddlets_observed_left_tail + 1]
                self.assertTrue(np.alltrue(paddlets_observed_cov_left == paddle_observed_cov_left))
            else:
                self.assertTrue("observed_cov" not in last_paddle_sample)
