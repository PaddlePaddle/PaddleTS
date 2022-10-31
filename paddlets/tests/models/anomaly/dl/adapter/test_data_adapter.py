# !/usr/bin/env python3
# -*- coding:utf-8 -*-

from paddlets.models.anomaly.dl.adapter.data_adapter import AnomalyDataAdapter
from paddlets.models.anomaly.dl.adapter.paddle_dataset_impl import AnomalyPaddleDatasetImpl
from paddlets import TSDataset, TimeSeries

import unittest
import pandas as pd
import numpy as np
from typing import Dict, Any, Set
import paddle.io


class TestDataAdapter(unittest.TestCase):
    def setUp(self):
        """
        unittest setup
        """
        self._adapter = AnomalyDataAdapter()
        super().setUp()

    def test_to_paddle_dataset(self):
        """
        Test DataAdapter.to_paddle_dataset()
        """
        # ##############################################
        # case 0 (good case)                           #
        # 1) Default in_chunk_len and sampling_stride. #
        # ##############################################
        observed_periods = target_periods = known_periods = 10

        # 0.1 Both numeric and categorical features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_contain_numeric=True,
            cov_dtypes_contain_categorical=True
        )
        # target / known_cov / static_cov are VALID to be None as anomaly detection models does NOT need them.
        tsdataset.target = None
        tsdataset.known_cov = None
        tsdataset.static_cov = None

        sample_ds = self._adapter.to_paddle_dataset(tsdataset)
        self._compare_tsdataset_and_sample_dataset(
            tsdataset=tsdataset,
            sample_ds=sample_ds,
            param={"in_chunk_len": sample_ds._observed_cov_chunk_len, "sampling_stride": sample_ds._sampling_stride}
        )

        # 0.2 ONLY numeric features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_contain_numeric=True,
            cov_dtypes_contain_categorical=False
        )
        # target / known_cov / static_cov are VALID to be None as anomaly detection models does NOT need them.
        tsdataset.target = None
        tsdataset.known_cov = None
        tsdataset.static_cov = None

        sample_ds = self._adapter.to_paddle_dataset(tsdataset)
        self._compare_tsdataset_and_sample_dataset(
            tsdataset=tsdataset,
            sample_ds=sample_ds,
            param={"in_chunk_len": sample_ds._observed_cov_chunk_len, "sampling_stride": sample_ds._sampling_stride}
        )

        # 0.3 ONLY categorical features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_contain_numeric=False,
            cov_dtypes_contain_categorical=True
        )
        # target / known_cov / static_cov are VALID to be None as anomaly detection models does NOT need them.
        tsdataset.target = None
        tsdataset.known_cov = None
        tsdataset.static_cov = None

        sample_ds = self._adapter.to_paddle_dataset(tsdataset)
        self._compare_tsdataset_and_sample_dataset(
            tsdataset=tsdataset,
            sample_ds=sample_ds,
            param={"in_chunk_len": sample_ds._observed_cov_chunk_len, "sampling_stride": sample_ds._sampling_stride}
        )

        ####################################################
        # case 1 (good case)                               #
        # 1) Non-Default in_chunk_len and sampling_stride. #
        ####################################################
        observed_periods = target_periods = known_periods = 10

        # 1.1 Both numeric and categorical features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_contain_numeric=True,
            cov_dtypes_contain_categorical=True
        )
        # target / known_cov / static_cov are VALID to be None as anomaly detection models does NOT need them.
        tsdataset.target = None
        tsdataset.known_cov = None
        tsdataset.static_cov = None

        param = {"in_chunk_len": 3, "sampling_stride": 3}
        sample_ds = self._adapter.to_paddle_dataset(tsdataset, **param)
        self._compare_tsdataset_and_sample_dataset(
            tsdataset=tsdataset,
            sample_ds=sample_ds,
            param=param
        )

        # 1.2 ONLY numeric features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_contain_numeric=True,
            cov_dtypes_contain_categorical=False
        )
        # target / known_cov / static_cov are VALID to be None as anomaly detection models does NOT need them.
        tsdataset.target = None
        tsdataset.known_cov = None
        tsdataset.static_cov = None

        param = {"in_chunk_len": 3, "sampling_stride": 3}
        sample_ds = self._adapter.to_paddle_dataset(tsdataset, **param)
        self._compare_tsdataset_and_sample_dataset(
            tsdataset=tsdataset,
            sample_ds=sample_ds,
            param=param
        )

        # 1.3 ONLY categorical features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_contain_numeric=False,
            cov_dtypes_contain_categorical=True
        )
        # target / known_cov / static_cov are VALID to be None as anomaly detection models does NOT need them.
        tsdataset.target = None
        tsdataset.known_cov = None
        tsdataset.static_cov = None

        param = {"in_chunk_len": 3, "sampling_stride": 3}
        sample_ds = self._adapter.to_paddle_dataset(tsdataset, **param)
        self._compare_tsdataset_and_sample_dataset(
            tsdataset=tsdataset,
            sample_ds=sample_ds,
            param=param
        )

        ######################################
        # case 2 (bad case)                  #
        # 1) TSDataset.observed_cov is None. #
        ######################################
        observed_periods = target_periods = known_periods = 10
        tsdataset = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)
        # Explicitly set observed cov to None to repro this bad case.
        tsdataset.observed_cov = None

        succeed = True
        try:
            _ = self._adapter.to_paddle_dataset(rawdataset=tsdataset)
        except ValueError:
            succeed = False
        self.assertFalse(succeed)

        #########################
        # case 3 (bad case)     #
        # 1) in_chunk_len <= 0. #
        #########################
        observed_periods = target_periods = known_periods = 10
        tsdataset = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)
        tsdataset.target = None
        tsdataset.known_cov = None
        
        succeed = True
        try:
            # in_chunk_len = 0.
            _ = self._adapter.to_paddle_dataset(rawdataset=tsdataset, in_chunk_len=0)
        except ValueError:
            succeed = False
        self.assertFalse(succeed)

        succeed = True
        try:
            # in_chunk_len < 0
            _ = self._adapter.to_paddle_dataset(rawdataset=tsdataset, in_chunk_len=-1)
        except ValueError:
            succeed = False
        self.assertFalse(succeed)
        
        ############################
        # case 4 (bad case)        #
        # 1) sampling_stride <= 0. #
        ############################
        observed_periods = target_periods = known_periods = 10
        tsdataset = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)
        tsdataset.target = None
        tsdataset.known_cov = None

        succeed = True
        try:
            # sampling_stride = 0.
            _ = self._adapter.to_paddle_dataset(rawdataset=tsdataset, sampling_stride=0)
        except ValueError:
            succeed = False
        self.assertFalse(succeed)

        succeed = True
        try:
            # sampling_stride < 0
            _ = self._adapter.to_paddle_dataset(rawdataset=tsdataset, sampling_stride=-1)
        except ValueError:
            succeed = False
        self.assertFalse(succeed)

        #######################################
        # case 5 (bad case)                   #
        # 1) observed_cov len < in_chunk_len. #
        #######################################
        observed_periods = target_periods = known_periods = 10
        tsdataset = self._build_mock_ts_dataset(target_periods, known_periods, observed_periods)
        tsdataset.target = None
        tsdataset.known_cov = None

        in_chunk_len = observed_periods + 1

        succeed = True
        try:
            # target observed_periods < in_chunk_len
            _ = self._adapter.to_paddle_dataset(rawdataset=tsdataset, in_chunk_len=in_chunk_len)
        except ValueError:
            succeed = False
        self.assertFalse(succeed)

    def test_to_paddle_dataloader(self):
        """
        Test DataAdapter.to_paddle_dataloader()
        """
        ######################
        # case 0 (good case) #
        ######################
        observed_periods = target_periods = known_periods = 10
        
        # 0.1 Both numeric and categorical features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_contain_numeric=True,
            cov_dtypes_contain_categorical=True
        )
        tsdataset.target = None
        tsdataset.known_cov = None
        tsdataset.static_cov = None
        
        sample_ds = self._adapter.to_paddle_dataset(tsdataset)
        batch_size = 2
        sample_dataloader = self._adapter.to_paddle_dataloader(sample_ds, batch_size, shuffle=False)
        
        good_keys = {"observed_cov_numeric", "observed_cov_categorical"}
        self._compare_sample_dataset_and_sample_dataloader(sample_ds, sample_dataloader, batch_size, good_keys)

        # 0.2 ONLY numeric features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_contain_numeric=True,
            cov_dtypes_contain_categorical=False
        )
        tsdataset.target = None
        tsdataset.known_cov = None
        tsdataset.static_cov = None

        sample_ds = self._adapter.to_paddle_dataset(tsdataset)
        batch_size = 2
        sample_dataloader = self._adapter.to_paddle_dataloader(sample_ds, batch_size, shuffle=False)

        good_keys = {"observed_cov_numeric"}
        self._compare_sample_dataset_and_sample_dataloader(sample_ds, sample_dataloader, batch_size, good_keys)

        # 0.3 ONLY categorical features.
        tsdataset = self._build_mock_ts_dataset(
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            cov_dtypes_contain_numeric=False,
            cov_dtypes_contain_categorical=True
        )
        tsdataset.target = None
        tsdataset.known_cov = None
        tsdataset.static_cov = None

        sample_ds = self._adapter.to_paddle_dataset(tsdataset)
        batch_size = 2
        sample_dataloader = self._adapter.to_paddle_dataloader(sample_ds, batch_size, shuffle=False)

        good_keys = {"observed_cov_categorical"}
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
        sample_ds: AnomalyPaddleDatasetImpl,
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

        in_chunk_len = param["in_chunk_len"]
        sampling_stride = param["sampling_stride"]
        observed_ts = tsdataset.get_observed_cov()
        static_cov = tsdataset.get_static_cov()

        first_sample_tail_idx = in_chunk_len - 1
        # Start compare.
        for sidx in range(len(sample_ds.samples)):
            curr_sample = sample_ds[sidx]
            tail_idx = first_sample_tail_idx + sidx * sampling_stride

            # past target
            self.assertTrue("past_target" not in curr_sample.keys())

            # future target
            self.assertTrue("future_target" not in curr_sample.keys())

            # known cov
            self.assertTrue("known_cov_numeric" not in curr_sample.keys())
            self.assertTrue("known_cov_categorical" not in curr_sample.keys())

            # observed cov
            observed_df = observed_ts.to_dataframe(copy=False)
            observed_start = tail_idx - in_chunk_len + 1
            observed_end = tail_idx + 1
            # numeric
            if "observed_cov_numeric" in curr_sample.keys():
                numeric_df = observed_df.select_dtypes(include=numeric_dtype)
                numeric_ndarray = numeric_df.to_numpy(copy=False)
                observed_numeric_ndarray = numeric_ndarray[observed_start:observed_end]
                # data ok.
                self.assertTrue(np.alltrue(observed_numeric_ndarray == curr_sample["observed_cov_numeric"]))
                # dtype ok.
                self.assertEqual(observed_numeric_ndarray.dtype, curr_sample["observed_cov_numeric"].dtype)
            # categorical
            if "observed_cov_categorical" in curr_sample.keys():
                categorical_df = observed_df.select_dtypes(include=categorical_dtype)
                categorical_ndarray = categorical_df.to_numpy(copy=False)
                observed_categorical_ndarray = categorical_ndarray[observed_start:observed_end]
                # data ok.
                self.assertTrue(np.alltrue(observed_categorical_ndarray == curr_sample["observed_cov_categorical"]))
                # dtype ok.
                self.assertEqual(observed_categorical_ndarray.dtype, curr_sample["observed_cov_categorical"].dtype)

            # static cov
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
            else:
                self.assertTrue("static_cov_numeric" not in curr_sample.keys())
                self.assertTrue("static_cov_categorical" not in curr_sample.keys())
    
    def _compare_sample_dataset_and_sample_dataloader(
        self,
        sample_ds: AnomalyPaddleDatasetImpl,
        sample_dataloader: paddle.io.DataLoader,
        batch_size: int,
        good_keys: Set[str]
    ):
        """Check if sample dataset matches batched sample dataloader."""
        all_keys = {"observed_cov_numeric", "observed_cov_categorical"}
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
                    # data ok
                    self.assertTrue(np.alltrue(dataloader_ndarray_sample == dataset_ndarray_sample))
                    # dtype ok
                    self.assertEqual(dataloader_ndarray_sample.dtype, dataset_ndarray_sample.dtype)
