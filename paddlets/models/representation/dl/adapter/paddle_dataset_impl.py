# !/usr/bin/env python3
# -*- coding:utf-8 -*-

from paddlets.datasets import TSDataset, TimeSeries
from paddlets.logger.logger import Logger, raise_if

from paddle.io import Dataset as PaddleDataset
import pandas as pd
import numpy as np
import math
from typing import List, Dict, Union

logger = Logger(__name__)


class ReprPaddleDatasetImpl(PaddleDataset):
    """
    An implementation of paddle Dataset.

    Note that any unused (known / observed) columns should be removed from the TSDataset before handled by this class.

    Args:
        rawdataset(TSDataset): Raw TSDataset to be converted.
        segment_size(int): The size of the loopback window, i.e., the number of time steps feed to the model.
        sampling_stride(int, optional): Time steps to stride over the i-th sample and (i+1)-th sample. More precisely,
            let `t` be the time index of target time series, `t[i]` be the start time of the i-th sample,
            `t[i+1]` be the start time of the (i+1)-th sample, then `sampling_stride` represents the result of
            `t[i+1] - t[i]`.
        fill_last_value(float, optional): The value used for filling last sample. Set to None if no need to fill.
    """
    def __init__(
        self,
        rawdataset: TSDataset,
        segment_size: int,
        sampling_stride: int,
        fill_last_value: Union[float, type(None)] = np.nan
    ):
        super(ReprPaddleDatasetImpl, self).__init__()

        self._rawdataset = rawdataset
        self._target_segment_size = segment_size
        self._known_cov_segment_size = self._target_segment_size
        self._observed_cov_segment_size = self._target_segment_size
        self._sampling_stride = sampling_stride
        self._fill_last_value = fill_last_value

        raise_if(rawdataset is None, "TSDataset must not be None.")
        raise_if(segment_size <= 0, f"segment_size ({segment_size}) must be positive integer.")
        raise_if(sampling_stride <= 0, f"sampling_stride ({sampling_stride}) must be positive integer.")

        # Raise if target / known cov / observed cov is invalid.
        self._validate_target_timeseries()
        self._validate_known_cov_timeseries()
        self._validate_observed_cov_timeseries()

        if self._fill_last_value is not None:
            # avoid inplace fill.
            self._rawdataset = self._fill_tsdataset(TSDataset.copy(self._rawdataset))

        self._samples = self._build_samples()

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        return self._samples[idx]

    def _build_samples(self) -> List[Dict[str, np.ndarray]]:
        """
        Internal method, builds samples.

        Returns:
            List[Dict[str, np.ndarray]]: A list of samples.

        Examples:
            .. code-block:: python

                # Scenario 1 - Need to fill the last sample with np.nan.
                # Given:
                segment_size = 3
                sampling_stride = 3
                fill_last_value = np.nan
                rawdataset = {
                    "target": [
                        [0.0],
                        [10.0],
                        [20.0],
                        [30.0],
                        [40.0],
                        [50.0],
                        [60.0]
                    ],
                    "known_cov": [
                        [0.0, 0.0],
                        [10.0, 100.0],
                        [20.0, 200.0],
                        [30.0, 300.0],
                        [40.0, 400.0],
                        [50.0, 500.0],
                        [60.0, 600.0]
                    ],
                    "observed_cov": [
                        [0.0],
                        [-1.0],
                        [-2.0],
                        [-3.0],
                        [-4.0],
                        [-5.0],
                        [-6.0]
                    ],
                    "static_cov": {"static0": 0.0, "static1": 1.0}
                }

                # Built samples:
                samples = [
                    # sample[0]
                    {
                        ###############
                        # past_target #
                        ###############
                        # row = _target_in_chunk_len = 3
                        # col = len(TSDataset._target.data.columns) = 1
                        "past_target": [
                            [0.0],
                            [10.0],
                            [20.0]
                        ],

                        #############
                        # known_cov #
                        #############
                        # numeric
                        # row = _target_in_chunk_len = 3
                        # col = len(TSDataset._known_cov.data.select_dtypes(include=np.float32)) = 1
                        "known_cov_numeric": [
                            [0.0, 0.0],
                            [10.0, 100.0],
                            [20.0, 200.0]
                        ],

                        # categorical (No such key, because input dataset does NOT contain any int-type known columns.)
                        # row = _target_in_chunk_len = 3
                        # col = len(TSDataset._known_cov.data.select_dtypes(include=np.int64)) = 0

                        ################
                        # observed_cov #
                        ################
                        # numeric
                        # row = _target_in_chunk_len = 3
                        # col = len(TSDataset._observed_cov.data.select_dtypes(include=np.float32)) = 1
                        "observed_cov_numeric": [
                            [0.0],
                            [-1.0],
                            [-2.0]
                        ]

                        # categorical (No such key, because input dataset does NOT contain any int-type observed col.)
                        # col = len(TSDataset._observed_cov.data.select_dtypes(include=np.int64)) = 0

                        ##############
                        # static_cov #
                        ##############
                        # numeric
                        # col = len(TSDataset._observed_cov.data.select_dtypes(include=np.float32)) = 2
                        "static_cov_numeric": [
                            # key-wise ascending sorted data.
                            [1.0, 2.0]
                        ],

                        # categorical (No such key, because input dataset does NOT contain any int-type static data.)
                        # col = len(TSDataset._observed_cov.data.select_dtypes(include=np.int64)) = 0
                    },

                    # sample[1]
                    {
                        "past_target": [
                            [30.0],
                            [40.0],
                            [50.0]
                        ],
                        "known_cov_numeric": [
                            [300.0],
                            [400.0],
                            [500.0]
                        ],
                        "observed_cov_numeric": [
                            [0.0],
                            [-1.0],
                            [-2.0]
                        ],
                        "static_cov_numeric": [
                            [1.0, 2.0]
                        ]
                    },

                    # sample[2] (i.e. last sample, filled with np.nan)
                    {
                        "past_target": [
                            [60.0],
                            [nan],
                            [nan]
                        ],
                        "known_cov_numeric": [
                            [60.0, 600.0],
                            [nan, nan],
                            [nan, nan]
                        ],
                        "observed_cov_numeric": [
                            [-6.0],
                            [nan],
                            [nan]
                        ],
                        "static_cov_numeric": [
                            [1.0, 2.0]
                        ]
                    }
                ]

            .. code-block:: python

                # Scenario 2 - Do NOT need to fill the last sample (i.e. fill_last_value = None).
                # Note that in this scenario, the last segment of target / known cov / observed will be skipped and
                # will NOT be filled.

                # Given:
                segment_size = 3
                sampling_stride = 3
                fill_last_value = None
                rawdataset = {
                    "target": [
                        [0.0],
                        [10.0],
                        [20.0],
                        [30.0],
                        [40.0],
                        [50.0],
                        [60.0]
                    ],
                    "known_cov": [
                        [0.0, 0.0],
                        [10.0, 100.0],
                        [20.0, 200.0],
                        [30.0, 300.0],
                        [40.0, 400.0],
                        [50.0, 500.0],
                        [60.0, 600.0]
                    ],
                    "observed_cov": [
                        [0.0],
                        [-1.0],
                        [-2.0],
                        [-3.0],
                        [-4.0],
                        [-5.0],
                        [-6.0]
                    ],
                    "static_cov": {"static0": 0.0, "static1": 1.0}
                }

                # Built samples:
                samples = [
                    # sample[0]
                    {
                        ###############
                        # past_target #
                        ###############
                        # row = _target_in_chunk_len = 3
                        # col = len(TSDataset._target.data.columns) = 1
                        "past_target": [
                            [0.0],
                            [10.0],
                            [20.0]
                        ],

                        #############
                        # known_cov #
                        #############
                        # numeric
                        # row = _target_in_chunk_len = 3
                        # col = len(TSDataset._known_cov.data.select_dtypes(include=np.float32)) = 1
                        "known_cov_numeric": [
                            [0.0, 0.0],
                            [10.0, 100.0],
                            [20.0, 200.0]
                        ],

                        # categorical (No such key, because input dataset does NOT contain any int-type known columns.)
                        # row = _target_in_chunk_len = 3
                        # col = len(TSDataset._known_cov.data.select_dtypes(include=np.int64)) = 0

                        ################
                        # observed_cov #
                        ################
                        # numeric
                        # row = _target_in_chunk_len = 3
                        # col = len(TSDataset._observed_cov.data.select_dtypes(include=np.float32)) = 1
                        "observed_cov_numeric": [
                            [0.0],
                            [-1.0],
                            [-2.0]
                        ]

                        # categorical (No such key, because input dataset does NOT contain any int-type observed col.)
                        # col = len(TSDataset._observed_cov.data.select_dtypes(include=np.int64)) = 0

                        ##############
                        # static_cov #
                        ##############
                        # numeric
                        # col = len(TSDataset._observed_cov.data.select_dtypes(include=np.float32)) = 2
                        "static_cov_numeric": [
                            # key-wise ascending sorted data.
                            [1.0, 2.0]
                        ],

                        # categorical (No such key, because input dataset does NOT contain any int-type static data.)
                        # col = len(TSDataset._observed_cov.data.select_dtypes(include=np.int64)) = 0
                    },

                    # sample[1] (as the remaining segment will NOT be filled, this will be the last sample.)
                    {
                        "past_target": [
                            [30.0],
                            [40.0],
                            [50.0]
                        ],
                        "known_cov_numeric": [
                            [300.0],
                            [400.0],
                            [500.0]
                        ],
                        "observed_cov_numeric": [
                            [0.0],
                            [-1.0],
                            [-2.0]
                        ],
                        "static_cov_numeric": [
                            [1.0, 2.0]
                        ]
                    }
                ]
        """
        numeric_dtype = np.float32
        categorical_dtype = np.int64

        target_ts = self._rawdataset.get_target()
        target_ndarray = target_ts.to_numpy(copy=False)

        # known cov (possibly be None)
        known_cov_ts = self._rawdataset.get_known_cov()
        known_cov_numeric_ndarray = None
        known_cov_categorical_ndarray = None
        if known_cov_ts is not None:
            known_cov_numeric_ndarray = self._build_ndarray_from_timeseries_by_dtype(
                timeseries=known_cov_ts,
                dtype=numeric_dtype
            )
            known_cov_categorical_ndarray = self._build_ndarray_from_timeseries_by_dtype(
                timeseries=known_cov_ts,
                dtype=categorical_dtype
            )

        # observed (possibly be None)
        observed_cov_ts = self._rawdataset.get_observed_cov()
        observed_cov_numeric_ndarray = None
        observed_cov_categorical_ndarray = None
        if observed_cov_ts is not None:
            observed_cov_numeric_ndarray = self._build_ndarray_from_timeseries_by_dtype(
                timeseries=observed_cov_ts,
                dtype=numeric_dtype
            )
            observed_cov_categorical_ndarray = self._build_ndarray_from_timeseries_by_dtype(
                timeseries=observed_cov_ts,
                dtype=categorical_dtype
            )

        # static cov (possibly be None)
        static_cov = self._rawdataset.get_static_cov()
        pre_computed_static_cov_numeric_for_single_sample = None
        pre_computed_static_cov_categorical_for_single_sample = None
        if static_cov is not None:
            static_cov_numeric = dict()
            static_cov_categorical = dict()
            for k, v in static_cov.items():
                if type(v) in {numeric_dtype, float}:
                    # built-in float type will be implicitly converted to numpy.float32 dtype.
                    static_cov_numeric[k] = numeric_dtype(v)
                if type(v) in {categorical_dtype, int}:
                    # built-in int type will be implicitly converted to numpy.int64 dtype.
                    static_cov_categorical[k] = categorical_dtype(v)

            pre_computed_static_cov_numeric_for_single_sample = self._build_static_cov_for_single_sample(
                static_cov_dict=static_cov_numeric
            )
            pre_computed_static_cov_categorical_for_single_sample = self._build_static_cov_for_single_sample(
                static_cov_dict=static_cov_categorical
            )

        samples = []
        tail_idx = self._target_segment_size - 1
        max_allowed_idx = len(self._rawdataset.get_target().time_index) - 1
        while tail_idx <= max_allowed_idx:
            sample = {
                "past_target": self._build_past_target_for_single_sample(
                    past_target_tail=tail_idx,
                    target_ndarray=target_ndarray
                )
            }

            # known_cov
            if known_cov_ts is not None:
                # numeric
                if 0 not in known_cov_numeric_ndarray.shape:
                    sample["known_cov_numeric"] = self._build_known_cov_for_single_sample(
                        known_cov_tail=tail_idx,
                        known_cov_ndarray=known_cov_numeric_ndarray
                    )
                # categorical
                if 0 not in known_cov_categorical_ndarray.shape:
                    sample["known_cov_numeric"] = self._build_known_cov_for_single_sample(
                        known_cov_tail=tail_idx,
                        known_cov_ndarray=known_cov_categorical_ndarray
                    )

            # observed_cov
            if observed_cov_ts is not None:
                # numeric
                if 0 not in observed_cov_numeric_ndarray.shape:
                    sample["observed_cov_numeric"] = self._build_observed_cov_for_single_sample(
                        observed_cov_tail=tail_idx,
                        observed_cov_ndarray=observed_cov_numeric_ndarray
                    )
                # categorical
                if 0 not in observed_cov_categorical_ndarray.shape:
                    sample["observed_cov_categorical"] = self._build_observed_cov_for_single_sample(
                        observed_cov_tail=tail_idx,
                        observed_cov_ndarray=observed_cov_categorical_ndarray
                    )

            # static_cov
            if static_cov is not None:
                # numeric
                if 0 not in pre_computed_static_cov_numeric_for_single_sample.shape:
                    sample["static_cov_numeric"] = pre_computed_static_cov_numeric_for_single_sample
                # categorical
                if 0 not in pre_computed_static_cov_categorical_for_single_sample.shape:
                    sample["static_cov_categorical"] = pre_computed_static_cov_categorical_for_single_sample

            samples.append(sample)

            # The predefined assert `sampling_stride >= 1` in the construct method ensures that `infinite while loop`
            # will Never occur.
            tail_idx += self._sampling_stride
        return samples

    def _validate_target_timeseries(self) -> None:
        target_ts = self._rawdataset.get_target()
        raise_if(target_ts is None, "dataset target Timeseries must not be None.")

        # fill np.int dtype is currently NOT supported.
        invalid_dtype = np.int64
        raise_if(
            len(target_ts.to_dataframe(copy=False).select_dtypes(invalid_dtype).columns) > 0,
            f"build target categorical(dtype = {invalid_dtype}) feature is currently NOT supported."
        )

        # This is to make sure that at least one sample can be built from the given target timeseries.
        raise_if(
            len(target_ts.time_index) < self._target_segment_size,
            f"TSDataset target Timeseries length ({len(target_ts.time_index)}) must >= {self._target_segment_size}."
        )

    def _validate_known_cov_timeseries(self) -> None:
        known_cov_ts = self._rawdataset.get_known_cov()
        if known_cov_ts is not None:
            invalid_dtype = np.int64
            raise_if(
                len(known_cov_ts.to_dataframe(copy=False).select_dtypes(invalid_dtype).columns) > 0,
                f"build known cov categorical(dtype = {invalid_dtype}) feature is currently NOT supported."
            )

            target_timeidx = self._rawdataset.get_target().time_index
            known_timeidx = known_cov_ts.time_index
            raise_if(
                known_timeidx[0] > target_timeidx[0],
                f"known cov timeindex[0] ({known_timeidx[0]}) must <= target timeindex[0] ({target_timeidx[0]})."
            )
            raise_if(
                known_timeidx[-1] != target_timeidx[-1],
                f"known cov timeindex[-1] ({known_timeidx[0]}) must == target timeindex[-1] ({target_timeidx[-1]})."
            )

    def _validate_observed_cov_timeseries(self) -> None:
        observed_cov_ts = self._rawdataset.get_observed_cov()
        if observed_cov_ts is not None:
            # fill np.int dtype is currently NOT supported.
            invalid_dtype = np.int64
            raise_if(
                len(observed_cov_ts.to_dataframe(copy=False).select_dtypes(invalid_dtype).columns) > 0,
                f"build observed cov categorical(dtype = {invalid_dtype}) feature is currently NOT supported."
            )

            target_timeidx = self._rawdataset.get_target().time_index
            observed_timeidx = observed_cov_ts.time_index
            raise_if(
                observed_timeidx[0] > target_timeidx[0],
                f"observed cov timeindex[0]({observed_timeidx[0]}) must <= target timeindex[0] ({target_timeidx[0]})."
            )
            raise_if(
                observed_timeidx[-1] != target_timeidx[-1],
                f"observed cov timeindex[-1]({observed_timeidx[-1]}) must == target timeindex[0]({target_timeidx[-1]})."
            )

    def _fill_tsdataset(self, tsdataset: TSDataset) -> TSDataset:
        # First, fill target
        filled_target_ts = self._fill_timeseries(tsdataset.get_target())
        tsdataset.set_target(target=filled_target_ts)

        # Second, fill known cov
        if tsdataset.get_known_cov() is not None:
            filled_known_cov_ts = self._fill_timeseries(tsdataset.get_known_cov())
            tsdataset.set_known_cov(known_cov=filled_known_cov_ts)

        # Third(Last), fill observed cov
        if tsdataset.get_observed_cov() is not None:
            filled_observed_cov_ts = self._fill_timeseries(tsdataset.get_observed_cov())
            tsdataset.set_observed_cov(observed_cov=filled_observed_cov_ts)

        return tsdataset

    def _fill_timeseries(self, raw_timeseries: TimeSeries) -> TimeSeries:
        # compute how long needs to be filled.
        last_sample_tail_timestamp = self._compute_last_sample_tail_timestamp()
        raw_timeindex = raw_timeseries.time_index
        if last_sample_tail_timestamp == raw_timeindex[-1]:
            # For example:
            # raw_timeindex = [7:00, 8:00, 9:00, 10:00]
            # last_sample_tail_timestamp = 10:00
            # Thus no need to fill.
            return raw_timeseries

        # need fill.
        raw_freq = pd.infer_freq(raw_timeindex)
        raw_df = raw_timeseries.to_dataframe(copy=False)
        raw_cols = raw_df.columns

        extra_timeindex = pd.date_range(
            start=raw_timeindex[-1],
            end=last_sample_tail_timestamp,
            freq=raw_freq
        )
        # remove first timestamp as it is duplicated with the last timestamp in the raw time index.
        extra_timeindex = extra_timeindex[1:]

        extra_ndarray = np.zeros(shape=(len(extra_timeindex), len(raw_cols)))
        extra_ndarray.fill(self._fill_last_value)
        extra_df = pd.DataFrame(
            data=extra_ndarray,
            index=extra_timeindex,
            columns=raw_cols
        )

        filled_df = pd.concat([raw_df, extra_df])
        return TimeSeries.load_from_dataframe(
            data=filled_df,
            freq=raw_freq,
            drop_tail_nan=False
        )

    def _compute_last_sample_tail_timestamp(self) -> pd.Timestamp:
        """
        compute last sample tail timestamp.

        If self._fill_last_value is None, the returned timestamp will be the last sample without filling, otherwise
        if it is not None, the returned timestamp will be the filled last sample.

        Step 1. compute the sample count that can be built from the raw tsdataset.
        The computation formula of the sample_cnt is as follows:
        a + b * (n - 1) <= c
        where:
        a = first_sample_tail_idx
        b = sampling_stride
        n = sample_cnt (Never contain the filled sample)
        c = max_target_idx

        Thus,
        n = math.floor((c - a) / b) + 1
        i.e.,
        sample_cnt = math.floor((max_target_idx - (segment_size - 1)) / sampling_stride) + 1

        Step 2. Compute the tail index of the last sample (include filled sample if self._fill_last_value is not None).
        The computation formula can be expressed as follows:
        c = a + b * (n - 1)
        where:
        a = first_sample_tail_idx
        b = sampling_stride
        n = sample_cnt (possibly contain the filled sample)
        c = max_target_idx

        Returns:
            pd.Timestamp: The tail timestamp of last sample, where the last sample can either be filled / unfilled.
        """
        target_timeindex = self._rawdataset.get_target().time_index
        max_target_idx = len(target_timeindex) - 1
        first_sample_tail_idx = self._target_segment_size - 1
        sample_cnt = math.floor((max_target_idx - first_sample_tail_idx) / self._sampling_stride) + 1
        last_sample_tail_idx = first_sample_tail_idx + self._sampling_stride * (sample_cnt - 1)

        if last_sample_tail_idx == max_target_idx:
            return target_timeindex[last_sample_tail_idx]

        if self._fill_last_value is None:
            return target_timeindex[last_sample_tail_idx]

        # need fill.
        sample_cnt += 1
        last_sample_tail_idx = first_sample_tail_idx + self._sampling_stride * (sample_cnt - 1)
        freq = pd.infer_freq(target_timeindex)
        # Here are 2 operations worth noticing:
        # 1. plus extra `1` when calling pd.date_range()
        # 2. remove the first element of extra_timeindex by calling extra_timeindex = extra_timeindex[1:]
        # The reason to do the above is that the first timestamp of extra_timeindex is duplicated with the last
        # timestamp of target_timeindex. To construct the extra timeindex, we must specify the start as the first
        # element of target_timeindex (which will cause the duplicate timestamp).
        # Similarly, to avoid the target_timeindex[-1] to be used twice, we must remove it from extra_timeindex.
        extra_timeindex = pd.date_range(
            start=target_timeindex[-1],
            periods=last_sample_tail_idx - (len(target_timeindex) - 1) + 1,
            freq=freq
        )
        return extra_timeindex[-1]

    def _build_past_target_for_single_sample(
        self,
        past_target_tail: int,
        target_ndarray: np.ndarray
    ):
        """
        Internal method, builds a past_target chunk for a single sample.

        Args:
            past_target_tail(int): the tail idx of past_target chunk of the same sample.
            target_ndarray(np.ndarray): an np.ndarray matrix.

        Returns:
            np.ndarray: built past_target chunk for the current single sample.
        """
        return target_ndarray[past_target_tail - self._target_segment_size + 1:past_target_tail + 1]

    def _build_known_cov_for_single_sample(self, known_cov_tail: int, known_cov_ndarray: np.ndarray) -> np.ndarray:
        """
        Internal method, builds a known_cov chunk for a single sample.

        Args:
            known_cov_tail(int): the tail idx of known cov chunk of the same sample.

        Returns:
            np.ndarray: built known cov chunk for the current single sample.
        """
        # As we know that known cov start timestamp might <= target start timestamp, thus needs to compute offset.
        target_start_timestamp = self._rawdataset.get_target().time_index[0]
        # the pre-check already guarantee that target_start_timestamp must be within the known_timeindex range.
        known_timeindex = self._rawdataset.get_known_cov().time_index
        offset = known_timeindex.get_loc(target_start_timestamp)

        start = offset + known_cov_tail - self._known_cov_segment_size + 1
        end = offset + known_cov_tail + 1
        return known_cov_ndarray[start:end]

    def _build_observed_cov_for_single_sample(
        self,
        observed_cov_tail: int,
        observed_cov_ndarray: np.ndarray
    ) -> np.ndarray:
        """
        Internal method, builds an observed_cov chunk for a single sample.

        Args:
            observed_cov_tail(int): the tail idx of observed cov chunk of the same sample.
            observed_cov_ndarray(np.ndarray, optional): an np.ndarray matrix.

        Returns:
            np.ndarray: built observed cov chunk for the current single sample.
        """
        # As we know that observed cov start timestamp might <= target start timestamp, thus needs to compute offset.
        target_start_timestamp = self._rawdataset.get_target().time_index[0]
        # the pre-check already guarantee that target_start_timestamp must be within the observed_timeindex range.
        observed_timeindex = self._rawdataset.get_observed_cov().time_index
        offset = observed_timeindex.get_loc(target_start_timestamp)

        start = offset + observed_cov_tail - self._observed_cov_segment_size + 1
        end = offset + observed_cov_tail + 1
        return observed_cov_ndarray[start:end]

    def _build_static_cov_for_single_sample(
        self,
        static_cov_dict: Dict[str, Union[np.float32, np.int64]]
    ) -> np.ndarray:
        """
        Internal method, build a numeric or categorical static_cov chunk for a single sample.

        Args:
            static_cov_dict(Dict[str, Union[np.float32, np.int64]]): a k-v static cov map, contains either
                int64-dtype-only subset of TSDataset._static_cov, or float32-dtype-only subset of TSDataset._static_cov.

        Returns:
            np.ndarray: built numeric or categorical static cov chunk for the current single sample.
        """
        # [(k1, v1), (k2, v2)]
        sorted_static_cov_list = sorted(static_cov_dict.items(), key=lambda t: t[0], reverse=False)

        # [[v1, v2]]
        return np.array([[t[1] for t in sorted_static_cov_list]])

    def _build_ndarray_from_timeseries_by_dtype(self, timeseries: TimeSeries, dtype: type) -> np.ndarray:
        """
        Internal method, extract dataframe from given timeseries with specified dtype, then return the converted
        numpy.ndarray.

        Args:
            timeseries(TimeSeries): TimeSeries object to be extracted.
            dtype(type]): dtype to be included when extract from given timeseries.

        Returns:
            np.ndarray: The ndarray object which is converted from the extracted dataframe from the given timeseries.

        Examples:

            .. code-block :: python

                # Note that in case if the given timeseries does NOT contain columns with given dtypes, the returned
                # ndarray.shape[1] = 0. See examples to get more details.

                # Given target time series only contains arrays of float type, but NOT contains arrays of int type.
                target_timeseries = {
                    "col_1_float": [1.0, 2.0, 3.0],
                    "col_2_float": [100.0, 200.0, 300.0]
                }

                # Thus, returned shape will be (3, 0), where the row-wise shape is NOT 0, but ONLY the column-wise
                # shape is 0.
                ndarray = self._build_ndarray_from_timeseries_by_dtypes(target_timeseries, np.int64)
                print(ndarray.shape)
                # (3, 0)
        """
        full_df = timeseries.to_dataframe(copy=False)
        extracted_df = full_df.select_dtypes(include=dtype)
        return extracted_df.to_numpy(copy=False)

    @property
    def samples(self):
        return self._samples
