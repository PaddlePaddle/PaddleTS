# !/usr/bin/env python3
# -*- coding:utf-8 -*-

from paddlets.datasets import TSDataset, TimeSeries
from paddlets.logger.logger import Logger, raise_if

import numpy as np
from typing import List, Dict, Union

logger = Logger(__name__)


class MLDataset(object):
    """
    Dataset for Anomaly machine learning models.

    Note that any unused (target / known) columns should be removed from the TSDataset before handled by this class.

    Args:
        rawdataset(TSDataset): Raw TSDataset to be converted.
        in_chunk_len(int): The size of the loopback window, i.e., the number of time steps feed to the model.
        sampling_stride(int, optional): Time steps to stride over the i-th sample and (i+1)-th sample. More precisely,
            let `t` be the time index of observed cov time series, `t[i]` be the start time of the i-th sample,
            `t[i+1]` be the start time of the (i+1)-th sample, then `sampling_stride` is equal to `t[i+1] - t[i]`.

    Examples:
        .. code-block:: python

            # 1) in_chunk_len examples
            # Given:
            tsdataset.observed_cov = [0, 1, 2, 3, 4]
            sampling_stride = 1

            # 1.1) If in_chunk_len = 1:
            samples = [
                [0],
                [1],
                [2],
                [3],
                [4]
            ]

            # 1.2) If in_chunk_len = 2:
            samples = [
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 4]
            ]

            # 1.3) If in_chunk_len = 3:
            samples = [
                [0, 1, 2],
                [1, 2, 3],
                [2, 3, 4]
            ]

        .. code-block:: python

            # 4) sampling_stride examples
            # Given:
            tsdataset.observed_cov = [0, 1, 2, 3, 4]
            in_chunk_len = 2

            # 4.1) If sampling_stride = 1:
            samples = [
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 4]
            ]

            # 4.2) If sampling_stride = 2:
            samples = [
                [0, 1],
                [2, 3]
            ]

            # 4.3) If sampling_stride = 3:
            samples = [
                [0, 1],
                [3, 4]
            ]
    """
    def __init__(
        self,
        rawdataset: TSDataset,
        in_chunk_len: int,
        sampling_stride: int
    ):
        self._rawdataset = rawdataset
        self._observed_cov_chunk_len = in_chunk_len
        self._sampling_stride = sampling_stride

        raise_if(rawdataset is None, "rawdataset must be specified.")
        raise_if(rawdataset.get_observed_cov() is None, "rawdataset.observed_cov must not be None.")
        raise_if(len(rawdataset.get_observed_cov().time_index) < 1, "rawdataset.observed_cov length must >= 1.")
        raise_if(in_chunk_len <= 0, f"in_chunk_len ({in_chunk_len}) must be positive integer.")
        raise_if(sampling_stride <= 0, f"sampling_stride ({sampling_stride}) must be positive integer.")

        # Validates input TSDataset, raises if input rawdataset invalid.
        # As anomaly detection only needs observed_cov, thus no need to validate known_cov or target.
        observed_cov_time_index = self._rawdataset.get_observed_cov().time_index
        raise_if(
            len(observed_cov_time_index) < in_chunk_len,
            f"""observed_cov timeseries length ({len(self._rawdataset.get_observed_cov().time_index)}) must be longer 
            than or equal to in_chunk_len ({in_chunk_len}) to ensure that at least one sample can be built."""
        )

        self._samples = self._build_samples()

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        return self._samples[idx]

    def _build_samples(self) -> List[Dict[str, np.ndarray]]:
        """
        Internal method, build samples for anomaly detection.

        Returns:
            List[Dict[str, np.ndarray]]: A list of samples.

        Examples:
            .. code-block:: python

                # Given:
                in_chunk_len = 2
                sampling_stride = 1
                rawdataset = {
                    "target": None,
                    "known_cov": None,
                    "observed_cov": [
                        [0, 0.0, 0],
                        [1, 10.0, 100],
                        [2, 20.0, 200],
                        [3, 30.0, 300],
                        [4, 40.0, 400]
                    ],
                    "static_cov": {"static0": 0, "static1": 1.0, "static2": 2, "static3": 3.0}
                }

            .. code-block:: python

                # Built samples:
                samples = [
                    # sample[0]
                    {
                        ################
                        # observed_cov #
                        ################
                        # numeric
                        # row = _observed_cov_chunk_len = 2
                        # col = len(TSDataset._observed_cov.data.select_dtypes(include=np.float32)) = 1
                        "observed_cov_numeric": [
                            [0.0],
                            [10.0]
                        ],

                        # categorical
                        # row = _observed_cov_chunk_len = 2
                        # col = len(TSDataset._observed_cov.data.select_dtypes(include=np.int64)) = 2
                        "observed_cov_categorical": [
                            [0, 0],
                            [1, 100]
                        ]
                    },
                    # sample[1]
                    {
                        "observed_cov_numeric": [
                            [10.0],
                            [20.0]
                        ],
                        "observed_cov_categorical": [
                            [1, 100],
                            [2, 200]
                        ]
                    },
                    # sample[2]
                    {
                        "observed_cov_numeric": [
                            [20.0],
                            [30.0]
                        ],
                        "observed_cov_categorical": [
                            [2, 200],
                            [3, 300]
                        ]
                    },
                    # sample[3] (i.e. last sample)
                    {
                        "observed_cov_numeric": [
                            [30.0],
                            [40.0]
                        ],
                        "observed_cov_categorical": [
                            [3, 300],
                            [4, 400]
                        ]
                    }
                ]
        """
        numeric_dtype = np.float32
        categorical_dtype = np.int64

        # observed cov (possibly be None)
        observed_cov_ts = self._rawdataset.get_observed_cov()
        observed_cov_numeric_ndarray = self._build_ndarray_from_timeseries_by_dtype(
            timeseries=observed_cov_ts,
            dtype=numeric_dtype
        )
        observed_cov_categorical_ndarray = self._build_ndarray_from_timeseries_by_dtype(
            timeseries=observed_cov_ts,
            dtype=categorical_dtype
        )

        samples = []
        # `observed_cov_tail` refers to the tail index of observed_cov chunk for each sample.
        observed_cov_tail = self._observed_cov_chunk_len - 1
        max_allowed_idx = len(self._rawdataset.get_observed_cov().time_index) - 1
        while observed_cov_tail <= max_allowed_idx:
            sample = dict()
            # observed_cov
            if 0 not in observed_cov_numeric_ndarray.shape:
                # numeric
                sample["observed_cov_numeric"] = self._build_observed_cov_for_single_sample(
                    observed_cov_tail=observed_cov_tail,
                    observed_cov_ndarray=observed_cov_numeric_ndarray
                )
            if 0 not in observed_cov_categorical_ndarray.shape:
                # categorical
                sample["observed_cov_categorical"] = self._build_observed_cov_for_single_sample(
                    observed_cov_tail=observed_cov_tail,
                    observed_cov_ndarray=observed_cov_categorical_ndarray
                )
            samples.append(sample)
            observed_cov_tail += self._sampling_stride
        return samples

    def _build_observed_cov_for_single_sample(
        self,
        observed_cov_tail: int,
        observed_cov_ndarray: np.ndarray
    ) -> np.ndarray:
        """
        Internal method, builds an observed_cov chunk for a single sample.

        Args:
            observed_cov_tail(int): the tail idx of observed_cov chunk of the same sample.
            observed_cov_ndarray(np.ndarray, optional): an np.ndarray matrix, as it comes from
                observed_cov_ts.to_numpy(), its value will be None if the passed known_cov_ts is None.

        Returns:
            np.ndarray: built observed cov chunk for the current single sample.
        """
        return observed_cov_ndarray[observed_cov_tail - self._observed_cov_chunk_len + 1:observed_cov_tail + 1]

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
