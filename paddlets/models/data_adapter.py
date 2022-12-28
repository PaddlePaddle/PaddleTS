# !/usr/bin/env python3
# -*- coding:utf-8 -*-

import paddle
import numpy as np
import pandas as pd
import math
from typing import List, Dict, Union, Callable, Optional, Tuple

from paddlets.datasets import TSDataset, TimeSeries
from paddlets.logger.logger import Logger, raise_if, raise_if_not

logger = Logger(__name__)


class SampleDataset(paddle.io.Dataset):
    """
    An implementation of paddle Dataset.

    The default time_window assumes each sample contains X (i.e. in_chunk), skip_chunk, and
    Y (i.e. out_chunk).

    If caller explicitly passes time_window parameter in, and time_window upper bound is larger than
    max standard timeseries (possibly be target or observed_cov) idx len, it means that each built sample will only
    contain X (i.e. in_chunk), but will not contain skip_chunk or Y (i.e. out_chunk).

    Args:
        rawdataset(TSDataset): Raw TSDataset to be converted.
        in_chunk_len(int): The size of the loopback window, i.e., the number of time steps feed to the model.
        out_chunk_len(int): The size of the forecasting horizon, i.e., the number of time steps output by the model.
        skip_chunk_len(int): Optional, the number of time steps between in_chunk and out_chunk for a single sample.
            The skip chunk is neither used as a feature (i.e. X) nor a label (i.e. Y) for a single sample. By
            default, it will NOT skip any time steps.
        sampling_stride(int): Time steps to stride over the i-th sample and (i+1)-th sample. More precisely,
            let `t` be the time index of target time series, `t[i]` be the start time of the i-th sample,
            `t[i+1]` be the start time of the (i+1)-th sample, then `sampling_stride` represents the result of
            `t[i+1] - t[i]`.
        fill_last_value(float, optional): The value used for filling last sample. Set to None if no need to fill.
            For any type `t` of fill_last_value that np.issubdtype(type(t), np.floating) or
            np.issubdtype(type(t), np.integer) is True are valid.
        time_window(Tuple, optional): A two-element-tuple-shaped time window that allows adapter to build samples.
            time_window[0] refers to the window lower bound, while time_window[1] refers to the window upper bound.
            Each element in the left-closed-and-right-closed interval refers to the TAIL index of each sample.

    Examples:
        .. code-block:: python

            # 1) in_chunk_len examples
            # Given:
            tsdataset.target = [0, 1, 2, 3, 4]
            skip_chunk_len = 0
            out_chunk_len = 1

            # 1.1) If in_chunk_len = 1, sample[0]:
            # X -> skip_chunk -> Y
            # (0) -> () -> (1)

            # 1.2) If in_chunk_len = 2, sample[0]:
            # X -> skip_chunk -> Y
            # (0, 1) -> () -> (2)

            # 1.3) If in_chunk_len = 3, sample[0]:
            # X -> skip_chunk -> Y
            # (0, 1, 2) -> () -> (3)

        .. code-block:: python

            # 2) out_chunk_len examples
            # Given:
            tsdataset.target = [0, 1, 2, 3, 4]
            in_chunk_len = 1
            skip_chunk_len = 0

            # 2.1) If out_chunk_len = 1, sample[0]:
            # X -> skip_chunk -> Y
            # (0) -> () -> (1)

            # 2.2) If out_chunk_len = 2, sample[0]:
            # X -> skip_chunk -> Y
            # (0) -> () -> (1, 2)

            # 2.3) If out_chunk_len = 3, sample[0]:
            # X -> skip_chunk -> Y
            # (0) -> () -> (1, 2, 3)

        .. code-block:: python

            # 3) skip_chunk_len examples
            # Given:
            tsdataset.target = [0, 1, 2, 3, 4]
            in_chunk_len = 1
            out_chunk_len = 1

            # 3.1) If skip_chunk_len = 0, sample[0]:
            # X -> skip_chunk -> Y
            # (0) -> () -> (1)

            # 3.2) If skip_chunk_len = 1, sample[0]:
            # X -> skip_chunk -> Y
            # (0) -> (1) -> (2)

            # 3.3) If skip_chunk_len = 2, sample[0]:
            # X -> skip_chunk -> Y
            # (0) -> (1, 2) -> (3)

            # 3.4) If skip_chunk_len = 3, sample[0]:
            # X -> skip_chunk -> Y
            # (0) -> (1, 2, 3) -> (4)

        .. code-block:: python

            # 4) sampling_stride examples
            # Given:
            tsdataset.target = [0, 1, 2, 3, 4]
            in_chunk_len = 1
            skip_chunk_len = 0
            out_chunk_len = 1

            # 4.1) If sampling_stride = 1, samples:
            # X -> skip_chunk -> Y
            # (0) -> () -> (1)
            # (1) -> () -> (2)
            # (2) -> () -> (3)
            # (3) -> () -> (4)

            # 4.2) If sampling_stride = 2, samples:
            # X -> skip_chunk -> Y
            # (0) -> () -> (1)
            # (2) -> () -> (3)

            # 4.3) If sampling_stride = 3, samples:
            # X -> skip_chunk -> Y
            # (0) -> () -> (1)
            # (3) -> () -> (4)

        .. code-block:: python

            # 5) time_window examples:
            # 5.1) The default time_window calculation formula is as follows:
            # time_window[0] = 0 + in_chunk_len + skip_chunk_len + (out_chunk_len - 1)
            # time_window[1] = max_target_idx
            #
            # Given:
            tsdataset.target = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            in_chunk_len = 4
            skip_chunk_len = 3
            out_chunk_len = 2
            sampling_stride = 1

            # The following equation holds:
            max_target_idx = tsdataset.target[-1] = 10

            # The default time_window is calculated as follows:
            time_window[0] = 0 + 2 + 3 + (4 - 1) = 5 + 3 = 8
            time_window[1] = max_target_idx = 10
            time_window = (8, 10)

            # 3 samples will be built in total:
            X -> Y
            (0, 1, 2, 3) -> (7, 8)
            (1, 2, 3, 4) -> (8, 9)
            (2, 3, 4, 5) -> (9, 10)


            # 5.2) Each element in time_window refers to the TAIL index of each sample, but NOT the HEAD index.
            # The following two scenarios shows how to pass in the expected time_window parameter to build samples.
            # Given:
            tsdataset.target = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            in_chunk_len = 4
            skip_chunk_len = 3
            out_chunk_len = 2

            # Scenario 5.2.1 - Suppose the following training samples are expected to be built:
            # X -> Y
            # (0, 1, 2, 3) -> (7, 8)
            # (1, 2, 3, 4) -> (8, 9)
            # (2, 3, 4, 5) -> (9, 10)

            # The 1st sample's tail index is 8
            # The 2nd sample's tail index is 9
            # The 3rd sample's tail index is 10

            # Thus, the time_window parameter should be as follows:
            time_window = (8, 10)

            # All other time_window showing up as follows are NOT correct:
            time_window = (0, 2)
            time_window = (0, 10)

            # Scenario 5.2.2 - Suppose the following predict sample is expected to be built:
            # X -> Y
            # (7, 8, 9, 10) -> (14, 15)

            # The first (i.e. the last) sample's tail index is 15;

            # Thus, the time_window parameter should be as follows:
            time_window = (15, 15)

            # 5.3) The calculation formula of the max allowed time_window upper bound is as follows:
            # time_window[1] <= len(tsdataset.target) - 1 + skip_chunk_len + out_chunk_len
            # The reason is that the built paddle.io.Dataset is used for a single call of :func: `model.predict`, as
            # it only allow for a single predict sample, any time_window upper bound larger than a single predict
            # sample's TAIL index will not be allowed because there is not enough target time series to build past
            # target time series chunk.
            #
            # Given:
            tsdataset.target = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            in_chunk_len = 4
            skip_chunk_len = 3
            out_chunk_len = 2

            # For a single :func:`model.predict` call:
            X = in_chunk = (7, 8, 9, 10)

            # max allowed time_window[1] is calculated as follows:
            time_window[1] <= len(tsdataset) - 1 + skip_chunk_len + out_chunk_len = 11 - 1 + 3 + 2 = 15

            # Note that time_window[1] (i.e. 15) is larger than the max_target_idx (i.e. 10), but this time_window
            # upper bound is still valid, because predict sample does not need skip_chunk (i.e.  [11, 12, 13]) or
            # out_chunk (i.e. [14, 15]).

            # Any values larger than 15 (i.e. 16) is invalid, because the existing target time series is NOT long
            # enough to build X for the prediction sample, see following example:
            # Given:
            time_window = (16, 16)

            # The calculated out_chunk = (15, 16)
            # The calculated skip_chunk = (12, 13, 14)

            # Thus, the in_chunk should be [8, 9, 10, 11]
            # However, the tail index of the calculated in_chunk 11 is beyond the max target time series
            # (i.e. tsdataset.target[-1] = 10), so current target timeseries cannot provide 11 to build this sample.
        """
    def __init__(
        self,
        rawdataset: TSDataset,
        in_chunk_len: int = 1,
        out_chunk_len: int = 0,
        skip_chunk_len: int = 0,
        sampling_stride: int = 1,
        fill_last_value: Optional[Union[np.floating, np.integer]] = None,
        time_window: Optional[Tuple[int, int]] = None
    ):
        super(SampleDataset, self).__init__()

        raise_if(rawdataset is None, "rawdataset must not be None.")
        raise_if(
            (rawdataset.target is None) and (rawdataset.observed_cov is None),
            "TSDataset.target and TSDataset.observed_cov cannot be None at same time."
        )
        raise_if(in_chunk_len <= 0, f"in_chunk_len ({in_chunk_len}) must > 0.")
        raise_if(skip_chunk_len < 0, f"skip_chunk_len ({skip_chunk_len}) must >= 0.")
        raise_if(out_chunk_len < 0, f"out_chunk_len ({out_chunk_len}) must >= 0.")
        raise_if(sampling_stride <= 0, f"sampling_stride ({sampling_stride}) must > 0.")
        raise_if(
            (time_window is not None) and (fill_last_value is not None),
            f"time_window ({time_window}) must not be set if fill_last_value ({fill_last_value}) is not None."
        )

        # models.utils::check_tsdataset() already guarantee that all float-like type will be converted
        # to the standard np.float32, similarly, all int-like type will be converted to the standard np.int64.
        # so here only need to focus on the below 2 std dtypes.
        self._numeric_dtype = np.float32
        self._categorical_dtype = np.int64

        self._rawdataset = rawdataset
        self._in_chunk_len = in_chunk_len
        self._out_chunk_len = out_chunk_len
        self._skip_chunk_len = skip_chunk_len
        self._sampling_stride = sampling_stride
        self._fill_last_value = fill_last_value
        self._std_timeseries_name, self._std_timeindex = self._compute_std_timeindex()
        self._validate_std_timeindex(time_window=time_window)
        self._time_window = time_window if time_window is not None else self._compute_default_time_window()
        self._validate_time_window()

        # validate dtype before filling timeseries.
        # [Rule 1] cannot fill int-like data in float-like timeseries.
        # [Rule 2] Similarly, cannot fill float-like data in int-like timeseries.
        ts_list = [
            ("target", self._rawdataset.target),
            ("known_cov", self._rawdataset.known_cov),
            ("observed_cov", self._rawdataset.observed_cov)
        ]
        for ts_tuple in ts_list:
            if ts_tuple[1] is None:
                continue
            ts_name, ts = ts_tuple
            numeric_df = ts.to_dataframe(copy=False).select_dtypes(include=self._numeric_dtype)
            categorical_df = ts.to_dataframe(copy=False).select_dtypes(include=self._categorical_dtype)
            # Reason why use np.issubdtype (but not use builtin isinstance) is that for all int-like / float-like
            # types, numpy.issubdtype will always return True (good), while isinstance will return False for python
            # builtin int/float type, along with np.int / np.float dtype (bad). So np.issubdtype() is more generic than
            # isinstance().
            # Below are types that both np.issubdtype and instance will return True:
            # np.float16
            # np.float32
            # np.float64
            # np.floating (used as base standard type for all float-like types)
            # python builtin int
            # np.int8
            # np.int16
            # np.int32
            # np.int64
            # np.integer
            #
            # Below are types that only np.issubdtype will return True, but isinstance will return False:
            # python builtin float
            # np.float (deprecated by numpy 1.20+)
            # python builtin int
            # np.int (deprecated by numpy 1.20+)
            if np.issubdtype(type(self._fill_last_value), np.integer):
                raise_if(
                    len(numeric_df.columns) > 0,
                    f"If numpy.issubdtype(fill_last_value, np.integer) is True, then " +
                    f"TSDataset.{ts_name} must not contain such float-like dtype columns: {numeric_df.columns}. " +
                    f"actual fill_last_value type: {type(self._fill_last_value)}"
                )
            if np.issubdtype(type(self._fill_last_value), np.floating):
                raise_if(
                    len(categorical_df.columns) > 0,
                    f"If numpy.issubdtype(fill_last_value, np.floating) is True, then "
                    f"TSDataset.{ts_name} must not contain such int-like dtype columns: {categorical_df.columns}. " +
                    f"actual fill_last_value type: {type(self._fill_last_value)}"
                )

        # fill timeseries.
        if self._fill_last_value is not None:
            # call TSDataset.copy() to avoid inplace fill.
            self._rawdataset = self._fill_tsdataset(TSDataset.copy(self._rawdataset))
            # after filled, std time index and time window must be re-computed based on filled timeseries.
            # Note that self._std_timeseries_name no need to be re computed.
            _, self._std_timeindex = self._compute_std_timeindex()
            self._validate_std_timeindex(time_window=self._time_window)

            # pre-check already guarantee that user cannot set fill_last_value and time_window at same time.
            # So we can ensure that time_window here must be default value.
            self._time_window = self._compute_default_time_window()
            self._validate_time_window()

        # perform the rest of the timeseries validation.
        self._validate_target_timeseries()
        self._validate_known_cov_timeseries()
        self._validate_observed_cov_timeseries()

        self.samples = self._build_samples()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        return self.samples[idx]

    def _build_samples(self) -> List[Dict[str, np.ndarray]]:
        """
        Internal method, builds samples.

        Returns:
            List[Dict[str, np.ndarray]]: A list of samples.

        Examples:
            .. code-block:: python

                # Given:
                in_chunk_len = 2
                skip_chunk_len = 1
                out_chunk_len = 2
                sampling_stride = 1
                time_window = (4, 7)
                rawdataset = {
                    "target": [
                        [0, 0.0],
                        [1, 10.0],
                        [2, 20.0],
                        [3, 30.0],
                        [4, 40.0],
                        [5, 50.0],
                        [6, 60.0],
                        [7, 70.0]
                    ],
                    "known_cov": [
                        [0, 0.0, 0],
                        [10, 100.0, 1000],
                        [20, 200.0, 2000],
                        [30, 300.0, 3000],
                        [40, 400.0, 4000],
                        [50, 500.0, 5000],
                        [60, 600.0, 6000],
                        [70, 700.0, 7000],
                        [80, 800.0, 8000]
                    ],
                    "observed_cov": [
                        [0],
                        [-1],
                        [-2],
                        [-3],
                        [-4],
                        [-5],
                        [-6],
                        [-7]
                    ],
                    "static_cov": {"static0": 0, "static1": 1.0, "static2": 2, "static3": 3.0}
                }

            .. code-block:: python

                # Built samples:
                samples = [
                    # sample[0]
                    {
                        ###############
                        # past_target #
                        ###############
                        # row = _target_in_chunk_len = 2
                        # col = len(TSDataset._target.data.columns) = 2
                        "past_target": [
                            [0.0, 0.0],
                            [1.0, 10.0]
                        ],

                        #################
                        # future_target #
                        #################
                        # Note that skip_chunk_len = 1 time steps are skipped between past_target and future_target.
                        # row = _target_out_chunk_len = 2
                        # col = len(TSDataset._target.data.columns) = 2
                        "future_target": [
                            [3.0, 30.0],
                            [4.0, 40.0]
                        ],

                        #############
                        # known_cov #
                        #############
                        # numeric
                        # Note that skip_chunk_len = 1 time steps are skipped between past_target and future_target.
                        # row = _target_in_chunk_len + _target_out_chunk_len = 2 + 2 = 4
                        # col = len(TSDataset._known_cov.data.select_dtypes(include=np.float32)) = 1
                        "known_cov_numeric": [
                            [0.0],
                            [100.0],
                            # Note: skip_chunk [20, 200.0, 2000] is skipped.
                            [300.0],
                            [400.0]
                        ],

                        # categorical
                        # Note that skip_chunk_len = 1 time steps are skipped between past_target and future_target.
                        # row = _target_in_chunk_len + _target_out_chunk_len = 2 + 2 = 4
                        # col = len(TSDataset._known_cov.data.select_dtypes(include=np.int64)) = 2
                        "known_cov_categorical": [
                            [0, 0],
                            [10, 1000],
                            # Note: skip_chunk [20, 200.0, 2000] is skipped.
                            [30, 3000],
                            [40, 4000]
                        ],

                        ################
                        # observed_cov #
                        ################
                        # numeric (None)
                        # NOTE: As no float-dtype column in TSDataset._observed_cov, thus the given TSDataset can NOT
                        # build observed_cov_numeric features, but can ONLY build observed_cov_categorical.

                        # categorical
                        # row = _observed_cov_chunk_len = 2
                        # col = len(TSDataset._observed_cov.data.select_dtypes(include=np.int64)) = 1
                        "observed_cov_categorical": [
                            [0],
                            [-1]
                        ],

                        ##############
                        # static_cov #
                        ##############
                        # numeric
                        # row = (fixed) 1.
                        # col = len([t for t in TSDataset._static_cov if type(t[1]) in [np.float32, float]]) = 2
                        "static_cov_numeric": [
                            # key-wise ascending sorted data.
                            [1.0, 3.0]
                        ],

                        # categorical
                        # row = (fixed) 1.
                        # col = len([t for t in TSDataset._static_cov if type(t[1]) in [np.int64, int]]) = 2
                        "static_cov_categorical": [
                            # key-wise ascending sorted data.
                            [0, 2]
                        ]
                    },
                    # sample[1]
                    {
                        "past_target": [
                            [1.0, 10.0],
                            [2.0, 20.0]
                        ],
                        "future_target": [
                            [4.0, 40.0],
                            [5.0, 50.0]
                        ],
                        "known_cov_numeric": [
                            [100.0],
                            [200.0],
                            [400.0],
                            [500.0]
                        ],
                        "known_cov_categorical": [
                            [10, 1000],
                            [20, 2000],
                            [40, 4000],
                            [50, 5000]
                        ],
                        "observed_cov_categorical": [
                            [-1],
                            [-2]
                        ],
                        "static_cov_numeric": [
                            [1.0, 3.0]
                        ],
                        "static_cov_categorical": [
                            [0, 2]
                        ]
                    },
                    # sample[2]
                    {
                        "past_target": [
                            [2.0, 20.0],
                            [3.0, 30.0]
                        ],
                        "future_target": [
                            [5.0, 50.0],
                            [6.0, 60.0]
                        ],
                        "known_cov_numeric": [
                            [200.0],
                            [300.0],
                            [500.0],
                            [600.0]
                        ],
                        "known_cov_categorical": [
                            [20, 2000],
                            [30, 3000],
                            [50, 5000],
                            [60, 6000]
                        ],
                        "observed_cov_categorical": [
                            [-2],
                            [-3]
                        ],
                        "static_cov_numeric": [
                            [1.0, 3.0]
                        ],
                        "static_cov_categorical": [
                            [0, 2]
                        ]
                    },
                    # sample[3] (i.e. last sample, future_target tail index = 7 reaches time_window upper bound)
                    {
                        "past_target": [
                            [3.0, 30.0],
                            [4.0, 40.0]
                        ],
                        "future_target": [
                            [6.0, 60.0],
                            [7.0, 70.0]
                        ],
                        "known_cov_numeric": [
                            [300.0],
                            [400.0],
                            [600.0],
                            [700.0]
                        ],
                        "known_cov_categorical": [
                            [30, 3000],
                            [40, 4000],
                            [60, 6000],
                            [70, 7000]
                        ],
                        "observed_cov_categorical": [
                            [-3],
                            [-4]
                        ],
                        "static_cov_numeric": [
                            [1.0, 3.0]
                        ],
                        "static_cov_categorical": [
                            [0, 2]
                        ]
                    }
                ]

            .. code-block:: python

                # Case 1 - in_chunk_len examples
                # Given:
                tsdataset.target = [0, 1, 2, 3, 4]
                skip_chunk_len = 0
                out_chunk_len = 1

                # If in_chunk_len = 1, sample[0]:
                # X -> skip_chunk -> Y
                # (0) -> () -> (1)

                # If in_chunk_len = 2, sample[0]:
                # X -> skip_chunk -> Y
                # (0, 1) -> () -> (2)

                # If in_chunk_len = 3, sample[0]:
                # X -> skip_chunk -> Y
                # (0, 1, 2) -> () -> (3)

            .. code-block:: python

                # Case 2 - out_chunk_len examples
                # Given:
                tsdataset.target = [0, 1, 2, 3, 4]
                in_chunk_len = 1
                skip_chunk_len = 0

                # If out_chunk_len = 1, sample[0]:
                # X -> skip_chunk -> Y
                # (0) -> () -> (1)

                # If out_chunk_len = 2, sample[0]:
                # X -> skip_chunk -> Y
                # (0) -> () -> (1, 2)

                # If out_chunk_len = 3, sample[0]:
                # X -> skip_chunk -> Y
                # (0) -> () -> (1, 2, 3)

            .. code-block:: python

                # Case 3 - skip_chunk_len examples
                # Given:
                tsdataset.target = [0, 1, 2, 3, 4]
                in_chunk_len = 1
                out_chunk_len = 1

                # If skip_chunk_len = 0, sample[0]:
                # X -> skip_chunk -> Y
                # (0) -> () -> (1)

                # If skip_chunk_len = 1, sample[0]:
                # X -> skip_chunk -> Y
                # (0) -> (1) -> (2)

                # If skip_chunk_len = 2, sample[0]:
                # X -> skip_chunk -> Y
                # (0) -> (1, 2) -> (3)

                # If skip_chunk_len = 3, sample[0]:
                # X -> skip_chunk -> Y
                # (0) -> (1, 2, 3) -> (4)

            .. code-block:: python

                # Case 4 - sampling_stride examples
                # Given:
                tsdataset.target = [0, 1, 2, 3, 4]
                in_chunk_len = 1
                skip_chunk_len = 0
                out_chunk_len = 1

                # If sampling_stride = 1:
                # samples:
                # X -> skip_chunk -> Y
                # (0) -> () -> (1)
                # (1) -> () -> (2)
                # (2) -> () -> (3)
                # (3) -> () -> (4)

                # If sampling_stride = 2, sample[0]:
                # samples:
                # X -> skip_chunk -> Y
                # (0) -> () -> (1)
                # (2) -> () -> (3)

                # If sampling_stride = 3, sample[0]:
                # samples:
                # X -> skip_chunk -> Y
                # (0) -> () -> (1)
                # (3) -> () -> (4)
                """
        # target (possibly be None for anomaly models)
        target_ts = self._rawdataset.target
        target_timeindex_offset = 0
        target_ndarray = None
        if target_ts is not None:
            target_timeindex_offset = self._compute_timeindex_offset(target_ts.time_index)
            target_ndarray = self._build_ndarray_from_timeseries_by_dtype(
                timeseries=target_ts,
                dtype=self._numeric_dtype
            )

        # known cov (possibly be None)
        known_cov_ts = self._rawdataset.known_cov
        known_cov_timeindex_offset = 0
        known_cov_numeric_ndarray = None
        known_cov_categorical_ndarray = None
        if known_cov_ts is not None:
            known_cov_timeindex_offset = self._compute_timeindex_offset(known_cov_ts.time_index)
            known_cov_numeric_ndarray = self._build_ndarray_from_timeseries_by_dtype(
                timeseries=known_cov_ts,
                dtype=self._numeric_dtype
            )
            known_cov_categorical_ndarray = self._build_ndarray_from_timeseries_by_dtype(
                timeseries=known_cov_ts,
                dtype=self._categorical_dtype
            )

        # observed cov (possibly be None)
        observed_cov_ts = self._rawdataset.observed_cov
        observed_cov_timeindex_offset = 0
        observed_cov_numeric_ndarray = None
        observed_cov_categorical_ndarray = None
        if observed_cov_ts is not None:
            observed_cov_timeindex_offset = self._compute_timeindex_offset(observed_cov_ts.time_index)
            observed_cov_numeric_ndarray = self._build_ndarray_from_timeseries_by_dtype(
                timeseries=observed_cov_ts,
                dtype=self._numeric_dtype
            )
            observed_cov_categorical_ndarray = self._build_ndarray_from_timeseries_by_dtype(
                timeseries=observed_cov_ts,
                dtype=self._categorical_dtype
            )

        # static cov (possibly be None)
        static_cov = self._rawdataset.static_cov
        pre_computed_static_cov_numeric_for_single_sample = None
        pre_computed_static_cov_categorical_for_single_sample = None
        if static_cov is not None:
            static_cov_numeric = dict()
            static_cov_categorical = dict()
            for k, v in static_cov.items():
                if type(v) in {self._numeric_dtype, float}:
                    # built-in float type will be implicitly converted to numpy.float32 dtype.
                    static_cov_numeric[k] = self._numeric_dtype(v)
                if type(v) in {self._categorical_dtype, int}:
                    # built-in int type will be implicitly converted to numpy.int64 dtype.
                    static_cov_categorical[k] = self._categorical_dtype(v)

            pre_computed_static_cov_numeric_for_single_sample = self._build_static_cov_for_single_sample(
                static_cov_dict=static_cov_numeric
            )
            pre_computed_static_cov_categorical_for_single_sample = self._build_static_cov_for_single_sample(
                static_cov_dict=static_cov_categorical
            )

        samples = []
        curr_sample_tail = self._time_window[0]
        # Because _time_window is left-closed-right-closed, thus using "<=" operator rather than "<".
        while curr_sample_tail <= self._time_window[1]:
            sample = dict()

            # target (future_target + past_target)
            if target_ts is not None:
                if 0 not in target_ndarray.shape:
                    # future target
                    if (self._time_window[1] <= len(self._std_timeindex) - 1) and (self._out_chunk_len > 0):
                        # ONLY fit api needs future_target, predict api does not need it.
                        sample["future_target"] = self._build_future_target_for_single_sample(
                            curr_sample_tail=curr_sample_tail,
                            timeindex_offset=target_timeindex_offset,
                            target_ndarray=target_ndarray
                        )

                    # past target
                    sample["past_target"] = self._build_past_target_for_single_sample(
                        curr_sample_tail=curr_sample_tail,
                        timeindex_offset=target_timeindex_offset,
                        target_ndarray=target_ndarray
                    )

            # known_cov
            if known_cov_ts is not None:
                # numeric
                if 0 not in known_cov_numeric_ndarray.shape:
                    sample["known_cov_numeric"] = self._build_known_cov_for_single_sample(
                        curr_sample_tail=curr_sample_tail,
                        timeindex_offset=known_cov_timeindex_offset,
                        known_cov_ndarray=known_cov_numeric_ndarray
                    )
                # categorical
                if 0 not in known_cov_categorical_ndarray.shape:
                    sample["known_cov_categorical"] = self._build_known_cov_for_single_sample(
                        curr_sample_tail=curr_sample_tail,
                        timeindex_offset=known_cov_timeindex_offset,
                        known_cov_ndarray=known_cov_categorical_ndarray
                    )

            # observed_cov
            if observed_cov_ts is not None:
                # numeric
                if 0 not in observed_cov_numeric_ndarray.shape:
                    sample["observed_cov_numeric"] = self._build_observed_cov_for_single_sample(
                        curr_sample_tail=curr_sample_tail,
                        timeindex_offset=observed_cov_timeindex_offset,
                        observed_cov_ndarray=observed_cov_numeric_ndarray
                    )
                # categorical
                if 0 not in observed_cov_categorical_ndarray.shape:
                    sample["observed_cov_categorical"] = self._build_observed_cov_for_single_sample(
                        curr_sample_tail=curr_sample_tail,
                        timeindex_offset=observed_cov_timeindex_offset,
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

            curr_sample_tail += self._sampling_stride
        return samples

    def _compute_std_timeindex(self) -> Tuple[str, pd.DatetimeIndex]:
        """
        Internal method, compute which time_index will be used as the standard time index.

        Currently, there are 3 model type, forecasting, representation and anomaly. For forecasting and representation
        models, it will use TSDataset.target.time_index as standard time index. For anomaly models, it will instead
        use TSDataset.observed_cov.time_index as standard time index.

        Therefore, before we make any computation (compute default time window, build model, etc.), we must firstly
        determine the standard time index.

        Returns:
            Tuple[str, pd.DatetimeIndex]: (timeseries_name, std_timeindex), where timeseries_name can be either
                target or observed_cov.
        """
        if self._rawdataset.target is not None:
            return "target", self._rawdataset.target.time_index
        return "observed_cov", self._rawdataset.observed_cov.time_index

    def _validate_std_timeindex(self, time_window: Optional[Tuple[int, int]]) -> None:
        """
        Internal method, ensuring the standard time index must be long enough to build at least one sample, raises
        if invalid.

        Note: Validating std_timeindex depends on time_window. However, instead of using self._time_window member,
        we explicitly pass time_window arg into this method, The reason is that for known reasons, initializing
        self._std_timeindex must prior to self._time_window, so if we need to use time_window before
        self._time_window was initialized, the only way is to pass the user provided time_window into this method.
        """
        valid_time_index_type = {pd.DatetimeIndex, pd.RangeIndex}
        raise_if(
            type(self._std_timeindex) not in valid_time_index_type,
            f"type(TSDataset.{self._std_timeseries_name}.time_index) ({type(self._std_timeindex)}) must be one of " +
            f"{valid_time_index_type}."
        )

        if time_window is None:
            # indicates that user did NOT explicitly set a valid time_window, thus no further check, return.
            return

        # need further check.
        one_sample_len = self._in_chunk_len + self._skip_chunk_len + self._out_chunk_len
        max_std_timeindex_idx = len(self._std_timeindex) - 1
        raise_if(
            (time_window[1] < max_std_timeindex_idx) and
            (self._fill_last_value is None) and
            (len(self._std_timeindex) < one_sample_len),
            f"If time_window[1] ({time_window[1]}) <= len(TSDataset.{self._std_timeseries_name}) - 1 " +
            f"({len(self._std_timeindex) - 1}) and "
            f"fill_last_value ({self._fill_last_value}) is None, then " +
            f"TSDataset.{self._std_timeseries_name} length ({len(self._std_timeindex)}) must >= " +
            f"in_chunk_len ({self._in_chunk_len}) + " +
            f"skip_chunk_len ({self._skip_chunk_len}) + " +
            f"out_chunk_len ({self._out_chunk_len}) " +
            f"to ensure that at least one sample can be built."
        )

    def _compute_default_time_window(self) -> Tuple[int, int]:
        """
        Internal method, compute default time window based on self._std_timeindex.

        Default time_window assumes each sample contains both X, skip_chunk and Y.
        """
        default_min_window = self._in_chunk_len + self._skip_chunk_len + self._out_chunk_len - 1
        # Note, this std time index is filled if fill_last_value is not None.
        default_max_window = len(self._std_timeindex) - 1
        return default_min_window, default_max_window

    def _validate_time_window(self) -> None:
        """
        Internal method, check if time_window is valid.

        Let w be the abbreviation of time_window, then:
        Case 1 - If built samples only contain X, but not contain Y, the following equation must hold:
        w[0] == w[1] == (max_std_idx + skip_chunk_len + out_chunk_len)

        Case 2 - If built samples contains both X and Y, then following Inequality must hold:
        (in_chunk_len + skip_chunk_len + out_chunk_len) - 1 <= w[0] <= w[1] <= (len(std_timeindex) - 1)
        """
        max_std_idx = len(self._std_timeindex) - 1

        # adapter does NOT fill samples for predict api, but will only fill samples for fit api.
        raise_if(
            (self._fill_last_value is not None) and (self._time_window[1] > max_std_idx),
            f"If fill_last_value ({self._fill_last_value}) is not None, time_window[1] ({self._time_window[1]}) must " +
            f"<= len(TSDataset.{self._std_timeseries_name}) - 1 ({max_std_idx})."
        )

        if self._time_window[1] > max_std_idx:
            # case 1 - samples ONLY contain X, not contain Y.
            only_allowed_window_not_contain_y = max_std_idx + self._skip_chunk_len + self._out_chunk_len
            raise_if_not(
                self._time_window[0] == self._time_window[1] == only_allowed_window_not_contain_y,
                f"if time_window[1] ({self._time_window[1]}) > " +
                f"len(TSDataset.{self._std_timeseries_name}) - 1 ({max_std_idx}), then " +
                f"(time_window[0] == time_window[1] == {only_allowed_window_not_contain_y}) must be True."
            )
        else:
            # case 2 - samples contain X and Y.
            min_allowed_window_contain_y = self._in_chunk_len + self._skip_chunk_len + self._out_chunk_len - 1
            raise_if_not(
                min_allowed_window_contain_y <= self._time_window[0] <= self._time_window[1] <= max_std_idx,
                f"if time_window[1] ({self._time_window[1]}) <= " +
                f"len(TSDataset.{self._std_timeseries_name}) - 1 ({max_std_idx}), then " +
                f"{min_allowed_window_contain_y} <= time_window[0] ({self._time_window[0]}) <= " +
                f"time_window[1] ({self._time_window[1]}) <= " +
                f"len(TSDataset.{self._std_timeseries_name}) - 1 ({max_std_idx}) must be True."
            )

    def _validate_target_timeseries(self) -> None:
        target_ts = self._rawdataset.target
        max_std_idx = len(self._std_timeindex) - 1
        if target_ts is not None:
            # 1 target must be long enough to build samples based on the range specified by time_window.
            sample_end_std_time = self._std_timeindex[min(self._time_window[1], max_std_idx)]
            sample_start_std_idx = \
                self._time_window[0] - \
                self._out_chunk_len - \
                self._skip_chunk_len - \
                self._in_chunk_len + \
                1

            sample_start_std_time = self._std_timeindex[sample_start_std_idx]
            raise_if_not(
                target_ts.start_time <= sample_start_std_time <= sample_end_std_time <= target_ts.end_time,
                f"The inequality must hold: " +
                f"TSDataset.target.start_time ({target_ts.start_time}) <= " +
                f"TSDataset.{self._std_timeseries_name}.time_index" +
                f"[(time_window[0] - out_chunk_len - skip_chunk_len - in_chunk_len + 1)] " +
                f"({sample_start_std_time}) <= " +
                f"TSDataset.{self._std_timeseries_name}.time_index" +
                f"[min(time_window[1], len(TSDataset.{self._std_timeseries_name}.time_index) - 1)] " +
                f"({sample_end_std_time}) <= " +
                f"TSDataset.target.end_time ({target_ts.end_time})."
            )

    def _validate_known_cov_timeseries(self) -> None:
        known_cov_ts = self._rawdataset.known_cov
        max_std_idx = len(self._std_timeindex) - 1
        if known_cov_ts is not None:
            # 1 known_cov must be long enough to build samples based on the range specified by time_window.
            if self._time_window[1] <= max_std_idx:
                sample_end_std_time = self._std_timeindex[self._time_window[1]]
            else:
                # pre-check already guarantee that std_timeindex type must be either pd.RangeIndex or pd.DateTimeIndex.
                if isinstance(self._std_timeindex, pd.DatetimeIndex):
                    exceeded_timesteps = self._time_window[1] - max_std_idx
                    # DateTimeIndex
                    exceeded_timeindex = pd.date_range(
                        start=self._std_timeindex[-1],
                        periods=exceeded_timesteps + 1,
                        freq=pd.infer_freq(self._std_timeindex)
                    )
                else:
                    # RangeIndex
                    # Note: RangeIndex.stop is right-opened, but time_window is right-closed, so stop param must + 1.
                    step = self._std_timeindex.step
                    exceeded_timeindex = pd.RangeIndex(
                        start=self._std_timeindex[-1],
                        stop=(self._time_window[1] + 1) * step,
                        step=step
                    )

                sample_end_std_time = exceeded_timeindex[-1]

            sample_start_std_idx = \
                self._time_window[0] - \
                self._out_chunk_len - \
                self._skip_chunk_len - \
                self._in_chunk_len + \
                1

            sample_start_std_time = self._std_timeindex[sample_start_std_idx]
            raise_if_not(
                known_cov_ts.start_time <= sample_start_std_time <= sample_end_std_time <= known_cov_ts.end_time,
                f"The inequality must hold: " +
                f"TSDataset.known_cov.start_time ({known_cov_ts.start_time}) <= " +
                f"TSDataset.{self._std_timeseries_name}.time_index" +
                f"[(time_window[0] - out_chunk_len - skip_chunk_len - in_chunk_len + 1)] " +
                f"({sample_start_std_time}) <= " +
                f"TSDataset.{self._std_timeseries_name}.time_index[time_window[1]] " +
                f"({sample_end_std_time}) <= " +
                f"TSDataset.known_cov.end_time ({known_cov_ts.end_time})."
            )

    def _validate_observed_cov_timeseries(self) -> None:
        observed_cov_ts = self._rawdataset.get_observed_cov()
        if observed_cov_ts is not None:
            # 1 observed_cov must be long enough to build samples based on the range specified by time_window.
            sample_end_std_idx = self._time_window[1] - self._out_chunk_len - self._skip_chunk_len
            sample_end_std_time = self._std_timeindex[sample_end_std_idx]

            sample_start_std_idx = \
                self._time_window[0] - \
                self._out_chunk_len - \
                self._skip_chunk_len - \
                self._in_chunk_len + \
                1
            sample_start_std_time = self._std_timeindex[sample_start_std_idx]

            raise_if_not(
                observed_cov_ts.start_time <= sample_start_std_time <= sample_end_std_time <= observed_cov_ts.end_time,
                f"The inequality must hold:" +
                f"TSDataset.observed_cov.start_time ({observed_cov_ts.start_time}) <= " +
                f"TSDataset.{self._std_timeseries_name}.time_index" +
                f"[(time_window[0] - out_chunk_len - skip_chunk_len - in_chunk_len + 1)] " +
                f"({sample_start_std_time}) <= " +
                f"TSDataset.{self._std_timeseries_name}.time_index" +
                f"[min(time_window[1], len(TSDataset.{self._std_timeseries_name}.time_index) - 1)] " +
                f"({sample_end_std_time}) <= " +
                f"TSDataset.observed_cov.end_time ({observed_cov_ts.end_time})."
            )

    def _fill_tsdataset(self, tsdataset: TSDataset) -> TSDataset:
        # First, fill target
        if tsdataset.target is not None:
            filled_target_ts = self._fill_timeseries(tsdataset.target)
            tsdataset.set_target(target=filled_target_ts)

        # Second, fill known cov
        if tsdataset.known_cov is not None:
            filled_known_cov_ts = self._fill_timeseries(tsdataset.known_cov)
            tsdataset.set_known_cov(known_cov=filled_known_cov_ts)

        # Third(Last), fill observed cov
        if tsdataset.observed_cov is not None:
            filled_observed_cov_ts = self._fill_timeseries(tsdataset.observed_cov)
            tsdataset.set_observed_cov(observed_cov=filled_observed_cov_ts)
        return tsdataset

    def _fill_timeseries(self, raw_timeseries: TimeSeries) -> TimeSeries:
        # compute how long needs to be filled.
        last_sample_tail_timestamp = self._compute_last_sample_tail_timestamp()
        raw_timeindex = raw_timeseries.time_index
        if last_sample_tail_timestamp <= raw_timeindex[-1]:
            # For example:
            # raw_timeindex = [7:00, 8:00, 9:00, 10:00]
            # last_sample_tail_timestamp = 10:00
            # Thus no need to fill.
            # Another example:
            # raw_timeindex = [7:00, 8:00, 9:00, 10:00, 11:00, 12:00]
            # last_sample_tail_timestamp = 10:00
            # Thus no need to fill too.
            return raw_timeseries

        # need fill.
        raw_df = raw_timeseries.to_dataframe(copy=False)
        raw_cols = raw_df.columns
        # there are 2 pre-guarantee worth noticing, these guarantee helps ensure that below computing extra timeindex
        # code is correct:
        # 1. pre-check already guarantee that std_timeindex can only be one of pd.DatetimeIndex / pd.RangeIndex.
        # 2. TSDataset already internally guarantee that raw_timeindex.time_index type must == std_timeindex type.
        if isinstance(self._std_timeindex, pd.DatetimeIndex):
            freq = pd.infer_freq(raw_timeindex)
            extra_timeindex = pd.date_range(
                start=raw_timeindex[-1],
                end=last_sample_tail_timestamp,
                freq=freq
            )
        else:
            # pd.RangeIndex
            step = raw_timeindex.step
            extra_timeindex = pd.RangeIndex(
                start=raw_timeindex[-1],
                # pd.date_range::end param is right-closed, but pd.RangeIndex::stop param is right-opened, so need + 1.
                stop=last_sample_tail_timestamp + 1,
                step=step
            )
        # remove first timestamp as it is duplicated with the last timestamp in the raw time index.
        extra_timeindex = extra_timeindex[1:]

        extra_ndarray = np.zeros(
            shape=(len(extra_timeindex), len(raw_cols)),
            dtype=raw_df.to_numpy(copy=False).dtype
        )
        extra_ndarray.fill(self._fill_last_value)
        extra_df = pd.DataFrame(
            data=extra_ndarray,
            index=extra_timeindex,
            columns=raw_cols
        )

        filled_df = pd.concat([raw_df, extra_df])
        return TimeSeries.load_from_dataframe(
            data=filled_df,
            freq=raw_timeindex.step if isinstance(self._std_timeindex, pd.RangeIndex) else pd.infer_freq(raw_timeindex),
            drop_tail_nan=False
        )

    def _compute_last_sample_tail_timestamp(self) -> Union[pd.Timestamp, int]:
        """
        compute last sample tail timestamp.

        If self._fill_last_value is None, the returned timestamp will be the last sample without filling, otherwise
        if it is not None, the returned timestamp will be the filled last sample.

        Step 1. compute the sample count that can be built from the raw tsdataset.
        The computation formula of the sample_cnt is as follows:
        a + b * (n - 1) <= c
        where:
        a = first_sample_tail_idx = time_window[0]
        b = sampling_stride
        n = sample_cnt (Never contain the filled sample)
        c = max_std_idx = len(self._std_timeindex) - 1

        Thus,
        n = math.floor((c - a) / b) + 1

        Step 2. Compute the tail index of the last sample (include filled sample if self._fill_last_value is not None).
        The computation formula can be expressed as follows:
        c = a + b * (n - 1)
        where:
        a = first_sample_tail_idx
        b = sampling_stride
        n = sample_cnt (possibly contain the filled sample)
        c = max_std_idx = len(self._std_timeindex) - 1

        Returns:
            Union[pd.Timestamp, int]: The tail timestamp of last sample, where the last sample can either be
            filled / unfilled. Note that if input type(std_timeindex) == pd.DatetimeIndex, this method will return
            pd.Timestamp, otherwise if type(std_timeindex) == pd.RangeIndex, this method will return int.
        """
        max_std_idx = len(self._std_timeindex) - 1
        first_sample_tail_idx = self._time_window[0]
        sample_cnt = math.floor((max_std_idx - first_sample_tail_idx) / self._sampling_stride) + 1
        last_sample_tail_idx = first_sample_tail_idx + self._sampling_stride * (sample_cnt - 1)

        if last_sample_tail_idx == max_std_idx:
            # no need to fill.
            return self._std_timeindex[last_sample_tail_idx]

        if self._fill_last_value is None:
            # no need to fill.
            return self._std_timeindex[last_sample_tail_idx]

        # need fill.
        sample_cnt += 1
        last_sample_tail_idx = first_sample_tail_idx + self._sampling_stride * (sample_cnt - 1)
        if isinstance(self._std_timeindex, pd.DatetimeIndex):
            freq = pd.infer_freq(self._std_timeindex)
            # Here are 2 operations worth noticing:
            # 1. plus extra `1` when calling pd.date_range()
            # 2. remove the first element of extra_timeindex by calling extra_timeindex = extra_timeindex[1:]
            # The reason to do the above is that the first timestamp of extra_timeindex is duplicated with the last
            # timestamp of self._std_timeindex. To construct the extra timeindex, we must specify the start as first
            # element of self._std_timeindex (which will cause the duplicate timestamp).
            # Similarly, to avoid the self._std_timeindex[-1] to be used twice, we must remove it from extra_timeindex.
            extra_timeindex = pd.date_range(
                start=self._std_timeindex[-1],
                periods=last_sample_tail_idx - max_std_idx + 1,
                freq=freq
            )
        # precheck already guarantee that type(std_timeindex) can only be one of DatetimeIndex, RangeIndex.
        # Refers to _validate_std_timeindex().
        else:
            step = self._std_timeindex.step
            # Because RangeIndex is left-closed-right-opened, so stop param must + 1
            extra_timeindex = pd.RangeIndex(start=self._std_timeindex[-1], stop=last_sample_tail_idx + 1, step=step)

        # return type can either be pd.Timestamp or int, depends on type(extra_timeindex).
        return extra_timeindex[-1]

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

    def _compute_timeindex_offset(self, time_index: Union[pd.DatetimeIndex, pd.RangeIndex]) -> int:
        """
        Internal method, compute the offset between given timeseries.time_index and std_timeindex.

        As we know that input timeseries.start_time might <= std_timeindex start timestamp, so needs to compute offset.
        Also, the pre-check already guarantee that input timeseries.start_time must be within std_timeindex range.

        Examples:
            case 1 - time_index type = pd.DatetimeIndex
            input_time_index = [8:00, 9:00, 10:00]
            std_timeindex = [9:00, 10:00, 11:00, 12:00]
            so offset = input_time_index.get_loc(std_timeindex[0]) = input_time_index.get_loc(9:00) = 1

            case 2 - time_index type = pd.RangeIndex
            input_time_index = [1, 3, 5, 7]
            std_timeindex = [5, 7, 9]
            so offset = input_time_index.get_loc(std_timeindex[0]) = input_time_index.get_loc(5) = 2
        """
        return time_index.get_loc(self._std_timeindex[0])

    def _build_future_target_for_single_sample(
        self,
        curr_sample_tail: int,
        timeindex_offset: int,
        target_ndarray: np.ndarray
    ) -> np.ndarray:
        """
        Internal method, builds a future_target chunk for a single sample.

        Args:
            curr_sample_tail(int): the tail idx of future_target chunk of the same sample, note that curr_sample_tail
                is based on self._std_timeindex.
            timeindex_offset(int): the offset between current timeseries(target) index and std_timeindex.
            target_ndarray(np.ndarray): an np.ndarray matrix.

        Returns:
            np.ndarray: built future_target chunk (Y) for the current single sample.
        """
        end = timeindex_offset + curr_sample_tail + 1
        start = (end - 1) - self._out_chunk_len + 1
        return target_ndarray[start:end]

    def _build_past_target_for_single_sample(
        self,
        curr_sample_tail: int,
        timeindex_offset: int,
        target_ndarray: np.ndarray
    ) -> np.ndarray:
        """
        Internal method, builds a past_target chunk for a single sample.

        Args:
            curr_sample_tail(int): the tail idx of future_target chunk of the same sample, note that curr_sample_tail
                is based on self._std_timeindex.
            timeindex_offset(int): the offset between current timeseries(target) index and std_timeindex.
            target_ndarray(np.ndarray): an np.ndarray matrix.

        Returns:
            np.ndarray: built past_target chunk for the current single sample.
        """
        end = timeindex_offset + curr_sample_tail - self._out_chunk_len - self._skip_chunk_len + 1
        start = (end - 1) - self._in_chunk_len + 1
        return target_ndarray[start:end]

    def _build_known_cov_for_single_sample(
        self,
        curr_sample_tail: int,
        timeindex_offset: int,
        known_cov_ndarray: np.ndarray
    ) -> np.ndarray:
        """
        Internal method, builds a known_cov chunk for a single sample.

        Args:
            curr_sample_tail(int): the tail idx of future_target chunk of the same sample, note that curr_sample_tail
                is based on self._std_timeindex.
            timeindex_offset(int): the offset between current timeseries(known_cov) index and std_timeindex.
            known_cov_ndarray(np.ndarray): an np.ndarray matrix comes from known_cov_ts.to_numpy().

        Returns:
            np.ndarray: built known cov chunk for the current single sample.
        """
        # known_cov can be combined with parts: left + right, while the skip_chunk will be SKIPPED.
        right_end = timeindex_offset + curr_sample_tail + 1
        right_start = (right_end - 1) - self._out_chunk_len + 1

        left_end = (right_end - 1) - self._out_chunk_len - self._skip_chunk_len + 1
        left_start = (left_end - 1) - self._in_chunk_len + 1
        return np.vstack(tup=(known_cov_ndarray[left_start:left_end], known_cov_ndarray[right_start:right_end]))

    def _build_observed_cov_for_single_sample(
        self,
        curr_sample_tail: int,
        timeindex_offset: int,
        observed_cov_ndarray: np.ndarray
    ) -> np.ndarray:
        """
        Internal method, builds an observed_cov chunk for a single sample.

        Args:
            curr_sample_tail(int): the tail idx of future_target chunk of the same sample, note that curr_sample_tail
                is based on self._std_timeindex.
            timeindex_offset(int): the offset between current timeseries(observed_cov) index and std_timeindex.
            observed_cov_ndarray(np.ndarray, optional): an np.ndarray matrix, as it comes from
                observed_cov_ts.to_numpy(), its value will be None if the passed known_cov_ts is None.

        Returns:
            np.ndarray: built observed cov chunk for the current single sample.
        """
        end = timeindex_offset + curr_sample_tail - self._out_chunk_len - self._skip_chunk_len + 1
        start = (end - 1) - self._in_chunk_len + 1
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


class MLDataLoader(object):
    """
    Machine learning Data loader, provides an iterable over the given SampleDataset.

    The MLDataLoader supports iterable-style datasets with single-process loading and optional user defined batch
    collation.

    Args:
        dataset(SampleDataset): SampleDataset to be built.
        batch_size(int): The number of samples for each batch.
        collate_fn(Callable, optional): A user defined collate function for each batch, optional.
    """
    def __init__(
        self,
        dataset: SampleDataset,
        batch_size: int,
        collate_fn: Optional[Callable[[List[Dict[str, np.ndarray]]], Dict[str, np.ndarray]]] = None
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.collate_fn = self._default_collate_fn if collate_fn is None else collate_fn
        self._start = 0
        self._end = self._next_end()

    def __iter__(self):
        return self

    def __next__(self):
        start = self._start
        end = self._end
        if self._start > len(self.dataset.samples) - 1:
            raise StopIteration
        self._start += self.batch_size
        self._end = self._next_end()

        return self.collate_fn(self.dataset.samples[start:end])

    def _next_end(self):
        # In case if the remaining number of iterable samples are less than batch_size, return the remaining samples.
        # Example:
        #
        # Given full_dataset = [1, 2, 3, 4, 5, 6, 7], batch_size = 3
        # thus:
        # first batch = [1, 2, 3]
        # second batch = [4, 5, 6]
        # third batch = [7]
        # After iterating the second batch, the remaining number of samples (i.e. 1) is less than batch_size (i.e. 3),
        # thus the last batch returns only 1 sample (i.e. [7]).
        return self._start + min(self.batch_size, len(self.dataset.samples[self._start:]))

    @staticmethod
    def _default_collate_fn(batch: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """
        Internal method that takes in a batch of data and puts the elements within the batch
        into a container (e.g. python built-in dict container) with an additional outer dimension - batch size.

        This is used as the default function for collation when `batch_size` parameter is passed in while `collate_fn`
        parameter is NOT.

        Args:
            batch(List[Dict[str, np.ndarray]]): A batch of data to collate.

        Returns:
            Dict[str, np.ndarray]: A collated batch of data with an additional outer dimension - batch size.
        """
        batch_size = len(batch)
        sample = batch[0]

        collated_batch = dict()
        for sidx in range(len(batch)):
            for k in sample.keys():
                if k not in collated_batch.keys():
                    collated_batch[k] = np.zeros(
                        shape=(batch_size, sample[k].shape[0], sample[k].shape[1]),
                        dtype=sample[k].dtype
                    )
                collated_batch[k][sidx] = batch[sidx][k]
        return collated_batch


class DataAdapter(object):
    """
    Data adapter for dl and ml models, converts TSDataset to SampleDataset and DataLoader.
    """
    def __init__(self):
        pass

    def to_sample_dataset(
        self,
        rawdataset: TSDataset,
        in_chunk_len: int = 1,
        out_chunk_len: int = 0,
        skip_chunk_len: int = 0,
        sampling_stride: int = 1,
        fill_last_value: Optional[Union[np.floating, np.integer]] = None,
        time_window: Optional[Tuple[int, int]] = None
    ) -> SampleDataset:
        """
        Convert TSDataset to SampleDataset.

        Args:
            rawdataset(TSDataset): Raw TSDataset to be converted.
            in_chunk_len(int): The size of the loopback window, i.e., the number of time steps feed to the model.
            out_chunk_len(int): The size of the forecasting horizon, i.e., the number of time steps output by the model.
            skip_chunk_len(int): Optional, the number of time steps between in_chunk and out_chunk for a single sample.
                The skip chunk is neither used as a feature (i.e. X) nor a label (i.e. Y) for a single sample. By
                default, it will NOT skip any time steps.
            sampling_stride(int): Time steps to stride over the i-th sample and (i+1)-th sample. More precisely,
                let `t` be the time index of target time series, `t[i]` be the start time of the i-th sample,
                `t[i+1]` be the start time of the (i+1)-th sample, then `sampling_stride` represents the result of
                `t[i+1] - t[i]`.
            fill_last_value(float, optional): The value used for filling last sample. Set to None if no need to fill.
                For any type `t` of fill_last_value that np.issubdtype(type(t), np.floating) or
                np.issubdtype(type(t), np.integer) is True are valid.
            time_window(Tuple, optional): A two-element-tuple-shaped time window that allows adapter to build samples.
                time_window[0] refers to the window lower bound, while time_window[1] refers to the window upper bound.
                Each element in the left-closed-and-right-closed interval refers to the TAIL index of each sample.

        Examples:
            .. code-block:: python

                samples = [
                    {
                        "past_target": np.ndarray(
                            shape=(in_chunk_len, target_col_num)
                        ),
                        "future_target": np.ndarray(
                            shape=(out_chunk_len, target_col_num)
                        ),
                        "known_cov_numeric": np.ndarray(
                            shape=(in_chunk_len + out_chunk_len, known_numeric_col_num)
                        ),
                        "known_cov_categorical": np.ndarray(
                            shape=(in_chunk_len + out_chunk_len, known_categorical_col_num)
                        ),
                        "observed_cov_numeric": np.ndarray(
                            shape=(in_chunk_len, observed_numeric_col_num)
                        ),
                        "observed_cov_categorical": np.ndarray(
                            shape=(in_chunk_len + out_chunk_len, observed_categorical_col_num)
                        ),
                        "static_cov_numeric": np.ndarray(
                            shape=(in_chunk_len + out_chunk_len, static_numeric_col_num)
                        ),
                        "static_cov_categorical": np.ndarray(
                            shape=(in_chunk_len + out_chunk_len, static_categorical_col_num)
                        )
                    },
                    # ...
                ]
        """
        return SampleDataset(
            rawdataset=rawdataset,
            in_chunk_len=in_chunk_len,
            out_chunk_len=out_chunk_len,
            skip_chunk_len=skip_chunk_len,
            sampling_stride=sampling_stride,
            time_window=time_window,
            fill_last_value=fill_last_value
        )

    def to_paddle_dataloader(
        self,
        sample_dataset: SampleDataset,
        batch_size: int,
        collate_fn: Callable = None,
        shuffle: bool = True,
        drop_last: bool = False
    ) -> paddle.io.DataLoader:
        """
        Convert SampleDataset to paddle DataLoader.

        Args:
            sample_dataset(SampleDataset): SampleDataset to be converted.
            batch_size(int): The number of samples for a single batch.
            collate_fn(Callable, optional): User-defined collate function for each batch, optional.
            shuffle(bool, optional): Whether to shuffle indices order before generating batch indices, default True.
            drop_last(bool, optional): Whether to discard when the remaining data does not meet a batch, default False.

        Returns:
            PaddleDataLoader: A built paddle DataLoader.

        Examples:
            .. code-block:: python

                dataloader = [
                    # 1st batch
                    {
                        "past_target": paddle.Tensor(
                            shape=(batch_size, in_chunk_len, target_col_num)
                        ),
                        "future_target": paddle.Tensor(
                            shape=(batch_size, out_chunk_len, target_col_num)
                        ),
                        "known_cov_numeric": paddle.Tensor(
                            shape=(batch_size, known_cov_chunk_len, known_cov_numeric_col_num)
                        ),
                        "known_cov_categorical": paddle.Tensor(
                            shape=(batch_size, known_cov_chunk_len, known_cov_categorical_col_num)
                        ),
                        "observed_cov_numeric": paddle.Tensor(
                            shape=(batch_size, observed_cov_chunk_len, observed_cov_col_num)
                        ),
                        "observed_cov_categorical": paddle.Tensor(
                            shape=(batch_size, observed_cov_chunk_len, observed_cov_categorical_col_num)
                        ),
                        "static_cov_numeric": paddle.Tensor(
                            shape=(batch_size, 1, static_cov_numeric_col_num)
                        ),
                        "static_cov_categorical": paddle.Tensor(
                            shape=(batch_size, 1, static_cov_categorical_col_num)
                        )
                    },

                    # ...
                ]
        """
        return paddle.io.DataLoader(
            dataset=sample_dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=shuffle,
            drop_last=drop_last
        )

    def to_ml_dataloader(
        self,
        sample_dataset: SampleDataset,
        batch_size: int,
        collate_fn: Callable = None
    ) -> MLDataLoader:
        """
        Convert SampleDataset to MLDataLoader.

        Args:
            sample_dataset(SampleDataset): SampleDataset to be converted.
            batch_size(int): The number of samples for a single batch.
            collate_fn(Callable, optional): User-defined collate function for each batch, optional.

        Returns:
            MLDataLoader: A built MLDataLoader.

        Examples:
            .. code-block:: python

                dataloader = [
                    # 1st batch
                    {
                        "past_target": np.ndarray(
                            shape=(batch_size, in_chunk_len, target_col_num)
                        ),
                        "future_target": np.ndarray(
                            shape=(batch_size, out_chunk_len, target_col_num)
                        ),
                        "known_cov_numeric": np.ndarray(
                            shape=(batch_size, known_cov_chunk_len, known_cov_numeric_col_num)
                        ),
                        "known_cov_categorical": np.ndarray(
                            shape=(batch_size, known_cov_chunk_len, known_cov_categorical_col_num)
                        ),
                        "observed_cov_numeric": np.ndarray(
                            shape=(batch_size, observed_cov_chunk_len, observed_cov_col_num)
                        ),
                        "observed_cov_categorical": np.ndarray(
                            shape=(batch_size, observed_cov_chunk_len, observed_cov_categorical_col_num)
                        ),
                        "static_cov_numeric": np.ndarray(
                            shape=(batch_size, 1, static_cov_numeric_col_num)
                        ),
                        "static_cov_categorical": np.ndarray(
                            shape=(batch_size, 1, static_cov_categorical_col_num)
                        )
                    },

                    # ...
                ]
        """
        return MLDataLoader(dataset=sample_dataset, batch_size=batch_size, collate_fn=collate_fn)
