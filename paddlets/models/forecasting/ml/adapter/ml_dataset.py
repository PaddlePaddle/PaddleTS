# !/usr/bin/env python3
# -*- coding:utf-8 -*-

from paddlets import TSDataset, TimeSeries
from paddlets.logger import raise_if

import numpy as np
from typing import List, Dict, Tuple, Optional, Union


class MLDataset(object):
    """
    Machine learning Dataset.

    1> The in_chunk_len can be divided into several case: in_chunk_len = 0 indicates that the ML model has been
    processed by lag transform; in_chunk_len > 0 indicates that the ML model has NOT been processed by lag
    transform; in_chunk_len < 0 is NOT allowed.

    2> The unused (known / observed) columns should be deleted before the dataset passed in.

    3> The default time_window assumes each sample contains X (i.e. in_chunk), skip_chunk, and
    Y (i.e. out_chunk).

    4> If caller explicitly passes time_window parameter in, and time_window upper bound is larger than
    len(TSDataset._target) - 1, it means that each built sample will only contain X (i.e. in_chunk), but
    will not contain skip_chunk or Y (i.e. out_chunk). This occurs only if caller wants to build a sample
    used for prediction, as only in this scenario the Y (i.e. out_chunk) is not required.

    Args:
        rawdataset(TSDataset): Raw TSDataset to be converted.
        in_chunk_len(int): The size of the loopback window, i.e., the number of time steps feed to the model.
        out_chunk_len(int): The size of the forecasting horizon, i.e., the number of time steps output by the model.
        skip_chunk_len(int): Optional, the number of time steps between in_chunk and out_chunk for a single sample.
            The skip chunk is neither used as a feature (i.e. X) nor a label (i.e. Y) for a single sample. By
            default, it will NOT skip any time steps.
        sampling_stride(int, optional): Time steps to stride over the i-th sample and (i+1)-th sample. More precisely,
            let `t` be the time index of target time series, `t[i]` be the start time of the i-th sample,
            `t[i+1]` be the start time of the (i+1)-th sample, then `sampling_stride` represents the result of
            `t[i+1] - t[i]`.
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
            # (i.e. tsdataset.target[-1] = 10), so current target time series cannot provide 11 to build this sample.
    """
    def __init__(
        self,
        rawdataset: TSDataset,
        in_chunk_len: int,
        out_chunk_len: int,
        skip_chunk_len: int,
        sampling_stride: int,
        time_window: Optional[Tuple] = None
    ):
        self._rawdataset = rawdataset
        self._target_in_chunk_len = in_chunk_len
        self._target_out_chunk_len = out_chunk_len
        self._target_skip_chunk_len = skip_chunk_len
        self._known_cov_chunk_len = self._target_in_chunk_len + self._target_out_chunk_len
        self._observed_cov_chunk_len = 1 if self._target_in_chunk_len == 0 else self._target_in_chunk_len
        self._sampling_stride = sampling_stride
        self._time_window = time_window

        raise_if(rawdataset is None, "TSDataset must not be None.")
        raise_if(rawdataset.get_target() is None, "TSDataset target Timeseries must not be None.")
        raise_if(len(rawdataset.get_target().time_index) < 1, "TSDataset target Timeseries length must >= 1.")
        raise_if(
            in_chunk_len < 0,
            "in_chunk_len must be non-negative integer, but %s is actually provided." % in_chunk_len
        )
        raise_if(
            skip_chunk_len < 0,
            "skip_chunk_len must be non-negative integer, but %s is actually provided." % skip_chunk_len
        )
        raise_if(
            out_chunk_len <= 0,
            "out_chunk_len must be positive integer, but %s is actually provided." % out_chunk_len
        )
        raise_if(
            sampling_stride <= 0,
            "sampling_stride must be positive integer, but %s is actually provided." % sampling_stride
        )

        # Compute a default time_window if caller does not provide it.
        if self._time_window is None:
            # The default time_window assumes each sample contains both X, skip_chunk and Y, thus requires the length
            # of the target timeseries must be greater than or equal to the sum of X, skip_chunk and Y.
            raise_if(
                len(rawdataset.get_target().time_index) < max(1, in_chunk_len) + skip_chunk_len + out_chunk_len,
                """If time_window is not specified, TSDataset target timeseries length must be equal or larger than 
                the sum of max(1, in_chunk_len), skip_chunk_len and out_chunk_len. 
                Current in_chunk_len = %s, skip_chunk_len = %s, out_chunk_len = %s.""" %
                (in_chunk_len, skip_chunk_len, out_chunk_len)
            )
            default_min_window = self._compute_min_allowed_window()
            default_max_window = len(rawdataset.get_target().time_index) - 1
            self._time_window = (default_min_window, default_max_window)

        min_allowed_window = self._compute_min_allowed_window()
        raise_if(
            self._time_window[0] < min_allowed_window,
            "time_window lower bound must be equal or larger than %s" % min_allowed_window
        )

        max_allowed_window = len(rawdataset.get_target().data) - 1 + skip_chunk_len + out_chunk_len
        raise_if(
            self._time_window[1] > max_allowed_window,
            "time window upper bound must be equal or smaller than %s" % max_allowed_window
        )

        # Validates input TSDataset, raises if the passed data is invalid.
        # Firstly, valid target timeseries.
        max_target_idx = len(rawdataset.get_target().time_index) - 1
        max_target_timestamp = rawdataset.get_target().time_index[max_target_idx]
        if self._time_window[1] > max_target_idx:
            # This `if` statement indicates that caller is building a sample only containing feature (i.e. X),
            # but NOT containing skip_chunk or label (i.e. Y).
            # Thus, as long as the target is long enough to build the X of a sample, it can be treated as valid.
            min_allowed_target_len = max(1, in_chunk_len)
        else:
            # This `else` statement indicates that caller is building a sample both containing feature (i.e. X),
            # skip_chunk and label (i.e. Y).
            # Thus, as long as the target is long enough to build a (X + skip + Y) sample, it can be treated as valid.
            min_allowed_target_len = max(1, in_chunk_len) + skip_chunk_len + out_chunk_len
        raise_if(
            len(rawdataset.get_target().time_index) < min_allowed_target_len,
            """Given TSDataset target timeseries length is too short to build even one sample, 
            actual time_window: (%s, %s), actual target timeseries length: %s, min allowed sample length: %s. 
            If time_window[1] > max target index, sample length includes Y but not includes X or skip chunk, 
            else if time_window[1] <= max target index, sample length includes both X and skip chunk and Y.""" %
            (
                self._time_window[0],
                self._time_window[1],
                len(rawdataset.get_target().time_index),
                min_allowed_target_len
            )
        )

        # Secondly, validates known_cov timeseries.
        target_timeindex = rawdataset.get_target().time_index
        if rawdataset.get_known_cov() is not None:
            known_timeindex = rawdataset.get_known_cov().time_index
            if self._time_window[1] > max_target_idx:
                # (Note that the following statement uses `w` as the abbreviation of `_time_window`).
                # This `if` statement indicates a scenario where the built sample only contains X, but NOT contains Y.
                # Known that the max known cov timestamp must be greater than or equal to the timestamp which
                # w[1] pointed to.
                # In the meantime, as in this case w[1] > max target index, thus, the known cov timeseries must be
                # longer than the target timeseries, refers to the following example compute process:
                # Known that the `TSDataset` requires that target_timeseries and known_timeseries must have same freq,
                # given:
                # target_timeindex = target_timeseries.time_index = [8:00, 9:00, 10:00]
                # known_timeindex = known_timeseries.time_index = [7:00, 8:00, 9:00, 10:00, 11:00]
                # in_chunk_len = 1
                # skip_chunk_len = 0
                # out_chunk_len = 2
                # w = [4, 4]
                # Thus, the timestamp of the predicted chunk can be calculated to be equal to [11:00, 12:00],
                # thus, requires max timestamp of known_timeseries must be equal or larger than 12:00.
                # Below is the calculation process based on the above mock data:
                # Firstly, get:
                # max_target_idx = len(target_timeindex) - 1 = 2, thus, max_target_timestamp = 10:00
                # Secondly, compute the index position of max_target_timestamp in known_timeindex, i.e.
                # max_target_timestamp_idx_in_known = known_timeindex.get_loc(10:00) = 3.
                # Thirdly, compute the extra required time steps to build features (i.e. X) of the current sample:
                # exceed_time_steps_in_known = time_window[1] - max_target_idx = 4 - 2 = 2
                # So far, known that max_target_timestamp_idx_in_known = 3, exceed_time_steps_in_known = 2, thus:
                # needs to ensure that the following equation holds:
                # len(known_timeindex[max_target_timestamp_idx_in_known:]) > exceed_time_steps_in_known
                # However, according the previously computed result,
                # len(known_timeindex[max_target_timestamp_idx_in_known:]) = len(known_timeindex[3:]) = 2 which is
                # NOT larger than exceed_time_steps_in_known (i.e. 2), thus, the above equation does NOT hold,
                # thus causes the current known cov timeseries failed to build known_cov_chunk features for the sample.
                # END.
                raise_if(
                    known_timeindex[-1] < max_target_timestamp,
                    """If time_window upper bound is larger than len(target timeseries) - 1, 
                    known_cov max timestamp must be equal or larger than target max timestamp. 
                    Current time_window: (%s, %s), len(target timeseries) - 1 = %s, 
                    known_cov max timestamp = %s, target max timestamp = %s""" %
                    (
                        self._time_window[0],
                        self._time_window[1],
                        len(target_timeindex) - 1,
                        known_timeindex[-1],
                        max_target_timestamp
                    )
                )
                # Compute the index position of the max_target_timestamp in known_timeindex.
                idx = known_timeindex.get_loc(max_target_timestamp)
                # Compute the extra time steps to build known_cov features of the current sample.
                exceeded_time_steps = self._time_window[1] - max_target_idx
                raise_if(
                    # Tips: the expression `len(a[x:] ) > b` and `len(a[x+1:]) >= b` have the same effect, however
                    # `a[x+1:]` requires extra `out of upper bound` check for `a`, which causes more code and less
                    # robustness, thus use `a[x:]` approach here.
                    len(known_timeindex[idx:]) <= exceeded_time_steps,
                    """known_cov length is too short to build known_cov chunk feature. 
                    It needs at least %s extra Timestamps after known_timeseries.time_index[%s:]""" %
                    (exceeded_time_steps, idx)
                )
            else:
                # This `else` indicates that the built samples contain both X, skip_chunk and Y.
                # Let `upper_window_timestamp` be the timestamp which time_window[1] pointed to, the following equation
                # needs to be held:
                # known_timeindex[-1] >= upper_window_timestamp
                # Otherwise the target timeseries will be too short to build the samples within the range specified by
                # time_window.
                upper_window_timestamp = target_timeindex[self._time_window[1]]
                raise_if(
                    known_timeindex[-1] < upper_window_timestamp,
                    """max known_cov timestamp must be equal or larger than time_window upper bound timestamp, 
                    actual max known_cov timestamp: %s, actual time_window upper bound timestamp: %s.""" %
                    (known_timeindex[-1], upper_window_timestamp)
                )

        # Thirdly(Lastly), validates observed_cov timeseries
        if rawdataset.get_observed_cov() is not None:
            observed_timeindex = rawdataset.get_observed_cov().time_index
            # Known that max observed_cov timestamp no need to be larger than the max target timeindex.
            # Thus exceed_time_steps is no need to be computed here, which is different from known_cov.
            if self._time_window[1] > max_target_idx:
                # This `if` indicates that it is a prediction scenario where the built sample only contains X, but NOT
                # contains skip_chunk or Y.
                # Thus, the observed_cov only needs to ensure that its max timestamp is always >= max target timestamp.
                raise_if(
                    observed_timeindex[-1] < max_target_timestamp,
                    """if time_window upper bound is larger than max target, the max observed timestamp must 
                    be equal or larger than max target timestamp so that observed timeseries is long enough to build 
                    samples. Actual max observed timestamp: %s, max target timestamp: %s, time_window: (%s, %s)""" %
                    (
                        observed_timeindex[-1],
                        max_target_timestamp,
                        self._time_window[1],
                        self._time_window[1]
                    )
                )
            else:
                # This `else` indicates that this is for fit scenario where the built samples contain both X,
                # skip_chunk and Y.
                last_sample_past_target_tail = self._time_window[1] - \
                    self._target_skip_chunk_len - \
                    self._target_out_chunk_len
                last_sample_past_target_tail_timestamp = target_timeindex[last_sample_past_target_tail]
                # Observed cov does not need to provide `future` features, thus only need to ensure that the max
                # observed cov timestamp is large enough to build the `observed` feature of the last sample, i.e. the
                # following equation needs to be held:
                raise_if(
                    observed_timeindex[-1] < last_sample_past_target_tail_timestamp,
                    """if time_window upper bound is equal or smaller than max target, the max observed timestamp must 
                    be equal or larger than timestamp the time_window upper bound pointed to, so that 
                    observed timeseries is long enough to build samples. Actual max observed timestamp: %s, 
                    max target timestamp: %s, time_window: (%s, %s)""" %
                    (
                        observed_timeindex[-1],
                        max_target_timestamp,
                        self._time_window[1],
                        self._time_window[1]
                    )
                )

        self._samples = self._build_samples()

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        # TODO
        # Currently the implementation build full data in the construct method, which will probably cause performance
        # waste if the number of the built full-data samples are much larger than the number model actually needed
        # while fitting.
        # Consider optimize this scenario later.
        return self._samples[idx]

    def _build_samples(self) -> List[Dict[str, np.ndarray]]:
        """
        Internal method, builds samples.

        Returns:
            List[Dict[str, np.ndarray]]: A list of samples.

        Examples:
            .. code-block:: python

                # 1) lag scenario (TSDataset has been processed by lag transform):
                # Given:
                in_chunk_len = 0 (in_chunk_len = 0 indicates that this is lag scenario.)
                skip_chunk_len = 1
                out_chunk_len = 2
                sampling_stride = 1
                time_window = (2, 5)
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
                    # Note that features originally in target timeseries will be processed and added to observed_cov.
                    # For example, the following 2nd and 3rd columns are originally in target time series, and then
                    # being processed by lag-transform and moved to observed_cov.
                    "observed_cov": [
                        # Note: np.NaN is float, cannot be converted to np.int64, so if the input tsdataset needs to
                        # be lag-transformed, in case if the original lagged column.dtype = int, then after lag
                        # transform, it must be another INT-convertible number rather than a np.NaN.
                        [0，int_convertible_num, NaN],
                        [-1, 0, 0.0],
                        [-2, 1, 10.0],
                        [-3, 2, 20.0],
                        [-4, 3, 30.0],
                        [-5, 4, 40.0],
                        [-6, 5, 50.0],
                        [-7, 6, 60.0]
                    ],
                    "static_cov": {"static0": 0, "static1": 1.0, "static2": 2, "static3": 3.0}
                }

                # Built samples:
                samples = [
                    # sample[0]
                    {
                        ###############
                        # past_target #
                        ###############
                        # Note: shape always be (0, 0) in lag scenario.
                        "past_target": np.array(shape=(0, 0)),

                        #################
                        # future_target #
                        #################
                        # Note that skip_chunk_len = 1 time steps are skipped between past_target and future_target.
                        # row = _target_out_chunk_len = 2
                        # col = len(TSDataset._target.data.columns) = 2
                        "future_target": [
                            [2,0, 20.0],
                            [3.0, 30.0]
                        ],

                        #############
                        # known_cov #
                        #############
                        # numeric
                        # row = _target_out_chunk_len = 2
                        # col = len(TSDataset._known_cov.data.select_dtypes(include=np.float32)) = 1
                        "known_cov_numeric": [
                            [200.0],
                            [300.0]
                        ],

                        # categorical
                        # row = _target_out_chunk_len = 2
                        # col = len(TSDataset._known_cov.data.select_dtypes(include=np.int64)) = 2
                        "known_cov_categorical": [
                            [20, 2000],
                            [30, 3000]
                        ],

                        ################
                        # observed_cov #
                        ################
                        # numeric
                        # row = _observed_cov_chunk_len = 1
                        # col = len(TSDataset._observed_cov.data.select_dtypes(include=np.int64)) = 1
                        "observed_cov_numeric": [
                            [NaN],
                        ],

                        # categorical
                        # row = _observed_cov_chunk_len = 1
                        # col = len(TSDataset._observed_cov.data.select_dtypes(include=np.int64)) = 2
                        "observed_cov_categorical": [
                            [0, int_convertible_num]
                        ],

                        ##############
                        # static_cov #
                        ##############
                        # numeric
                        # row = (fixed) 1.
                        # col = len(TSDataset._observed_cov.data.select_dtypes(include=np.float32)) = 2
                        "static_cov_numeric": [
                            # key-wise ascending sorted data.
                            [1.0, 3.0]
                        ],

                        # categorical
                        # row = (fixed) 1.
                        # col = len(TSDataset._observed_cov.data.select_dtypes(include=np.int64)) = 2
                        "static_cov_categorical": [
                            # key-wise ascending sorted data.
                            [0, 2]
                        ]
                    },

                    # sample[1]
                    {
                        "past_target": np.array(shape=(0, 0)),
                        "future_target": [
                            [3.0, 30.0],
                            [4.0, 40.0]
                        ],
                        "known_cov_numeric": [
                            [300.0],
                            [400.0]
                        ],
                        "known_cov_categorical": [
                            [30, 3000],
                            [40, 4000]
                        ],
                        "observed_cov_numeric": [
                            [0.0],
                        ],
                        "observed_cov_categorical": [
                            [-1, 0]
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
                        "past_target": np.array(shape=(0, 0)),
                        "future_target": [
                            [4.0, 40.0],
                            [5.0, 50.0]
                        ],
                        "known_cov_numeric": [
                            [400.0],
                            [500.0]
                        ],
                        "known_cov_categorical": [
                            [40, 4000],
                            [50, 5000]
                        ],
                        "observed_cov_numeric": [
                            [1.0],
                        ],
                        "observed_cov_categorical": [
                            [-2, 1]
                        ],
                        "static_cov_numeric": [
                            [1.0, 3.0]
                        ],
                        "static_cov_categorical": [
                            [0, 2]
                        ]
                    },

                    # sample[3]
                    {
                        "past_target": np.array(shape=(0, 0)),
                        "future_target": [
                            [5.0, 50.0],
                            [6.0, 60.0]
                        ],
                        "known_cov_numeric": [
                            [500.0],
                            [600.0]
                        ],
                        "known_cov_categorical": [
                            [50, 5000],
                            [60, 6000]
                        ],
                        "observed_cov_numeric": [
                            [2.0],
                        ],
                        "observed_cov_categorical": [
                            [-3, 2]
                        ],
                        "static_cov_numeric": [
                            [1.0, 3.0]
                        ],
                        "static_cov_categorical": [
                            [0, 2]
                        ]
                    },

                    sample[4] (i.e. last sample, future_target tail index = 7 reaches time_window upper bound)
                    {
                        "past_target": np.array(shape=(0, 0)),
                        "future_target": [
                            [6.0, 60.0],
                            [7.0, 70.0]
                        ],
                        "known_cov_numeric": [
                            [600.0],
                            [700.0]
                        ],
                        "known_cov_categorical": [
                            [60, 6000],
                            [70, 7000]
                        ],
                        "observed_cov_numeric": [
                            [3.0],
                        ],
                        "observed_cov_categorical": [
                            [-4, 3]
                        ],
                        "static_cov_numeric": [
                            [1.0, 3.0]
                        ],
                        "static_cov_categorical": [
                            [0, 2]
                        ]
                    }
                ]

                # 2) not lag scenario (TSDataset has NOT been processed by lag transform):
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
                        # col = len(TSDataset._target.data.columns) = 1
                        "past_target": [
                            [0.0, 0.0],
                            [1.0, 10.0]
                        ],

                        #################
                        # future_target #
                        #################
                        # numeric
                        # Note that skip_chunk_len = 1 time steps are skipped between past_target and future_target.
                        # row = _target_out_chunk_len = 2
                        # col = len(TSDataset._target.data.columns) = 1
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
                        # col = len(TSDataset._observed_cov.data.select_dtypes(include=np.float32)) = 2
                        "static_cov_numeric": [
                            # key-wise ascending sorted data.
                            [1.0, 3.0]
                        ],

                        # categorical
                        # row = (fixed) 1.
                        # col = len(TSDataset._observed_cov.data.select_dtypes(include=np.int64)) = 2
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
        """
        numeric_dtype = np.float32
        categorical_dtype = np.int64

        # target
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
        # `future_target_tail` refers to the tail index of the future_target chunk for each sample.
        future_target_tail = self._time_window[0]
        # Because _time_window is left-closed-right-closed, thus using `<=` operator rather than `<`.
        while future_target_tail <= self._time_window[1]:
            past_target_tail = future_target_tail - self._target_out_chunk_len - self._target_skip_chunk_len
            sample = {
                "future_target": self._build_future_target_for_single_sample(
                    future_target_tail=future_target_tail,
                    target_ndarray=target_ndarray
                ),
                "past_target": self._build_past_target_for_single_sample(
                    past_target_tail=past_target_tail,
                    target_ndarray=target_ndarray
                )
            }

            # known_cov
            if known_cov_ts is not None:
                # numeric
                if 0 not in known_cov_numeric_ndarray.shape:
                    sample["known_cov_numeric"] = self._build_known_cov_for_single_sample(
                        future_target_tail=future_target_tail,
                        known_cov_ndarray=known_cov_numeric_ndarray
                    )
                # categorical
                if 0 not in known_cov_categorical_ndarray.shape:
                    sample["known_cov_categorical"] = self._build_known_cov_for_single_sample(
                        future_target_tail=future_target_tail,
                        known_cov_ndarray=known_cov_categorical_ndarray
                    )

            # observed_cov
            if observed_cov_ts is not None:
                # numeric
                if 0 not in observed_cov_numeric_ndarray.shape:
                    sample["observed_cov_numeric"] = self._build_observed_cov_for_single_sample(
                        past_target_tail=past_target_tail,
                        observed_cov_ndarray=observed_cov_numeric_ndarray
                    )
                # categorical
                if 0 not in observed_cov_categorical_ndarray.shape:
                    sample["observed_cov_categorical"] = self._build_observed_cov_for_single_sample(
                        past_target_tail=past_target_tail,
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

            # The predefined `sampling_stride >= 1 assertion` in the construct method ensures that `infinite while loop`
            # will Not occur.
            future_target_tail += self._sampling_stride
        return samples

    def _compute_min_allowed_window(self) -> int:
        """
        Internal method, used for computing min allowed window lower bound based on given in/skip/out chunk len.

        Consider lag-transform case which will cause _target_in_chunk_len equal to zero, thus use
        max(1, self._target_in_chunk_len) to ensure that in_chunk will hold at least 1 time unit.

        Returns:
            int: Computed min allowed window lower bound.
        """
        return max(1, self._target_in_chunk_len) + self._target_skip_chunk_len + self._target_out_chunk_len - 1

    def _build_future_target_for_single_sample(
        self,
        future_target_tail: int,
        target_ndarray: np.ndarray
    ) -> np.ndarray:
        """
        Internal method, builds a numeric or categorical future_target chunk for a single sample.

        Args:
            future_target_tail(int): the tail idx of future_target chunk of the same sample.
            target_ndarray(np.ndarray): an np.ndarray matrix.

        Returns:
            np.ndarray: built numeric or categorical future_target chunk (Y) for the current single sample.
        """
        # sample only contains  X, but not contains skip_chunk and Y, thus filled with all zeros ndarray.
        if future_target_tail > len(self._rawdataset.get_target().time_index) - 1:
            return np.zeros(shape=(0, 0))

        # samples contain both X, skip_chunk and Y.
        return target_ndarray[future_target_tail - self._target_out_chunk_len + 1:future_target_tail + 1]

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
        if self._target_in_chunk_len == 0:
            # lag case.
            return np.zeros(shape=(0, 0))
        # not-lag case.
        return target_ndarray[past_target_tail - self._target_in_chunk_len + 1:past_target_tail + 1]

    def _build_known_cov_for_single_sample(
        self,
        future_target_tail: int,
        known_cov_ndarray: np.ndarray
    ) -> np.ndarray:
        """
        Internal method, builds a known_cov chunk for a single sample.

        Args:
            future_target_tail(int): the tail idx of future_target chunk of the same sample.
            known_cov_ndarray(np.ndarray): an np.ndarray matrix comes from known_cov_ts.to_numpy().

        Returns:
            np.ndarray: built known cov chunk for the current single sample.
        """
        target_ts = self._rawdataset.get_target()
        known_cov_ts = self._rawdataset.get_known_cov()
        if future_target_tail > len(target_ts.time_index) - 1:
            max_target_timestamp = target_ts.time_index[-1]
            # compute the index position of max_target_timestamp in known_cov.
            max_target_timestamp_idx_in_known = known_cov_ts.time_index.get_loc(max_target_timestamp)
            known_cov_right_tail = max_target_timestamp_idx_in_known + \
                self._target_skip_chunk_len + \
                self._target_out_chunk_len
        else:
            future_target_tail_timestamp = target_ts.time_index[future_target_tail]
            known_cov_right_tail = known_cov_ts.time_index.get_loc(future_target_tail_timestamp)
        # right
        known_cov_right = \
            known_cov_ndarray[known_cov_right_tail - self._target_out_chunk_len + 1:known_cov_right_tail + 1]
        # left
        known_cov_left_tail = known_cov_right_tail - self._target_out_chunk_len - self._target_skip_chunk_len
        known_cov_left = \
            known_cov_ndarray[known_cov_left_tail - self._target_in_chunk_len + 1:known_cov_left_tail + 1]
        # known = right + left
        return np.vstack((known_cov_left, known_cov_right))

    def _build_observed_cov_for_single_sample(
        self,
        past_target_tail: int,
        observed_cov_ndarray: np.ndarray
    ) -> np.ndarray:
        """
        Internal method, builds an observed_cov chunk for a single sample.

        Args:
            past_target_tail(int): the tail idx of past_target chunk of the same sample.
            observed_cov_ndarray(np.ndarray, optional): an np.ndarray matrix, as it comes from
                observed_cov_ts.to_numpy(), its value will be None if the passed known_cov_ts is None.

        Returns:
            np.ndarray: built observed cov chunk for the current single sample.
        """
        past_target_tail_timestamp = self._rawdataset.get_target().time_index[past_target_tail]
        observed_cov_tail = self._rawdataset.get_observed_cov().time_index.get_loc(past_target_tail_timestamp)
        return observed_cov_ndarray[observed_cov_tail - self._observed_cov_chunk_len + 1:observed_cov_tail + 1]

    def _build_static_cov_for_single_sample(
        self,
        static_cov_dict: Dict[str, Union[np.float32, np.int64]]
    ) -> np.ndarray:
        """
        Internal method, build static_cov chunk for a single sample.

        Args:
            static_cov_dict(Dict[str, Union[np.float32, np.int64]]): a k-v static cov map.

        Returns:
            np.ndarray: built static cov chunk for the current single sample.
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

    @samples.setter
    def samples(self, samples):
        self._samples = samples
