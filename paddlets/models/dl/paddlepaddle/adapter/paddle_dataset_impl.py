# !/usr/bin/env python3
# -*- coding:utf-8 -*-

from paddlets.datasets import TSDataset, TimeSeries
from paddlets.logger.logger import Logger, raise_if

import paddle
from paddle.io import Dataset as PaddleDataset
import numpy as np
from typing import List, Dict, Tuple, Optional

logger = Logger(__name__)


class PaddleDatasetImpl(PaddleDataset):
    """
    An implementation of :class:`paddle.io.Dataset`.

    1> Any unused (known / observed) columns should be removed from the TSDataset before handled by this class.

    2> The default time_window assumes each sample contains X (i.e. in_chunk), skip_chunk, and
    Y (i.e. out_chunk).

    3> If caller explicitly passes time_window parameter in, and time_window upper bound is larger than
    len(TSDataset._target) - 1, it means that each built sample will only contain X (i.e. in_chunk), but
    will not contain skip_chunk or Y (i.e. out_chunk).

    Args:
        rawdataset(TSDataset): Raw :class:`~paddlets.TSDataset` for building :class:`paddle.io.Dataset`.
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

    Attributes:
        _supported_paddle_versions(Set[str]): A set of paddle module versions to support.
        _rawdataset(TSDataset) Raw :class:`~paddlets.TSDataset` for building :class:`paddle.io.Dataset`.
        _target_in_chunk_len(int): The size of the loopback window, i.e., the number of time steps feed to the model.
        _target_out_chunk_len(int): The size of the forecasting horizon, i.e. the number of time steps output by
            the model.
        _target_skip_chunk_len(int): The number of time steps between in_chunk and out_chunk for a single sample.
            The skip chunk is neither used as a feature (i.e. X) nor a label (i.e. Y) for a single sample. By
            default, it will NOT skip any time steps.
        _known_cov_chunk_len(int): The length of known covariates time series chunk for a single sample.
        _observed_cov_chunk_len(int): The length of observed covariates time series chunk for a single sample.
        _sampling_stride(int): Time steps to stride over the i-th sample and (i+1)-th sample.
        _time_window(Tuple, optional): A two-element-tuple-shaped time window that allows adapter to build samples.
            time_window[0] refers to the window lower bound, while time_window[1] refers to the window upper bound.
            Each element in the left-closed-and-right-closed interval refers to the TAIL index of each sample.
        _samples(List[Dict[str, np.ndarray]]): The built samples.

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
        super(PaddleDatasetImpl, self).__init__()

        self._supported_paddle_versions = {"2.2.0", "2.3.0"}
        self._curr_paddle_version = paddle.__version__
        self._rawdataset = rawdataset
        self._target_in_chunk_len = in_chunk_len
        self._target_out_chunk_len = out_chunk_len
        self._target_skip_chunk_len = skip_chunk_len
        self._known_cov_chunk_len = self._target_in_chunk_len + self._target_out_chunk_len
        self._observed_cov_chunk_len = self._target_in_chunk_len
        self._sampling_stride = sampling_stride
        self._time_window = time_window

        if self._curr_paddle_version not in self._supported_paddle_versions:
            logger.logger.info("recommended to use paddlepaddle >= 2.3.0")
        raise_if(rawdataset is None, "TSDataset must be specified.")
        raise_if(rawdataset.get_target() is None, "dataset target Timeseries must not be None.")
        raise_if(len(rawdataset.get_target().time_index) < 1, "TSDataset target Timeseries length must >= 1.")
        raise_if(
            in_chunk_len <= 0,
            "in_chunk_len must be positive integer, but %s is actually provided." % in_chunk_len
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
                len(rawdataset.get_target().time_index) < in_chunk_len + skip_chunk_len + out_chunk_len,
                """If time_window is not specified, TSDataset target timeseries length must be equal or larger than 
                the sum of in_chunk_len, skip_chunk_len and out_chunk_len. 
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

        # Validates input TSDataset, raises if input rawdataset invalid.
        # Firstly, validates target timeseries.
        max_target_idx = len(rawdataset.get_target().time_index) - 1
        max_target_timestamp = rawdataset.get_target().time_index[max_target_idx]
        if self._time_window[1] > max_target_idx:
            # This `if` statement indicates that caller is building a sample only containing feature (i.e. X),
            # but NOT containing skip_chunk or label (i.e. Y).
            # Thus, as long as the target is long enough to build the X of a sample, it can be treated as valid.
            min_allowed_target_len = in_chunk_len
        else:
            # This `else` statement indicates that caller is building a sample both containing feature (i.e. X),
            # skip_chunk and label (i.e. Y).
            # Thus, as long as the target is long enough to build a (X + skip + Y) sample, it can be treated as valid.
            min_allowed_target_len = in_chunk_len + skip_chunk_len + out_chunk_len
        raise_if(
            len(rawdataset.get_target().time_index) < min_allowed_target_len,
            """Given TSDataset target timeseries length is too short to build even one sample, 
            actual time_window: (%s, %s), actual target timeseries length: %s, min allowed sample length: %s. 
            If time_window[1] > max target index, sample length includes X but not includes Y or skip chunk, 
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
                    # `pd.Timestamp` data type can be compared using `<` operator directly.
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
                max_target_timestamp_idx_in_known = known_timeindex.get_loc(max_target_timestamp)
                # Compute the extra time steps to build known_cov features of the current sample.
                exceeded_time_steps = self._time_window[1] - max_target_idx
                raise_if(
                    # Tips: the expression `len(a[x:] ) > b` and `len(a[x+1:]) >= b` have the same effect, however
                    # `a[x+1:]` requires extra `out of upper bound` check for `a`, which causes more code and less
                    # robustness, thus use `a[x:]` approach here.
                    len(known_timeindex[max_target_timestamp_idx_in_known:]) <= exceeded_time_steps,
                    """known_cov length is too short to build known_cov chunk feature. 
                    It needs at least %s extra Timestamps after known_timeseries.time_index[%s:]""" %
                    (exceeded_time_steps, max_target_timestamp_idx_in_known)
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
                    """If time_window[1] <= len(target_timeindex) - 1, 
                    known_timeindex[-1] must >= target_timeindex[window[1]]. 
                    Actual known_timeindex[-1]: %s, actual target_timeindex[window[1]]: %s.""" %
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

                # Given:
                in_chunk_len = 2
                skip_chunk_len = 1
                out_chunk_len = 2
                sampling_stride = 1
                time_window = (4, 7)
                rawdataset = {
                    "target": [
                        [0, 0],
                        [1, 10],
                        [2, 20],
                        [3, 30],
                        [4, 40],
                        [5, 50],
                        [6, 60],
                        [7, 70]
                    ],
                    "known_cov": [
                        [0, 0, 0],
                        [10, 100, 1000],
                        [20, 200, 2000],
                        [30, 300, 3000],
                        [40, 400, 4000],
                        [50, 500, 5000],
                        [60, 600, 6000],
                        [70, 700, 7000],
                        [80, 800, 8000]
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
                    "static_cov": {"f": 1, "g": 2}
                }

            .. code-block:: python

                # Built samples:
                samples = [
                    # sample[0]
                    {
                        # past target time series chunk, totally contains _target_in_chunk_len time steps.
                        "past_target": [
                            [0, 0],
                            [1, 10]
                        ],
                        # future target time series chunk (i.e. Y), contains _target_out_chunk_len time steps.
                        # Note that skip_chunk_len = 1 time steps are skipped between past_target and future_target.
                        "future_target": [
                            [3, 30],
                            [4, 40]
                        ],
                        # known covariates time series chunk, totally contains _known_cov_chunk_len time steps.
                        # Note that skip_chunk_len = 1 time steps are skipped between past_target and future_target.
                        "known_cov": [
                            [0, 0, 0],
                            [10, 100, 1000],
                            # Note: skip_chunk [20, 200, 2000] is skipped between [10, 100, 1000] and [30, 300, 3000].
                            [30, 300, 3000],
                            [40, 400, 4000]
                        ],
                        # observed covariates time series chunk, totally contains _observed_cov_chunk_len time steps.
                        "observed_cov": [
                            [0],
                            [-1]
                        ]
                    },
                    # sample[1]
                    {
                        "past_target": [
                            [1, 10]
                            [2, 20]
                        ],
                        "future_target": [
                            [4, 40],
                            [5, 50]
                        ],
                        "known_cov": [
                            [10, 100, 1000],
                            [20, 200, 2000],
                            [40, 400, 4000],
                            [50, 500, 5000]
                        ],
                        "observed_cov": [
                            [-1],
                            [-2]
                        ]
                    },
                    # sample[2]
                    {
                        "past_target": [
                            [2, 30]
                            [3, 30]
                        ],
                        "future_target": [
                            [5, 50],
                            [6, 60],
                        ],
                        "known_cov": [
                            [20, 200, 2000],
                            [30, 300, 3000],
                            [50, 500, 5000],
                            [60, 600, 6000]
                        ],
                        "observed_cov": [
                            [-2],
                            [-3]
                        ]
                    },
                    # sample[3] (i.e. last sample, future_target tail index = 7 reaches time_window upper bound)
                    {
                        "past_target": [
                            [3, 30]
                            [4, 40]
                        ],
                        "future_target": [
                            [6, 60],
                            [7, 70]
                        ],
                        "known_cov": [
                            [30, 300, 3000],
                            [40, 400, 4000],
                            [60, 600, 6000],
                            [70, 700, 7000]
                        ],
                        "observed_cov": [
                            [-3],
                            [-4]
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
        target_ts = self._rawdataset.get_target()
        target_ndarray = target_ts.to_numpy(copy=False)

        # Consider the case where covariates is None.
        # As it is not mandatory for the models to use the covariates as features, thus covariates are VALID to be None.
        known_cov_ts = self._rawdataset.get_known_cov()
        known_cov_ndarray = None
        if known_cov_ts is not None:
            known_cov_ndarray = known_cov_ts.to_numpy(copy=False)

        observed_cov_ts = self._rawdataset.get_observed_cov()
        observed_cov_ndarray = None
        if observed_cov_ts is not None:
            observed_cov_ndarray = observed_cov_ts.to_numpy(copy=False)

        samples = []
        # `future_target_tail` refers to the tail index of the future_target chunk for each sample.
        future_target_tail = self._time_window[0]
        # Because _time_window is left-closed-right-closed, thus using `<=` operator rather than `<`.
        while future_target_tail <= self._time_window[1]:
            past_target_tail = future_target_tail - self._target_out_chunk_len - self._target_skip_chunk_len
            sample = {
                "future_target": self._build_future_target_for_single_sample(
                    future_target_tail=future_target_tail,
                    target_ts=target_ts,
                    target_ndarray=target_ndarray
                    ),
                "past_target": self._build_past_target_for_single_sample(
                    past_target_tail=past_target_tail,
                    target_ndarray=target_ndarray
                ),
                "known_cov": self._build_known_cov_for_single_sample(
                    future_target_tail=future_target_tail,
                    target_ts=target_ts,
                    known_cov_ts=known_cov_ts,
                    known_cov_ndarray=known_cov_ndarray
                ),
                "observed_cov": self._build_observed_cov_for_single_sample(
                    past_target_tail=past_target_tail,
                    target_ts=target_ts,
                    observed_cov_ts=observed_cov_ts,
                    observed_cov_ndarray=observed_cov_ndarray
                )
            }
            samples.append(sample)

            # The predefined `sampling_stride >= 1 assertion` in the construct method ensures that `infinite while loop`
            # will Not occur.
            future_target_tail += self._sampling_stride
        return samples

    def _compute_min_allowed_window(self) -> int:
        """
        Internal method, computes min allowed window.

        Returns:
            int: computed min allowed window.
        """
        return self._target_in_chunk_len + self._target_skip_chunk_len + self._target_out_chunk_len - 1

    def _build_future_target_for_single_sample(
        self,
        future_target_tail: int,
        target_ts: TimeSeries,
        target_ndarray: np.ndarray
    ) -> np.ndarray:
        """
        Internal method, builds a future_target chunk for a single sample.

        Args:
            future_target_tail(int): the tail idx of future_target chunk of the same sample.
            target_ts(TimeSeries): a target TimeSeries.
            target_ndarray(np.ndarray): an np.ndarray matrix.

        Returns:
            np.ndarray: built future_target chunk (Y) for the current single sample.
        """
        # Assumes the built samples contains Y by default.
        future_target = target_ndarray[future_target_tail - self._target_out_chunk_len + 1:future_target_tail + 1]
        if future_target_tail > len(target_ts.time_index) - 1:
            # `tail index > max target time index` indicates that no need to contain Y, thus returns empty ndarray.
            future_target = self._build_tensor_convertible_empty_ndarray(compatible=True)
        return future_target

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
        return target_ndarray[past_target_tail - self._target_in_chunk_len + 1:past_target_tail + 1]

    def _build_known_cov_for_single_sample(
        self,
        future_target_tail: int,
        target_ts: TimeSeries,
        known_cov_ts: Optional[TimeSeries] = None,
        known_cov_ndarray: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Internal method, builds a known_cov chunk for a single sample.

        Args:
            future_target_tail(int): the tail idx of future_target chunk of the same sample.
            target_ts(TimeSeries): a target TimeSeries.
            known_cov_ts(TimeSeries, optional): a known_cov TimeSeries, it can be None.
            known_cov_ndarray(np.ndarray, optional): an np.ndarray matrix, as it comes from
                known_cov_ts.to_numpy(), its value will be None if the passed known_cov_ts is None.

        Returns:
            np.ndarray: built known cov chunk for the current single sample.
        """
        # If known_cov timeseries is None, to avoid the failure of the conversion from paddle.Dataset to
        # paddle.DataLoader, we need to fill the empty ndarray with np.NaN because paddle.Tensor cannot be converted
        # from a python built-in None object, but can be converted from a np.ndarray filled with np.NaN.
        known_cov = self._build_tensor_convertible_empty_ndarray(True)
        # Build known_cov.
        # known_cov = left + right, where left = (in, skip), right = (skip, out).
        if known_cov_ts is not None:
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
            # known_cov = right + left
            known_cov = np.vstack((known_cov_left, known_cov_right))
        return known_cov

    def _build_observed_cov_for_single_sample(
        self,
        past_target_tail: int,
        target_ts: TimeSeries,
        observed_cov_ts: Optional[TimeSeries] = None,
        observed_cov_ndarray: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Internal method, builds an observed_cov chunk for a single sample.

        Args:
            past_target_tail(int): the tail idx of past_target chunk of the same sample.
            target_ts(TimeSeries): a target TimeSeries.
            observed_cov_ts(TimeSeries, optional): a observed_cov TimeSeries, it can be None.
            observed_cov_ndarray(np.ndarray, optional): an np.ndarray matrix, as it comes from
                observed_cov_ts.to_numpy(), its value will be None if the passed known_cov_ts is None.

        Returns:
            np.ndarray: built observed cov chunk for the current single sample.
        """
        # If known_cov timeseries is None, to avoid the failure of the conversion from paddle.Dataset to
        # paddle.DataLoader, we need to fill the empty ndarray with np.NaN because paddle.Tensor cannot be converted
        # from a python built-in None object, but can be converted from a np.ndarray filled with np.NaN.
        observed_cov = self._build_tensor_convertible_empty_ndarray(True)
        # Build observed_cov.
        if observed_cov_ts is not None:
            past_target_tail_timestamp = target_ts.time_index[past_target_tail]
            observed_cov_tail = observed_cov_ts.time_index.get_loc(past_target_tail_timestamp)
            observed_cov = \
                observed_cov_ndarray[observed_cov_tail - self._observed_cov_chunk_len + 1:observed_cov_tail + 1]
        return observed_cov

    def _build_tensor_convertible_empty_ndarray(self, compatible: bool = False):
        """
        Internal method, build a default empty ndarray that supports convert to a paddle.Tensor.

        For paddle==2.3.0, an empty np.ndarray refers to a ndarray with (0, 0) shape. For paddle==2.2.0, an empty
        np.ndarray refers to a ndarray with (1, 1) shape and filled with np.nan. The reason is that for paddle 2.2.0,
        a np.ndarray with (0, 0) shape cannot be converted to a paddle.Tensor.

        Args:
            compatible(bool, optional): A flag to decide whether to support old paddle version. if set to True, will
                support all paddle versions included in self._supported_paddle_versions.

        Returns:
            np.ndarray: An empty np.ndarray.
        """
        # np.zeros() is the original workaround as it works for paddle=2.3.0. However, this workaround is not working
        # for paddle=2.2.0.
        # The root cause is that paddle.Tensor cannot be converted from a ndarray of shape(0, 0) for version 2.0.0.
        # To support both 2.2.0 and 2.3.0, we use a ndarray of shape(1, 1) as a workaround for paddle 2.2.0.
        # From a long-term perspective, only 2.3.0 will be supported, the code for 2.2.0 will be deprecated later.
        ndarray = np.zeros(shape=(0, 0))
        if compatible is True:
            if self._curr_paddle_version == "2.2.0":
                ndarray = np.zeros(shape=(1, 1))
                ndarray.fill(np.nan)
        return ndarray

    @property
    def samples(self):
        return self._samples
