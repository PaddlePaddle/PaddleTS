# !/usr/bin/env python3
# -*- coding:utf-8 -*-

from paddlets.models.forecasting.ml.adapter.ml_dataset import MLDataset
from typing import List, Dict, Callable, Optional
import numpy as np


class MLDataLoader(object):
    """
    Machine learning Data loader, provides an iterable over the given MLDataset.

    The :class:`~paddlets.models.ml.adapter.MLDataLoader` supports iterable-style datasets with single-process loading and
    optional user defined batch collation.

    Args:
        dataset(MLDataset): MLDataset for building :class:`~paddlets.models.ml.adapter.MLDataLoader`.
        batch_size(int): The number of samples for each batch.
        collate_fn(Callable, optional): A user defined collate function for each batch, optional.

    Attributes:
        _dataset(MLDataset): MLDataset for building :class:`~paddlets.models.ml.adapter.MLDataLoader`.
        _batch_size(int): The number of samples for each batch.
        _collate_fn(Callable): A user defined collate function for each batch, optional.
        _start(int): The start index of the current batch in the full MLDataset, updated per iteration.
        _end(int): The end index of the current batch in the full MLDataset, updated per iteration.
    """
    def __init__(
        self,
        dataset: MLDataset,
        batch_size: int,
        collate_fn: Optional[Callable[[List[Dict[str, np.ndarray]]], Dict[str, np.ndarray]]] = None
    ):
        self._dataset = dataset
        self._batch_size = batch_size
        self._collate_fn = self._default_collate_fn if collate_fn is None else collate_fn
        self._start = 0
        self._end = self._next_end()

    def __iter__(self):
        return self

    def __next__(self):
        start = self._start
        end = self._end
        if self._start > len(self._dataset.samples) - 1:
            raise StopIteration
        self._start += self._batch_size
        self._end = self._next_end()

        return self._collate_fn(self._dataset.samples[start:end])

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
        return self._start + min(self._batch_size, len(self._dataset.samples[self._start:]))

    @staticmethod
    def _default_collate_fn(minibatch: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """
        Internal method that takes in a batch of data and puts the elements within the batch
        into a container (e.g. python built-in dict container) with an additional outer dimension - batch size.

        This is used as the default function for collation when `batch_size` parameter is passed in while `collate_fn`
        parameter is NOT.

        Args:
            minibatch(List[Dict[str, np.ndarray]]): A batch of data to collate.

        Returns:
            Dict[str, np.ndarray]: A collated batch of data with an additional outer dimension - batch size.
        """
        batch_size = len(minibatch)

        target_in_chunk_len = minibatch[0]["past_target"].shape[0]
        target_out_chunk_len = minibatch[0]["future_target"].shape[0]
        # Tips:
        # In case if known/observed timeSeries is None, the shape of known_cov/observed_cov will be (0, 0).
        # Thus, in this case, the known/observed chunk len cannot be computed by target input/output chunk, but need
        # to be represented by the corresponding shape of known/observed cov.
        known_cov_chunk_len = minibatch[0]["known_cov"].shape[0]
        observed_cov_chunk_len = minibatch[0]["observed_cov"].shape[0]

        # Tips:
        # In some cases, the past_target.shape are NOT equivalent to future_target.shape, see following example:
        # 1) In `Lag + build training sample` case, past_target.shape is ALWAYS equal to (0, 0), while
        # future_target.shape[1] = TSDataset._target.columns.
        # 2) In `Not-Lat + build predict sample` case, past_target.shape[1] = TSDataset._target.columns, while
        # future_target.shape is ALWAYS equal to (1, 1).
        # Thus, past_target_col_num 和 future_target_col_num need to be computed separately.
        past_target_col_num = minibatch[0]["past_target"].shape[1]
        future_target_col_num = minibatch[0]["future_target"].shape[1]
        known_col_num = minibatch[0]["known_cov"].shape[1]
        observed_col_num = minibatch[0]["observed_cov"].shape[1]

        collated_minibatch = dict()
        collated_minibatch["past_target"] = np.zeros(
            shape=(batch_size, target_in_chunk_len, past_target_col_num)
        )
        collated_minibatch["future_target"] = np.zeros(
            shape=(batch_size, target_out_chunk_len, future_target_col_num)
        )
        collated_minibatch["known_cov"] = np.zeros(
            shape=(batch_size, known_cov_chunk_len, known_col_num)
        )
        collated_minibatch["observed_cov"] = np.zeros(
            shape=(batch_size, observed_cov_chunk_len, observed_col_num)
        )
        for sidx in range(len(minibatch)):
            collated_minibatch["past_target"][sidx] = minibatch[sidx]["past_target"]
            collated_minibatch["future_target"][sidx] = minibatch[sidx]["future_target"]
            collated_minibatch["known_cov"][sidx] = minibatch[sidx]["known_cov"]
            collated_minibatch["observed_cov"][sidx] = minibatch[sidx]["observed_cov"]
        return collated_minibatch

    @property
    def dataset(self):
        return self._dataset

    @property
    def collate_fn(self):
        return self._collate_fn

    @property
    def batch_size(self):
        return self._batch_size
