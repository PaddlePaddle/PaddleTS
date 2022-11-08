# !/usr/bin/env python3
# -*- coding:utf-8 -*-

from paddlets.models.forecasting.ml.adapter.ml_dataset import MLDataset
from typing import List, Dict, Callable, Optional
import numpy as np


class MLDataLoader(object):
    """
    Machine learning Data loader, provides an iterable over the given MLDataset.

    The MLDataLoader supports iterable-style datasets with single-process loading and optional user defined batch
    collation.

    Args:
        dataset(MLDataset): MLDataset for building MLDataLoader.
        batch_size(int): The number of samples for each batch.
        collate_fn(Callable, optional): A user defined collate function for each batch, optional.
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

    @property
    def dataset(self):
        return self._dataset

    @property
    def collate_fn(self):
        return self._collate_fn

    @property
    def batch_size(self):
        return self._batch_size
