# !/usr/bin/env python3
# -*- coding:utf-8 -*-
import numpy as np

from paddlets.models.classify.dl.adapter.paddle_dataset_impl import ClassifyPaddleDatasetImpl
from paddlets.datasets import TSDataset
from paddlets.logger import Logger

from paddle.io import DataLoader as PaddleDataLoader
from typing import List, Callable, Tuple, Optional

logger = Logger(__name__)


class ClassifyDataAdapter(object):
    """
    Data adapter, converts `TSDataset` to `paddle.io.Dataset` and `paddle.io.DataLoader`.
    """
    def __init__(self):
        pass

    def to_paddle_dataset(
        self,
        rawdatasets: List[TSDataset],
        labels: np.ndarray
    ) -> ClassifyPaddleDatasetImpl:
        """
        Converts :class:`TSDataset` to :class:`paddle.io.Dataset`.

        Args:
            rawdataset(TSDataset): Raw TSDataset for converting to :class:`paddle.io.Dataset`.
            labels:(np.ndarray) : The data class labels

        Returns:
            PaddleDatasetImpl: A built PaddleDatasetImpl.
        """
        return ClassifyPaddleDatasetImpl(
            rawdatasets=rawdatasets,
            labels = labels
        )

    def to_paddle_dataloader(
        self,
        paddle_dataset: ClassifyPaddleDatasetImpl,
        batch_size: int,
        collate_fn: Callable = None,
        shuffle: bool = True
    ) -> PaddleDataLoader:
        """
        Converts :class:`paddle.io.Dataset` to :class:`paddle.io.DataLoader`.

        Args:
            paddle_dataset(PaddleDatasetImpl): Raw :class:`TSDataset` for building :class:`paddle.io.DataLoader`.
            batch_size(int): The number of samples for a single batch.
            collate_fn(Callable, optional): User-defined collate function for each batch, optional.
            shuffle(bool, optional): Whether to shuffle indices order before generating batch indices, default True.

        Returns:
            PaddleDataLoader: A built paddle DataLoader.
        """
        return PaddleDataLoader(dataset=paddle_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle)
