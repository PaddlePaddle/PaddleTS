# !/usr/bin/env python3
# -*- coding:utf-8 -*-

from paddlets.models.representation.dl.adapter.paddle_dataset_impl import ReprPaddleDatasetImpl
from paddlets.datasets import TSDataset
from paddlets.logger import Logger

from paddle.io import DataLoader as PaddleDataLoader
from typing import Callable, Union
import numpy as np

logger = Logger(__name__)


class ReprDataAdapter(object):
    """
    Data adapter, converts :class:`paddlets.TSDataset` to :class:`paddle.io.Dataset` and :class:`paddle.io.DataLoader`.
    """
    def __init__(self):
        pass

    def to_paddle_dataset(
        self,
        rawdataset: TSDataset,
        segment_size: int = 1,
        sampling_stride: int = 1,
        fill_last_value: Union[float, type(None)] = np.nan
    ) -> ReprPaddleDatasetImpl:
        """
        Converts :class:`paddlets.TSDataset` to :class:`paddle.io.Dataset`.

        Args:
            rawdataset(TSDataset): Raw TSDataset for converting to :class:`paddle.io.Dataset`.
            segment_size(int): The size of the loopback window, i.e., the number of time steps feed to the model.
            sampling_stride(int, optional): Time steps to stride over the i-th sample and (i+1)-th sample.
                More precisely, let `t` be the time index of target time series,
                `t[i]` be the start time of the i-th sample, `t[i+1]` be the start time of the (i+1)-th sample, then
                `sampling_stride` represents the result of `t[i+1] - t[i]`.
            fill_last_value(float): value to fill in the last sample. Set to None if no need to fill. Note
                that isinstance(np.NaN, float) is True.
        Returns:
            PaddleDatasetImpl: A built PaddleDatasetImpl.
        """
        return ReprPaddleDatasetImpl(
            rawdataset=rawdataset,
            segment_size=segment_size,
            sampling_stride=sampling_stride,
            fill_last_value=fill_last_value
        )

    def to_paddle_dataloader(
        self,
        paddle_dataset: ReprPaddleDatasetImpl,
        batch_size: int,
        collate_fn: Callable = None,
        shuffle: bool = True
    ) -> PaddleDataLoader:
        """
        Converts :class:`paddle.io.Dataset` to :class:`paddle.io.DataLoader`.

        Args:
            paddle_dataset(PaddleDatasetImpl): Raw :class:`~paddlets.TSDataset` for building :class:`paddle.io.DataLoader`.
            batch_size(int): The number of samples for a single batch.
            collate_fn(Callable, optional): User-defined collate function for each batch, optional.
            shuffle(bool, optional): Whether to shuffle indices order before generating batch indices, default True.

        Returns:
            PaddleDataLoader: A built paddle DataLoader.

        Examples:
            .. code-block:: python

                # Given:
                batch_size = 4
                segment_size = 3
                target_col_num = 2 (target column number, e.g. ["t0", "t1"])
                known_cov_col_num = 3 (known covariates column number, e.g. ["k0", "k1", "k2"])
                observed_cov_col_num = 1 (observed covariates column number, e.g. ["obs0"])

                # Built DataLoader instance:
                dataloader = [
                    # 1st batch
                    {
                        "past_target": paddle.Tensor(shape=(batch_size, segment_size, target_col_num))
                        "known_cov": paddle.Tensor(shape=(batch_size, segment_size, known_cov_col_num)),
                        "observed_cov": paddle.Tensor(shape=(batch_size, segment_size, observed_cov_col_num))
                    },

                    # ...

                    # N-th batch
                    {
                        "past_target": paddle.Tensor(shape=(batch_size, segment_size, target_col_num))
                        "known_cov": paddle.Tensor(shape=(batch_size, segment_size, known_cov_col_num)),
                        "observed_cov": paddle.Tensor(shape=(batch_size, segment_size, observed_cov_col_num))
                    }
                ]
        """
        return PaddleDataLoader(dataset=paddle_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle)
