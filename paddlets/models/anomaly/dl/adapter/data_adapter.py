# !/usr/bin/env python3
# -*- coding:utf-8 -*-

from paddlets.models.anomaly.dl.adapter.paddle_dataset_impl import AnomalyPaddleDatasetImpl
from paddlets.datasets import TSDataset
from paddlets.logger import Logger

from paddle.io import DataLoader as PaddleDataLoader
from typing import Callable

logger = Logger(__name__)


class AnomalyDataAdapter(object):
    """
    Data adapter, convert TSDataset to paddle Dataset and paddle DataLoader.
    """
    def __init__(self):
        pass

    def to_paddle_dataset(
        self,
        rawdataset: TSDataset,
        in_chunk_len: int = 1,
        sampling_stride: int = 1
    ) -> AnomalyPaddleDatasetImpl:
        """
        Convert TSDataset to paddle Dataset.

        Args:
            rawdataset(TSDataset): Raw TSDataset to be converted.
            in_chunk_len(int): The size of the loopback window, i.e., the number of time steps feed to the model.
            sampling_stride(int, optional): Time steps to stride over the i-th sample and (i+1)-th sample.
                More precisely, let `t` be the time index of target time series,
                `t[i]` be the start time of the i-th sample, `t[i+1]` be the start time of the (i+1)-th sample, then
                `sampling_stride` represents the result of `t[i+1] - t[i]`.

        Returns:
            AnomalyPaddleDatasetImpl: A built PaddleDatasetImpl for anomaly detection models.
        """
        return AnomalyPaddleDatasetImpl(
            rawdataset=rawdataset,
            in_chunk_len=in_chunk_len,
            sampling_stride=sampling_stride
        )

    def to_paddle_dataloader(
        self,
        paddle_dataset: AnomalyPaddleDatasetImpl,
        batch_size: int,
        collate_fn: Callable = None,
        shuffle: bool = True
    ) -> PaddleDataLoader:
        """
        Convert paddle Dataset to paddle DataLoader.

        Args:
            paddle_dataset(PaddleDatasetImpl): paddle Dataset to be converted.
            batch_size(int): The number of samples for a single batch.
            collate_fn(Callable, optional): User-defined collate function for each batch, optional.
            shuffle(bool, optional): Whether to shuffle indices order before generating batch indices, default True.

        Returns:
            PaddleDataLoader: A built paddle DataLoader.

        Examples:
            .. code-block:: python

                # Given:
                batch_size = 4
                in_chunk_len = 3
                observed_cov_chunk_len = in_chunk_len = 3
                observed_cov_numeric_col_num = 1 (observed covariates column number with numeric dtype)
                observed_cov_categorical_col_num = 0 (observed covariates column number with categorical dtype)

                # Built DataLoader instance:
                dataloader = [
                    # 1st batch
                    {
                        "observed_cov": paddle.Tensor(
                            shape=(batch_size, observed_cov_chunk_len, observed_cov_numeric_col_num)
                        )
                    },

                    # ...

                    # N-th batch
                    {
                        "observed_cov": paddle.Tensor(
                            shape=(batch_size, observed_cov_chunk_len, observed_cov_numeric_col_num)
                        )
                    }
                ]
        """
        return PaddleDataLoader(dataset=paddle_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle)
