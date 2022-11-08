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
    Data adapter, converts TSDataset to paddle Dataset and paddle DataLoader.
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
        Convert TSDataset to paddle Dataset.

        Args:
            rawdataset(TSDataset): Raw TSDataset to be converted.
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
        shuffle: bool = True,
        drop_last: bool = False
    ) -> PaddleDataLoader:
        """
        Converts paddle Dataset to paddle DataLoader.

        Args:
            paddle_dataset(PaddleDatasetImpl): paddle Dataset to be converted.
            batch_size(int): The number of samples for a single batch.
            collate_fn(Callable, optional): User-defined collate function for each batch, optional.
            shuffle(bool, optional): Whether to shuffle indices order before generating batch indices, default True.
            drop_last(bool, optional): Whether to discard when the remaining data does not meet a batch, default False.

        Returns:
            PaddleDataLoader: A built paddle DataLoader.

        Examples:
            .. code-block:: python

                # Given:
                batch_size = 4
                segment_size = 3
                target_col_num = 2 (target column number)
                known_cov_numeric_col_num = 3 (known covariates column number with numeric dtype)
                known_cov_categorical_col_num = 0 (known covariates column number with categorical dtype)
                observed_cov_numeric_col_num = 1 (observed covariates column number with numeric dtype)
                observed_cov_categorical_col_num = 0 (observed covariates column number with categorical dtype)
                static_cov_numeric_col_num = 1 (static covariates column number with numeric dtype)
                static_cov_categorical_col_num = 0 (static covariates column number with categorical dtype)

                # Built DataLoader instance:
                dataloader = [
                    # 1st batch
                    {
                        "past_target": paddle.Tensor(
                            shape=(batch_size, in_chunk_len, target_col_num)
                        ),
                        "known_cov_numeric": paddle.Tensor(
                            shape=(batch_size, known_cov_chunk_len, known_cov_numeric_col_num)
                        ),
                        "observed_cov_numeric": paddle.Tensor(
                            shape=(batch_size, observed_cov_chunk_len, observed_cov_col_num)
                        ),
                        "static_cov_numeric": paddle.Tensor(
                            shape=(batch_size, 1, static_cov_numeric_col_num)
                        )
                    },

                    # ...

                    # N-th batch
                    {
                        "past_target": paddle.Tensor(
                            shape=(batch_size, in_chunk_len, target_col_num)
                        ),
                        "known_cov_numeric": paddle.Tensor(
                            shape=(batch_size, known_cov_chunk_len, known_cov_numeric_col_num)
                        ),
                        "observed_cov_numeric": paddle.Tensor(
                            shape=(batch_size, observed_cov_chunk_len, observed_cov_col_num)
                        ),
                        "static_cov_numeric": paddle.Tensor(
                            shape=(batch_size, 1, static_cov_numeric_col_num)
                        )
                    }
                ]
        """
        return PaddleDataLoader(
            dataset=paddle_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle, drop_last=drop_last
        )
