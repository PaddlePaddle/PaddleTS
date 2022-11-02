# !/usr/bin/env python3
# -*- coding:utf-8 -*-

from paddlets.models.forecasting.dl.adapter.paddle_dataset_impl import PaddleDatasetImpl
from paddlets.datasets import TSDataset
from paddlets.logger import Logger

from paddle.io import DataLoader as PaddleDataLoader
from typing import Callable, Tuple, Optional

logger = Logger(__name__)


class DataAdapter(object):
    """
    Data adapter, converts TSDataset to paddle Dataset and paddle DataLoader.
    """
    def __init__(self):
        pass

    def to_paddle_dataset(
        self,
        rawdataset: TSDataset,
        in_chunk_len: int = 1,
        out_chunk_len: int = 1,
        skip_chunk_len: int = 0,
        sampling_stride: int = 1,
        time_window: Optional[Tuple] = None
    ) -> PaddleDatasetImpl:
        """
        Convert TSDataset to paddle Dataset.

        Args:
            rawdataset(TSDataset): Raw TSDataset to be converted.
            in_chunk_len(int): The size of the loopback window, i.e., the number of time steps feed to the model.
            out_chunk_len(int): The size of the forecasting horizon, i.e., the number of time steps output by the model.
            skip_chunk_len(int): Optional, the number of time steps between in_chunk and out_chunk for a single sample.
                The skip chunk is neither used as a feature (i.e. X) nor a label (i.e. Y) for a single sample. By
                default, it will NOT skip any time steps.
            sampling_stride(int, optional): Time steps to stride over the i-th sample and (i+1)-th sample.
                More precisely, let `t` be the time index of target time series,
                `t[i]` be the start time of the i-th sample, `t[i+1]` be the start time of the (i+1)-th sample, then
                `sampling_stride` represents the result of `t[i+1] - t[i]`.
            time_window(Tuple, optional): A two-element-tuple-shaped time window that allows adapter to build samples.
                time_window[0] refers to the window lower bound, while time_window[1] refers to the window upper bound.
                Each element in the left-closed-and-right-closed interval refers to the TAIL index of each sample.

        Returns:
            PaddleDatasetImpl: A built PaddleDatasetImpl.
        """
        return PaddleDatasetImpl(
            rawdataset=rawdataset,
            in_chunk_len=in_chunk_len,
            out_chunk_len=out_chunk_len,
            skip_chunk_len=skip_chunk_len,
            sampling_stride=sampling_stride,
            time_window=time_window
        )

    def to_paddle_dataloader(
        self,
        paddle_dataset: PaddleDatasetImpl,
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
                out_chunk_len = 2
                known_cov_chunk_len = in_chunk_len + out_chunk_len = 3 + 2 = 5
                observed_cov_chunk_len = in_chunk_len = 3
                target_col_num = 2 (target column number)
                known_cov_numeric_col_num = 3 (known covariates column number with numeric dtype)
                known_cov_categorical_col_num = 1 (known covariates column number with categorical dtype)
                observed_cov_numeric_col_num = 1 (observed covariates column number with numeric dtype)
                observed_cov_categorical_col_num = 2 (observed covariates column number with categorical dtype)
                static_cov_numeric_col_num = 1 (static covariates column number with numeric dtype)
                static_cov_categorical_col_num = 1 (static covariates column number with categorical dtype)

                # Built DataLoader instance:
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

                    # N-th batch
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
                    }
                ]
        """
        return PaddleDataLoader(dataset=paddle_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle)
