# !/usr/bin/env python3
# -*- coding:utf-8 -*-
import paddle.io

from paddlets.models.dl.paddlepaddle.adapter.paddle_dataset_impl import PaddleDatasetImpl
from paddlets.datasets import TSDataset
from paddlets.logger import Logger

from paddle.io import DataLoader as PaddleDataLoader
from typing import Callable, Tuple, Optional

logger = Logger(__name__)


class DataAdapter(object):
    """
    Data adapter, converts :class:`paddlets.TSDataset` to :class:`paddle.io.Dataset` and :class:`paddle.io.DataLoader`.
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
        Converts :class:`paddlets.TSDataset` to :class:`paddle.io.Dataset`.

        Args:
            rawdataset(TSDataset): Raw TSDataset for converting to :class:`paddle.io.Dataset`.
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
        Converts :class:`paddle.io.Dataset` to :class:`paddle.io.DataLoader`.

        Args:
            paddle_dataset(PaddleDatasetImpl): Raw :class:`~paddlets.TSDataset` for building :class:`paddle.io.DataLoader`.
            batch_size(int): The number of samples for a single batch.
            collate_fn(Callable, optional): User-defined collate function for each batch, optional.
            shuffle(bool, optional): Whether to shuffle indices order before generating batch indices, default True.
                TODO: add this argument to :func:`__init__` construct method allow caller to set its value.

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
                target_col_num = 2 (target column number, e.g. ["t0", "t1"])
                known_cov_col_num = 3 (known covariates column number, e.g. ["k0", "k1", "k2"])
                observed_cov_col_num = 1 (observed covariates column number, e.g. ["obs0"])

                # Built DataLoader instance:
                dataloader = [
                    # 1st batch
                    {
                        "past_target": paddle.Tensor(shape=(batch_size, in_chunk_len, target_col_num)),
                        "future_target": paddle.Tensor(shape=(batch_size, out_chunk_len, target_col_num)),
                        "known_cov": paddle.Tensor(shape=(batch_size, known_cov_chunk_len, known_cov_col_num)),
                        "observed_cov": paddle.Tensor(shape=(batch_size, observed_cov_chunk_len, observed_cov_col_num))
                    },

                    # ...

                    # N-th batch
                    {
                        "past_target": paddle.Tensor(shape=(batch_size, in_chunk_len, target_col_num)),
                        "future_target": paddle.Tensor(shape=(batch_size, out_chunk_len, target_col_num)),
                        "known_cov": paddle.Tensor(shape=(batch_size, known_cov_chunk_len, known_cov_col_num)),
                        "observed_cov": paddle.Tensor(shape=(batch_size, observed_cov_chunk_len, observed_cov_col_num))
                    }
                ]
        """
        return PaddleDataLoader(dataset=paddle_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle)
