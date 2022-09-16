# !/usr/bin/env python3
# -*- coding:utf-8 -*-

from paddlets import TSDataset
from paddlets.models.forecasting.ml.adapter.ml_dataset import MLDataset
from paddlets.models.forecasting.ml.adapter.ml_dataloader import MLDataLoader

from typing import Optional, Callable, Tuple


class DataAdapter(object):
    """
    Data adapter, converts paddlets.datasets.tsdataset.TSDataset to MLDataset and MLDataLoader.
    """
    def __init__(self):
        pass

    def to_ml_dataset(
        self,
        rawdataset: TSDataset,
        in_chunk_len: int = 1,
        out_chunk_len: int = 1,
        skip_chunk_len: int = 0,
        sampling_stride: int = 1,
        time_window: Optional[Tuple] = None
    ) -> MLDataset:
        """
        Converts :class:`~paddlets.TSDataset` to :class:`~paddlets.models.ml.adapter.MLDataset`.

        Args:
            rawdataset(TSDataset): Raw TSDataset for converting to :class:`~paddlets.models.ml.adapter.MLDataset`.
            in_chunk_len(int): The length of past target time series chunk for a single sample.
            out_chunk_len(int): The length of future target time series chunk for a single sample.
            skip_chunk_len(int): The length of time series chunk between past and future target for a single sample.
                The skip chunk are neither used as feature (i.e. X) nor label (i.e. Y) for a single sample.
            sampling_stride(int, optional): Time steps to stride over the i-th sample and (i+1)-th sample.
            time_window(Tuple, optional): A two-element-tuple-shaped time window that allows adapter to build samples.
                time_window[0] refers to the window lower bound, while time_window[1] refers to the window upper bound.
                Each element in the left-closed-and-right-closed interval refers to the TAIL index of each sample.

        Returns:
            PaddleDatasetImpl: A built MLDataset.
        """
        return MLDataset(
            rawdataset=rawdataset,
            in_chunk_len=in_chunk_len,
            out_chunk_len=out_chunk_len,
            skip_chunk_len=skip_chunk_len,
            sampling_stride=sampling_stride,
            time_window=time_window
        )

    def to_ml_dataloader(
        self,
        ml_dataset: MLDataset,
        batch_size: int,
        collate_fn: Optional[Callable] = None
    ) -> MLDataLoader:
        """
        Converts :class:`~paddlets.models.ml.adapter.MLDataset` to :class:`~paddlets.models.ml.adapter.MLDataLoader`.

        Args:
            ml_dataset(MLDataset): Raw TSDataset for converting to :class:`~paddlets.models.ml.adapter.MLDataLoader`.
            batch_size(int): The number of samples for a single batch.
            collate_fn(Callable, optional): User defined collate function for each batch, optional.

        Returns:
            MLDataLoader: A built MLDataLoader.

        Examples:
            .. code-block:: python

                # Given:
                batch_size = 4
                in_chunk_len = 3
                out_chunk_len = 1
                known_cov_chunk_len = in_chunk_len + out_chunk_len = 3 + 1 = 4
                observed_cov_chunk_len = in_chunk_len = 3
                target_col_num = 2 (target column number, e.g. ["t0", "t1"])
                known_cov_col_num = 3 (known covariates column number, e.g. ["k0", "k1", "k2"])
                observed_cov_col_num = 1 (observed covariates column number, e.g. ["obs0"])

                # Built MLDataLoader instance:
                dataloader = [
                    # 1st batch
                    {
                        "past_target": np.ndarray(shape=(batch_size, in_chunk_len, target_col_num)),
                        "future_target": np.ndarray(shape=(batch_size, out_chunk_len, target_col_num)),
                        "known_cov": np.ndarray(shape=(batch_size, known_cov_chunk_len, known_cov_col_num)),
                        "observed_cov": np.ndarray(shape=(batch_size, observed_cov_chunk_len, observed_cov_col_num))
                    },

                    # ...

                    # N-th batch
                    {
                        "past_target": np.ndarray(shape=(batch_size, in_chunk_len, target_col_num)),
                        "future_target": np.ndarray(shape=(batch_size, out_chunk_len, target_col_num)),
                        "known_cov": np.ndarray(shape=(batch_size, known_cov_chunk_len, known_cov_col_num)),
                        "observed_cov": np.ndarray(shape=(batch_size, observed_cov_chunk_len, observed_cov_col_num))
                    }
                ]
                """
        if collate_fn is None:
            return MLDataLoader(dataset=ml_dataset, batch_size=batch_size)
        return MLDataLoader(dataset=ml_dataset, batch_size=batch_size, collate_fn=collate_fn)
