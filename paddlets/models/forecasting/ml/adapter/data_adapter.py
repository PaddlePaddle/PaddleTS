# !/usr/bin/env python3
# -*- coding:utf-8 -*-

from paddlets import TSDataset
from paddlets.models.forecasting.ml.adapter.ml_dataset import MLDataset
from paddlets.models.forecasting.ml.adapter.ml_dataloader import MLDataLoader

from typing import Optional, Callable, Tuple


class DataAdapter(object):
    """
    Data adapter, converts TSDataset to MLDataset and MLDataLoader.
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
        Convert TSDataset to MLDataset.

        Args:
            rawdataset(TSDataset): Raw TSDataset to be converted.
            in_chunk_len(int): The size of the loopback window, i.e., the number of time steps feed to the model.
            out_chunk_len(int): The size of the forecasting horizon, i.e., the number of time steps output by the model.
            skip_chunk_len(int): Optional, the number of time steps between in_chunk and out_chunk for a single sample.
                The skip chunk is neither used as a feature (i.e. X) nor a label (i.e. Y) for a single sample. By
                default, it will NOT skip any time steps.
            sampling_stride(int, optional): Time steps to stride over the i-th sample and (i+1)-th sample.
                More precisely, let `t` be the time index of target time series, `t[i]` be the start time of the i-th
                sample, `t[i+1]` be the start time of the (i+1)-th sample, then `sampling_stride` represents the result
                of `t[i+1] - t[i]`.
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
        Convert MLDataset to MLDataLoader.

        Args:
            ml_dataset(MLDataset): MLDataset to be converted.
            batch_size(int): The number of samples for a single batch.
            collate_fn(Callable, optional): User defined collate function for each batch, optional.

        Returns:
            MLDataLoader: A built MLDataLoader.

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

                # Built MLDataLoader instance:
                dataloader = [
                    # 1st batch
                    {
                        "past_target": np.ndarray(
                            shape=(batch_size, in_chunk_len, target_col_num)
                        ),
                        "future_target": np.ndarray(
                            shape=(batch_size, out_chunk_len, target_col_num)
                        ),
                        "known_cov_numeric": np.ndarray(
                            shape=(batch_size, known_cov_chunk_len, known_cov_numeric_col_num)
                        ),
                        "known_cov_categorical": np.ndarray(
                            shape=(batch_size, known_cov_chunk_len, known_cov_categorical_col_num)
                        ),
                        "observed_cov_numeric": np.ndarray(
                            shape=(batch_size, observed_cov_chunk_len, observed_cov_col_num)
                        ),
                        "observed_cov_categorical": np.ndarray(
                            shape=(batch_size, observed_cov_chunk_len, observed_cov_categorical_col_num)
                        ),
                        "static_cov_numeric": np.ndarray(
                            shape=(batch_size, 1, static_cov_numeric_col_num)
                        ),
                        "static_cov_categorical": np.ndarray(
                            shape=(batch_size, 1, static_cov_categorical_col_num)
                        )
                    },

                    # ...

                    # N-th batch
                    {
                        "past_target": np.ndarray(
                            shape=(batch_size, in_chunk_len, target_col_num)
                        ),
                        "future_target": np.ndarray(
                            shape=(batch_size, out_chunk_len, target_col_num)
                        ),
                        "known_cov_numeric": np.ndarray(
                            shape=(batch_size, known_cov_chunk_len, known_cov_numeric_col_num)
                        ),
                        "known_cov_categorical": np.ndarray(
                            shape=(batch_size, known_cov_chunk_len, known_cov_categorical_col_num)
                        ),
                        "observed_cov_numeric": np.ndarray(
                            shape=(batch_size, observed_cov_chunk_len, observed_cov_col_num)
                        ),
                        "observed_cov_categorical": np.ndarray(
                            shape=(batch_size, observed_cov_chunk_len, observed_cov_categorical_col_num)
                        ),
                        "static_cov_numeric": np.ndarray(
                            shape=(batch_size, 1, static_cov_numeric_col_num)
                        ),
                        "static_cov_categorical": np.ndarray(
                            shape=(batch_size, 1, static_cov_categorical_col_num)
                        )
                    }
                ]
                """
        return MLDataLoader(dataset=ml_dataset, batch_size=batch_size, collate_fn=collate_fn)
