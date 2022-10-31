# !/usr/bin/env python3
# -*- coding:utf-8 -*-

from paddlets import TSDataset
from paddlets.models.anomaly.ml.adapter.ml_dataset import MLDataset
from paddlets.models.forecasting.ml.adapter.ml_dataloader import MLDataLoader

import numpy as np
from typing import Callable, List, Dict


def anomaly_collate_fn(batch: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """
    Internal method that takes in a batch of data and puts the elements within the batch
    into a container (e.g. python built-in dict container) with an additional outer dimension - batch size.

    This is for collating a list of anomaly samples to mini-batches according to the passed batch_size param.

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


class DataAdapter(object):
    """
    Data adapter, converts bts.datasets.tsdataset.TSDataset to MLDataset and MLDataLoader.
    """
    def __init__(self):
        pass

    def to_ml_dataset(self, rawdataset: TSDataset, in_chunk_len: int = 1, sampling_stride: int = 1) -> MLDataset:
        """
        Converts bts.TSDataset to bts.models.anomaly.ml.adapter.MLDataset.

        Args:
            rawdataset(TSDataset): TSDataset to be converted.
            in_chunk_len(int): The length of past target time series chunk for a single sample.
            sampling_stride(int, optional): Time steps to stride over the i-th sample and (i+1)-th sample.

        Returns:
            PaddleDatasetImpl: A built MLDataset.
        """
        return MLDataset(
            rawdataset=rawdataset,
            in_chunk_len=in_chunk_len,
            sampling_stride=sampling_stride
        )

    def to_ml_dataloader(
        self,
        ml_dataset: MLDataset,
        batch_size: int,
        collate_fn: Callable = anomaly_collate_fn
    ) -> MLDataLoader:
        """
        Converts bts.models.anomaly.ml.adapter.MLDataset to bts.models.forecasting.ml.adapter.MLDataLoader.

        Args:
            ml_dataset(MLDataset): TSDataset to be converted.
            batch_size(int): Number of samples for a single batch.
            collate_fn(Callable): User defined collate function for each batch.

        Returns:
            MLDataLoader: A built MLDataLoader.

        Examples:
            .. code-block:: python

                # Given:
                batch_size = 4
                in_chunk_len = 3
                observed_cov_chunk_len = in_chunk_len = 3
                observed_cov_col_num = 1 (observed covariates column number, e.g. ["obs0"])

                # Built MLDataLoader instance:
                dataloader = [
                    # 1st batch
                    {
                        "observed_cov": np.ndarray(shape=(batch_size, observed_cov_chunk_len, observed_cov_col_num)),
                        "static_cov": np.ndarray(shape=(batch_size, 1, static_cov_col_num))
                    },

                    # ...

                    # N-th batch
                    {
                        "observed_cov": np.ndarray(shape=(batch_size, observed_cov_chunk_len, observed_cov_col_num)),
                        "static_cov": np.ndarray(shape=(batch_size, 1, static_cov_col_num))
                    }
                ]
                """
        return MLDataLoader(dataset=ml_dataset, batch_size=batch_size, collate_fn=collate_fn)
