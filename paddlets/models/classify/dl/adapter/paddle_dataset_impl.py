# !/usr/bin/env python3
# -*- coding:utf-8 -*-

from paddlets.datasets import TSDataset, TimeSeries
from paddlets.logger.logger import Logger, raise_if

from paddle.io import Dataset as PaddleDataset
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
from typing import List, Dict, Tuple, Optional, Union

logger = Logger(__name__)


class ClassifyPaddleDatasetImpl(PaddleDataset):
    """
    An implementation of :class:`paddle.io.Dataset`.

    Args:
        rawdatasets(List[TSDataset]): List[TSDataset] for building :class:`paddle.io.Dataset`.
        labels:(np.ndarray) : The data class labels
    """

    def __init__(
            self,
            rawdatasets: List[TSDataset],
            labels: np.ndarray
    ):
        super(ClassifyPaddleDatasetImpl, self).__init__()

        self._rawdatasets = rawdatasets
        self._labels = [] if labels is None else labels
        self.classes_ = []  # unique labels
        self.n_classes_ = 0  # number of unique labels

        raise_if(self._rawdatasets is None or len(self._rawdatasets) == 0, "TSDataset must be specified.")
        raise_if(0 < len(self._labels) != len(self._rawdatasets), "TSDatasets length must be equal to labels length.")
        raise_if(self._rawdatasets[0].get_target() is None, "dataset target Timeseries must not be None.")
        raise_if(len(self._rawdatasets[0].get_target().time_index) < 1, "TSDataset target Timeseries length must >= 1.")

        self._samples = self._build_samples()

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        # TODO
        # Currently the implementation build full data in the construct method, which will probably cause performance
        # waste if the number of the built full-data samples are much larger than the number model actually needed
        # while fitting.
        # Consider optimize this scenario later.
        return self._samples[idx]

    def _build_samples(self) -> List[Dict[str, np.ndarray]]:
        """
        Internal method, builds samples.

        Returns:
            List[Dict[str, np.ndarray]]: A list of samples.

        """
        labels = []
        if self._labels is not None and len(self._labels) > 0:
            labels = self.format_labels(self._labels)

        samples = []
        for i in range(len(self._rawdatasets)):
            sample = dict()
            target_ts = self._rawdatasets[i].get_target()
            target_ndarray = target_ts.to_numpy(copy=False)
            sample["features"] = target_ndarray
            sample["label"] = [] if len(self._labels) == 0 else labels[i]
            samples.append(sample)

        return samples

    def format_labels(self, labels):
        """Convert label to required format."""
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(labels)
        self.classes_ = self.label_encoder.classes_
        self.n_classes_ = len(self.classes_)
        y = y.reshape(len(y), 1)
        self.onehot_encoder = OneHotEncoder(sparse=False, categories="auto")
        y = self.onehot_encoder.fit_transform(y).astype(np.float32)
        return y

    @property
    def samples(self):
        return self._samples

    @samples.setter
    def samples(self, samples):
        self._samples = samples
