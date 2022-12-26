#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import abc

from typing import List, Optional
import numpy as np

from paddlets.datasets import TSDataset


class BaseClassifier(abc.ABC):
    """
    Base class for all classifier.
    """
    def __init__(self):
        pass

    @abc.abstractmethod
    def fit(
        self,
        train_tsdatasets: List[TSDataset],
        train_labels: np.ndarray,
        valid_tsdatasets: Optional[List[TSDataset]] = None,
        valid_labels: Optional[np.ndarray] = None
    ):
        """
        Fit a BaseClassifier instance.

        Any non-abstract classes inherited from this class should implement this method.

        Args: 
            train_tsdataset(TSDataset): Train set.
            train_labels:(np.ndarray) : The train data class labels
            valid_tsdataset(TSDataset|None): Eval set, used for early stopping.
            valid_labels:(np.ndarray) : The valid data class labels
        """
        pass

    @abc.abstractmethod
    def predict(
        self,
        tsdatasets: List[TSDataset]
    ) -> np.ndarray:
        """
        Predict labels. Results are output as ndarray.

        Args:
            tsdataset(List[TSDataset]) : Data to be predicted.
        Returns:
            np.ndarray.
        """
        pass

    @abc.abstractmethod
    def predict_proba(
        self,
        tsdatasets: List[TSDataset]
    ) -> np.ndarray:
        """
        Find probability estimates for each class for all cases.
        Results are output as ndarray.

        Args:
            tsdataset(List[TSDataset]) : Data to be predicted.
        Returns:
            np.ndarray.
        """
        pass

    @abc.abstractmethod
    def save(self, path: str) -> None:
        """
        Saves a BaseClassifier instance to a disk file.

        Any non-abstract classes inherited from this class should implement this method.

        Args:
            path(str): A path string containing a model file name.
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def load(path: str) -> "BaseClassifier":
        """
        Loads a :class:`~/paddlets.models.classify.base.BaseClassifier` instance from a file.

        Any non-abstract classes inherited from this class should implement this method.

        Args:
            path(str): A path string containing a model file name.

        Returns:
            BaseClassifier: A loaded model.
        """
        pass
