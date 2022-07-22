# !/usr/bin/env python3
# -*- coding:utf-8 -*-

from paddlets import TSDataset

import abc


class BaseTransform(object, metaclass=abc.ABCMeta):
    """
    Base class for all data transformation classes (named `transformers` in this module)

    Any subclass or transformer needs to inherit from this base class and
    implement :func:`fit`, :func:`transform` and :func:`fit_transform` methods.
    """
    def __init__(self):
        pass

    @abc.abstractmethod
    def fit(self, dataset: TSDataset):
        """
        Learn the parameters from the dataset needed by the transformer.

        Any non-abstract class inherited from this class should implement this method.

        The parameters fitted by this method is transformer-specific. For example, the `MinMaxScaler` needs to 
        compute the MIN and MAX, and the `StandardScaler` needs to compute the MEAN and STD (standard deviation)
        from the dataset. 

        Args:
            dataset(TSDataset): dataset from which to fit the transformer.
        """
        pass

    @abc.abstractmethod
    def transform(
        self,
        dataset: TSDataset,
        inplace: bool = False
    ) -> TSDataset:
        """
        Apply the fitted transformer on the dataset

        Any non-abstract class inherited from this class should implement this method.

        Args:
            dataset(TSDataset): dataset to be transformed.
            inplace(bool, optional): Set to True to perform inplace transformation. Default is False.
            
        Returns:
            TSDataset: transformed dataset.
        """
        pass

    @abc.abstractmethod
    def fit_transform(
        self,
        dataset: TSDataset,
        inplace: bool = False
    ) -> TSDataset:
        """
        Combine the above fit and transform into one method, firstly fitting the transformer from the dataset 
        and then applying the fitted transformer on the dataset.

        Any non-abstract class inherited from this class should implement this method.

        Args:
            dataset(TSDataset): dataset to process.
            inplace(bool, optional): Set to True to perform inplace transformation. Default is False.

        Returns:
            TSDataset: transformed data.
        """
        pass

    def inverse_transform(
        self,
        dataset: TSDataset,
        inplace: bool = False
    ) -> TSDataset:
        """
        Inversely transform the dataset output by the `transform` method.

        Differ from other abstract methods, this method is not decorated by abc.abstractmethod. The reason is that not
        all the transformations can be transformed back inversely, thus, it is neither possible nor mandatory
        for all sub classes inherited from this base class to implement this method.

        In general, other modules such as Pipeline will possibly call this method WITHOUT knowing if the called
        transform instance has implemented this method. To work around this, instead of simply using `pass`
        expression as the default placeholder, this method raises a NotImplementedError to enable the callers
        (e.g. Pipeline) to use try-except mechanism to identify those data transformation operators that do NOT 
        implement this method.

        Args:
            dataset(TSDataset): dataset to be inversely transformed.
            inplace(bool, optional): Set to True to perform inplace transformation. Default is False.

        Returns:
            TSDataset: inverserly transformed dataset.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError
