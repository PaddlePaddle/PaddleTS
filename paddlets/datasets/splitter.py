#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from abc import abstractclassmethod, ABCMeta
from typing import List, Dict, Any, Callable, Optional, Tuple, Union
import numbers

import pandas as pd
import numpy as np

from paddlets.datasets import TSDataset
from paddlets.logger import Logger, raise_if, raise_if_not

logger = Logger(__file__)

class SplitterBase(metaclass=ABCMeta):
    """
    Base class for all splitter.
    
    Args:
        skip_size(int): Series to be skipped between train data and test data, equal to 0 by default.
        verbose(bool): Whehter trun on the verbose mode, set to True by default.
    
    Return:
        None

    Raise:
        None
    
    """

    def __init__(self,
                 skip_size: int = 0,
                 verbose: bool = True) -> None:
        self._skip_size = skip_size
        self._verbose = verbose

    def split(self, dataset: TSDataset,
              return_index: bool = False) -> Union[TSDataset, pd.DatetimeIndex, pd.RangeIndex]:
        """
        Split TSdataset.
        
        Args:
            dataset(TS): Dataset to be splitted.
            return_index(bool): Return index or return TSDataset, set to False by default
        
        Return:
            TSDataset|pd.DatetimeIndex|pd.RangeIndex

        Raise:
            ValueError
        
        """
        target = dataset.get_target()
        raise_if(target is None, "Target is None!")
        observed_cov = dataset.get_observed_cov()
        n_samples = len(target)
        splits = self._get_splits(n_samples)
        indices = target._data.index

        for train, test in splits:
            if isinstance(indices, pd.DatetimeIndex):
                train_index = indices[train]
                test_index = indices[test]
                train_index.freq = indices.freqstr
                test_index.freq = indices.freqstr
            elif isinstance(indices, pd.RangeIndex):
                train_index = pd.RangeIndex(indices[0] + train[0] * indices.step, indices[0] + train[-1] * indices.step + indices.step, indices.step)
                test_index = pd.RangeIndex(indices[0] + test[0] * indices.step, indices[0] + test[-1] * indices.step + indices.step, indices.step)
            if return_index:
                yield (train_index, test_index)
            else:
                known_cov = dataset.get_known_cov()
                static_cov = dataset.get_static_cov()
                target_train = target[train_index]
                target_test = target[test_index]
                observed_cov_train = observed_cov[train_index] if observed_cov else None
                observed_cov_test = observed_cov[test_index] if observed_cov else None
                yield (TSDataset(target_train, observed_cov_train, known_cov, static_cov),
                       TSDataset(target_test, observed_cov_test, known_cov, static_cov)
                       )

    @abstractclassmethod
    def _get_splits(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get split indices.
        
        Args:
            None

        Return:
            Tuple[np.ndarray, np.ndarray]

        Raise:
            None
        
        """
        pass


    def _check_params(self):
        """
        Check params
        
        Args:
            None

        Return:
            None

        Raise:
            ValueError
        
        """

        raise_if_not(isinstance(self._skip_size, numbers.Integral),
                     "The number of skip_size must be of Integral type. "
                     "%s of type %s was passed." % (self._skip_size, type(self._skip_size))
                     )
        raise_if(self._skip_size < 0,
                 "skip_size should not be a  negitive integer, got skip_size={0} instead.".format(self._skip_size))

        raise_if_not(isinstance(self._verbose, bool),
                     "Param verbose must be of bool type. "
                     "%s of type %s was passed." % (self._verbose, type(self._verbose))
                     )


class HoldoutSplitter(SplitterBase):
    """
    Holdout splitter

    Split TSDataset  into an training set and a  test set.

    Args:
        test_size(int|float): Test_size, can be int or float, int represent data length, type float represent data ratio.
        skip_size(int): Series to be skipped between train data and test data.
        verbose(bool): Whehter trun on the verbose mode, set to True by default.
    
    Return:
        None
    
    Example:
        1) For example for ``test_size = 5`` , other param set to default.
        here is a representation of the folds:

        .. code-block:: python

            I * * * * * * * x x x x xI

        ``*`` = training fold.
        ``x`` = test fold.

        2) For example for ``test_size = 0.5``, other param set to default.
        here is a representation of the folds:

        .. code-block:: python

            I * * * * * * x x x x x xI

        ``*`` = training fold.
        ``x`` = test fold.

    """

    def __init__(self,
                 test_size: Union[int, float],
                 skip_size: int = 0,
                 verbose: bool = True) -> None:
        super().__init__(skip_size, verbose)
        self._test_size = test_size
        self._check_params()

    def _get_splits(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get split indices.
        
        Args:
            n_samples(int):Sample length

        Return:
            Tuple[np.ndarray, np.ndarray]

        Raise:
            None
        
        """
        skip_size = self._skip_size
        test_size = self._test_size

        if isinstance(self._test_size, float):
            test_size = self._convert_test(test_size, n_samples)

        raise_if(n_samples <= test_size, "Test size should <= data target length")
        raise_if(n_samples <= test_size + skip_size, "Test size + skip size should <= data target length")

        split_point = n_samples - test_size
        indices = np.arange(n_samples)
        yield (
            indices[:split_point],
            indices[split_point + skip_size:],
        )

    def _convert_test(self, test_size: float, n_samples: int):
        """
        Transform float test_size to int
        
        Args:
            test_size(float): float test size
            n_samples: length of data.

        Return:
            int: test size

        Raise:
            ValueError
        
        """
        raise_if_not(
            0.0 < test_size < 1.0,
            "`test_size` (float) should be between 0.0 and 1.0."
        )
        return int(n_samples * test_size)

    def _check_params(self):
        super()._check_params()
        raise_if_not(self._test_size, "please input test data size")


class ExpandingWindowSplitter(SplitterBase):
    """
    Expanding Window Splitter

    Split time series repeatedly into an growing training set and a fixed-size test set.

    Args:
        n_splits: Number of folds, not None.
        test_size(int|float|None): Test data size, test_size = n_samples // (n_splits+1) when test_size = None.
        skip_size(int): Series to be skipped between train data and test data.
        max_train_size(int): Max train size.
        verbose(bool): Whehter trun on the verbose mode, set to True by default.
    
    Return:
        None

    Raise:
        None

    Example:

        1) For example for ``n_splits = 5``, other params set to default.
        By default, test_size = n_samples // (n_splits+1) = 12//(5+1) = 2
        here is a representation of the folds:

        .. code-block:: python

            I * * x x - - - - - - - -I
            I * * * * x x - - - - - -I
            I * * * * * * x x - - - -I
            I * * * * * * * * x x - -I
            I * * * * * * * * * * x xI

        ``*`` = training fold.
        ``x`` = test fold.

        2) For example for ``n_splits = 5``, ``test_size = 1`` other params set to default.
        here is a representation of the folds:

        .. code-block:: python

            I * * * * * * * x - - - -I
            I * * * * * * * * x - - -I
            I * * * * * * * * * x - -I
            I * * * * * * * * * * x -I
            I * * * * * * * * * * * xI

        ``*`` = training fold.
        ``x`` = test fold.

        3) For example for ``n_splits = 5``, ``test_size = 1``, ``skip_size = 1``, other params set to default.
        here is a representation of the folds:

        .. code-block:: python

            I * * * * * * - x - - - -I
            I * * * * * * * - x - - -I
            I * * * * * * * * - x - -I
            I * * * * * * * * * - x -I
            I * * * * * * * * * * - xI

        ``*`` = training fold.
        ``x`` = test fold.

        4) For example for ``n_splits = 5``, ``test_size = 1``, ``max_train_size = 5``,other params set to default.
        here is a representation of the folds:

        .. code-block:: python

            I * * * * * - - x - - - -I
            I * * * * * - - - x - - -I
            I * * * * * - - - - x - -I
            I * * * * * - - - - - x -I
            I * * * * * - - - - - - xI

        ``*`` = training fold.
        ``x`` = test fold.
        
    """

    def __init__(self, n_splits: int = 5,
                 test_size: Union[int, None] = None,
                 skip_size: int = 0,
                 max_train_size: int = None,
                 verbose: bool = True) -> None:

        super().__init__(skip_size, verbose)
        self._test_size = test_size
        self._max_train_size = max_train_size
        self._n_splits = n_splits
        self._check_params()

    def _get_splits(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get split indices.
        
        Args:
            n_samples(int):Sample length

        Return:
            Tuple[np.ndarray, np.ndarray]

        Raise:
            None
        
        """
        n_splits = self._n_splits
        max_train_size = self._max_train_size
        skip_size = self._skip_size
        test_size = self._test_size

        if not test_size:
            test_size = n_samples // (n_splits + 1)
            if self._verbose:
                logger.info(f"Did not set test_size, set to data.len// (n_splits+1) by default,test_size = {test_size}")

        raise_if(n_samples <= test_size, "Test size should <= data target length")
        raise_if(n_samples <= test_size + skip_size, "Test size + skip size should <= data target length")
        raise_if(n_splits > n_samples,
                 f"Cannot have number of folds={n_splits} greater than the number of samples={n_samples}.")
        raise_if(n_samples - test_size * n_splits - skip_size <= 0,
                 f"(test_size({test_size}) - skip_size({skip_size}))*n_splits({n_splits}) can not equal or greater than the number of samples({n_samples})")

        indices = np.arange(n_samples)
        test_starts = range(n_samples - test_size * n_splits - skip_size, n_samples, test_size)

        for start in test_starts:
            if max_train_size is not None and max_train_size < start:
                yield (
                    indices[:max_train_size],
                    indices[start + skip_size: start + skip_size + test_size],
                )
            else:
                yield (
                    indices[:start],
                    indices[start + skip_size: start + skip_size + test_size],
                )

    def _check_params(self):
        """
        Check params
        
        Args:
            None

        Return:
            None

        Raise:
            ValueError
        
        """
        super()._check_params()
        if self._max_train_size:
            raise_if_not(isinstance(self._test_size, numbers.Integral),
                         "The number of max_train_size must be of Integral type. "
                         "%s of type %s was passed." % (self._max_train_size, type(self._max_train_size))
                         )
            raise_if(self._max_train_size <= 0,
                     "max_train_size should be a positive integer, got max_train_size={0} instead.".format(
                         self._max_train_size))
            if self._skip_size != 0:
                raise_if(self._max_train_size, "skip size and max train size can not set simultaneously")

        raise_if_not(isinstance(self._n_splits, numbers.Integral),
                     "The number of folds must be of Integral type. "
                     "%s of type %s was passed." % (self._n_splits, type(self._n_splits))
                     )
        raise_if(self._n_splits < 1,
                 "cross-validation requires at least 1 splits, got n_splits={0} instead.".format(self._n_splits))

    @property
    def get_n_splits(self) -> int:
        """
        Get n_splits
        
        Args:
            None

        Return:
            int: n_splits

        Raise:
            None
        
        """
        return self._n_splits


class SlideWindowSplitter(SplitterBase):
    """
    Slide Window Splitter

    Split time series repeatedly into a fixed-length training and test set.

    Args:
        train_size(int): Train data size, not None.
        test_size(int|float|None): Test data size, not None.
        step_size: Step size between two folds, equal to test_size if None.
        skip_size(int): Series to be skipped between train data and test data.
        verbose(bool): Whehter trun on the verbose mode, set to True by default.
    
    Return:
        None

    Raise:
        ValuError
    
    Example:
        1) For example for ``train_size = 5``, ``test_size = 2``, other params set to default.
        here is a representation of the folds:

        .. code-block:: python

            I * * * * * x x - - - - -I
            I - - * * * * * x x - - -I
            I - - - - * * * * * x x -I

        ``*`` = training fold.
        ``x`` = test fold.

        2) For example for ``train_size = 5``, ``test_size = 2``, ``step_size = 1``, other params set to default.
        here is a representation of the folds:

        .. code-block:: python

            I * * * * * x x - - - - -I
            I - * * * * * x x - - - -I
            I - - * * * * * x x - - -I
            I - - - * * * * * x x - -I
            I - - - - * * * * * x x -I
            I - - - - - * * * * * x xI

        ``*`` = training fold.
        ``x`` = test fold.

        3) For example for ``n_splits = 5``, ``test_size = 2``, ``skip_size = 1``, other param set to default.
        here is a representation of the folds:

        .. code-block:: python

            I * * * * * - x x - - - -I
            I - - * * * * * - x x - -I
            I - - - - * * * * * - x xI

        ``*`` = training fold.
        ``x`` = test fold.

    """

    def __init__(self,
                 train_size: int,
                 test_size: int,
                 step_size: int = None,
                 skip_size: int = 0,
                 verbose: bool = True) -> None:

        super().__init__(skip_size, verbose)
        self._test_size = test_size
        self._train_size = train_size
        if step_size:
            self._step_size = step_size
        else:
            self._step_size = test_size
            if self._verbose:
                logger.info(f"Did not set window move step_size, set step_size = test_size by default, step_size= {step_size}")
        self._check_params()

    def _get_splits(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get split indices.
        
        Args:
            n_samples(int):Sample length

        Return:
            Tuple[np.ndarray, np.ndarray]

        Raise:
            None
        
        """
        skip_size = self._skip_size
        train_size = self._train_size
        test_size = self._test_size
        step_size = self._step_size

        raise_if(n_samples <= test_size, "Test size should <= data target length")
        raise_if(n_samples <= test_size + skip_size, "Test size + skip size should <= data target length")
        raise_if(train_size > n_samples, "Train size exceed the data length!")
        raise_if(train_size + test_size > n_samples,
                     "Train size + Test size exceed the data length!")
        raise_if(train_size + test_size + skip_size > n_samples,
                     "Train size + Test size + skip_size exceed the data length!")

        test_starts = range(train_size + skip_size, n_samples, step_size)

        indices = np.arange(n_samples)
        for start in test_starts:
            yield (
                indices[start - train_size:start],
                indices[start + skip_size: start + skip_size + test_size],
            )

    def _check_params(self):
        """
        Check params
        
        Args:
            None

        Return:
            None

        Raise:
            ValueError
        
        """
        super()._check_params()
        raise_if_not(isinstance(self._test_size, numbers.Integral),
                     "Test size must be of Integral type. "
                     "%s of type %s was passed." % (self._test_size, type(self._test_size))
                     )
        raise_if(self._test_size < 1,
                 "Test size should be a positive integer, got test_size={0} instead.".format(self._test_size))

        raise_if_not(isinstance(self._train_size, numbers.Integral),
                     "Trian size must be of Integral type. "
                     "%s of type %s was passed." % (self._train_size, type(self._train_size))
                     )
        raise_if(self._train_size < 1,
                 "Trian size should be a positive integer, got train_size={0} instead.".format(self._train_size))

        raise_if_not(isinstance(self._step_size, numbers.Integral),
                     "Step must be of Integral type. "
                     "%s of type %s was passed." % (self._step_size, type(self._step_size))
                     )
        raise_if(self._step_size < 1,
                 "step should be a positive integer, got step_size={0} instead.".format(self._step_size))


