# !/usr/bin/env python3
# -*- coding:utf-8 -*-

import abc
from typing import Union, List, Optional

from paddlets.transform.base import BaseTransform
from paddlets.datasets.tsdataset import TimeSeries, TSDataset
from paddlets.logger import Logger, raise_if_not, raise_if, raise_log
from paddlets.logger.logger import log_decorator

logger = Logger(__name__)


class LagFeatureGenerator(BaseTransform):
    """
    Transform columns into lag features
    
    Args:
        cols(str): Name of feature columns to transform. default target cols. 
        lag_points(int): Number of lag points. 
        down_samples(int): Sample freq of lag points. 
        suffix(str): suffix of new feature columns
        
    Returns:
        None
    """
    def __init__(self,
        cols: Union[str, List[str]]=None,
        lag_points: int = 0,
        down_samples: int = 1,
        suffix: str = '_before'
    ):
        super(LagFeatureGenerator, self).__init__()
        self.lag_points = lag_points
        self.down_samples = down_samples
        self.feature_names = cols
        self.suffix = suffix
        self.need_previous_data = True
        self.n_rows_pre_data_need = lag_points

    def fit_one(self, dataset: TSDataset):
        """
        This transformer does not need to be fitted.

        Args:
            dataset(TSDataset): Dataset to be fitted.
        
        Returns:
            TimeFeatureGenerator
        """
        return self

    def transform_one(self, dataset: TSDataset, inplace: bool = False) -> TSDataset:
        """
        Transform target column to lag features.
        
        Args:
            dataset(TSDataset): Dataset to be transformed.
            inplace(bool): Whether to perform the transformation inplace. default=False
        
        Returns:
            TSDataset
        """
        #Whether to replace data
        new_ts = dataset
        if not inplace:
            new_ts = dataset.copy()
        target_cols = new_ts.get_target().columns
        self.feature_names = target_cols if self.feature_names is None else self.feature_names
        if isinstance(self.feature_names, str):
            self.feature_names = [self.feature_names]
        for feature_name in self.feature_names:
            #Get series
            tcov = new_ts.get_item_from_column(feature_name)
            #Generate lag feature content
            for index in range(self.down_samples, self.lag_points + 1, self.down_samples):
                v = tcov.data[feature_name].shift(index)
                if feature_name in target_cols:
                    new_ts.set_column(feature_name + '%s_%d' % (self.suffix, index), v, 'observed_cov')
                else:
                    tcov.data[feature_name + '%s_%d' % (self.suffix, index)] = v
        return new_ts
