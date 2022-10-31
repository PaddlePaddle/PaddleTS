# !/usr/bin/env python3
# -*- coding:utf-8 -*-

import abc
from typing import Union, List, Optional

from paddlets.transform.base import BaseTransform
from paddlets.datasets.tsdataset import TimeSeries, TSDataset
from paddlets.logger import Logger, raise_if_not, raise_if, raise_log
from paddlets.logger.logger import log_decorator

logger = Logger(__name__)


class DifferenceFeatureGenerator(BaseTransform):
    """
    Transform columns into difference features
    
    Args:
        cols(str): Name of feature columns to transform. default target cols. 
        difference_points(int): Number of difference points. 
        down_samples(int): Sample freq of difference points. 
        suffix(str): suffix of new feature columns
        
    Returns:
        None
    """
    def __init__(self,
        cols: Union[str, List[str]],
        difference_points: int = 0,
        down_samples: int = 1,
        suffix: str = '_diff'
    ):
        super(DifferenceFeatureGenerator, self).__init__()
        self.difference_points = difference_points
        self.down_samples = down_samples
        self.feature_names = cols
        self.suffix = suffix
        self.need_previous_data = True
        self.n_rows_pre_data_need = difference_points
        
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
        Transform target column to difference features.
        
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
        if isinstance(self.feature_names, str):
            self.feature_names = [self.feature_names]
        for feature_name in self.feature_names:
            #Get series
            tcov = new_ts.get_item_from_column(feature_name)
            #Adjust whether to be target
            raise_if(feature_name in new_ts.target.columns, "The target value should't be differentiated!")
            #Generate difference feature content
            for index in range(self.down_samples, self.difference_points + 1, self.down_samples):
                v = tcov.data[feature_name].diff(index)
                tcov.data[feature_name + '%s_%d' % (self.suffix, index)] = v
        return new_ts
