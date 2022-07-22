# !/usr/bin/env python3
# -*- coding:utf-8 -*-

from typing import Union, List, Optional, Tuple

import copy
import pickle

from bts.models import base
from bts.datasets.tsdataset import TimeSeries, TSDataset
from bts.logger import Logger, raise_if_not, raise_if, raise_log
from bts.logger.logger import log_decorator

logger = Logger(__name__)

class Pipeline(base.Trainable):
    """
    Pipeline：实现任务序列封装，顺序执行

    Args:
        steps(List[Tuple[object, str]]): Pipeline中最后一个之外的所有estimators都必须是变换器（transformers），最后一个estimator可以是任意类型
        例如：pipeline=[(CountEncoder, {params}), (RfeSelect, {params}), (LogisticRegression, {params})]
        """
        
    def __init__(self, steps: List[Tuple[object, str]]):
        raise_if(steps is None, ValueError("steps must not be None"))
        for e in steps:
            if 2 != len(e):
                raise_log(ValueError("The expected length of the tuple is 2, but actual element len: %s" % len(e)))
        
        self._steps = steps
        self._fitted = False
        self._model = None
        
        #tranform实例化
        self._transform_list = []
        for index in range(len(self._steps) - 1):
            e = self._steps[index]
            transform_params = e[-1] 
            #初始化
            try:
                transform = e[0](**transform_params)
            except Exception as e:
                raise_log(ValueError("init error: %s" % (str(e))))             
            self._transform_list.append(transform)
        
        #last_estimator
        try:
            last_object = self._steps[-1][0](**self._steps[-1][-1]) 
        except Exception as e:
            raise_log(ValueError("init error: %s" % (str(e))))
        if hasattr(last_object, "fit_transform"):
            self._transform_list.append(last_object)
        else:
            self._model = last_object
            
    @log_decorator
    def fit(
        self,
        train_data: TSDataset,
        valid_data: Optional[TSDataset] = None):
        """
        pipeline训练入口

        Args:
            train_data(TSDataset): 训练集, 不论传入单份数据集, 还是多份数据集列表，该接口都只会训练一个模型, 而
                不是训练多个模型. 传入训练集列表在大多场景下都是为了支持多组数据联合训练一个模型的场景
            valid_data(TSDataset, optional): 验证集(可选)
        
        Returns:
            Pipeline
            This estimator
        """
        #transform
        for transform in self._transform_list:
            train_data = transform.fit_transform(train_data)
            if valid_data:
                valid_data = transform.fit_transform(valid_data)
        
        #last_estimator
        if self._model:
            if valid_data:
                self._model.fit(train_data, valid_data)
            else:
                self._model.fit(train_data)
            self._fitted = True
        return self


    def predict(self, data: TSDataset) -> TSDataset:
        """
        pipeline预测接口

        Args:
            data(TSDataset): 待预测的数据
        Returns:
            预测结果
        """
        if not self._fitted:
            raise_log(ValueError("please do fit first!"))
        for transform in self._transform_list:
            data = transform.transform(data)
        return self._model.predict(data)
       

    def predict_proba(self, data: TSDataset) -> TSDataset:
        """
        pipeline预测概率接口

        Args:
            data(TSDataset): 待预测的数据
        Returns:
            预测结果
        """
        if not self._fitted:
            raise_log(ValueError("please do fit first!"))
        for transform in self._transform_list:
            data = transform.transform(data)
        return self._model.predict_proba(data)        
    
    def save(self, path):
        """
        pipeline保存
        
        Args:
            path(str): 保存路径目录
        """
        #TODO
        pass

    
    def load(self, path):
        """
        pipeline保存
        
        Args:
            path(str): 保存路径目录
        """
        #TODO
        pass