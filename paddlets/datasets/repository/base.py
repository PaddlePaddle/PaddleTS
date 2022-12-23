# !/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
内置时序数据集相关操作
"""

from typing import Any, Callable, List, Optional, Sequence, Tuple, Union, Dict
import os

import pandas as pd
import numpy as np

from paddlets.logger import raise_if_not
from paddlets import TSDataset, TimeSeries
from paddlets.datasets.repository._data_config import ETTh1Dataset
from paddlets.datasets.repository._data_config import ETTm1Dataset
from paddlets.datasets.repository._data_config import ECLDataset
from paddlets.datasets.repository._data_config import WTHDataset
from paddlets.datasets.repository._data_config import UNIWTHDataset
from paddlets.datasets.repository._data_config import NABTEMPDataset
from paddlets.datasets.repository._data_config import PSMTRAINDataset
from paddlets.datasets.repository._data_config import PSMTESTDataset
from paddlets.datasets.repository._data_config import BasicMotionsTrainTDataset
from paddlets.datasets.repository._data_config import BasicMotionsTestDataset


DATASETS = {
    UNIWTHDataset.name: UNIWTHDataset,
    ETTh1Dataset.name: ETTh1Dataset,
    ETTm1Dataset.name: ETTm1Dataset,
    ECLDataset.name: ECLDataset,
    WTHDataset.name: WTHDataset,
    NABTEMPDataset.name: NABTEMPDataset,
    PSMTRAINDataset.name: PSMTRAINDataset,
    PSMTESTDataset.name: PSMTESTDataset,
    BasicMotionsTrainTDataset.name: BasicMotionsTrainTDataset,
    BasicMotionsTestDataset.name: BasicMotionsTestDataset
}

def dataset_list() -> List[str]:
    """
    获取paddlets内置时序数据集名称列表

    Returns:
        List(str): 数据集名称列表
    """
    return list(DATASETS.keys())

def get_dataset(name: str) -> Union["TSDataset", List["TSDataset"], Tuple[List["TSDataset"], List[Any]]]:
    """
    基于名称获取内置数据集
    
    Args:
        name(str): 数据集名称, 可以从dataset_list获取的列表中选取

    Returns:
        Union["TSDataset", List["TSDataset"], Tuple[List["TSDataset"], List[Any]]]: 基于内置数据集构建好的TSDataset对象或者对象列表
        
    """
    raise_if_not(
        name in DATASETS,
        f"Invaild dataset name: {name}"
    )
    dataset = DATASETS[name]
    path = dataset.path
    df = pd.read_csv(path)
    if dataset.type == 'classification':
        data_list = TSDataset.load_from_dataframe(df, **dataset.load_param)
        y_label = []
        for dataset in data_list:
            y_label.append(dataset.static_cov['label'])
            dataset.static_cov = None
        y_label = np.array(y_label)
        return (data_list, y_label)
    else:
        return TSDataset.load_from_dataframe(df, **dataset.load_param)
