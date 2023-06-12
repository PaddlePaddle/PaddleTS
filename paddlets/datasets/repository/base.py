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
from paddlets.datasets.repository._data_config import ETTh2Dataset
from paddlets.datasets.repository._data_config import ETTm1Dataset
from paddlets.datasets.repository._data_config import ETTm2Dataset
from paddlets.datasets.repository._data_config import ECLDataset
from paddlets.datasets.repository._data_config import WTHDataset
from paddlets.datasets.repository._data_config import UNIWTHDataset
from paddlets.datasets.repository._data_config import NABTEMPDataset
from paddlets.datasets.repository._data_config import PSMTRAINDataset
from paddlets.datasets.repository._data_config import PSMTESTDataset
from paddlets.datasets.repository._data_config import BasicMotionsTrainTDataset
from paddlets.datasets.repository._data_config import BasicMotionsTestDataset
from paddlets.datasets.repository._data_config import SMDTestDataset
from paddlets.datasets.repository._data_config import SMDTrainDataset
from paddlets.datasets.repository._data_config import SMAPTrainDataset
from paddlets.datasets.repository._data_config import SMAPTestDataset
from paddlets.datasets.repository._data_config import MSLTrainDataset
from paddlets.datasets.repository._data_config import MSLTestDataset
from paddlets.datasets.repository._data_config import SWATTrainDataset
from paddlets.datasets.repository._data_config import SWATTestDataset
from paddlets.datasets.repository._data_config import EthanolConcentrationTrainTDataset
from paddlets.datasets.repository._data_config import EthanolConcentrationTestTDataset
from paddlets.datasets.repository._data_config import FaceDetectionTrainDataset
from paddlets.datasets.repository._data_config import FaceDetectionTestDataset
from paddlets.datasets.repository._data_config import HandwritingTrainDataset
from paddlets.datasets.repository._data_config import HandwritingTestDataset
from paddlets.datasets.repository._data_config import HeartbeatTrainDataset
from paddlets.datasets.repository._data_config import HeartbeatTestDataset
from paddlets.datasets.repository._data_config import JapaneseVowelsTrainDataset
from paddlets.datasets.repository._data_config import JapaneseVowelsTestDataset
from paddlets.datasets.repository._data_config import PEMSSFTrainDataset
from paddlets.datasets.repository._data_config import PEMSSFTestDataset
from paddlets.datasets.repository._data_config import SelfRegulationSCP1TrainDataset
from paddlets.datasets.repository._data_config import SelfRegulationSCP1TestDataset
from paddlets.datasets.repository._data_config import SelfRegulationSCP2TrainDataset
from paddlets.datasets.repository._data_config import SelfRegulationSCP2TestDataset
from paddlets.datasets.repository._data_config import SpokenArabicDigitsTrainDataset
from paddlets.datasets.repository._data_config import SpokenArabicDigitsTestDataset
from paddlets.datasets.repository._data_config import UWaveGestureLibraryTrainDataset
from paddlets.datasets.repository._data_config import UWaveGestureLibraryTestDataset
from paddlets.datasets.repository._data_config import TrafficDataset
from paddlets.datasets.repository._data_config import ILIDataset
from paddlets.datasets.repository._data_config import ExchangeDataset
from paddlets.datasets.repository._data_config import WeatherDataset
from paddlets.datasets.repository._data_config import M4YearlyTrainDataset
from paddlets.datasets.repository._data_config import M4YearlyTestDataset
from paddlets.datasets.repository._data_config import M4WeeklyTrainDataset
from paddlets.datasets.repository._data_config import M4WeeklyTestDataset
from paddlets.datasets.repository._data_config import M4QuarterlyTrainDataset
from paddlets.datasets.repository._data_config import M4QuarterlyTestDataset
from paddlets.datasets.repository._data_config import M4MonthlyTrainDataset
from paddlets.datasets.repository._data_config import M4MonthlyTestDataset
from paddlets.datasets.repository._data_config import M4DailyTrainDataset
from paddlets.datasets.repository._data_config import M4DailyTestDataset
from paddlets.datasets.repository._data_config import M4HourlyTrainDataset
from paddlets.datasets.repository._data_config import M4HourlyTestDataset

DATASETS = {
    UNIWTHDataset.name: UNIWTHDataset,
    ETTh1Dataset.name: ETTh1Dataset,
    ETTh2Dataset.name: ETTh2Dataset,
    ETTm1Dataset.name: ETTm1Dataset,
    ETTm2Dataset.name: ETTm2Dataset,
    ECLDataset.name: ECLDataset,
    WTHDataset.name: WTHDataset,
    TrafficDataset.name: TrafficDataset,
    ILIDataset.name: ILIDataset,
    ExchangeDataset.name: ExchangeDataset,
    WeatherDataset.name: WeatherDataset,
    M4YearlyTrainDataset.name: M4YearlyTrainDataset,
    M4YearlyTestDataset.name: M4YearlyTestDataset,
    M4WeeklyTrainDataset.name: M4WeeklyTrainDataset,
    M4WeeklyTestDataset.name: M4WeeklyTestDataset,
    M4QuarterlyTrainDataset.name: M4QuarterlyTrainDataset,
    M4QuarterlyTestDataset.name: M4QuarterlyTestDataset,
    M4MonthlyTrainDataset.name: M4MonthlyTrainDataset,
    M4MonthlyTestDataset.name: M4MonthlyTestDataset,
    M4HourlyTrainDataset.name: M4HourlyTrainDataset,
    M4HourlyTestDataset.name: M4HourlyTestDataset,
    M4DailyTrainDataset.name: M4DailyTrainDataset,
    M4DailyTestDataset.name: M4DailyTestDataset,
    NABTEMPDataset.name: NABTEMPDataset,
    PSMTRAINDataset.name: PSMTRAINDataset,
    PSMTESTDataset.name: PSMTESTDataset,
    SMDTrainDataset.name: SMDTrainDataset,
    SMDTestDataset.name: SMDTestDataset,
    SMAPTrainDataset.name: SMAPTrainDataset,
    SMAPTestDataset.name: SMAPTestDataset,
    MSLTrainDataset.name: MSLTrainDataset,
    MSLTestDataset.name: MSLTestDataset,
    BasicMotionsTrainTDataset.name: BasicMotionsTrainTDataset,
    BasicMotionsTestDataset.name: BasicMotionsTestDataset,
    SWATTrainDataset.name: SWATTrainDataset,
    SWATTestDataset.name: SWATTestDataset,
    EthanolConcentrationTrainTDataset.name: EthanolConcentrationTrainTDataset,
    EthanolConcentrationTestTDataset.name: EthanolConcentrationTestTDataset,
    FaceDetectionTrainDataset.name: FaceDetectionTrainDataset,
    FaceDetectionTestDataset.name: FaceDetectionTestDataset,
    HandwritingTrainDataset.name: HandwritingTrainDataset,
    HandwritingTestDataset.name: HandwritingTestDataset,
    HeartbeatTrainDataset.name: HeartbeatTrainDataset,
    HeartbeatTestDataset.name: HeartbeatTestDataset,
    JapaneseVowelsTrainDataset.name: JapaneseVowelsTrainDataset,
    JapaneseVowelsTestDataset.name: JapaneseVowelsTestDataset,
    PEMSSFTrainDataset.name: PEMSSFTrainDataset,
    PEMSSFTestDataset.name: PEMSSFTestDataset,
    SelfRegulationSCP1TrainDataset.name: SelfRegulationSCP1TrainDataset,
    SelfRegulationSCP1TestDataset.name: SelfRegulationSCP1TestDataset,
    SelfRegulationSCP2TrainDataset.name: SelfRegulationSCP2TrainDataset,
    SelfRegulationSCP2TestDataset.name: SelfRegulationSCP2TestDataset,
    SpokenArabicDigitsTrainDataset.name: SpokenArabicDigitsTrainDataset,
    SpokenArabicDigitsTestDataset.name: SpokenArabicDigitsTestDataset,
    UWaveGestureLibraryTrainDataset.name: UWaveGestureLibraryTrainDataset,
    UWaveGestureLibraryTestDataset.name: UWaveGestureLibraryTestDataset,
}


def dataset_list() -> List[str]:
    """
    获取paddlets内置时序数据集名称列表

    Returns:
        List(str): 数据集名称列表
    """
    return list(DATASETS.keys())


def get_dataset(name: str, split=None) -> Union["TSDataset", List[
        "TSDataset"], Tuple[List["TSDataset"], List[Any]]]:
    """
    基于名称获取内置数据集
    
    Args:
        name(str): 数据集名称, 可以从dataset_list获取的列表中选取

    Returns:
        Union["TSDataset", List["TSDataset"], Tuple[List["TSDataset"], List[Any]]]: 基于内置数据集构建好的TSDataset对象或者对象列表
        
    """
    raise_if_not(name in DATASETS, f"Invaild dataset name: {name}")
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
        if not split:
            return TSDataset.load_from_dataframe(df, **dataset.load_param)
        else:
            ts_list = []
            for _, point in split.items():
                ts_list.append(
                    TSDataset.load_from_dataframe(df[point[0]:point[1]], **
                                                  dataset.load_param))
            return ts_list
