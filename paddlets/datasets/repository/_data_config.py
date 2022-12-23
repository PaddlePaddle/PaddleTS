# !/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
Public data sets for time series.
"""

from collections import namedtuple


DatasetConfig = namedtuple('DatasetConfig', [
    "name",
    "type",
    "path",
    "load_param"
])

# 1> ETT data with 1 hour frequency, https://github.com/zhouhaoyi/ETDataset
ETTh1Dataset = DatasetConfig(
    name = "ETTh1",
    type = "forecasting",
    path = "https://bj.bcebos.com/paddlets/ETTh1.csv",
    load_param = {
        "target_cols": "OT",
        "time_col": "date",
        "observed_cov_cols": ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"],
        "freq": "1h",
        "dtype": "float32"
    }
)

# 2> ETT data with 15 minutes frequency, https://github.com/zhouhaoyi/ETDataset
ETTm1Dataset = DatasetConfig(
    name = "ETTm1",
    type = "forecasting",
    path = "https://bj.bcebos.com/paddlets/ETTm1.csv",
    load_param = {
        "target_cols": "OT",
        "time_col": "date",
        "observed_cov_cols": ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"],
        "freq": "15T",
        "dtype": "float32"
    }
)

# 3> Fixed ECL data, https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014
ECLDataset = DatasetConfig(
    name = "ECL",
    type = "forecasting",
    path = "https://bj.bcebos.com/paddlets/ECL.csv",
    load_param = {
        "target_cols": "MT_320",
        "time_col": "date",
        "observed_cov_cols": ["MT_{:0>3}".format(x) for x in range(320)],
        "freq": "1h",
        "dtype": "float32"
    }
)

# 4> Fixed weather data, https://www.ncei.noaa.gov/data/local-climatological-data/
WTHDataset = DatasetConfig(
    name = "WTH",
    type = "forecasting",
    path = "https://bj.bcebos.com/paddlets/WTH.csv",
    load_param = {
        "target_cols": "WetBulbCelsius",
        "time_col": "date",
        "observed_cov_cols": ["Visibility", "DryBulbFarenheit", "DryBulbCelsius", "WetBulbFarenheit", \
                              "DewPointFarenheit", "DewPointCelsius", "RelativeHumidity", "WindSpeed", \
                              "WindDirection", "StationPressure", "Altimeter"],
        "freq": "1h",
        "dtype": "float32"
    }
)

UNIWTHDataset = DatasetConfig(
    name = "UNI_WTH",
    type = "forecasting",
    path = "https://bj.bcebos.com/paddlets/UNI_WTH.csv",
    load_param = {
        "target_cols": "WetBulbCelsius",
        "time_col": "date",
        "freq": "1h",
        "dtype": "float32"
    }  
)

# 6> nab machine temperature data, https://github.com/numenta/NAB/blob/master/data/realKnownCause/machine_temperature_system_failure.csv
NABTEMPDataset = DatasetConfig(
    name = "NAB_TEMP",
    type = "anomaly",
    path = "https://bj.bcebos.com/paddlets/NAB_TEMP.csv",
    load_param = {
        "label_col": "label",
        "time_col": "timestamp",
        "feature_cols": ["value"],
        "freq": "5T"
    }
)

# 7> psm train data, https://cloud.tsinghua.edu.cn/d/9605612594f0423f891e/files/?p=%2FPSM%2Ftrain.csv
PSMTRAINDataset = DatasetConfig(
    name = "psm_train",
    type = "anomaly",
    path = "https://bj.bcebos.com/paddlets/psm_train.csv",
    load_param = {
        "time_col": "timestamp",
        "feature_cols": ["feature_" + str(i) for i in range(25)],
        "freq": 1
    }
)

# 8> psm test data, https://cloud.tsinghua.edu.cn/d/9605612594f0423f891e/files/?p=%2FPSM%2Ftest.csv
PSMTESTDataset = DatasetConfig(
    name = "psm_test",
    type = "anomaly",
    path = "https://bj.bcebos.com/paddlets/psm_test.csv",
    load_param = {
        "label_col": "label",
        "time_col": "timestamp",
        "feature_cols": ["feature_" + str(i) for i in range(25)],
        "freq": 1
    }
)

# 9> BasicMotions_Test.csv
BasicMotionsTestDataset = DatasetConfig(
    name = "BasicMotions_Test",
    type = "classification",
    path = "https://bj.bcebos.com/paddlets/BasicMotions_Test.csv",
    load_param = {
        "time_col": "index",
        "group_id": "7", 
        "target_cols": ["0", "1", "2", "3", "4", "5"], 
        "static_cov_cols": ["label"]
    }
)

# 10> BasicMotions_Train.csv
BasicMotionsTrainTDataset = DatasetConfig(
    name = "BasicMotions_Train",
    type = "classification",
    path = "https://bj.bcebos.com/paddlets/BasicMotions_Train.csv",
    load_param = {
        "time_col": "index",
        "group_id": "7", 
        "target_cols": ["0", "1", "2", "3", "4", "5"], 
        "static_cov_cols": ["label"]
    }
)
