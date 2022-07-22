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
    type = "origin",
    path = "https://bj.bcebos.com/paddlets/ETTh1.csv",
    load_param = {
        "target_cols": "OT",
        "time_col": "date",
        "observed_cov_cols": ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"],
        "freq": "1h"
    }
)

# 2> ETT data with 15 minutes frequency, https://github.com/zhouhaoyi/ETDataset
ETTm1Dataset = DatasetConfig(
    name = "ETTm1",
    type = "origin",
    path = "https://bj.bcebos.com/paddlets/ETTm1.csv",
    load_param = {
        "target_cols": "OT",
        "time_col": "date",
        "observed_cov_cols": ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"],
        "freq": "15T"
    }
)

# 3> Fixed ECL data, https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014
ECLDataset = DatasetConfig(
    name = "ECL",
    type = "origin",
    path = "https://bj.bcebos.com/paddlets/ECL.csv",
    load_param = {
        "target_cols": "MT_320",
        "time_col": "date",
        "observed_cov_cols": ["MT_{:0>3}".format(x) for x in range(320)],
        "freq": "1h"
    }
)

# 4> Fixed weather data, https://www.ncei.noaa.gov/data/local-climatological-data/
WTHDataset = DatasetConfig(
    name = "WTH",
    type = "origin",
    path = "https://bj.bcebos.com/paddlets/WTH.csv",
    load_param = {
        "target_cols": "WetBulbCelsius",
        "time_col": "date",
        "observed_cov_cols": ["Visibility", "DryBulbFarenheit", "DryBulbCelsius", "WetBulbFarenheit", \
                              "DewPointFarenheit", "DewPointCelsius", "RelativeHumidity", "WindSpeed", \
                              "WindDirection", "StationPressure", "Altimeter"],
        "freq": "1h"
    }
)

UNIWTHDataset = DatasetConfig(
    name = "UNI_WTH",
    type = "origin", #local代表本地文件，origin代表远程文件
    path = "https://bj.bcebos.com/paddlets/UNI_WTH.csv",
    load_param = {
        "target_cols": "WetBulbCelsius",
        "time_col": "date",
        "freq": "1h"
    }  
)

