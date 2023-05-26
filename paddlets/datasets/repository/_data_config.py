# !/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Public data sets for time series.
"""

from collections import namedtuple

DatasetConfig = namedtuple('DatasetConfig',
                           ["name", "type", "path", "load_param"])

# 1> ETT data with 1 hour frequency, https://github.com/zhouhaoyi/ETDataset
ETTh1Dataset = DatasetConfig(
    name="ETTh1",
    type="forecasting",
    path="https://bj.bcebos.com/paddlets/ETTh1.csv",
    load_param={
        "target_cols": "OT",
        "time_col": "date",
        "observed_cov_cols": ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"],
        "freq": "1h",
        "dtype": "float32"
    })

# 2> ETT data with 1 hour frequency, https://github.com/zhouhaoyi/ETDataset
ETTh2Dataset = DatasetConfig(
    name="ETTh2",
    type="forecasting",
    path="https://bj.bcebos.com/paddlets/ETTh2.csv",
    load_param={
        "target_cols": "OT",
        "time_col": "date",
        "observed_cov_cols": ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"],
        "freq": "1h",
        "dtype": "float32"
    })

# 3> ETT data with 15 minutes frequency, https://github.com/zhouhaoyi/ETDataset
ETTm1Dataset = DatasetConfig(
    name="ETTm1",
    type="forecasting",
    path="https://bj.bcebos.com/paddlets/ETTm1.csv",
    load_param={
        "target_cols": "OT",
        "time_col": "date",
        "observed_cov_cols": ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"],
        "freq": "15T",
        "dtype": "float32"
    })

# 4> ETT data with 15 minutes frequency, https://github.com/zhouhaoyi/ETDataset
ETTm2Dataset = DatasetConfig(
    name="ETTm2",
    type="forecasting",
    path="https://bj.bcebos.com/paddlets/ETTm2.csv",
    load_param={
        "target_cols": "OT",
        "time_col": "date",
        "observed_cov_cols": ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"],
        "freq": "15T",
        "dtype": "float32"
    })

# 5> Fixed ECL data, https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014
ECLDataset = DatasetConfig(
    name="ECL",
    type="forecasting",
    path="https://bj.bcebos.com/paddlets/ECL.csv",
    load_param={
        "target_cols": "MT_320",
        "time_col": "date",
        "observed_cov_cols": ["MT_{:0>3}".format(x) for x in range(320)],
        "freq": "1h",
        "dtype": "float32"
    })

# 6> Fixed weather data, https://www.ncei.noaa.gov/data/local-climatological-data/
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
    name="UNI_WTH",
    type="forecasting",
    path="https://bj.bcebos.com/paddlets/UNI_WTH.csv",
    load_param={
        "target_cols": "WetBulbCelsius",
        "time_col": "date",
        "freq": "1h",
        "dtype": "float32"
    })

# 7> Traffic data, https://pems.dot.ca.gov/
TrafficDataset = DatasetConfig(
    name="Traffic",
    type="forecasting",
    path="https://bj.bcebos.com/paddlets/traffic.csv",
    load_param={
        "target_cols": "OT",
        "time_col": "date",
        "observed_cov_cols": ["{}".format(x) for x in range(861)],
        "freq": "1h",
        "dtype": "float32"
    })

# 8> ILIness data, https://gis.cdc.gov/grasp/fluview/fluportaldashboard.html
ILIDataset = DatasetConfig(
    name = "ILI",
    type = "forecasting",
    path = "https://bj.bcebos.com/paddlets/national_illness.csv",
    load_param = {
        "target_cols": "OT",
        "time_col": "date",
        "observed_cov_cols": ['% WEIGHTED ILI', '%UNWEIGHTED ILI', \
                              'AGE 0-4', 'AGE 5-24', 'ILITOTAL', 'NUM. OF PROVIDERS'],
        "freq": "7D",
        "dtype": "float32"
    }
)

# 9> Exchange-rate data, https://github.com/laiguokun/multivariate-time-series-data
ExchangeDataset = DatasetConfig(
    name="Exchange",
    type="forecasting",
    path="https://bj.bcebos.com/paddlets/exchange_rate.csv",
    load_param={
        "target_cols": "OT",
        "time_col": "date",
        "observed_cov_cols": ['0', '1', '2', '3', '4', '5', '6'],
        "freq": "1D",
        "dtype": "float32"
    })

# 10> Weather data, https://www.bgc-jena.mpg.de/wetter/
WeatherDataset = DatasetConfig(
    name = "Weather",
    type = "forecasting",
    path = "https://bj.bcebos.com/paddlets/weather.csv",
    load_param = {
        "target_cols": "OT",
        "time_col": "date",
        "observed_cov_cols": ['p (mbar)','T (degC)', 'Tpot (K)', 'Tdew (degC)','rh (%)',\
                            'VPmax (mbar)', 'VPact (mbar)','VPdef (mbar)','sh (g/kg)','H2OC (mmol/mol)', \
                            'rho (g/m**3)','wv (m/s)','max. wv (m/s)','wd (deg)','rain (mm)','raining (s)', \
                            'SWDR (W/m)', 'PAR (_ol/m/s)', 'max. PAR (_ol/m/s)','Tlog (degC)'],
        "freq": "10T",
        "dtype": "float32"
    }
)

# 11> M4 data, https://github.com/M4Competition/ M4-methods/tree/master/Dataset
M4YearTrainDataset = DatasetConfig(
    name="M4-Yearly-train",
    type="forecasting",
    path="https://bj.bcebos.com/paddlets/m4/Yearly-train-t.csv",
    load_param={"freq": 1,
                "dtype": "float32"})

M4YearTestDataset = DatasetConfig(
    name="M4-Yearly-test",
    type="forecasting",
    path="https://bj.bcebos.com/paddlets/m4/Yearly-test-t.csv",
    load_param={"freq": 1,
                "dtype": "float32"})

M4WeekTrainDataset = DatasetConfig(
    name="M4-Weekly-train",
    type="forecasting",
    path="https://bj.bcebos.com/paddlets/m4/Weekly-train-t.csv",
    load_param={"freq": 1,
                "dtype": "float32"})

M4WeekTestDataset = DatasetConfig(
    name="M4-Weekly-test",
    type="forecasting",
    path="https://bj.bcebos.com/paddlets/m4/Weekly-test-t.csv",
    load_param={"freq": 1,
                "dtype": "float32"})

M4QuarterTrainDataset = DatasetConfig(
    name="M4-Quarterly-train",
    type="forecasting",
    path="https://bj.bcebos.com/paddlets/m4/Quarterly-train-t.csv",
    load_param={"freq": 1,
                "dtype": "float32"})

M4QuarterTestDataset = DatasetConfig(
    name="M4-Quarterly-test",
    type="forecasting",
    path="https://bj.bcebos.com/paddlets/m4/Quarterly-test-t.csv",
    load_param={"freq": 1,
                "dtype": "float32"})

M4MonthTrainDataset = DatasetConfig(
    name="M4-Monthly-train",
    type="forecasting",
    path="https://bj.bcebos.com/paddlets/m4/Monthly-train-t.csv",
    load_param={"freq": 1,
                "dtype": "float32"})

M4MonthTestDataset = DatasetConfig(
    name="M4-Monthly-test",
    type="forecasting",
    path="https://bj.bcebos.com/paddlets/m4/Monthly-test-t.csv",
    load_param={"freq": 1,
                "dtype": "float32"})

M4HourTrainDataset = DatasetConfig(
    name="M4-Hourly-train",
    type="forecasting",
    path="https://bj.bcebos.com/paddlets/m4/Hourly-train-t.csv",
    load_param={"freq": 1,
                "dtype": "float32"})

M4HourTestDataset = DatasetConfig(
    name="M4-Hourly-test",
    type="forecasting",
    path="https://bj.bcebos.com/paddlets/m4/Hourly-test-t.csv",
    load_param={"freq": 1,
                "dtype": "float32"})

M4DaiTrainDataset = DatasetConfig(
    name="M4-Daily-train",
    type="forecasting",
    path="https://bj.bcebos.com/paddlets/m4/Daily-train-t.csv",
    load_param={"freq": 1,
                "dtype": "float32"})

M4DaiTestDataset = DatasetConfig(
    name="M4-Daily-test",
    type="forecasting",
    path="https://bj.bcebos.com/paddlets/m4/Daily-test-t.csv",
    load_param={"freq": 1,
                "dtype": "float32"})

# 6> nab machine temperature data, https://github.com/numenta/NAB/blob/master/data/realKnownCause/machine_temperature_system_failure.csv
NABTEMPDataset = DatasetConfig(
    name="NAB_TEMP",
    type="anomaly",
    path="https://bj.bcebos.com/paddlets/NAB_TEMP.csv",
    load_param={
        "label_col": "label",
        "time_col": "timestamp",
        "feature_cols": ["value"],
        "freq": "5T"
    })

# 7> psm train data, https://cloud.tsinghua.edu.cn/d/9605612594f0423f891e/files/?p=%2FPSM%2Ftrain.csv
PSMTRAINDataset = DatasetConfig(
    name="psm_train",
    type="anomaly",
    path="https://bj.bcebos.com/paddlets/psm_train.csv",
    load_param={
        "time_col": "timestamp",
        "feature_cols": ["feature_" + str(i) for i in range(25)],
        "freq": 1
    })

# 8> psm test data, https://cloud.tsinghua.edu.cn/d/9605612594f0423f891e/files/?p=%2FPSM%2Ftest.csv
PSMTESTDataset = DatasetConfig(
    name="psm_test",
    type="anomaly",
    path="https://bj.bcebos.com/paddlets/psm_test.csv",
    load_param={
        "label_col": "label",
        "time_col": "timestamp",
        "feature_cols": ["feature_" + str(i) for i in range(25)],
        "freq": 1
    })

# 9> BasicMotions_Test.csv
BasicMotionsTestDataset = DatasetConfig(
    name="BasicMotions_Test",
    type="classification",
    path="https://bj.bcebos.com/paddlets/BasicMotions_Test.csv",
    load_param={
        "time_col": "index",
        "group_id": "7",
        "target_cols": ["0", "1", "2", "3", "4", "5"],
        "static_cov_cols": ["label"]
    })

# 10> BasicMotions_Train.csv
BasicMotionsTrainTDataset = DatasetConfig(
    name="BasicMotions_Train",
    type="classification",
    path="https://bj.bcebos.com/paddlets/BasicMotions_Train.csv",
    load_param={
        "time_col": "index",
        "group_id": "7",
        "target_cols": ["0", "1", "2", "3", "4", "5"],
        "static_cov_cols": ["label"]
    })
