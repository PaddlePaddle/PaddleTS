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

# 11> SDM_test data
SMDTestDataset = DatasetConfig(
    name = "smd_test",
    type = "anomaly",
    path = "https://paddlets.bj.bcebos.com/SMD_test.csv",
    load_param = {
        "label_col": "label",
        "feature_cols": [str(i) for i in range(38)],
        "freq": 1
    }
)

# 12> SMAP_train 
SMAPTrainDataset = DatasetConfig(
    name = "smap_train",
    type = "anomaly",
    path = "https://paddlets.bj.bcebos.com/SMAP_train.csv",
    load_param = {
        "feature_cols": [str(i) for i in range(25)],
        "freq": 1
    }
)

# 12> SMAP_test
SMAPTestDataset = DatasetConfig(
    name = "smap_test",
    type = "anomaly",
    path = "https://paddlets.bj.bcebos.com/SMAP_test.csv",
    load_param = {
        "label_col": "label",
        "feature_cols": [str(i) for i in range(25)],
        "freq": 1
    }
)

# 13> MSL_train
MSLTrainDataset = DatasetConfig(
    name = "msl_train",
    type = "anomaly",
    path = "https://paddlets.bj.bcebos.com/MSL_train.csv",
    load_param = {
        "feature_cols": [str(i) for i in range(55)],
        "freq": 1
    }
)

# 13> MSL_test
MSLTestDataset = DatasetConfig(
    name = "msl_test",
    type = "anomaly",
    path = "https://paddlets.bj.bcebos.com/MSL_test.csv",
    load_param = {
        "label_col": "label",
        "feature_cols": [str(i) for i in range(55)],
        "freq": 1
    }
)

# 14> SwaT train
SWATTrainDataset = DatasetConfig(
    name = "swat_train",
    type = "anomaly",
    path = "https://paddlets.bj.bcebos.com/swat_train.csv",
    load_param = {
        "feature_cols": ['FIT101', 'LIT101', ' MV101', 'P101', 'P102', ' AIT201', 'AIT202',
            'AIT203', 'FIT201', ' MV201', ' P201', ' P202', 'P203', ' P204', 'P205',
            'P206', 'DPIT301', 'FIT301', 'LIT301', 'MV301', 'MV302', ' MV303',
            'MV304', 'P301', 'P302', 'AIT401', 'AIT402', 'FIT401', 'LIT401', 'P401',
            'P402', 'P403', 'P404', 'UV401', 'AIT501', 'AIT502', 'AIT503', 'AIT504',
            'FIT501', 'FIT502', 'FIT503', 'FIT504', 'P501', 'P502', 'PIT501',
            'PIT502', 'PIT503', 'FIT601', 'P601', 'P602', 'P603'],
        "freq": 1
    }
)

# 14> SwaT test
SWATTestDataset = DatasetConfig(
    name = "swat_test",
    type = "anomaly",
    path = "https://paddlets.bj.bcebos.com/swat_test.csv",
    load_param = {
        "label_col": "Normal/Attack",
        "feature_cols": ['FIT101', 'LIT101', ' MV101', 'P101', 'P102', ' AIT201', 'AIT202',
            'AIT203', 'FIT201', ' MV201', ' P201', ' P202', 'P203', ' P204', 'P205',
            'P206', 'DPIT301', 'FIT301', 'LIT301', 'MV301', 'MV302', ' MV303',
            'MV304', 'P301', 'P302', 'AIT401', 'AIT402', 'FIT401', 'LIT401', 'P401',
            'P402', 'P403', 'P404', 'UV401', 'AIT501', 'AIT502', 'AIT503', 'AIT504',
            'FIT501', 'FIT502', 'FIT503', 'FIT504', 'P501', 'P502', 'PIT501',
            'PIT502', 'PIT503', 'FIT601', 'P601', 'P602', 'P603'],
        "freq": 1
    }
)


# 15> BasicMotions_Train.csv
EthanolConcentrationTrainTDataset = DatasetConfig(
    name = "EthanolConcentration_Train",
    type = "classification",
    path = "https://paddlets.bj.bcebos.com/classification/EthanolConcentration_TRAIN.csv",
    load_param = {
        "time_col": "time",
        "group_id": "group_id", 
        "target_cols": ['dim_0', 'dim_1', 'dim_2'], 
        "static_cov_cols": ["label"]
    }
)

EthanolConcentrationTestTDataset = DatasetConfig(
    name = "EthanolConcentration_Test",
    type = "classification",
    path = "https://paddlets.bj.bcebos.com/classification/EthanolConcentration_TEST.csv",
    load_param = {
        "time_col": "time",
        "group_id": "group_id", 
        "target_cols": ['dim_0', 'dim_1', 'dim_2'], 
        "static_cov_cols": ["label"]
    }
)

FaceDetectionTrainDataset = DatasetConfig(
    name = "FaceDetection_Train",
    type = "classification",
    path = "https://paddlets.bj.bcebos.com/classification/FaceDetection_TRAIN.csv",
    load_param = {
        "time_col": "time",
        "group_id": "group_id", 
        "target_cols": ["dim_" + str(i) for i in range(144)],
        "static_cov_cols": ["label"]
    }
)

FaceDetectionTestDataset = DatasetConfig(
    name = "FaceDetection_Train",
    type = "classification",
    path = "https://paddlets.bj.bcebos.com/classification/FaceDetection_TEST.csv",
    load_param = {
        "time_col": "time",
        "group_id": "group_id", 
        "target_cols": ["dim_" + str(i) for i in range(144)],
        "static_cov_cols": ["label"]
    }
)

HandwritingTrainDataset = DatasetConfig(
    name = "Handwriting_Train",
    type = "classification",
    path = "https://paddlets.bj.bcebos.com/classification/Handwriting_TRAIN.csv",
    load_param = {
        "time_col": "time",
        "group_id": "group_id", 
        "target_cols": ['dim_0', 'dim_1', 'dim_2'], 
        "static_cov_cols": ["label"]
    }
)

HandwritingTestDataset = DatasetConfig(
    name = "Handwriting_Test",
    type = "classification",
    path = "https://paddlets.bj.bcebos.com/classification/Handwriting_TEST.csv",
    load_param = {
        "time_col": "time",
        "group_id": "group_id", 
        "target_cols":['dim_0', 'dim_1', 'dim_2'], 
        "static_cov_cols": ["label"]
    }
)

HeartbeatTrainDataset = DatasetConfig(
    name = "Heartbeat_Train",
    type = "classification",
    path = "https://paddlets.bj.bcebos.com/classification/Heartbeat_TRAIN.csv",
    load_param = {
        "time_col": "time",
        "group_id": "group_id", 
        "target_cols": ["dim_" + str(i) for i in range(61)],
        "static_cov_cols": ["label"]
    }
)

HeartbeatTestDataset = DatasetConfig(
    name = "Heartbeat_Test",
    type = "classification",
    path = "https://paddlets.bj.bcebos.com/classification/Heartbeat_TEST.csv",
    load_param = {
        "time_col": "time",
        "group_id": "group_id", 
        "target_cols": ["dim_" + str(i) for i in range(61)],
        "static_cov_cols": ["label"]
    }
)

JapaneseVowelsTrainDataset = DatasetConfig(
    name = "JapaneseVowels_Train",
    type = "classification",
    path = "https://paddlets.bj.bcebos.com/classification/JapaneseVowels_TRAIN.csv",
    load_param = {
        "time_col": "time",
        "group_id": "group_id", 
        "target_cols": ["dim_" + str(i) for i in range(12)],
        "static_cov_cols": ["label"]
    }
)

JapaneseVowelsTestDataset = DatasetConfig(
    name = "JapaneseVowels_Test",
    type = "classification",
    path = "https://paddlets.bj.bcebos.com/classification/JapaneseVowels_TEST.csv",
    load_param = {
        "time_col": "time",
        "group_id": "group_id", 
        "target_cols": ["dim_" + str(i) for i in range(12)],
        "static_cov_cols": ["label"]
    }
)

PEMSSFTrainDataset = DatasetConfig(
    name = "PEMSSF_Train",
    type = "classification",
    path = "https://paddlets.bj.bcebos.com/classification/PEMS-SF_TRAIN.csv",
    load_param = {
        "time_col": "time",
        "group_id": "group_id", 
        "target_cols": ["dim_" + str(i) for i in range(12)],
        "static_cov_cols": ["label"]
    }
)

PEMSSFTestDataset = DatasetConfig(
    name = "PEMSSF_Test",
    type = "classification",
    path = "https://paddlets.bj.bcebos.com/classification/PEMS-SF_TEST.csv",
    load_param = {
        "time_col": "time",
        "group_id": "group_id", 
        "target_cols": ["dim_" + str(i) for i in range(12)],
        "static_cov_cols": ["label"]
    }
)

SelfRegulationSCP1TrainDataset = DatasetConfig(
    name = "SelfRegulationSCP1_Train",
    type = "classification",
    path = "https://paddlets.bj.bcebos.com/classification/SelfRegulationSCP1_TRAIN.csv",
    load_param = {
        "time_col": "time",
        "group_id": "group_id", 
        "target_cols": ["dim_" + str(i) for i in range(6)],
        "static_cov_cols": ["label"]
    }
)

SelfRegulationSCP1TestDataset = DatasetConfig(
    name = "SelfRegulationSCP1_Test",
    type = "classification",
    path = "https://paddlets.bj.bcebos.com/classification/SelfRegulationSCP1_TEST.csv",
    load_param = {
        "time_col": "time",
        "group_id": "group_id", 
        "target_cols": ["dim_" + str(i) for i in range(6)],
        "static_cov_cols": ["label"]
    }
)

SelfRegulationSCP2TrainDataset = DatasetConfig(
    name = "SelfRegulationSCP2_Train",
    type = "classification",
    path = "https://paddlets.bj.bcebos.com/classification/SelfRegulationSCP2_TRAIN.csv",
    load_param = {
        "time_col": "time",
        "group_id": "group_id", 
        "target_cols": ["dim_" + str(i) for i in range(7)],
        "static_cov_cols": ["label"]
    }
)

SelfRegulationSCP2TestDataset = DatasetConfig(
    name = "SelfRegulationSCP2_Test",
    type = "classification",
    path = "https://paddlets.bj.bcebos.com/classification/SelfRegulationSCP2_TEST.csv",
    load_param = {
        "time_col": "time",
        "group_id": "group_id", 
        "target_cols": ["dim_" + str(i) for i in range(7)],
        "static_cov_cols": ["label"]
    }
)

SpokenArabicDigitsTrainDataset = DatasetConfig(
    name = "SpokenArabicDigits_Train",
    type = "classification",
    path = "https://paddlets.bj.bcebos.com/classification/SpokenArabicDigits_TRAIN.csv",
    load_param = {
        "time_col": "time",
        "group_id": "group_id", 
        "target_cols": ["dim_" + str(i) for i in range(7)],
        "static_cov_cols": ["label"]
    }
)

SpokenArabicDigitsTestDataset = DatasetConfig(
    name = "SpokenArabicDigits_Test",
    type = "classification",
    path = "https://paddlets.bj.bcebos.com/classification/SpokenArabicDigits_TEST.csv",
    load_param = {
        "time_col": "time",
        "group_id": "group_id", 
        "target_cols": ["dim_" + str(i) for i in range(7)],
        "static_cov_cols": ["label"]
    }
)

UWaveGestureLibraryTrainDataset = DatasetConfig(
    name = "UWaveGestureLibrary_Train",
    type = "classification",
    path = "https://paddlets.bj.bcebos.com/classification/UWaveGestureLibrary_TRAIN.csv",
    load_param = {
        "time_col": "time",
        "group_id": "group_id", 
        "target_cols": ["dim_" + str(i) for i in range(3)],
        "static_cov_cols": ["label"]
    }
)

UWaveGestureLibraryTestDataset = DatasetConfig(
    name = "UWaveGestureLibrary_Test",
    type = "classification",
    path = "https://paddlets.bj.bcebos.com/classification/UWaveGestureLibrary_TEST.csv",
    load_param = {
        "time_col": "time",
        "group_id": "group_id", 
        "target_cols": ["dim_" + str(i) for i in range(3)],
        "static_cov_cols": ["label"]
    }
)
