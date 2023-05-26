Supported Datasets
==================

PaddleTS currently supports dozens of datasets including the four major tasks of time series prediction, time series anomaly detection, time series imputation, and time series classification.

Time Series Prediction
----------------------

Time series forecasting is one of the most important tasks in time series. At present, we integrate data from multiple scenarios such as electricity, weather, disease, and exchange rate for time series forecasting.

1. ETT-small
^^^^^^^^^^^^
* Data source: https://github.com/zhouhaoyi/ETDataset
* Data brief introduction: Power transformer data containing 7 variables, used to forecast electricity demand in various regions. Divided into minute-level and hour-level data
* Dataset names: ETTh1, ETTh2, ETTm1, ETTm2.

2. Weather
^^^^^^^^^^
* Data source: https://www.bgc-jena.mpg.de/wetter
* Data introduction: contains 22 variables sampled weather data every ten minutes, used to predict the weather.
* Dataset name: Weather.

3. ILI
^^^^^^
* Data source: https://gis.cdc.gov/grasp/fluview/fluportaldashboard.html
* Data brief introduction: Contains daily sampling of 22 variables, influenza disease data spanning 19 years, used to predict the proportion of influenza patients.
* Dataset name: ILI.

4. Traffic
^^^^^^^^^^
* Data source: https://pems.dot.ca.gov
* Data brief introduction: Contains 862 sensors, hourly road occupancy ratio.
* Dataset name: Traffic.

5. Exchange
^^^^^^^^^^^
* Data source: https://github.com/laiguokun/multivariate-time-series-data
* Data profile: Contains daily sampled exchange rate data spanning 36 years.
* Dataset name: Exchange.

6. ECL
^^^^^^
* Data source: https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014
* Data introduction: Contains hourly electricity consumption data of 321 customers spanning 4 years, used to predict electricity demand in various regions.
* Dataset name: ECL.

6. M4
^^^^^^
* Data source: https://www.kaggle.com/datasets/yogesh94/m4-forecasting-competition-dataset
* Data introduction: 100,000 pieces of univariate data including demography, finance, industry, macroeconomics, microeconomics, etc.
* Dataset name: M4-Yearly-train, M4-Yearly-test, M4-Monthly-train, M4-Monthly-test, M4-Weekly-train, M4-Weekly-test, M4-Daily-train, M4-Daily-test, M4-Hourly-train, M4-Hourly-test.

7. WTH
^^^^^^
* Data source: https://www.ncei.noaa.gov/data/local-climatological-data/
* Data introduction: hourly weather data with 12 variables spanning 4 years, used to predict the weather in various regions.
* Dataset names: WTH, UNI_WTH.


Time Series Anomaly Detection
-----------------------------

Detecting anomalies from monitoring data is vital to industrial maintenance.  We provide widely-used anomaly detection benchmarks: SMD, MSL, SMAP, SWaT, PSM, covering service monitoring, space & earth exploration, and water treatment applications.

1. SMD 
^^^^^^^^^^^^
* Data source: https://github.com/NetManAIOps/OmniAnomaly/tree/master/ServerMachineDataset
* Data brief introduction: SMD (Server Machine Dataset is a 5-week-long dataset that is collected from a large Internet company with 38 dimensions.
* Dataset names: smd_train, smd_test.

2. SMAP 
^^^^^^^^^^^^
* Data source: https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv
* Data brief introduction: SMAP is from NASA with 25 dimensions, which contain the telemetry anomaly data derived from the Incident Surprise Anomaly (ISA) reports of spacecraft monitoring systems.
* Dataset names: smap_train, smap_test.

3. MSL 
^^^^^^^^^^^^
* Data source: https://s3-us-west-2.amazonaws.com/telemanom/data.zip
* Data brief introduction: MSL (Mars Science Laboratory) is from NASA with 55 dimensions, which contain the telemetry anomaly data derived from the Incident Surprise Anomaly (ISA) reports of spacecraft monitoring systems.
* Dataset names: msl_train, msl_test.

4. SWAT 
^^^^^^^^^^^^
* Data source: https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info
* Data brief introduction: ) SWaT(Secure Water Treatment) is obtained from 51 sensors of the critical infrastructure system under continuous operations.
* Dataset names: swat_train, swat_test.

5. PSM 
^^^^^^^^^^^^
* Data source: https://cloud.tsinghua.edu.cn/d/9605612594f0423f891e/files/?p=%2FPSM%2Ftrain.csv
* Data brief introduction: PSM (Pooled Server Metrics) is collected internally from multiple application server nodes at eBay with 26 dimensions.
* Dataset names: psm_train, psm_test.

6. NAB_TEMP
^^^^^^^^^^^^
* Data source: https://github.com/numenta/NAB
* Data brief introduction: The Numenta Anomaly Benchmark (NAB) provides streaming data to research anomaly detection algorithms. NAB_TEMP is the temperature dataset.
* Dataset names: NAB_TEMP


Time Series Classification
---------------------------


1. UEA
^^^^^^^^^^^^
* Data source: https://www.timeseriesclassification.com/index.php
* Data brief introduction: UEA Time Series Classification dataset includes 10 multivariate datasets, covering the gesture, action and audio recognition, medical diagnosis by heartbeat monitoring and other practical tasks. 
* Dataset names: EthanolConcentration_Train, EthanolConcentration_Test, FaceDetection_Train, FaceDetection_Test, Handwriting_Train, Handwriting_Test, Heartbeat_Train, Heartbeat_Test, JapaneseVowels_Train, JapaneseVowels_Test, PEMSSF_Train, PEMSSF_Test, SelfRegulationSCP1_Train, SelfRegulationSCP1_Test, SelfRegulationSCP2_Train, SelfRegulationSCP2_Test, SpokenArabicDigits_Train, SpokenArabicDigits_Test, UWaveGestureLibrary_Train, UWaveGestureLibrary_Test.
    

2. BasicMotions
^^^^^^^^^^^^^^^
* Data source: https://timeseriesclassification.com/description.php?Dataset=BasicMotions
* Data brief introduction: The data was generated as part of a student project where four students performed four activities whilst wearing a smart watch. here are classes: walking, resting, running and badminton. Participants were required to record motion a total of five times, and the data is sampled once every tenth of a second, for a ten second period.
* Dataset names: BasicMotions_Train, BasicMotions_Test.