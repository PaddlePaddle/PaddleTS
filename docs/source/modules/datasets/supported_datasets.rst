Supported Datasets
==================

PaddleTS currently supports dozens of datasets including the four major tasks of timing prediction, timing anomaly detection, timing completion, and timing classification.

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