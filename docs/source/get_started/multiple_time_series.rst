======================================
Joint Training of Multiple Time Series
======================================

When we need to train and predict multiple time series data, one of the methods is to create a model for each group of time series data for independent training and prediction; but in many practical case, we hope to create only one model with multiple time series data, which can better improve efficiency and obtain better model effects.
For the needs of joint training of Multiple Time Series, PaddleTS provides full process support from data load, data transformation and model training.

1. Multiple Time Series Load
----------------------------

PaddleTS support automatically load data groups based on the `group_id` in the original data, such as device ID or other attributes, with the same `group_id` represents a group of time series. 
The time indexes of different groups of time series can be repeated.


.. code:: python

   #Build DataFrame with group_id
   
   import pandas as pd
   import numpy as np
   sample = pd.DataFrame(np.random.randn(200, 3), columns=['a', 'c', 'd'])
   sample['id'] = pd.Series([0]*80 + [1]*120, name='id')

   #Load TSDatasets by group_id
   from paddlets import TSDataset
   tsdatasets = TSDataset.load_from_dataframe(
       df=sample,
       group_id='id',
       target_cols='a',
       observed_cov_cols=['c', 'd'],
       #static_cov_cols='id'
   )
   
   print(f" The type of tsdatasets is {type(tsdatasets)},\n \
   and the length of tsdatasets is {len(tsdatasets)},\n \
   the length of first tsdataset target is {len(tsdatasets[0].target)},\n \
   the length of second tsdataset target is {len(tsdatasets[1].target)}")
   # The type of tsdatasets is <class 'list'>,
   # and the length of tsdatasets is 2,
   # the length of first tsdataset target is 80,
   # the length of second tsdataset target is 120


Whether to pass `static_cov_cols` and assign them as `group_id` is optional. If set, the `group_id` will be added to training model as a static covariate. 
Note that not all PaddleTS models currently support the input of static covariates. Please refer to the introduction of the model section for details.

One can also load time series data from different groups to build TSDatasets separately, and then form a List for subsequent joint training. 
It should be noted that all TSDatasets in joint training require data homogeneity, that is, the columns and dtypes attributes are the same.

.. code:: python

   ts1 = TSDataset.load_from_dataframe(...)
   ts2 = TSDataset.load_from_dataframe(...)
   #ts1.columns == ts2.columns and ts1.dtypes = ts2.dtypes
   tsdatasets = [ts1, ts2]


2. Multiple Time Series Transformation
--------------------------------------

When data transformation is required before joint training of multiple time series data, such as normalization, feature generation and other operations, joint transformation of data is usually required to ensure the consistency of input data. 
The Transform module of PaddleTS supports the TSDataset arrays input and joint transformation of multiple time series data.

.. code:: python

   from paddlets.transform import MinMaxScaler
   min_max_scaler = MinMaxScaler()
   tsdatasets = min_max_scaler.fit_transform(tsdatasets)


3. Joint Training of Multiple Time Series
-----------------------------------------

The models in PaddleTS forecasting supports the TSDataset arrays input and joint training of multiple time series data.

.. code:: python

   from paddlets.models.forecasting import MLPRegressor
   mlp = MLPRegressor(in_chunk_len=10, out_chunk_len=2)
   mlp.fit(tsdatasets)

   for tsdataset in tsdatasets:
       print(mlp.predict(tsdataset))

   #           a
   #80  0.546383
   #81  0.513985
   #           a
   #200  0.511116
   #201  0.590263


4. Pipeline for Multiple Time Series
------------------------------------

The Pipeline in PaddleTS also supports the TSDataset arrays input.

.. code:: python

   from paddlets import Pipeline
   pipeline = Pipeline([
       (MinMaxScaler, {}),
       (MLPRegressor, {"in_chunk_len": 10, "out_chunk_len": 2})
   ])
   pipeline.fit(tsdatasets)

   for tsdataset in tsdatasets:
       print(pipeline.predict(tsdataset))

   #           a
   #80  0.344289
   #81  0.255014
   #           a
   #200  0.272490
   #201  0.842059

