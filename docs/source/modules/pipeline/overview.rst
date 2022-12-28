========
Pipeline
========

The pipeline is designed to build a workflow for time series modeling which may be comprised of a set of
`transformers <../transform/overview.html>`_ and a model.

1. Pipeline: Chain transformers
====================================

There is often a sequence of steps in processing the time series data. Pipeline can be used for chaining multiple
`transformers <../transform/overview.html>`_ into one.

1.1. Prepare data
-------------------

.. code:: python

    >>> import pandas as pd
    >>> import numpy as np
    >>> from paddlets.datasets.tsdataset import TimeSeries, TSDataset
    >>> target = TimeSeries.load_from_dataframe(
    >>>     pd.Series(
    >>>         np.random.randn(10).astype(np.float32),
    >>>         index=pd.date_range("2022-01-01", periods=10, freq="15T"),
    >>>         name="target"
    >>>     ))
    >>> observed_cov = TimeSeries.load_from_dataframe(
    >>>     pd.DataFrame(
    >>>         np.random.randn(11, 2).astype(np.float32),
    >>>         index=pd.date_range("2022-01-01", periods=11, freq="15T"),
    >>>         columns=["observed_a", "observed_b"]
    >>>     ))
    >>> known_cov = TimeSeries.load_from_dataframe(
    >>>     pd.DataFrame(
    >>>         np.random.randn(15, 2).astype(np.float32),
    >>>         index=pd.date_range("2022-01-01", periods=15, freq="15T"),
    >>>         columns=["known_c", "known_d"]
    >>>     ))
    >>> tsdataset = TSDataset(target, observed_cov, known_cov)
    >>> train_dataset, test_dataset = tsdataset.split(0.7)
    >>> train_dataset
                           target  observed_a  observed_b   known_c   known_d
    2022-01-01 00:00:00  0.222311   -0.277376    0.546331 -1.408227  0.662035
    2022-01-01 00:15:00  0.317041    0.854092   -1.857899  0.314928 -0.767439
    2022-01-01 00:30:00  1.513104    0.379383    0.850350 -0.909959 -1.331936
    2022-01-01 00:45:00  0.598694   -0.445081   -1.326147  0.749286  1.723710
    2022-01-01 01:00:00 -0.387747   -0.621718   -1.689694  1.437675 -0.621165
    2022-01-01 01:15:00  0.476731    0.890895    0.058239 -0.487614  0.668113
    2022-01-01 01:30:00  0.219381   -0.684207   -0.001203 -0.199150  1.221772
    2022-01-01 01:45:00       NaN         NaN         NaN  1.413992 -0.452255
    2022-01-01 02:00:00       NaN         NaN         NaN  0.228248  0.102397
    2022-01-01 02:15:00       NaN         NaN         NaN  0.687319  0.240901
    2022-01-01 02:30:00       NaN         NaN         NaN  0.075458 -0.922555
    2022-01-01 02:45:00       NaN         NaN         NaN -1.718082  0.362322
    2022-01-01 03:00:00       NaN         NaN         NaN -0.126352  1.376127
    2022-01-01 03:15:00       NaN         NaN         NaN  1.508944 -2.041886
    2022-01-01 03:30:00       NaN         NaN         NaN -0.852201  0.476529

    >>> test_dataset
                           target  observed_a  observed_b   known_c   known_d
    2022-01-01 01:45:00  1.861470    0.342490    2.156751  1.413992 -0.452255
    2022-01-01 02:00:00 -0.942302    0.198202    0.527947  0.228248  0.102397
    2022-01-01 02:15:00  0.754010   -1.400311   -0.105994  0.687319  0.240901
    2022-01-01 02:30:00       NaN    1.359987    0.119287  0.075458 -0.922555
    2022-01-01 02:45:00       NaN         NaN         NaN -1.718082  0.362322
    2022-01-01 03:00:00       NaN         NaN         NaN -0.126352  1.376127
    2022-01-01 03:15:00       NaN         NaN         NaN  1.508944 -2.041886
    2022-01-01 03:30:00       NaN         NaN         NaN -0.852201  0.476529


For this data, we might want to do anomaly detection on both observed co-variates and known co-variates using `KSigma` then generate the time
feature using `TimeFeatureGenerator`.

1.2. Construct
--------------------

The pipeline is built using a list of (key, value) pairs, where the key is the class name of `transformers <../transform/overview.html>`_ and value is
the init parameter of `transformers <../transform/overview.html>`_.

This pipeline is comprised of the following:

    - A `KSigma <../../api/paddlets.transform.ksigma.html>`_ transformer to detect outliers.
    - A `TimeFeatureGenerator <../../api/paddlets.transform.time_feature.html>`_ transformer to generate time features.

.. code:: python

    >>> from paddlets.pipeline.pipeline import Pipeline
    >>> from paddlets.transform import KSigma, TimeFeatureGenerator
    >>> pipeline = Pipeline([(KSigma, {"cols":["observed_a", "observed_b", "known_c", "known_d"], "k": 1}), (TimeFeatureGenerator, {})])

1.3. Transform
---------------------

Fit pipeline and perform the transformation.

.. code:: python

    >>> pipeline.fit(train_dataset)
    >>> tsdataset_preprocessed = pipeline.transform(test_dataset)
    >>> tsdataset_preprocessed
                           target  observed_a  observed_b   known_c   known_d  year  month  day  weekday  hour  quarter  dayofyear  weekofyear  is_holiday  is_workday
    2022-01-01 01:45:00  1.861470    0.342490   -0.488575  0.047618 -0.452255  2022      1    1        5     1        1          1          52         1.0         0.0
    2022-01-01 02:00:00 -0.942302    0.198202    0.527947  0.228248  0.102397  2022      1    1        5     2        1          1          52         1.0         0.0
    2022-01-01 02:15:00  0.754010    0.013713   -0.105994  0.687319  0.240901  2022      1    1        5     2        1          1          52         1.0         0.0
    2022-01-01 02:30:00       NaN    0.013713    0.119287  0.075458 -0.922555  2022      1    1        5     2        1          1          52         1.0         0.0
    2022-01-01 02:45:00       NaN         NaN         NaN  0.047618  0.362322  2022      1    1        5     2        1          1          52         1.0         0.0
    2022-01-01 03:00:00       NaN         NaN         NaN -0.126352  0.046445  2022      1    1        5     3        1          1          52         1.0         0.0
    2022-01-01 03:15:00       NaN         NaN         NaN  0.047618  0.046445  2022      1    1        5     3        1          1          52         1.0         0.0
    2022-01-01 03:30:00       NaN         NaN         NaN -0.852201  0.476529  2022      1    1        5     3        1          1          52         1.0         0.0


2. Pipeline: Chain model
=============================

The last object of a pipeline may be a model, then you can only call `fit` once on your data to `fit` whole steps in your
pipeline.

2.1. Construct
------------------

This pipeline is comprised of the following:

    - A `KSigma <../../api/paddlets.transform.ksigma.html>`_ transformer to detect outliers.
    - A `TimeFeatureGenerator <../../api/paddlets.transform.time_feature.html>`_ transformer to generate time features.
    - A `MLPRegressor <../../api/paddlets.models.forecasting.dl.mlp.html>`_ to build a model on given time series data.

.. code:: python

    >>> from paddlets.models.forecasting import MLPRegressor
    >>> mlp_params = {
    >>>     'in_chunk_len': 3,
    >>>     'out_chunk_len': 2,
    >>>     'skip_chunk_len': 0,
    >>>     'eval_metrics': ["mse", "mae"]
    >>> }
    >>> pipeline = Pipeline([(KSigma, {"cols":["observed_a", "observed_b", "known_c", "known_d"], "k": 1}), (MLPRegressor, mlp_params)])

2.2. Fit pipeline and make predictions
----------------------------------------------------

You can use `pipeline.predict` for time series forecasting or use `recursive_predict` for recursive multi-step time series
forecasting after fitting the pipeline:

.. code:: python

    >>> pipeline.fit(train_dataset)
    >>> predicted_results = pipeline.predict(train_dataset)
    >>> predicted_results
                           target
    2022-01-01 01:45:00 -0.034728
    2022-01-01 02:00:00  0.156984

2.3. Recursive predict
----------------------------------------------------

The recursive strategy involves applying `pipeline.predict` method iteratively for multi-step time series forecasting.
The predicted results from the current call will be appended to the given `TSDataset` object and will appear in the
loopback window for the next call.

Note that `pipeline.recursive_predict` is not supported when `pipeline.skip_chunk` != 0.

Note that each call of `pipeline.predict` will return a result of length `out_chunk_len`, so `pipeline.recursive_predict`
will be called ceiling(`predict_length`/`out_chunk_len`) times to meet the required length. For example, the `out_chunk_length`
of the pipeline mentioned before is 2, but `recursive_predict` allows you to set `predict_length` as 5 or more:

.. code:: python

    >>> train_dataset.set_observed_cov(TimeSeries.concat([train_dataset.observed_cov, test_dataset.observed_cov]))
    >>> train_dataset
                           target  observed_a  observed_b   known_c   known_d
    2022-01-01 00:00:00  0.222311   -0.277376    0.546331 -1.408227  0.662035
    2022-01-01 00:15:00  0.317041    0.854092   -1.857899  0.314928 -0.767439
    2022-01-01 00:30:00  1.513104    0.379383    0.850350 -0.909959 -1.331936
    2022-01-01 00:45:00  0.598694   -0.445081   -1.326147  0.749286  1.723710
    2022-01-01 01:00:00 -0.387747   -0.621718   -1.689694  1.437675 -0.621165
    2022-01-01 01:15:00  0.476731    0.890895    0.058239 -0.487614  0.668113
    2022-01-01 01:30:00  0.219381   -0.684207   -0.001203 -0.199150  1.221772
    2022-01-01 01:45:00       NaN    0.342490    2.156751  1.413992 -0.452255
    2022-01-01 02:00:00       NaN    0.198202    0.527947  0.228248  0.102397
    2022-01-01 02:15:00       NaN   -1.400311   -0.105994  0.687319  0.240901
    2022-01-01 02:30:00       NaN    1.359987    0.119287  0.075458 -0.922555
    2022-01-01 02:45:00       NaN         NaN         NaN -1.718082  0.362322
    2022-01-01 03:00:00       NaN         NaN         NaN -0.126352  1.376127
    2022-01-01 03:15:00       NaN         NaN         NaN  1.508944 -2.041886
    2022-01-01 03:30:00       NaN         NaN         NaN -0.852201  0.476529

    >>> recursive_predicted_results = pipeline.recursive_predict(train_dataset, predict_length=5)
    >>> recursive_predicted_results
                           target
    2022-01-01 01:45:00 -0.034728
    2022-01-01 02:00:00  0.156984
    2022-01-01 02:15:00  0.290443
    2022-01-01 02:30:00 -0.007422
    2022-01-01 02:45:00  0.025956

When known_cov or observed_cov exists, the length of known_cov must be greater than or equal to the number of
`recursive prediction steps * prediction length` and the length of observed_cov must be greater than or equal to number
of `(recursive prediction steps - 1) * prediction` length to meet the needs of feature construction.

**Note**: The prediction errors are accumulated such that the performance of prediction will degrade as the prediction
time horizon increases.

For detailed usage, please refer to
`API: pipeline.recursive_predict <../../api/paddlets.pipeline.pipeline.html#paddlets.pipeline.pipeline.Pipeline.recursive_predict>`_

3. Pipeline: Persistence
=============================

Similar to other PaddleTS models, `Pipeline` provides save() and load() functions to support persistence.

3.1. Save
----------------------------------------------------

    >>> pipeline.save(path="./")

3.2. Load
----------------------------------------------------

    >>> pipeline.load(path="./")
    >>> predicted_results = pipeline.predict(train_dataset)