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
    >>>         np.random.randn(10, 2).astype(np.float32),
    >>>         index=pd.date_range("2022-01-01", periods=10, freq="15T"),
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
    2022-01-01 00:00:00  1.044207    0.693854   -0.175082 -1.890724  1.364049
    2022-01-01 00:15:00 -1.011910   -0.471103    0.001718 -0.307345  0.100475
    2022-01-01 00:30:00 -0.953874    0.473357   -0.620498  0.426992 -1.976870
    2022-01-01 00:45:00  0.811422    0.679846    0.401000  1.026558 -0.281426
    2022-01-01 01:00:00 -0.343195   -0.657656    2.705243  0.997001  0.856995
    2022-01-01 01:15:00  1.992397   -0.281009    1.666330  1.136454  0.154997
    2022-01-01 01:30:00  0.070085   -0.291660    0.449088  0.034393 -1.101410
    2022-01-01 01:45:00       NaN         NaN         NaN  0.387557  0.557443
    2022-01-01 02:00:00       NaN         NaN         NaN -0.927851  1.425830
    2022-01-01 02:15:00       NaN         NaN         NaN  0.056782 -0.112722
    2022-01-01 02:30:00       NaN         NaN         NaN  0.545976 -2.725172
    2022-01-01 02:45:00       NaN         NaN         NaN  0.436852 -2.653046
    2022-01-01 03:00:00       NaN         NaN         NaN  0.579058 -1.125973
    2022-01-01 03:15:00       NaN         NaN         NaN -0.480516  1.002109
    2022-01-01 03:30:00       NaN         NaN         NaN  0.220903  0.325239
    >>> test_dataset
                           target  observed_a  observed_b   known_c   known_d
    2022-01-01 00:00:00       NaN         NaN         NaN -1.890724  1.364049
    2022-01-01 00:15:00       NaN         NaN         NaN -0.307345  0.100475
    2022-01-01 00:30:00       NaN         NaN         NaN  0.426992 -1.976870
    2022-01-01 00:45:00       NaN         NaN         NaN  1.026558 -0.281426
    2022-01-01 01:00:00       NaN         NaN         NaN  0.997001  0.856995
    2022-01-01 01:15:00       NaN         NaN         NaN  1.136454  0.154997
    2022-01-01 01:30:00       NaN         NaN         NaN  0.034393 -1.101410
    2022-01-01 01:45:00 -0.833165   -1.301204   -1.073407  0.387557  0.557443
    2022-01-01 02:00:00  0.767193    1.543675    0.985133 -0.927851  1.425830
    2022-01-01 02:15:00  0.458591    0.453419   -2.523344  0.056782 -0.112722
    2022-01-01 02:30:00       NaN         NaN         NaN  0.545976 -2.725172
    2022-01-01 02:45:00       NaN         NaN         NaN  0.436852 -2.653046
    2022-01-01 03:00:00       NaN         NaN         NaN  0.579058 -1.125973
    2022-01-01 03:15:00       NaN         NaN         NaN -0.480516  1.002109
    2022-01-01 03:30:00       NaN         NaN         NaN  0.220903  0.325239



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
    2022-01-01 00:00:00       NaN         NaN         NaN  0.149473 -0.279299  2022      1    1        5     0        1          1          52         1.0         0.0
    2022-01-01 00:15:00       NaN         NaN         NaN -0.307345  0.100475  2022      1    1        5     0        1          1          52         1.0         0.0
    2022-01-01 00:30:00       NaN         NaN         NaN  0.426992 -0.279299  2022      1    1        5     0        1          1          52         1.0         0.0
    2022-01-01 00:45:00       NaN         NaN         NaN  0.149473 -0.281426  2022      1    1        5     0        1          1          52         1.0         0.0
    2022-01-01 01:00:00       NaN         NaN         NaN  0.149473  0.856995  2022      1    1        5     1        1          1          52         1.0         0.0
    2022-01-01 01:15:00       NaN         NaN         NaN  0.149473  0.154997  2022      1    1        5     1        1          1          52         1.0         0.0
    2022-01-01 01:30:00       NaN         NaN         NaN  0.034393 -1.101410  2022      1    1        5     1        1          1          52         1.0         0.0
    2022-01-01 01:45:00 -0.833165    0.020804    0.632543  0.387557  0.557443  2022      1    1        5     1        1          1          52         1.0         0.0
    2022-01-01 02:00:00  0.767193    0.020804    0.985133  0.149473 -0.279299  2022      1    1        5     2        1          1          52         1.0         0.0
    2022-01-01 02:15:00  0.458591    0.453419    0.632543  0.056782 -0.112722  2022      1    1        5     2        1          1          52         1.0         0.0
    2022-01-01 02:30:00       NaN         NaN         NaN  0.545976 -0.279299  2022      1    1        5     2        1          1          52         1.0         0.0
    2022-01-01 02:45:00       NaN         NaN         NaN  0.436852 -0.279299  2022      1    1        5     2        1          1          52         1.0         0.0
    2022-01-01 03:00:00       NaN         NaN         NaN  0.579058 -1.125973  2022      1    1        5     3        1          1          52         1.0         0.0
    2022-01-01 03:15:00       NaN         NaN         NaN -0.480516  1.002109  2022      1    1        5     3        1          1          52         1.0         0.0
    2022-01-01 03:30:00       NaN         NaN         NaN  0.220903  0.325239  2022      1    1        5     3        1          1          52         1.0         0.0

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
    >>> pipeline = Pipeline([(KSigma, {"cols":["observed_a", "observed_b", "known_c", "known_d"], "k": 1}), (TimeFeatureGenerator, {}), (MLPRegressor, mlp_params)])

2.2. Fit pipeline and make predictions
----------------------------------------------------

You can use `pipeline.predict` for time series forecasting or use `recursive_predict` for recursive multi-step time series
forecasting after fitting the pipeline:

.. code:: python

    >>> pipeline.fit(train_dataset)
    >>> predicted_results = pipeline.predict(train_dataset)
    >>> predicted_results
                           target
    2022-01-01 01:45:00  2.543621
    2022-01-01 02:00:00 -0.368826

2.3. Recursive predict
----------------------------------------------------

The recursive strategy involves applying `pipeline.predict` method iteratively for multi-step time series forecasting.
The predicted results from the current call will be appended to the given `TSDataset` object and will appear in the
loopback window for the next call.

Note that each call of `pipeline.predict` will return a result of length `out_chunk_len`, so `pipeline.recursive_predict`
will be called ceiling(`predict_length`/`out_chunk_len`) times to meet the required length. For example, the `out_chunk_length`
of the pipeline mentioned before is 2, but `recursive_predict` allows you to set `predict_length` as 5 or more:

    >>> recursive_predicted_results = pipeline.recursive_predict(train_dataset, predict_length=5)
    >>> recursive_predicted_results
                           target
    2022-01-01 01:45:00  2.543621
    2022-01-01 02:00:00 -0.368826
    2022-01-01 02:15:00  3.192380
    2022-01-01 02:30:00 -0.752583
    2022-01-01 02:45:00  4.176333

Note that `pipeline.recursive_predict` is not supported when `pipeline.skip_chunk` != 0.

**Note**: The prediction errors are accumulated such that the performance of prediction will degrade as the prediction
time horizon increases.

For detailed usage, please refer to
`API: pipeline.recursive_predict <../../api/paddlets.pipeline.pipeline.html#paddlets.pipeline.pipeline.Pipeline.recursive_predict>`_
