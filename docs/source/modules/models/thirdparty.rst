==================
Third-party Model
==================

PaddleTS provides `make_ml_model <../../api/paddlets.models.ml_model_wrapper.html#paddlets.models.ml_model_wrapper.make_ml_model>`_ interface
allow users to build time series models based on `scikit-learn <https://scikit-learn.org>`_ for time series forecasting,
`pyod <https://pyod.readthedocs.io/en/latest>`_ for time series anomaly detection, respectively.
With this ability, users only need to develop a small piece of code to verify the feasibility and performance of their time-series-related ideas,
which significantly improve the efficiency.

1. Third-party Model Integration for Time Series Forecasting
==============================================================

1.1 Minimal Example
--------------------

Below is an example of how to make models based on
`sklearn.neighbors.KNeighborsRegressor <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html>`_ for time series forecasting.

.. code-block:: python

    from paddlets.datasets.repository import get_dataset
    from paddlets.models.ml_model_wrapper import make_ml_model

    from sklearn.neighbors import KNeighborsRegressor

    # prepare data
    tsdataset = get_dataset("UNI_WTH")

    # make model based on sklearn.neighbors.KNeighborsRegressor
    model = make_ml_model(
        in_chunk_len=3,
        out_chunk_len=1,
        model_class=KNeighborsRegressor
    )

    # fit
    model.fit(train_data=tsdataset)

    # predict
    predicted_ds = model.predict(tsdataset)
    #             WetBulbCelsius
    # 2014-01-01           -1.72


1.2 Convert MLDataLoader to Trainable / Predictable ndarray
------------------------------------------------------------

The third-party library such as `scikit-learn <https://scikit-learn.org>`_ usually accepts numpy.ndarray data as `fit` and `predict` method inputs,
while PaddleTS uses `paddlets.models.forecasting.ml.adapter.ml_dataloader.MLDataLoader` to represent trainable / predictable time series data.
Thus, `make_ml_model` provides 2 optional arguments `udf_ml_dataloader_to_fit_ndarray` and `udf_ml_dataloader_to_predict_ndarray` allow users to
convert `MLDataLoader` to an `numpy.ndarray` object.

By default, `make_ml_model` uses
`default_sklearn_ml_dataloader_to_fit_ndarray <../../api/paddlets.models.ml_model_wrapper.html#paddlets.models.ml_model_wrapper.default_sklearn_ml_dataloader_to_fit_ndarray>`_ and
`default_sklearn_ml_dataloader_to_predict_ndarray <../../api/paddlets.models.ml_model_wrapper.html#paddlets.models.ml_model_wrapper.default_sklearn_ml_dataloader_to_predict_ndarray>`_
to convert MLDataLoader to `numpy.ndarray` for `fit` and `predict` method, respectively.
Also, users are able to develop user-defined convert functions to get expected trainable / predictable output.

.. code-block:: python

    from paddlets.datasets.repository import get_dataset
    from paddlets.models.forecasting.ml.adapter.ml_dataloader import MLDataLoader
    from paddlets.models.ml_model_wrapper import make_ml_model

    from sklearn.neighbors import KNeighborsRegressor

    # prepare data
    tsdataset = get_dataset("UNI_WTH")

    # develop user-defined convert functions
    def udf_ml_dataloader_to_fit_ndarray(
        ml_dataloader: MLDataLoader,
        model_init_params: Dict[str, Any],
        in_chunk_len: int,
        skip_chunk_len: int,
        out_chunk_len: int
    ):
        # build and return converted numpy.ndarray object that sklearn model fit method accepts.
        pass

    def udf_ml_dataloader_to_predict_ndarray(
        ml_dataloader: MLDataLoader,
        model_init_params: Dict[str, Any],
        in_chunk_len: int,
        skip_chunk_len: int,
        out_chunk_len: int
    ):
        # build and return converted numpy.ndarray object that sklearn model predict method accepts.
        pass

    # pass the above 2 udf arguments to make_ml_model
    model = make_ml_model(
        in_chunk_len=3,
        out_chunk_len=1,
        model_class=KNeighborsRegressor,
        udf_ml_dataloader_to_fit_ndarray=udf_ml_dataloader_to_fit_ndarray,
        udf_ml_dataloader_to_fit_ndarray=udf_ml_dataloader_to_predict_ndarray
    )

    # fit
    model.fit(train_data=tsdataset)

    # predict
    predicted_ds = model.predict(tsdataset)

1.3 Multi-step forecasting
----------------------------

The time series models also support multi-timestep forecasting by calling
`recursive_predict <../../api/paddlets.models.base.html#paddlets.models.base.BaseModel.recursive_predict>`_ .

.. code-block:: python

    from paddlets.datasets.repository import get_dataset
    from paddlets.models.forecasting.ml.ml_model_wrapper import make_ml_model

    # prepare data
    tsdataset = get_dataset("UNI_WTH")

    # make model
    model = make_ml_model(
        in_chunk_len=3,
        out_chunk_len=1,
        model_class=KNeighborsRegressor
    )

    # fit
    model.fit(train_data=tsdataset)

    # recursively predict
    recursively_predicted_ds = model.recursive_predict(tsdataset=tsdataset, predict_length=4)
    #                      WetBulbCelsius
    # 2014-01-01 00:00:00           -1.72
    # 2014-01-01 01:00:00           -1.88
    # 2014-01-01 02:00:00           -2.18
    # 2014-01-01 03:00:00           -2.44


2 Third-party Model Integration for Time Series Anomaly Detection
===================================================================

2.1 Minimal Example
--------------------

Below is an example of how to make models based on
`pyod.models.KNN <https://github.com/yzhao062/pyod/blob/master/pyod/models/knn.py>`_ for time series anomaly detection.
As it has the unified interface as making time series forecasting models, you may refer to above section 1.1 to know
how to define udf functions to customize the input ndarray of the built model.


.. code-block:: python

    from paddlets.datasets.repository import get_dataset
    from paddlets.models.ml_model_wrapper import make_ml_model

    from pyod.models.knn import KNN

    # prepare data
    tsdataset = get_dataset("WTH")

    # make model based on pyod.models.knn.KNN
    model = make_ml_model(
        in_chunk_len=3,
        model_class=KNN
    )

    # fit
    model.fit(train_data=tsdataset)

    # predict
    predicted_ds = model.predict(tsdataset)
    #                      WetBulbCelsius
    # date
    # 2010-01-01 02:00:00               0
    # 2010-01-01 03:00:00               0
    # 2010-01-01 04:00:00               1
    # 2010-01-01 05:00:00               0
    # 2010-01-01 06:00:00               0
    # ...                             ...
    # 2013-12-31 19:00:00               1
    # 2013-12-31 20:00:00               1
    # 2013-12-31 21:00:00               1
    # 2013-12-31 22:00:00               1
    # 2013-12-31 23:00:00               1

    # [35062 rows x 1 columns]
