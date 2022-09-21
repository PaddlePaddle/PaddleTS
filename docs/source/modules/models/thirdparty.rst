==================
Third-party Model
==================

PaddleTS allows users to implement time series models based on third party models and verify the feasibility and performance efficiently.

`scikit-learn <https://scikit-learn.org>`_ is currently the only supported third-party library for PaddleTS.

1. Make Time Series Model Based On Third-party Model
=====================================================

PaddleTS provides `make_ml_model <../../api/paddlets.models.forecasting.ml.ml_model_wrapper.html#paddlets.models.forecasting.ml.ml_model_wrapper.make_ml_model>`_ interface
that allows users to build time series models by simply specifying a third party model class and relevant parameters without extra development.

1.1 Minimal Example
--------------------

Below is an example of how to make time series models based on
`sklearn.neighbors.KNeighborsRegressor <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html>`_ .

.. code-block:: python

    from paddlets.datasets.repository import get_dataset
    from paddlets.models.forecasting.ml.ml_model_wrapper import make_ml_model

    from sklearn.neighbors import KNeighborsRegressor

    # prepare data
    paddlets_ds = get_dataset("UNI_WTH")

    # make model based on sklearn.neighbors.KNeighborsRegressor
    model = make_ml_model(
        in_chunk_len=3,
        out_chunk_len=1,
        model_class=KNeighborsRegressor
    )

    # fit
    model.fit(train_data=paddlets_ds)

    # predict
    predicted_ds = model.predict(paddlets_ds)
    #             WetBulbCelsius
    # 2014-01-01           -1.72


1.2 Convert MLDataLoader to Trainable / Predictable ndarray
------------------------------------------------------------

The third-party library such as `scikit-learn <https://scikit-learn.org>`_ usually accepts numpy.ndarray data as `fit` and `predict` method inputs,
while PaddleTS uses `paddlets.models.forecasting.ml.adapter.ml_dataloader.MLDataLoader` to represent trainable / predictable time series data.
Thus, `make_ml_model` provides 2 optional arguments `udf_ml_dataloader_to_fit_ndarray` and `udf_ml_dataloader_to_predict_ndarray` allow users to
convert `MLDataLoader` to an `numpy.ndarray` object.

By default, `make_ml_model` uses
`default_ml_dataloader_to_fit_ndarray <../../api/paddlets.models.forecasting.ml.ml_model_wrapper.html#paddlets.models.ml.ml_model_wrapper.default_ml_dataloader_to_fit_ndarray>`_ and
`default_ml_dataloader_to_predict_ndarray <../../api/paddlets.models.forecasting.ml.ml_model_wrapper.html#paddlets.models.ml.ml_model_wrapper.default_ml_dataloader_to_predict_ndarray>`_
to convert MLDataLoader to `numpy.ndarray` for `fit` and `predict` method, respectively.
Also, users are able to develop user-defined convert functions to get expected trainable / predictable output.

.. code-block:: python

    from paddlets.datasets.repository import get_dataset
    from paddlets.models.forecasting.ml.adapter.ml_dataloader import MLDataLoader
    from paddlets.models.forecasting.ml.ml_model_wrapper import make_ml_model

    from sklearn.neighbors import KNeighborsRegressor

    # prepare data
    paddlets_ds = get_dataset("UNI_WTH")

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
    model.fit(train_data=paddlets_ds)

    # predict
    predicted_ds = model.predict(paddlets_ds)

2. Multi-step forecasting
==========================

The time series models also support multi-timestep forecasting by calling
`recursive_predict <../../api/paddlets.models.base.html#paddlets.models.base.BaseModel.recursive_predict>`_ .

.. code-block:: python

    from paddlets.datasets.repository import get_dataset
    from paddlets.models.forecasting.ml.ml_model_wrapper import make_ml_model

    # prepare data
    paddlets_ds = get_dataset("UNI_WTH")

    # make model
    model = make_ml_model(
        in_chunk_len=3,
        out_chunk_len=1,
        model_class=KNeighborsRegressor
    )

    # fit
    model.fit(train_data=paddlets_ds)

    # recursively predict
    recursively_predicted_ds = model.recursive_predict(tsdataset=paddlets_ds, predict_length=4)
    #                      WetBulbCelsius
    # 2014-01-01 00:00:00           -1.72
    # 2014-01-01 01:00:00           -1.88
    # 2014-01-01 02:00:00           -2.18
    # 2014-01-01 03:00:00           -2.44
