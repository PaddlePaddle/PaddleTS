==========
Transform
==========

Before reading this document, it is recommended to first read the `TSDataset Document <../datasets/overview.html>`_ to understand the design of `TSDataset`.
Simply speaking, `TSDataset` is a unified time series data structure throughout the modeling lifecycle.
It introduces several fundamental but important time series related concepts such as
``target`` (including ``past_target`` and ``future_target``) and ``covariates`` (including ``known_covariates`` and ``observed_covariates``).
A good understanding of those concepts would be helpful for deep diving into this documentation and building well-performed models.

PaddleTS provides a series of operators for data transformation, which can be applied to fill missing values, normalize the data, or
encode the data in columns of TSDataset, etc.
Each transformation operator is implemented as a class with three methods: ``fit``, ``transform``, and ``fit_transform``.
The fit method is usually used to learn the parameters for data transformation(e.g. ``mean`` and ``std`` for data normalization),
the transform method is used to apply the data transformation based on the learnt parameters,
and the fit_transform method combines tha above two steps into a single step.

Currently, PaddleTS supports the following data transformations:

- `fill <../../api/paddlets.transform.fill.html>`_
- `ksigma <../../api/paddlets.transform.ksigma.html>`_
- `min_max_scaler <../../api/paddlets.transform.sklearn_transforms.html#paddlets.transform.sklearn_transforms.MinMaxScaler>`_
- `standard_scaler <../../api/paddlets.transform.sklearn_transforms.html#paddlets.transform.sklearn_transforms.MinMaxScaler>`_
- `onehot <../../api/paddlets.transform.sklearn_transforms.html#paddlets.transform.sklearn_transforms.OneHot>`_
- `ordinal <../../api/paddlets.transform.sklearn_transforms.html#paddlets.transform.sklearn_transforms.Ordinal>`_
- `statistical <../../api/paddlets.transform.statistical.html>`_
- `time_feature <../../api/paddlets.transform.time_feature.html>`_

1. Example
===========

The below demo takes `OneHot <../../api/paddlets.transform.sklearn_transforms.html#paddlets.transform.sklearn_transforms.OneHot>`_ as
an example to illustrates the basic usage of data transformations.

.. code-block:: python

    from paddlets import TSDataset
    from paddlets.transform import OneHot
    
    # 1 prepare the data
    data = TSDataset.load_from_csv("/path/to/data.csv")
    
    # 2 init the onehot encoder
    encoder = OneHot(cols=["Gender"])
    
    # 3 fit the encoder
    encoder.fit(dataset=data)
    
    # 4 do transformation with the learnt encoder
    transformed_data = encoder.transform(data, inplace=False)


Alternatively, one can also call the combined ``fit_transform`` to achieve the same purpose:

.. code-block:: python

    from paddlets import TSDataset
    from paddlets.transform import OneHot

    # 1 prepare the data
    data = TSDataset.load_from_csv("/path/to/data.csv")

    # 2 init the onehot encoder transform instance.
    encoder = Onehot(cols=["Gender"])

    # 3 fit + transform simultaneously
    transformed_data = encoder.fit_transform(dataset=data)
