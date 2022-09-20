========
AutoTS
========

AutoTS is an automated machine learning tool for PaddleTS.

It frees the user from selecting hyperparameters for PaddleTS models or PaddleTS pipelines.

1. Installation
====================================

.. code:: python

    pip install paddlets[autots]

2. Quickstart
===============

2.1. Prepare Data
--------------------

.. code:: python

    from paddlets.datasets.repository import get_dataset
    tsdataset = get_dataset("UNI_WTH")

2.2. Construct and Fitting
----------------------------

With four lines of code, we initialize an `AutoTS` model with `MLPRegressor` and perform model fitting. Autots will
automatically optimize hyperparameters during the fitting process.

.. code:: python

    from paddlets.models.forecasting import MLPRegressor
    from paddlets.automl.autots import AutoTS
    autots_model = AutoTS(MLPRegressor, 96, 2)
    autots_model.fit(tsdataset)

3. Search Space
==================

For hyperparameter optimization, you can define a search space, or we also provide built-in recommended search space
for the PaddleTS models if you do not define a search space.

You can specify how your hyperparameters are sampled and define valid ranges for hyperparameters.

The following is an example of a autots pipeline which specifying the search space:

.. code:: python

    from ray.tune import uniform, qrandint, choice
    sp = {
        "Fill": {
            "cols": ['WetBulbCelsius'],
            "method": choice(['max', 'min', 'mean', 'median', 'pre', 'next', 'zero']),
            "value": uniform(0.1, 0.9),
            "window_size": qrandint(20, 50, q=1)
        },
        "MLPRegressor": {
            "batch_size": qrandint(16, 64, q=16),
            "use_bn": choice([True, False]),
            "max_epochs": qrandint(10, 50, q=10)
        }
    }
    autots_model = AutoTS([Fill, MLPRegressor], 25, 2, search_space=sp, sampling_stride=25)
    autots_model.fit(tsdataset, n_trails=1)
    sp = autots_model.search_space()
    predicted = autots_model.predict(tsdataset)

Search space API can refer to: https://docs.ray.io/en/latest/tune/api_docs/search_space.html

4. Search Algorithms
============================

Search Algorithms are wrappers around open-source optimization libraries.

We have built in the following algorithms：
    ["Random", "CMAES", "TPE", "CFO", "BlendSearch", "Bayes"]

For more details about those optimization libraries, please refer to their documentation.

You can specify different algorithms as follows:

.. code:: python

    autots_model = AutoTS(MLPRegressor, 96, 2, search_space="CMAES")

If no search algorithm is specified, "TPE" will be used as default.
