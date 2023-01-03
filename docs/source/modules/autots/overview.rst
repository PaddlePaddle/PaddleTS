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
    train_tsdataset, valid_tsdataset = tsdataset.split(0.3)

2.2. Construct and Fitting
----------------------------

With four lines of code, we initialize an `AutoTS` model with `MLPRegressor` and perform model fitting. Autots will
automatically optimize hyperparameters during the fitting process.

.. code:: python

    from paddlets.models.forecasting import MLPRegressor
    from paddlets.automl.autots import AutoTS
    autots_model = AutoTS(MLPRegressor, 96, 24, sampling_stride=24)
    autots_model.fit(train_tsdataset, valid_tsdataset)

2.3. Persistence
----------------------------

Although `AutoTS` itself does not provide persistence support, we can save the best estimator it found after hyperparameter optimization

.. code:: python

    # Method 1
    best_estimator = autots_model.fit(train_tsdataset, valid_tsdataset)
    best_estimator.save(path="./autots_best_estimator_m1")

    # Method 2
    best_estimator = autots_model.best_estimator()
    best_estimator.save(path="./autots_best_estimator_m2")


3. Search Space
==================

3.1. Run With Specified Search Space
--------------------------------------

For hyperparameter optimization, you can define a search space, or we also provide built-in recommended search space
for the PaddleTS models if you do not define a search space.

You can specify how your hyperparameters are sampled and define valid ranges for hyperparameters.

The following is an example of a autots pipeline which specifying the search space:

.. code:: python

    from ray.tune import uniform, qrandint, choice
    from paddlets.transform import Fill

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
    autots_model = AutoTS([Fill, MLPRegressor], 96, 24, search_space=sp, sampling_stride=24)
    autots_model.fit(tsdataset)
    sp = autots_model.search_space()
    predicted = autots_model.predict(tsdataset)

Search space API can refer to: https://docs.ray.io/en/latest/tune/api_docs/search_space.html

3.2. Get Default Search Space Using The Search Space Configer
---------------------------------------------------------------

In order to make it easier for users to use AutoTS, we provide the `SearchSpaceConfiger`, which has built-in
recommended search space for the PaddleTS models.

Algorithms that have been adapted to `SearchSpaceConfiger` are
    ["MLPRegressor", "RNNBlockRegressor", "NBEATSModel", "NHiTSModel", "LSTNetRegressor", "TransformerModel", "TCNRegressor", "InformerModel", "DeepARModel"]

- Get the search space in the form of a string

.. code:: python

    >>> from paddlets.automl.autots import SearchSpaceConfiger
    >>> from paddlets.models.forecasting import MLPRegressor
    >>> configer = SearchSpaceConfiger()
    >>> sp_str = configer.recommend(MLPRegressor)
    >>> print(sp_str)
    The recommended search space are as follows:
    =======================================================
    from ray.tune import uniform, quniform, loguniform, qloguniform, randn, qrandn, randint, qrandint, lograndint, qlograndint, choice
    recommended_sp = {
            "hidden_config": choice([[64], [64, 64], [64, 64, 64], [128], [128, 128], [128, 128, 128]]),
            "use_bn": choice([True, False]),
            "batch_size": qrandint(8, 128, q=8),
            "max_epochs": qrandint(30, 600, q=30),
            "optimizer_params": {
                    "learning_rate": uniform(0.0001, 0.01)
            },
            "patience": qrandint(5, 50, q=5)
    }
    =====================================================
    Please note that the **USER_DEFINED_SEARCH_SPACE** parameters need to be set by the user


- Get the search space in the form of a dict

.. code:: python

    >>> sp_dict = configer.get_default_search_space(MLPRegressor)
    >>> from pprint import pprint as print
    >>> print(sp_dict)
    {'batch_size': <ray.tune.sample.Integer object at 0x7f88bef520a0>,
     'hidden_config': <ray.tune.sample.Categorical object at 0x7f88bef45fd0>,
     'max_epochs': <ray.tune.sample.Integer object at 0x7f88bef52e80>,
     'optimizer_params': {'learning_rate': <ray.tune.sample.Float object at 0x7f88bef521c0>},
     'patience': <ray.tune.sample.Integer object at 0x7f88bef52070>,
     'use_bn': <ray.tune.sample.Categorical object at 0x7f88bef52250>}


4. Search Algorithms
============================

Search Algorithms are wrappers around open-source optimization libraries.

We have built in the following algorithms：
    ["Random", "CMAES", "TPE", "CFO", "BlendSearch", "Bayes"]

For more details about those optimization libraries, please refer to their documentation.

You can specify different algorithms as follows:

.. code:: python

    autots_model = AutoTS(MLPRegressor, 96, 2, search_alg="CMAES")

If no search algorithm is specified, "TPE" will be used as default.

5. Parallelism and Resources
============================

The function `AutoTS.fit()` will run n_trials (defaulting to 20) times trials during hyperparameter optimization, which means
to sample n_trials sets of hyperparameters from the hyperparameter space.

Parallelism is determined by `cpu_resource`, `gpu_resource`, and `max_concurrent_trials`.

The `max_concurrent_trials` (defaulting to 1) controls the maximum number of trials running concurrently.

.. code:: python

    # If you have 4 CPUs on your machine, this will run 2 concurrent trials at a time.
    autots.fit(train_tsdataset, valid_tsdataset, cpu_resource=2)

    # If you have 4 CPUs on your machine, this will run 1 trial at a time.
    autots.fit(train_tsdataset, valid_tsdataset, cpu_resource=4)

    # Fractional values are also supported, (i.e., cpu_resource=0.5, which means running 8 concurrent trials at a time).
    autots.fit(train_tsdataset, valid_tsdataset, cpu_resource=0.5)


5.1. How To Leverage GPUs?
---------------------------

To leverage GPUs, you must set `gpu_resource` (defaulting to 0) in `AutoTS.fit()` and set CUDA_VISIBLE_DEVICES.

**Note that GPUs will not be assigned if you do not specify them (gpu_resource, defaulting to 0).**

.. code:: python

    import os
    # If you have 8 GPUs, this will run 8 trials at once.
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    autots.fit(train_tsdataset, valid_tsdataset, cpu_resource=1, gpu_resource=1)

    # If you have 4 CPUs on your machine and 1 GPU, this will run 1 trial at a time.
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    autots.fit(train_tsdataset, valid_tsdataset, cpu_resource=2, gpu_resource=1)

More details refer to: https://docs.ray.io/en/latest/tune/tutorials/tune-resources.html

6. Log and Temporary Files
============================

The parameter `local_dir` in `AutoTS()` can specify a dir to save training results to (Defaulting to `./`, and the
results directory is defaulting to `./ray_results`).

6.1. Temporary Files
-----------------------

Ray is a dependencies of AutoTS, and the root temporary directory for the Ray process is an OS-specific conventional location, e.g., “/tmp/ray”.
Due to a known issue with Ray, AutoTS did not specify the root temporary directory for Ray, while specifying it would cause a startup failure.

**Please clean up the temporary directory by yourself.**

Depending on the system, the temporary file may be stored in folders such as /tmp or /usr/tmp or `tmp` based on the
system root directory. The temporary folder can be set by the environment variable RAY_TMPDIR/TMPDIR, or using
tempfile.gettempdir().