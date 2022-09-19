Welcome to PaddleTS
======================

PaddleTS is an easy to use Python library for deep time series modeling,
focusing on the state-of-the-art deep neural network models based on 
PaddlePaddle deep learning framework. It aims to provide great flexibility 
and excellent user experiences for practitioners and professionals. It’s featured with:

* A unified data structure named TSDataset for representing time series data with one or multiple target variables and optional different kinds of covariates (e.g. known covariates, observed covariates, and static covariates, etc.);

* A base model class named PaddleBaseModelImpl , which inherits from the PaddleBaseModel and further encapsulates some routine procedures (e.g. data loading, callbacks setup, loss computation, training loop control, etc.) and allows developers to focus on the implementation of network architectures when developing new models;

* A set of state-of-the-art deep learning models (e.g. NBEATS, NHiTS, LSTNet, TCN, Transformer, etc);

* A set of transformation operators for data preprocessing (e.g. missing values/outliers handling, one-hot encoding, normalization, and automatic date/time-related covariate generation, etc.);

* A set of analysis operators for quick data exploration (e.g. basic statistics and summary).

* Automatic hyper-parameter tuning.

* Third-party ML models & data transformations integration (e.g., scikit-learn).

* Run models on GPU devices.

* TS2Vec, a time series representation learning model.

* DeepAR, a deep time series model for probabilistic forecasting.

In future, more advanced features will be coming, including:

* Time series anomaly detection;
* More time series representation learning models；
* More probabilistic forecasting models;
* Scenario-specific pipelines which aim to provide an end-to-end solution for solving real-world business problems;
* And more.


Project GitHub: https://github.com/PaddlePaddle/PaddleTS


.. toctree::
    :maxdepth: 1
    :caption: Get Started

    Get Started <source/get_started/get_started.rst>
    Run On GPU <source/get_started/run_on_gpu.rst>

.. toctree::
    :maxdepth: 1
    :caption: Installation

    Installation <source/installation/overview.rst>

.. toctree::
    :maxdepth: 1
    :caption: Dataset

    Dataset <source/modules/datasets/overview.rst>


.. toctree::
    :maxdepth: 1
    :caption: Transform

    Transform <source/modules/transform/overview.md>
    Third-Party And User-Define Transform <source/modules/transform/thirdparty_userdefine.rst>

.. toctree::
    :maxdepth: 1
    :caption: Models

    Overview <source/modules/models/overview.rst>
    Third-party Model <source/modules/models/thirdparty.rst>


.. toctree::
    :maxdepth: 1
    :caption: Metrics

    Metrics <source/modules/metrics/overview.md>


.. toctree::
    :maxdepth: 1
    :caption: Pipeline

    Pipeline <source/modules/pipeline/overview.rst>

.. toctree::
    :maxdepth: 1
    :caption: Analysis

    Analysis <source/modules/analysis/overview.md>

.. toctree::
    :maxdepth: 1
    :caption: Backtest

    Backtest <source/modules/backtest/overview.md>

.. toctree::
    :maxdepth: 1
    :caption: AutoTS

    AutoTS <source/modules/autots/overview.rst>

.. toctree::
    :maxdepth: 1
    :caption: API

    paddlets.analysis <source/api/paddlets.analysis.rst>
    paddlets.automl <source/api/paddlets.automl.rst>
    paddlets.datasets <source/api/paddlets.datasets.rst>
    paddlets.metrics <source/api/paddlets.metrics.rst>
    paddlets.models <source/api/paddlets.models.rst>
    paddlets.pipeline <source/api/paddlets.pipeline.rst>
    paddlets.transform <source/api/paddlets.transform.rst>
    paddlets.utils <source/api/paddlets.utils.rst>

