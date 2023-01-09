Welcome to PaddleTS
======================

PaddleTS is an easy-to-use Python library for deep time series modeling,
focusing on the state-of-the-art deep neural network models based on 
PaddlePaddle deep learning framework. It aims to provide great flexibility 
and excellent user experiences for practitioners and professionals. It’s featured with:

* A unified data structure named TSDataset for representing time series data with one
  or multiple target variables and optional different kinds of covariates
  (e.g. known covariates, observed covariates, static covariates, etc.)

* A base model class named PaddleBaseModelImpl , which inherits from the PaddleBaseModel
  and further encapsulates some routine procedures (e.g. data loading, callbacks setup,
  loss computation, training loop control, etc.) and allows developers to focus on
  the implementation of network architectures when developing new models

* A set of state-of-the-art deep learning models containing
  NBEATS, NHiTS, LSTNet, TCN, Transformer, DeepAR(Probabilistic), Informer, etc. for forecasting, TS2Vec for representation

* A set of transformation operators for data preprocessing (e.g. missing values/outliers handling,
  one-hot encoding, normalization, and automatic date/time-related covariates generation, etc.)

* A set of analysis operators for quick data exploration (e.g. basic statistics and summary)

* Automatic time series modeling module (AutoTS) which supports mainstream Hyper Parameter Optimization algorithms and shows significant improvement on multiple models and datasets

* Third-party (e.g. scikit-learn) ML models & data transformations integration

Recently updated:

* Released a new time series representation model, i.e. Contrastive Learning of Disentangled Seasonal-trend Representations(CoST)

* Time series anomaly detection model supported, with three deep models released, including AE(AutoEncoder), VAE(Variational AutoEncoder), and AnomalyTransformer
* Third-party `pyod <https://github.com/yzhao062/pyod>`_ ML models integration supported

* Support time series model ensemble with two types of ensemble forecaster, StackingEnsembleForecaster and WeightingEnsembleForecaster proposed

* RNN time series forecasting model supports categorical features and static covariates

* New representation forecaster to support representation models to solve time series forecasting task

* Support joint training of multiple time series datasets


In the future, more advanced features will be coming, including:

* More time series anomaly detection models
* More time series representation learning models
* More probabilistic forecasting models
* Scenario-specific pipelines which aim to provide an end-to-end solution for solving real-world business problems
* And more

Project GitHub: https://github.com/PaddlePaddle/PaddleTS


.. toctree::
    :maxdepth: 1
    :caption: Get Started

    Get Started <source/get_started/get_started.rst>
    Run On GPU <source/get_started/run_on_gpu.rst>
    Joint Training of Multiple Time Series <source/get_started/multiple_time_series.rst>

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

    Forecasting <source/modules/models/overview.rst>
    Third-party Model <source/modules/models/thirdparty.rst>
    Probability Forecasting <source/modules/models/probability_forecasting.rst>
    Representation  <source/modules/models/representation.rst>
    Anomaly Detection <source/modules/models/anomaly.rst>
    Classification <source/modules/models/classify.rst>
    Paddle Inference <source/modules/models/paddle_inference.rst>


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
    :caption: Ensemble

    EnsembleForecaster <source/modules/ensemble/ensemble_forecaster.rst>
    EnsembleAnomaly <source/modules/ensemble/ensemble_anomaly.rst>

.. toctree::
    :maxdepth: 1
    :caption: XAI

    XAI <source/modules/xai/overview.rst>

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
    paddlets.ensemble <source/api/paddlets.ensemble.rst>
    paddlets.utils <source/api/paddlets.utils.rst>
    paddlets.xai <source/api/paddlets.xai.rst>

