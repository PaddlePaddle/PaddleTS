[简体中文](./README_cn.md) |  **English**

<p align="center">
  <img src="docs/static/images/logo/paddlets-readme-logo.png" align="middle" width=500>
<p>

------------------------------------------------------------------------------------------

<p align="center">
  <a href="https://github.com/PaddlePaddle/PaddleTS/graphs/contributors"><img src="https://img.shields.io/github/contributors/PaddlePaddle/PaddleNLP?color=9ea"></a>
  <a href=""><img src="https://img.shields.io/badge/python-3.7+-aff.svg"></a>
  <a href=""><img src="https://img.shields.io/badge/paddlepaddle-2.3.0+-aff.svg"></a>
  <a href="https://github.com/PaddlePaddle/PaddleTS/commits"><img src="https://img.shields.io/github/commit-activity/m/PaddlePaddle/PaddleTS?color=3af"></a>
  <a href="https://github.com/PaddlePaddle/PaddleTS/issues"><img src="https://img.shields.io/github/issues/PaddlePaddle/PaddleTS?color=9cc"></a>
</p>

--------------------------------------------------------------------------------


PaddleTS - PaddlePaddle-based Time Series Modeling in Python

PaddleTS is an easy to use Python library for deep time series modeling,
    focusing on the state-of-the-art deep neural network models based on 
    PaddlePaddle deep learning framework. It aims to provide great flexibility 
    and excellent user experiences for practitioners and professionals. It’s featured with:

* A unified data structure named TSDataset for representing time series data with one 
    or multiple target variables and optional different kinds of covariates 
    (e.g. known covariates, observed covariates, and static covariates, etc.);
* A base model class named PaddleBaseModelImpl , which inherits from the PaddleBaseModel 
    and further encapsulates some routine procedures (e.g. data loading, callbacks setup, 
    loss computation, training loop control, etc.) and allows developers to focus on 
    the implementation of network architectures when developing new models;
* A set of state-of-the-art deep learning models (e.g. NBEATS, NHiTS, LSTNet, TCN, Transformer, etc);
* A set of transformation operators for data preprocessing (e.g. missing values/outliers handling, 
    one-hot encoding, normalization, and automatic date/time-related covariate generation, etc.);
* A set of analysis operators for quick data exploration (e.g. basic statistics and summary).

In future, more advanced features will be coming, including:

* Automatic hyper-parameter tuning;
* Time series representation learning models；
* Add support for probabilistic forecasting;
* Scenario-specific pipelines which aim to provide an end-to-end solution for solving real-world business problems;
* And more.


## More About PaddleTS

Specifically, PaddleTS consists of the following modules:

| Module                                                                                                     | Description                                                                                                                     |
|------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|
| [**paddlets.datasets**](https://paddlets.readthedocs.io/en/latest/source/modules/datasets/overview.html)   | A uniform time series data object provides data representation and manipulation capabilities throughout the modeling lifecycle. |
| [**paddlets.transform**](https://paddlets.readthedocs.io/en/latest/source/modules/transform/overview.html) | A data processing module containing a set of time series specific data transformations to meet a wide variety of needs.         |
| [**paddlets.models**](https://paddlets.readthedocs.io/en/latest/source/modules/models/overview.html)       | A time series modeling module deeply integrated with paddlepaddle framework for maximum flexibility.                            |
| [**paddlets.pipeline**](https://paddlets.readthedocs.io/en/latest/source/modules/pipeline/overview.html)   | A module designed to build a workflow for time series modeling which may be comprised of a set of transformations and a model.  |
| [**paddlets.metrics**](https://paddlets.readthedocs.io/en/latest/source/modules/metrics/overview.html)     | A module for measuring the performance of a model.                                                                              |
| [**paddlets.analysis**](https://paddlets.readthedocs.io/en/latest/source/modules/analysis/overview.html)   | A module provides a variety of analyzers for time series data inspection.                                                       |
| [**paddlets.utils**](https://paddlets.readthedocs.io/en/latest/source/modules/backtest/overview.html)             | A module contains utility functions such as backtest.                                                                           |


## Installation

### Prerequisites

* python >= 3.7
* paddlepaddle >= 2.3

Commands to install paddlets via pip wheels:
```bash
pip install paddlets
```

To get more details, please see at [Installation](https://paddlets.readthedocs.io/en/latest/source/installation/overview.html)


## Documentation

Please refer to the following documents to get more in depth information.

* [Get Started](https://paddlets.readthedocs.io/en/latest/source/get_started/get_started.html)

* [API Reference](https://paddlets.readthedocs.io/en/latest/source/api/paddlets.analysis.html)


## Releases and Contributing

We appreciate all contributions, please let us know if you encounter a bug by [filing an issue](https://github.com/PaddlePaddle/PaddleTS/issues).

If you are planning to contribute back bug-fixes, please do so without any further discussion.

If you plan to contribute new features, utility functions, or extensions to the core, please first open an issue and discuss the feature with us.
Sending a PR without discussion might end up resulting in a rejected PR because we might be taking the core in a different direction than you might be aware of.


## License
PaddleTS has an Apache-style license, as found in the [LICENSE](LICENSE) file.
