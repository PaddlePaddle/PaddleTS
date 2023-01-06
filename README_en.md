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
    NBEATS, NHiTS, LSTNet, TCN, Transformer, DeepAR, Informer, etc. for forecasting, 
    TS2Vec, CoST, etc. for representation,
    AutoEncoder, VAE, AnomalyTransformer, etc. for anomaly detection
* A set of transformation operators for data preprocessing (e.g. missing values/outliers handling, 
    one-hot encoding, normalization, and automatic date/time-related covariates generation, etc.)
* A set of analysis operators for quick data exploration (e.g. basic statistics and summary)
* Automatic time series modeling module (AutoTS) which supports mainstream Hyper Parameter Optimization algorithms and shows significant improvement on multiple models and datasets
* Third-party (e.g. scikit-learn, [pyod](https://github.com/yzhao062/pyod)) ML models & data transformations integration
* Time series model ensemble

Recently updated:

* PaddleTS now supports time series classification
* PaddleTS releases 6 new time series models. 
  USAD(UnSupervised Anomaly Detection) and MTAD-GAT(Multivariate Time-series Anomaly Detection via Graph Attention Network) for anomaly detection,
  CNN and Inception Time for time series classification, 
  SCINet(Sample Convolution and Interaction Network) and TFT(Temporal Fusion Transformer) for forecasting
* [Paddle Inference](https://www.paddlepaddle.org.cn/paddle/paddleinference) is now available for PaddleTS time series forecasting and anomaly detection
* PaddleTS now supports both model-agnostic and model-specific explanation
* PaddleTS now supports representation-based time series cluster and classification

Please also see [release notes](https://github.com/PaddlePaddle/PaddleTS/wiki/Release-Notes) to get exhaustive update lists.

In the future, more advanced features will be coming, including:

* More time series models
* Scenario-specific pipelines which aim to provide an end-to-end solution for solving real-world business problems
* And more


## More About PaddleTS

Specifically, PaddleTS consists of the following modules:


| Module                                                                                                                    | Description                                                                                   |
|---------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|
| [**paddlets.datasets**](https://paddlets.readthedocs.io/en/latest/source/modules/datasets/overview.html)                  | Unified time series representation (TSDataset) and data repository with pre-built TSDatasets. |
| [**paddlets.autots**](https://paddlets.readthedocs.io/en/latest/source/modules/autots/overview.html)                      | Automatic hyper-parameter tuning.                                                             |
| [**paddlets.transform**](https://paddlets.readthedocs.io/en/latest/source/modules/transform/overview.html)                | Data preprocessing and data transformations.                                                  |
| [**paddlets.models.forecasting**](https://paddlets.readthedocs.io/en/latest/source/modules/models/overview.html)          | PaddlePaddle-based deep neural network models for time series forecasting.                    |
| [**paddlets.models.representation**](https://paddlets.readthedocs.io/en/latest/source/modules/models/representation.html) | PaddlePaddle-based deep neural network models for time series representation.                 |
| [**paddlets.models.anomaly**](https://paddlets.readthedocs.io/en/latest/source/modules/models/anomaly.html)               | PaddlePaddle-based deep neural network models for time series anomaly detection.              |
| [**paddlets.models.classify**](https://paddlets.readthedocs.io/en/latest/source/api/paddlets.models.classify.html)        | PaddlePaddle-based deep neural network models for time series classification.                 |
| [**paddlets.pipeline**](https://paddlets.readthedocs.io/en/latest/source/modules/pipeline/overview.html)                  | Pipeline for building time series analysis and modeling workflows.                            |
| [**paddlets.metrics**](https://paddlets.readthedocs.io/en/latest/source/modules/metrics/overview.html)                    | Metrics for measuring the performance of a model.                                             |
| [**paddlets.analysis**](https://paddlets.readthedocs.io/en/latest/source/modules/analysis/overview.html)                  | Quick data exploration and advanced data analysis.                                            |
| [**paddlets.ensemble**](https://paddlets.readthedocs.io/en/latest/source/modules/ensemble/overview.html)                  | Time series ensemble methods.                                                                 |
| [**paddlets.xai**](https://paddlets.readthedocs.io/en/latest/source/api/paddlets.xai.html)                                | Model-agnostic and model-specific explanation for time series modeling.                       |
| [**paddlets.utils**](https://paddlets.readthedocs.io/en/latest/source/modules/backtest/overview.html)                     | Utility functions.                                                                            |


## Installation

### Prerequisites

* python >= 3.7
* paddlepaddle >= 2.3

Install paddlets via pip:
```bash
pip install paddlets
```

To get more detailed information, please refer to [Installation](https://paddlets.readthedocs.io/en/latest/source/installation/overview.html).


## Documentation

* [Get Started](https://paddlets.readthedocs.io/en/latest/source/get_started/get_started.html)

* [API Reference](https://paddlets.readthedocs.io/en/latest/source/api/paddlets.analysis.html)


## Community

Feel free to scan below WeChat QR code to join PaddleTS community for technical discussion with PaddleTS maintainers and community members:
<p align="center">
    <img src="docs/static/images/wechat_qrcode/wechat_qrcode.jpg" align="middle" height=300 width=300>
</p>

## Contributions

We appreciate all kinds of contributions. Please let us know if you encounter any bug by [filing an issue](https://github.com/PaddlePaddle/PaddleTS/issues).

If you are willing to contribute back bug-fixes, please go ahead without any further discussion.

If you plan to contribute new features, utility functions, or extensions to the core, please first open an issue and discuss with us.
Sending a PR without discussion might end up with rejection because we might be taking the core in a different direction than you might be aware of.


## License
PaddleTS has an Apache-style license, as found in the [LICENSE](LICENSE) file.
