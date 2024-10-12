**简体中文** | [English](./README_en.md)

<p align="center">
  <img src="docs/static/images/logo/paddlets-readme-logo.png" align="middle"  width="500">
<p>

------------------------------------------------------------------------------------------

<p align="center">
  <a href="https://github.com/PaddlePaddle/PaddleTS/graphs/contributors"><img src="https://img.shields.io/github/contributors/PaddlePaddle/PaddleTS?color=9ea"></a>
  <a href=""><img src="https://img.shields.io/badge/python-3.7+-aff.svg"></a>
  <a href=""><img src="https://img.shields.io/badge/paddlepaddle-2.3.0+-aff.svg"></a>
  <a href="https://github.com/PaddlePaddle/PaddleTS/commits"><img src="https://img.shields.io/github/commit-activity/m/PaddlePaddle/PaddleTS?color=3af"></a>
  <a href="https://github.com/PaddlePaddle/PaddleTS/issues"><img src="https://img.shields.io/github/issues/PaddlePaddle/PaddleTS?color=9cc"></a>
</p>

--------------------------------------------------------------------------------

PaddleTS 是一个易用的深度时序建模的Python库，它基于飞桨深度学习框架PaddlePaddle，专注业界领先的深度模型，旨在为领域专家和行业用户提供可扩展的时序建模能力和便捷易用的用户体验。PaddleTS 的主要特性包括：

* 设计统一数据结构，实现对多样化时序数据的表达，支持单目标与多目标变量，支持多类型协变量
* 封装基础模型功能，如数据加载、回调设置、损失函数、训练过程控制等公共方法，帮助开发者在新模型开发过程中专注网络结构本身
* 内置业界领先的深度学习模型，包括NBEATS、NHiTS、LSTNet、TCN、Transformer、DeepAR、Informer等时序预测模型，
  TS2Vec、CoST等时序表征模型，以及
  Autoencoder、VAE、AnomalyTransformer等时序异常检测模型
* 内置多样化的数据转换算子，支持数据处理与转换，包括缺失值填充、异常值处理、归一化、时间相关的协变量提取等
* 内置经典的数据分析算子，帮助开发者便捷实现数据探索，包括数据统计量信息及数据摘要等功能
* 自动模型调优AutoTS，支持多类型HPO(Hyper Parameter Optimization)算法，在多个模型和数据集上展现显著调优效果
* 第三方机器学习模型及数据转换模块自动集成，支持包括sklearn、[pyod](https://github.com/yzhao062/pyod)等第三方库的时序应用
* 支持在GPU设备上运行基于PaddlePaddle的时序模型
* 时序模型集成学习能力

📣 **近期更新**

* 🔥 飞桨低代码开发工具PaddleX，依托于PaddleTS的先进技术，支持时序分析领域的低代码开发能力
  * 🎨 [**模型丰富一键调用**](docs/paddlex/quick_start.md)：将时序预测、时序异常检测和时序分类涉及的**13个模型**整合为3条模型产线，通过极简的**Python API一键调用**，快速体验模型效果。此外，同一套API，也支持图像分类、图像分割、目标检测、文本图像智能分析、通用OCR等共计**200+模型**，形成20+单功能模块，方便开发者进行**模型组合使用**。
  * 🚀 [**提高效率降低门槛**](docs/paddlex/overview.md)：提供基于**统一命令**和**图形界面**两种方式，实现模型简洁高效的使用、组合与定制。支持**高性能部署、服务化部署和端侧部署**等多种部署方式。此外，对于各种主流硬件如**英伟达GPU、昆仑芯、昇腾、寒武纪和海光**等，进行模型开发时，都可以**无缝切换**。

* 🔥 增加7个时序预测前沿算法[**DLinear、NLinear、RLinear、Nonstationary、PatchTST、TiDE、TimesNet**](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta1/docs/module_usage/tutorials/time_series_modules/time_series_forecasting.md)，5个时序异常检测前沿算法[**AutoEncoder_ad、DLinear_ad、Nonstationary_ad、PatchTST_ad、TimesNet_ad**](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta1/docs/module_usage/tutorials/time_series_modules/time_series_anomaly_detection.md)和1个时序分类算法[**TimesNet_cls**](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta1/docs/module_usage/tutorials/time_series_modules/time_series_classification.md)。
  

* 全新发布6个深度时序模型。
  USAD(UnSupervised Anomaly Detection)与MTAD_GAT(Multivariate Time-series Anomaly Detection via Graph Attention Network)异常检测模型，
  CNN与Inception Time时序分类模型，
  SCINet(Sample Convolution and Interaction Network)与TFT(Temporal Fusion Transformer)时序预测模型


## ⚡ [快速开始](docs/paddlex/quick_start.md)

## 🔥 [低代码开发](docs/paddlex/overview.md)

## 📖 文档

* [开始使用](https://paddlets.readthedocs.io/zh_CN/latest/source/get_started/get_started.html)

* [API文档](https://paddlets.readthedocs.io/zh_CN/latest/source/api/paddlets.analysis.html)

## 📝 关于 PaddleTS

具体来说，PaddleTS 时序库包含以下子模块：

| 模块                                                                                                                           | 简述                                     |
|------------------------------------------------------------------------------------------------------------------------------|----------------------------------------|
| [**paddlets.datasets**](https://paddlets.readthedocs.io/zh_CN/latest/source/modules/datasets/overview.html)                  | 时序数据模块，统一的时序数据结构和预定义的数据处理方法            |
| [**paddlets.autots**](https://paddlets.readthedocs.io/en/latest/source/modules/autots/overview.html)                         | 自动超参寻优                                 |
| [**paddlets.transform**](https://paddlets.readthedocs.io/zh_CN/latest/source/modules/transform/overview.html)                | 数据转换模块，提供数据预处理和特征工程相关能力                |
| [**paddlets.models.forecasting**](https://paddlets.readthedocs.io/zh_CN/latest/source/modules/models/overview.html)          | 时序模型模块，基于飞桨深度学习框架PaddlePaddle的时序预测模型   |
| [**paddlets.models.representation**](https://paddlets.readthedocs.io/zh_CN/latest/source/modules/models/representation.html) | 时序模型模块，基于飞桨深度学习框架PaddlePaddle的时序表征模型   |
| [**paddlets.models.anomaly**](https://paddlets.readthedocs.io/zh_CN/latest/source/modules/models/anomaly.html)               | 时序模型模块，基于飞桨深度学习框架PaddlePaddle的时序异常检测模型 |
| [**paddlets.models.classify**](https://paddlets.readthedocs.io/zh_CN/latest/source/api/paddlets.models.classify.html)        | 时序模型模块，基于飞桨深度学习框架PaddlePaddle的时序分类模型   |
| [**paddlets.pipeline**](https://paddlets.readthedocs.io/zh_CN/latest/source/modules/pipeline/overview.html)                  | 建模任务流模块，支持特征工程、模型训练、模型评估的任务流实现         |
| [**paddlets.metrics**](https://paddlets.readthedocs.io/zh_CN/latest/source/modules/metrics/overview.html)                    | 效果评估模块，提供多维度模型评估能力                     |
| [**paddlets.analysis**](https://paddlets.readthedocs.io/zh_CN/latest/source/modules/analysis/overview.html)                  | 数据分析模块，提供高效的时序特色数据分析能力                 |
| [**paddlets.ensemble**](https://paddlets.readthedocs.io/zh_CN/latest/source/modules/ensemble/overview.html)                  | 时序集成学习模块，基于模型集成提供时序预测能力                |
| [**paddlets.xai**](https://paddlets.readthedocs.io/zh_CN/latest/source/api/paddlets.xai.html)                                | 时序模型可解释性模块                             |
| [**paddlets.utils**](https://paddlets.readthedocs.io/zh_CN/latest/source/modules/backtest/overview.html)                     | 工具集模块，提供回测等基础功能                        |


## 代码发布与贡献

我们非常感谢每一位代码贡献者。如果您发现任何Bug，请随时通过[提交issue](https://github.com/PaddlePaddle/PaddleTS/issues)的方式告知我们。

如果您计划贡献涉及新功能、工具类函数、或者扩展PaddleTS的核心组件相关的代码，请您在提交代码之前先[提交issue](https://github.com/PaddlePaddle/PaddleTS/issues)，并针对此次提交的功能与我们进行讨论。

如果在没有讨论的情况下直接发起的PR请求，可能会导致此次PR请求被拒绝。原因是对于您提交的PR涉及的模块，我们也许希望该模块朝着另一个不同的方向发展。


## 许可证
PaddleTS 使用Apache风格的许可证, 可参考 [LICENSE](LICENSE) 文件.
