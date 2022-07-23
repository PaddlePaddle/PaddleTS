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
* 内置业界领先的深度学习模型，如NBEATS， NHiTS, LSTNet, TCN, Transformer等
* 内置经典数据转换算子，支持数据处理与转换，包括缺失值填充、异常值处理、归一化、时间相关的协变量提取等
* 内置时序数据分析算子，帮助开发者便捷实现数据探索，包括数据统计量信息及数据摘要等功能

未来，更多的高级特性会进一步发布，包括但不限于：
* 自动超参寻优
* 时序表征模型
* 概率预测模型
* 场景化Pipeline，支持端到端真实场景解决方案


## 关于 PaddleTS

具体来说，PaddleTS 时序库包含以下子模块：

| 模块                                                                                                   | 简述                                              |
|------------------------------------------------------------------------------------------------------|-------------------------------------------------|
| [**paddlets.datasets**](https://paddlets.readthedocs.io/zh_CN/latest/source/modules/datasets.html)   | 一个标准的时序数据对象，在整个建模生命周期内提供时序数据的表示和操作能力。           |
| [**paddlets.transform**](https://paddlets.readthedocs.io/zh_CN/latest/source/modules/transform.html) | 一个时序数据转换模块，提供数据预处理和特征工程相关的处理能力，满足多种多样的时序数据处理需求。 |
| [**paddlets.models**](https://paddlets.readthedocs.io/zh_CN/latest/source/modules/models.html)       | 一个与PaddlePaddle框架深度集成的时序模型组件，提供业界领先的时序建模能力      |
| [**paddlets.pipeline**](https://paddlets.readthedocs.io/zh_CN/latest/source/modules/pipeline.html)   | 一个用于为时序建模任务构建工作流的模块。其通常可以由一组数据转换模块和一个时序模型组成。    |
| [**paddlets.metrics**](https://paddlets.readthedocs.io/zh_CN/latest/source/modules/metrics.html)     | 一个用于计算模型性能指标的模块。                                |
| [**paddlets.analysis**](https://paddlets.readthedocs.io/zh_CN/latest/source/modules/analysis.html)   | 一个提供多种分析组件，帮助完成时序数据分析的模块。                       |
| [**paddlets.utils**](https://paddlets.readthedocs.io/zh_CN/latest/source/modules/utils.html)         | 一个工具集模块，提供诸如回测等功能。                              |


## 安装

### 前置条件

* python >= 3.7
* paddlepaddle >= 2.3

pip 安装 paddlets 命令如下：
```bash
pip install paddlets
```

更多安装方式请参考：[环境安装](https://paddlets.readthedocs.io/zh_CN/latest/source/installation/installation.html)


## 文档

[开始使用](https://paddlets.readthedocs.io/zh_CN/latest/source/tutorials)
[API文档](https://paddlets.readthedocs.io/zh_CN/latest/source/api)


## 代码发布与贡献

我们非常感谢每一位代码贡献者。如果您发现任何Bug，请随时通过[提交issue](https://github.com/PaddlePaddle/PaddleTS/issues)的方式告知我们。

如果您计划贡献涉及新功能、工具类函数、或者扩展PaddleTS的核心组件相关的代码，请您在提交代码之前先[提交issue](https://github.com/PaddlePaddle/PaddleTS/issues)，并针对此次提交的功能与我们进行讨论。

如果在没有讨论的情况下直接发起的PR请求，可能会导致此次PR请求被拒绝。原因是对于您提交的PR涉及的模块，我们也许希望该模块朝着另一个不同的方向发展。


## 许可证
PaddleTS 使用Apache风格的许可证, 可参考 [LICENSE](LICENSE) 文件.
