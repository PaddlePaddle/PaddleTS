# 快速开始

>**说明：**
>* 飞桨低代码开发工具[PaddleX](https://github.com/PaddlePaddle/PaddleX/tree/release/3.0-beta1)，依托于PaddleTS的先进技术，支持了时序分析领域的**低代码全流程**开发能力。通过低代码全流程开发，可实现简单且高效的模型使用、组合与定制。
>* PaddleX 致力于实现产线级别的模型训练、推理与部署。模型产线是指一系列预定义好的、针对特定AI任务的开发流程，其中包含能够独立完成某类任务的单模型（单功能模块）组合。本文档提供**时序分析相关产线**的快速使用，单功能模块的快速使用以及更多功能请参考[PaddleTS低代码全流程开发](./overview.md)中相关章节。


### 🛠️ 安装

> ❗安装PaddleX前请先确保您有基础的**Python运行环境**。
* **安装PaddlePaddle**
```bash
# cpu
python -m pip install paddlepaddle==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# gpu，该命令仅适用于 CUDA 版本为 11.8 的机器环境
python -m pip install paddlepaddle-gpu==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# gpu，该命令仅适用于 CUDA 版本为 12.3 的机器环境
python -m pip install paddlepaddle-gpu==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cu123/
```
> ❗ 更多飞桨 Wheel 版本请参考[飞桨官网](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)。

* **安装PaddleX**

```bash
pip install https://paddle-model-ecology.bj.bcebos.com/paddlex/whl/paddlex-3.0.0b1-py3-none-any.whl
```

> ❗ 更多安装方式参考[PaddleX安装教程](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta1/docs/installation/installation.md)

### 💻 命令行使用

一行命令即可快速体验产线效果，统一的命令行格式为：

```bash
paddlex --pipeline [产线名称] --input [输入图片] --device [运行设备]
```

只需指定三个参数：
* `pipeline`：产线名称
* `input`：待处理的输入图片的本地路径或URL
* `device`: 使用的GPU序号（例如`gpu:0`表示使用第0块GPU），也可选择使用CPU（`cpu`）


以时序预测产线为例：
```bash
paddlex --pipeline ts_fc --input https://paddle-model-ecology.bj.bcebos.com/paddlex/ts/demo_ts/ts_fc.csv --device gpu:0
```
运行后，得到的结果为：

```bash
{'ts_path': '/root/.paddlex/predict_input/ts_fc.csv', 'forecast':                            OT
date  
2018-06-26 20:00:00  9.586131
2018-06-26 21:00:00  9.379762
2018-06-26 22:00:00  9.252275
2018-06-26 23:00:00  9.249993
2018-06-27 00:00:00  9.164998
...                       ...
2018-06-30 15:00:00  8.830340
2018-06-30 16:00:00  9.291553
2018-06-30 17:00:00  9.097666
2018-06-30 18:00:00  8.905430
2018-06-30 19:00:00  8.993793

[96 rows x 1 columns]}
```


其他产线的命令行使用，只需将`pipeline`参数调整为相应产线的名称。下面列出了每个产线对应的命令：


| 产线名称      | 使用命令                                                                                                                                                                                             |
|-----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 通用时序预测    | `paddlex --pipeline ts_fc --input https://paddle-model-ecology.bj.bcebos.com/paddlex/ts/demo_ts/ts_fc.csv --device gpu:0`                                                                    |
| 通用时序异常检测  | `paddlex --pipeline ts_ad --input https://paddle-model-ecology.bj.bcebos.com/paddlex/ts/demo_ts/ts_ad.cs --device gpu:0` |
| 通用时序分类    | `paddlex --pipeline ts_cls --input https://paddle-model-ecology.bj.bcebos.com/paddlex/ts/demo_ts/ts_cls.csv --device gpu:0`       |



### 📝 Python脚本使用

几行代码即可完成产线的快速推理，以时序分类产线为例：
```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="ts_cls")

output = pipeline.predict("ts_cls.csv")
for res in output:
    res.print() ## 打印预测的结构化输出
    res.save_to_csv("./output/") ## 保存csv格式结果
```

运行后得到的结果为：

```bash
{'ts_path': '/root/.paddlex/predict_input/ts_cls.csv', 'classification':         classid     score
sample  
0             0  0.617688}
````

执行了如下几个步骤：

* `create_pipeline()` 实例化产线对象
* 传入图片并调用产线对象的`predict` 方法进行推理预测
* 对预测结果进行处理

其他产线的Python脚本使用，只需将`create_pipeline()`方法的`pipeline`参数调整为相应产线的名称。下面列出了每个产线对应的参数名称及详细的使用解释：

| 产线名称     | 对应参数                 | 详细说明 |
|----------|----------------------|------|
| 通用时序预测       | `ts_fc` | [通用时序预测产线Python脚本使用说明](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta1/docs/pipeline_usage/tutorials/time_series_pipelines/time_series_forecasting.md) |
| 通用时序异常检测   | `ts_ad` | [通用时序异常检测产线Python脚本使用说明](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta1/docs/pipeline_usage/tutorials/time_series_pipelines/time_series_anomaly_detection.md) |
| 通用时序分类       | `ts_cls` | [通用时序分类产线Python脚本使用说明](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta1/docs/pipeline_usage/tutorials/time_series_pipelines/time_series_classification.md) |
