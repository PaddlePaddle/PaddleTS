========================
Paddle Inference Support
========================

The vast majority of models in PaddleTS support Paddle Inference. For the instruction and functions of Paddle Inference, please refer to
`Paddle Inference <https://paddleinference.paddlepaddle.org.cn/master/product_introduction/inference_intro.html>`_ . 
PaddleTS supports export of native Paddle network models for the deployment of Paddle Inference. 
To simplify the process, PaddleTS provides the Python tool to build input data, so users can build input data for Paddle Inference easily.


1. Build and Train Model
========================

.. code-block:: python

    from paddlets.datasets.repository import get_dataset
    from paddlets.models.forecasting.dl.rnn import RNNBlockRegressor

    # prepare data
    dataset = get_dataset("WTH")

    rnn = RNNBlockRegressor(
        in_chunk_len=4, 
        out_chunk_len=2,
        max_epochs=10
    )

    #fit
    rnn.fit(dataset)

    #predict
    rnn.predict(dataset)

    #                      WetBulbCelsius
    # 2014-01-01 00:00:00       -1.436116
    # 2014-01-01 01:00:00       -2.057547

2 Save model
============

`network_model` and `dygraph_to_static` parameters are added in `save` interfaces of all PaddleTS time-series forecasting and anomaly detection models.
The `network_model` parameter controls which objects are dumped. The default value of network_model is False, which means the dumped files can be only used by `PaddleTS.predict` interface. If True, additional files which can be used by `paddle inference` will be dumped. 
`dygraph_to_static` converts the dumped model from a dynamic graph to a static one, and it works only when network_model is True. 
For more information, please refer to `dygraph_to_static <https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/jit/index_cn.html>`_. 


.. code-block:: python
    
    rnn.save("./rnn", network_model=True, dygraph_to_static=True)

    # dump file names
    # ./rnn.pdmodel
    # ./rnn.pdiparams
    # ./rnn_model_meta

The preceding code snippet is an example to dump model files after saving. `rnn.pdmodel` and `rnn.pdiparams` are native paddle models and parameters, which can be used for `Paddle Inference`.
At the meanwhile, PaddleTS generates `rnn_model_meta` file for model description with input data types and metadata of shape. Users can deploy the app correctly in an easy way.

3. Paddle Inference
===================

With the dumped model in step 2, users can deploy models by `Paddle Inference`. Here's the example.

3.1 Load model
---------------

.. code-block:: python

    import paddle.inference as paddle_infer
    config = paddle_infer.Config("./rnn.pdmodel", "./rnn.pdiparams")
    predictor = paddle_infer.create_predictor(config)
    input_names = predictor.get_input_names()
    print(f"input_name: f{input_names}")

    # input_name: f['observed_cov_numeric', 'past_target']

    import json
    with open("./rnn_model_meta") as f:
        json_data = json.load(f)
        print(json_data)
    
    # {'model_type': 'forecasting', 'ancestor_classname_set': ['RNNBlockRegressor', 'PaddleBaseModelImpl', 'PaddleBaseModel', 'BaseModel', 'Trainable', 'ABC', 'object'], 'modulename': 'paddlets.models.forecasting.dl.rnn', 'size': {'in_chunk_len': 4, 'out_chunk_len': 2, 'skip_chunk_len': 0}, 'input_data': {'past_target': [None, 4, 1], 'observed_cov_numeric': [None, 4, 11]}}

As the above code snippet shows, we can build the input based on input_name, which contains attributes of the data (target、known_cov、observed_cov).
In addition to input_name, rnn_model_meta contains input types, the shape format of data, original `in_chunk_len` and `out_chunk_len` and so on.
With these information, users can build input data correctly and easily.

3.2 Build Input Tensor
----------------------

PaddleTS also has built-in functions to build Paddle Inference input automatically.

.. code-block:: python

    from paddlets.utils.utils import build_ts_infer_input
    
    input_data = build_ts_infer_input(dataset, "./rnn_model_meta")

    for key, value in json_data['input_data'].items():
        input_handle1 = predictor.get_input_handle(key)
        #set batch_size=1
        value[0] = 1
        input_handle1.reshape(value)
        input_handle1.copy_from_cpu(input_data[key])


3.3 Inference
-------------

.. code-block:: python

    predictor.run()
    output_names = predictor.get_output_names()
    output_handle = predictor.get_output_handle(output_names[0])
    output_data = output_handle.copy_to_cpu() 
    print(output_data)

    # [[[-1.436116 ]
    #   [-2.0575469]]]

The results of above code snippet are basically consistent with the results of predict in chapter 1.
 