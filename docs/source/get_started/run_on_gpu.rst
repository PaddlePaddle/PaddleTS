============
Run on GPU
============

PaddleTS uses PaddlePaddle framework to build deep time series models. Since PaddlePaddle provides GPU capability, it is
quite easy to fit and predict PaddleTS models on GPU.


1. Prerequisites
==================

There are few prerequisites before running a PaddleTS time series model on Nvidia GPU devices:

- Verify the system has Nvidia GPU and relevant Driver(s) installed.
  See `Nvidia Installation Guide <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html>`__
  to get more details.
- Verify the system supported gpu-version of PaddlePaddle installed.
  See `PaddlePaddle-gpu Installation Guide <https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html>`__
  to get more details.

In the meantime, for known reason, it is currently not feasible to run the built-in
`NHiTS Model <../api/paddlets.models.forecasting.dl.nhits.html>`_
on GPU, you can expect a future fix.

2. Example
============

Generally, there are three steps to run PaddleTS deep time series models on GPU:

- Get available GPU devices in your system.
- Choose a GPU device to use.
- Execute your program to fit and predict model on GPU.

See below step-by-step instructions to get details.

2.1 Get available GPU devices
-------------------------------

Assume the system already have GPU and its driver installed.
You may run `nvidia-smi` command to retrieve a list of the GPU devices containing detailed state information.

Below is a sample output. Briefly, it indicates the following:

- There are 4 Nvidia A30 GPU devices available.
- Each device has 24258MiB free memory to use.
- Currently no running process occupying any devices.

.. code-block:: shell

    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 470.82.01    Driver Version: 470.82.01    CUDA Version: 11.4     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |                               |                      |               MIG M. |
    |===============================+======================+======================|
    |   0  NVIDIA A30          On   | 00000000:17:00.0 Off |                    0 |
    | N/A   33C    P0    31W / 165W |      3MiB / 24258MiB |      0%      Default |
    |                               |                      |             Disabled |
    +-------------------------------+----------------------+----------------------+
    |   1  NVIDIA A30          On   | 00000000:65:00.0 Off |                    0 |
    | N/A   35C    P0    29W / 165W |      3MiB / 24258MiB |      0%      Default |
    |                               |                      |             Disabled |
    +-------------------------------+----------------------+----------------------+
    |   2  NVIDIA A30          On   | 00000000:B1:00.0 Off |                    0 |
    | N/A   33C    P0    28W / 165W |      3MiB / 24258MiB |      0%      Default |
    |                               |                      |             Disabled |
    +-------------------------------+----------------------+----------------------+
    |   3  NVIDIA A30          On   | 00000000:E3:00.0 Off |                    0 |
    | N/A   35C    P0    30W / 165W |      3MiB / 24258MiB |      0%      Default |
    |                               |                      |             Disabled |
    +-------------------------------+----------------------+----------------------+

    +-----------------------------------------------------------------------------+
    | Processes:                                                                  |
    |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
    |        ID   ID                                                   Usage      |
    |=============================================================================|
    |  No running processes found                                                 |
    +-----------------------------------------------------------------------------+

2.2 Explicitly set GPU devices to use
--------------------------------------

Nvidia provides `CUDA_VISIBLE_DEVICES` environment variable to rearrange the installed CUDA devices that will be visible to a CUDA application.
Suppose there are totally 4 GPUs {0, 1, 2, 3} available in your system, given the scenario that only the device 0 will be used,
thus you may run `export CUDA_VISIBLE_DEVICES=0` in the Linux shell to explicitly make the device 0 visible to a CUDA application.

If you run `echo $CUDA_VISIBLE_DEVICES`, the output `0` indicates that we choose to use the device 0 to fit and predict time series model.

See `Nvidia CUDA_VISIBLE_DEVICES <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#cuda-visible-devices>`__ to get more details.


2.3 Install GPU-capable PaddleTS
--------------------------------------

There are currently 2 ways to setup environment:

- pip
- docker


2.3.1 pip install
~~~~~~~~~~~~~~~~~~

Before installing PaddleTS, it is required to first install the
`gpu-capable paddlepaddle <https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html#gpu>`__

.. code-block:: shell

    pip install paddlepaddle-gpu


Now install the latest version of PaddleTS by running the following:

.. code-block:: shell

    pip install paddlets

2.3.2 docker
~~~~~~~~~~~~~~

It is required to follow the
`Nvidia Container Toolkit Installation Guide <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html>`__
to install the nvidia-docker engine.

Now we can pull the gpu-capable docker image.

.. code-block:: shell

    nvidia-docker pull registry.baidubce.com/paddlets/paddlets:0.2.0-gpu-paddle2.3.2-cuda11.2-cudnn8


2.3 Use GPU device to fit and predict models
----------------------------------------------

After completing the above, the rest steps to fit and predict the model are identical to the ones on CPU.
See `Get Started <../modules/datasets/overview.html>`_ to get more details.

.. code-block:: python

    import numpy as np

    from paddlets.datasets.repository import get_dataset
    from paddlets.transform.normalization import StandardScaler
    from paddlets.models.forecasting import MLPRegressor

    np.random.seed(2022)

    # prepare data
    tsdataset = get_dataset("WTH")
    ts_train, ts_val_test = ts.split("2012-03-31 23:00:00")
    ts_val, ts_test = ts_val_test.split("2013-02-28 23:00:00")

    # transform
    scaler = StandardScaler()
    scaler.fit(ts_train)
    ts_train_scaled = scaler.transform(ts_train)
    ts_val_scaled = scaler.transform(ts_val)
    ts_test_scaled = scaler.transform(ts_test)
    ts_val_test_scaled = scaler.transform(ts_val_test)

    # model
    model = MLPRegressor(
         in_chunk_len=7 * 24,
         out_chunk_len=24,
         skip_chunk_len=0,
         sampling_stride=24,
         eval_metrics=["mse", "mae"],
         batch_size=32,
         max_epochs=1000,
         patience=100,
         use_bn=True,
         seed=2022
    )

    model.fit(ts_train_scaled, ts_val_scaled)

    predicted_tsdataset = model.predict(ts_val_test_scaled)

    print(predicted_tsdataset)

    #                      WetBulbCelsius
    # 2014-01-01 00:00:00       -0.124221
    # 2014-01-01 01:00:00       -0.184970
    # 2014-01-01 02:00:00       -0.398122
    # 2014-01-01 03:00:00       -0.500016
    # 2014-01-01 04:00:00       -0.350443
    # 2014-01-01 05:00:00       -0.580986
    # 2014-01-01 06:00:00       -0.482264
    # 2014-01-01 07:00:00       -0.413248
    # 2014-01-01 08:00:00       -0.451982
    # 2014-01-01 09:00:00       -0.471430
    # 2014-01-01 10:00:00       -0.427212
    # 2014-01-01 11:00:00       -0.264509
    # 2014-01-01 12:00:00       -0.308266
    # 2014-01-01 13:00:00       -0.386270
    # 2014-01-01 14:00:00       -0.261341
    # 2014-01-01 15:00:00       -0.492441
    # 2014-01-01 16:00:00       -0.497322
    # 2014-01-01 17:00:00       -0.628926
    # 2014-01-01 18:00:00       -0.528971
    # 2014-01-01 19:00:00       -0.588881
    # 2014-01-01 20:00:00       -0.860580
    # 2014-01-01 21:00:00       -0.742121
    # 2014-01-01 22:00:00       -0.819053
    # 2014-01-01 23:00:00       -0.875322
