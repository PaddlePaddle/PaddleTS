=============
Installation
=============



Environment Requirements
====================================
Python: 3.7/3.8/3.9 

System: Windows/MacOS/Ubuntu/CentOS


Preparations
====================================

1）Install PaddlePaddle 

The PaddleTS is built upon the `PaddlePaddle <https://www.paddlepaddle.org.cn/>`__ deep learning framework, 
thus it is recommended to install the PaddlePaddle first. To successfully run the PaddleTS, a PaddlePaddle 
version no earlier than v2.3 is required.

If you have ``PaddlePaddle`` installed already, please skip this step.

Run below command to install PaddlePaddle via pip:
::

    python -m pip install paddlepaddle==2.3.0 -i https://mirror.baidu.com/pypi/simple 

You may also refer to the PaddlePaddle website for a detailed installation guide:

  `PaddlePaddle Install Guide <https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/compile/linux-compile.html>`__

2）Update protobuf (Only needed if paddle version < 2.3.1)

The version of protobuf required is 3.19.x - 3.20.x.
If your protobuf version is earlier than v3.19 or later than v3.20, you need to upgrade or downgrade it.

Check protobuf version
::

    pip list  

Update protobuf
::

    pip install protobuf == 3.19.0  

Install PaddleTS 
====================================
It is strongly recommended that Mac M1 users use the python environment in conda to install paddlets. 
The original python of M1 may encounter problems during the installation process.

Install PaddleTS with pip
----------------------------------

::

    python -m pip install paddlets

Install PaddleTS with docker
====================================
`Docker <https://docs.docker.com/engine/install/>`_ needs to be installed locally.

1）Pull PaddleTS image 
----------------------------------

::

    docker pull registry.baidubce.com/paddlets:latest

2）Start container
-----------------------

::

    docker run --name paddle_docker -it -v $PWD:/paddle registry.baidubce.com/paddlets:latest  /bin/bash
