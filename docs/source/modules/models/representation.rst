=============================
Representation Model Tutorial
=============================

The representation model (TS2Vect) is one of the self-supervised models, mainly hoping to learn a general feature expression for downstream tasks. The current mainstream self-supervised learning mainly includes Generative-based and Contrastive-based methods, TS2Vect is a Self-Supervised Model Based on Contrastive Method

The use of self-supervised models is divided into two phases:
    - Pre-training with unlabeled data, independent of downstream tasks
    - Fine-tune on downstream tasks using labeled data

TS2Vect follows the usage paradigm of self-supervised models:
    - Representational model training
    - Use the output of the representation model for the downstream task (the downstream task of the current case is the prediction task)

For the sake of accommodating both beginners and experienced developers, there are two ways to use representation tasks:
    - A pipeline that combines the representation model and downstream tasks, which is very friendly to beginners
    - Decoupling the representational model and downstream tasks, showing in detail how the representational model and downstream tasks are used in combination

Method one
=================

A pipeline that combines the representation model and downstream tasks

Prepare the data
================
.. code-block:: python

   import numpy as np
   np.random.seed(2022)
   import pandas as pd

   import paddle
   paddle.seed(2022)
   from paddlets.datasets import TimeSeries, TSDataset
   from paddlets.models.representation.dl.ts2vec import TS2Vec
   from paddlets.models.common.callbacks.callbacks import Callback
   from paddlets.datasets.repository import get_dataset, dataset_list
   from paddlets.models.representation.task.repr_forecasting import ReprForecasting

   data = get_dataset('ETTh1')
   data, _ = data.split('2016-09-22 06:00:00')
   train_data, test_data = data.split('2016-09-21 05:00:00')

Training 
========
.. code-block:: python

   ts2vec_params = {"segment_size": 200, 
                    "repr_dims": 320,
                    "batch_size": 32,
                    "sampling_stride": 200,
                    "max_epochs": 20}
   model = ReprForecasting(in_chunk_len=200,
                           out_chunk_len=24,
                           sampling_stride=1,
                           repr_model=TS2Vec,
                           repr_model_params=ts2vec_params)
   model.fit(train_data)

Prediction
==========
.. code-block:: python

   model.predict(train_data)

Backtest
========
.. code-block:: python

   from paddlets.utils.backtest import backtest
   score, predicts = backtest(
            data,
            model, 
            start="2016-09-21 06:00:00", 
            predict_window=24, 
            stride=24,
            return_predicts=True)


Method two
=================

Decoupling the representational model and downstream tasks. It's divided into two stages, the first stage is representation model training and prediction, and the second stage is the training and prediction of downstream tasks

1. The first stage:
===================
    - Training of the representation model
    - Output of training set and test set representation results


1.1 Prepare the data
====================
.. code-block:: python

   import numpy as np
   np.random.seed(2022)
   import pandas as pd

   import paddle
   paddle.seed(2022)
   from paddlets.datasets import TimeSeries, TSDataset
   from paddlets.models.representation.dl.ts2vec import TS2Vec
   from paddlets.models.common.callbacks.callbacks import Callback
   from paddlets.datasets.repository import get_dataset, dataset_list

   data = get_dataset('ETTh1')
   data, _ = data.split('2016-09-22 06:00:00')
   train_data, test_data = data.split('2016-09-21 05:00:00')

1.2 Training of the representation model
========================================
.. code-block:: python

   # initialize the TS2Vect object
   ts2vec = TS2Vec(
    segment_size=200,
    repr_dims=320,
    batch_size=32,
    max_epochs=20,
   )

   # training
   ts2vec.fit(train_data)

1.3 Output of training set and test set representation results
==============================================================
.. code-block:: python

   sliding_len = 200 # Use past sliding_len length points to infer the representation of the current point in time
   all_reprs = ts2vec.encode(data, sliding_len=sliding_len) 
   split_tag = len(train_data['OT'])
   train_reprs = all_reprs[:, :split_tag]
   test_reprs = all_reprs[:, split_tag:]

2. The second stage:
=======================
    - Build training and test samples for regression models
    - training and prediction

2.1 Build training and test samples for regression models
=========================================================
.. code-block:: python

   # generate samples
   def generate_pred_samples(features, data, pred_len, drop=0):
       n = data.shape[1]
       features = features[:, :-pred_len]
       labels = np.stack([ data[:, i:1+n+i-pred_len] for i in range(pred_len)], axis=2)[:, 1:]
       features = features[:, drop:]
       labels = labels[:, drop:]
       return features.reshape(-1, features.shape[-1]), \
                labels.reshape(-1, labels.shape[2]*labels.shape[3])

   pre_len = 24 # prediction lengths

   # generate training samples
   train_to_numpy = train_data.to_numpy()
   train_to_numpy = np.expand_dims(train_to_numpy, 0) # keep the same dimensions as the encode output
   train_features, train_labels = generate_pred_samples(train_reprs, train_to_numpy, pre_len, drop=sliding_len)

   # generate test samples
   test_to_numpy = test_data.to_numpy()
   test_to_numpy = np.expand_dims(test_to_numpy, 0) 
   test_features, test_labels = generate_pred_samples(test_reprs, test_to_numpy, pre_len) 

2.2 Training and prediction
===========================
.. code-block:: python

   # training
   from sklearn.linear_model import Ridge
   lr = Ridge(alpha=0.1)
   lr.fit(train_features, train_labels)

   # predict
   test_pred = lr.predict(test_features)
