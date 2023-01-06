=====================
EnsembleAnomaly 
=====================

1. Prepare data
==================================

The built-in API: `get_dataset` is used to load nab_temp dataset: `NAB_TEMP`.

.. code-block:: python

   from paddlets.datasets.repository import get_dataset

   ts_data = get_dataset('NAB_TEMP') # label_col: 'label', feature_cols: 'value'


2. Data processing
==================================

Set up training set (top 15%), and standardize the data.

.. code-block:: python

   import paddle
   import numpy as np
   from paddlets.transform import StandardScaler

   #set seed
   seed = 2022
   paddle.seed(seed)
   np.random.seed(seed)
   train_tsdata, test_tsdata = ts_data.split(0.15)

   #standardize
   scaler = StandardScaler('value')
   scaler.fit(train_tsdata)
   train_tsdata_scaled = scaler.transform(train_tsdata)
   test_tsdata_scaled = scaler.transform(test_tsdata)

3. Prepare Models
==================================
Prepare base models for ensemble model. 




.. code:: python

   from paddlets.models.anomaly import AutoEncoder
   from paddlets.models.anomaly import VAE
   ae_params = {"max_epochs":100}
   vae_params = {"max_epochs":100}


4. Construct and Fitting
===================================

EnsembleAnomaly use a aggragate function to aggragate base model predictions, use "mean" mode by default.
More infomation about EnsembleAnomaly  please read `EnsembleAnomaly doc <../../api/paddlets.ensemble.weighting_ensemble.html>`_ .

Example1 

.. code:: python

    from paddlets.ensemble import WeightingEnsembleAnomaly

    model = WeightingEnsembleAnomaly(
    in_chunk_len=2,
    estimators=[(AutoEncoder, ae_params),(VAE, vae_params)],
    mode = "voting")

    model.fit(train_tsdata_scaled)


5. Model prediction and evaluation
=======================================

Use the trained model for prediction and evaluation.

.. code-block:: python

   from paddlets.metrics import F1,ACC,Precision,Recall
   
   pred_label = model.predict(test_tsdata_scaled)
   lable_name = pred_label.target.data.columns[0]
   f1 = F1()(test_tsdata, pred_label)
   precision = Precision()(test_tsdata, pred_label)
   recall = Recall()(test_tsdata, pred_label)
   print ('f1: ', f1[lable_name])
   print ('precision: ', precision[lable_name])
   print ('recall: ', recall[lable_name])
