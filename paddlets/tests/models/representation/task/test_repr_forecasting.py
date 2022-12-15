# !/usr/bin/env python3
# -*- coding:utf-8 -*-
 
from base64 import encode
import pandas as pd
import numpy as np
import shutil
 
import unittest
from unittest import TestCase
 
from paddlets import TimeSeries, TSDataset
from paddlets.models.representation.task.repr_forecasting import ReprForecasting
from paddlets.models.representation.dl.ts2vec import TS2Vec
from paddlets.datasets.repository import get_dataset, dataset_list
from paddlets.utils import backtest
from paddlets.transform.sklearn_transforms import StandardScaler
from paddlets.metrics import MAE
 
from sklearn.ensemble import GradientBoostingRegressor
 
 
class TestReprForcaster(TestCase):
    def setUp(self):
        """unittest function
        """
        super().setUp()
        np.random.seed(2022)
 
        target = TimeSeries.load_from_dataframe(
            pd.DataFrame(
                np.random.randn(200, 2).astype(np.float32),
                index=pd.date_range("2022-01-01", periods=200, freq="15T"),
                columns=["a1", "a2"]
            ))
 
        observed_cov = TimeSeries.load_from_dataframe(
            pd.DataFrame(
                np.random.randn(200, 2).astype(np.float32),
                index=pd.date_range("2022-01-01", periods=200, freq="15T"),
                columns=["b", "c"]
            ))
        known_cov = TimeSeries.load_from_dataframe(
            pd.DataFrame(
                np.random.randn(200, 2).astype(np.float32),
                index=pd.date_range("2022-01-01", periods=200, freq="15T"),
                columns=["b1", "c1"]
            ))
        static_cov = {"f": 1., "g": 2.}
        # multi target
        self.tsdataset = TSDataset(target, observed_cov, known_cov, static_cov)
        self.ts = self.tsdataset
 
    def test_repr_forecasting(self):
        # case1
        ts2vec_params = {"segment_size": 20,  # maximum sequence length
                         "repr_dims": 10,
                         "batch_size": 4,
                         "max_epochs": 1}
        model = ReprForecasting(in_chunk_len=10,
                                out_chunk_len=1,
                                skip_chunk_len=0,
                                repr_model=TS2Vec,
                                repr_model_params=ts2vec_params)
 
        model.fit(self.tsdataset)
        predictions = model.predict(self.tsdataset)
        assert predictions.target.data.shape[0] == 1
        assert predictions.target.data.shape[1] == 2
        
        # case2 skip != 0
        ts2vec_params = {"segment_size": 20,  # maximum sequence length
                         "repr_dims": 10,
                         "batch_size": 4,
                         "max_epochs": 1}
        model = ReprForecasting(in_chunk_len=10,
                                out_chunk_len=1,
                                skip_chunk_len=2,
                                repr_model=TS2Vec,
                                repr_model_params=ts2vec_params)
 
        model.fit(self.tsdataset)
        predictions = model.predict(self.tsdataset)
        assert predictions.target.data.shape[0] == 1
        assert predictions.target.data.shape[1] == 2
        # case3 in_chunk = 10
        ts2vec_params = {"segment_size": 20,  # maximum sequence length
                         "repr_dims": 10,
                         "batch_size": 4,
                         "max_epochs": 1}
        model = ReprForecasting(in_chunk_len=10,
                                out_chunk_len=1,
                                skip_chunk_len=2,
                                repr_model=TS2Vec,
                                repr_model_params=ts2vec_params)
 
        model.fit(self.tsdataset)
        predictions = model.predict(self.tsdataset)
        assert predictions.target.data.shape[0] == 1
        assert predictions.target.data.shape[1] == 2
        # case4 add encode params
        ts2vec_params = {"segment_size": 20,  # maximum sequence length
                         "repr_dims": 10,
                         "batch_size": 4,
                         "max_epochs": 1}
        encode_params = {
            "mask": "mask_last"
        }
        model = ReprForecasting(in_chunk_len=1,
                                out_chunk_len=1,
                                skip_chunk_len=2,
                                repr_model=TS2Vec,
                                repr_model_params=ts2vec_params,
                                encode_params=encode_params)
 
        model.fit(self.tsdataset)
        predictions = model.predict(self.tsdataset)
        assert predictions.target.data.shape[0] == 1
        assert predictions.target.data.shape[1] == 2
        # case5 add sampling_stride
        ts2vec_params = {"segment_size": 20,  # maximum sequence length
                         "repr_dims": 10,
                         "batch_size": 4,
                         "max_epochs": 1}
        encode_params = {
            "mask": "mask_last"
        }
        model = ReprForecasting(in_chunk_len=1,
                                out_chunk_len=1,
                                skip_chunk_len=2,
                                sampling_stride=2,
                                repr_model=TS2Vec,
                                repr_model_params=ts2vec_params,
                                encode_params=encode_params)
 
        model.fit(self.tsdataset)
        predictions = model.predict(self.tsdataset)
        assert predictions.target.data.shape[0] == 1
        assert predictions.target.data.shape[1] == 2
        '''
        '''
        # case6 badcase add err final_learner
        ts2vec_params = {"segment_size": 20,  # maximum sequence length
                         "repr_dims": 10,
                         "batch_size": 4,
                         "max_epochs": 1}
        encode_params = {
            "mask": "mask_last"
        }
        with self.assertRaises(ValueError):
            model = ReprForecasting(in_chunk_len=1,
                                out_chunk_len=1,
                                skip_chunk_len=2,
                                sampling_stride=2,
                                repr_model=TS2Vec,
                                downstream_learner='err_final_learner',
                                repr_model_params=ts2vec_params,
                                encode_params=encode_params)
        # case7 add right final_learner
        ts2vec_params = {"segment_size": 20,  # maximum sequence length
                         "repr_dims": 10,
                         "batch_size": 4,
                         "max_epochs": 1}
        encode_params = {
            "mask": "mask_last"
        }
        model = ReprForecasting(in_chunk_len=1,
                                out_chunk_len=1,
                                skip_chunk_len=2,
                                sampling_stride=2,
                                repr_model=TS2Vec,
                                downstream_learner=GradientBoostingRegressor(max_depth=5),
                                repr_model_params=ts2vec_params,
                                encode_params=encode_params)
 
        model.fit(self.tsdataset)
        predictions = model.predict(self.tsdataset)
        assert predictions.target.data.shape[0] == 1
        assert predictions.target.data.shape[1] == 2

 
    
    def test_pipeline_backtest(self):
        #case1 ts2vec
        ts2vec_params = {"segment_size": 20,  # maximum sequence length
                         "sampling_stride":50,
                         "repr_dims": 10,
                         "batch_size": 4,
                         "max_epochs": 1}
        model = ReprForecasting(in_chunk_len=10,
                                out_chunk_len=100,
                                skip_chunk_len=0,
                                repr_model=TS2Vec,
                                repr_model_params=ts2vec_params)
        model.fit(self.ts)
        #model.fit(self.ts_train_scaled)
        repr_mae, ts_pred = backtest(data=self.ts,
                           model=model,
                           #start="2013-07-01 00:00:00",  # the point after "start" as the first point
                           #start="2013-02-28 23:00:00",  # the point after "start" as the first point
                           metric=MAE(),
                           predict_window=100,  # respect to out_chunk_len
                           stride=100,  # respect to sampling_stride
                           return_predicts=True  #
                           )

         #case2 CoST
        cost_params = {#"segment_size": 20,
                         "sampling_stride":50,
                         "repr_dims": 10,
                         "batch_size": 4,
                         "max_epochs": 1}
        from paddlets.models.representation import CoST
        model2 = ReprForecasting(in_chunk_len=10,
                                out_chunk_len=100,
                                skip_chunk_len=0,
                                repr_model=CoST,
                                repr_model_params=cost_params)
        model2.fit(self.ts)
        #model.fit(self.ts_train_scaled)
        repr_mae, ts_pred = backtest(data=self.ts,
                           model=model,
                           metric=MAE(),
                           predict_window=100,  # respect to out_chunk_len
                           stride=100,  # respect to sampling_stride
                           return_predicts=True  #
                           )
        
    def test_save_and_load(self):
        ts2vec_params = {"segment_size": 20,  # maximum sequence length
                         "repr_dims": 10,
                         "batch_size": 4,
                         "max_epochs": 1}
        model = ReprForecasting(in_chunk_len=10,
                                out_chunk_len=1,
                                skip_chunk_len=2,
                                repr_model=TS2Vec,
                                repr_model_params=ts2vec_params)
        model.fit(self.tsdataset)
        model.save(path='./tmp/repr1/')
        model1 = model.load(path='./tmp/repr1/')
        predictions1 = model1.predict(self.tsdataset)
        predictions = model.predict(self.tsdataset)
        print('predictions, predictions1', predictions, predictions1)
        shutil.rmtree('./tmp/repr1/')
        

    def test_generate_meta_data(self):
        ts2vec_params = {"segment_size": 20,  # maximum sequence length
                         "repr_dims": 10,
                         "batch_size": 4,
                         "max_epochs": 1}
        model = ReprForecasting(in_chunk_len=10,
                                out_chunk_len=1,
                                skip_chunk_len=0,
                                repr_model=TS2Vec,
                                repr_model_params=ts2vec_params)
 
        x_meta, y_meta = model._generate_fit_meta_data(self.tsdataset)
        assert len(x_meta) == 199
        assert len(y_meta) == 199
 
        ts2vec_params = {"segment_size": 20,  # maximum sequence length
                         "repr_dims": 10,
                         "batch_size": 4,
                         "max_epochs": 1}
        model = ReprForecasting(in_chunk_len=10,
                                out_chunk_len=20,
                                skip_chunk_len=0,
                                repr_model=TS2Vec,
                                repr_model_params=ts2vec_params)
 
        x_meta, y_meta = model._generate_fit_meta_data(self.tsdataset)
        assert len(x_meta) == 180
        assert len(y_meta) == 180
 
        ts2vec_params = {"segment_size": 20,  # maximum sequence length
                         "repr_dims": 10,
                         "batch_size": 4,
                         "max_epochs": 1}
        model = ReprForecasting(in_chunk_len=10,
                                out_chunk_len=20,
                                skip_chunk_len=0,
                                sampling_stride=5,
                                repr_model=TS2Vec,
                                repr_model_params=ts2vec_params)
 
        x_meta, y_meta = model._generate_fit_meta_data(self.tsdataset)
        assert len(x_meta) == 36
        assert len(y_meta) == 36

 
if __name__ == "__main__":
    unittest.main()