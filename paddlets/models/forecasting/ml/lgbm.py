# !/usr/bin/env python3
# -*- coding:utf-8 -*-


from typing import List
from typing import Optional, List, Union
from itertools import product

import numpy as np
import lightgbm as lgbm

from paddlets.datasets.tsdataset import TSDataset
from paddlets.logger import Logger
from paddlets.logger import raise_log
from paddlets.models.forecasting.ml.ml_base import MLBaseModel
from paddlets.models.forecasting.ml.adapter.data_adapter import DataAdapter
from paddlets.models.utils import to_tsdataset, check_tsdataset

logger = Logger(__name__)


class LGBM(MLBaseModel):
    """
    A typical method of GBDT, compared with xgboost, 
    has been accelerated and optimized in terms of memory occupation and computing speed, with about four differences:
        Exclusive Feature Bundling, leaf-wise, Histogram algorithm, Goss.

    Args:
        in_chunk_len(int): Sample length of training model
        skip_chunk_len(int): Length skipped between label and feature.
        sampling_stride(int, optional): For the entire data set, this parameter specifies the time step spanned between
            two adjacent samples when obtaining samples, The number of time points that differ between the starting time
            point t(i) of the i-th sample and the starting time point t(i+1) of the i+1-th sample 
            sampling_stride = t(i+1) - t(i) 
        params(dict): Parameters for training.
        num_boost_round(int): Number of boosting iterations.
        valid_names: Names of valid_data
        fobj: Customized objective function.
        feval: Customized evaluation function.
        init_model: Filename of LightGBM model or Booster instance used for continue training.
        feature_name: Feature names. If ‘auto’ and data is pandas DataFrame, data columns names are used.
        categorical_feature: Categorical features. If list of int, interpreted as indices. 
            If list of strings, interpreted as feature names (need to specifyfeature_name as well). 
            If ‘auto’ and data is pandas DataFrame, pandas categorical columns are used.
        early_stopping_rounds: Activates early stopping. The model will train until the validation score stops improving. 
            Requires at least one validation data and one metric. If there’s more than one, will check all of them. If 
            early stopping occurs, the model will add best_iteration field.
        evals_result: This dictionary used to store all evaluation results of all the items in valid_sets.
        verbose_eval: Requires at least one validation data. If True, the eval metric on the valid set is printed at each 
            boosting stage. If int, the eval metric on the valid set is printed at every verbose_eval boosting stage. The 
            last boosting stage or the boosting stage found by using early_stopping_rounds is also printed.
        keep_training_booster: Whether the returned Booster will be used to keep training. If False, the returned value 
            will be converted into _InnerPredictor before returning. You can still use _InnerPredictor as init_model for 
            future continue training.
        callbacks: List of callback functions that are applied at each iteration. 
            See Callbacks in Python API for more information.

    Return:
        None
    """
    def __init__(self, 
                 in_chunk_len: int=0, 
                 out_chunk_len: int=1,
                 skip_chunk_len: int=0, 
                 sampling_stride: int=1,
                 params: Optional[dict]=None, 
                 num_boost_round: int=100, 
                 valid_names: str='valid set',
                 fobj=None, 
                 feval=None, 
                 init_model=None, 
                 feature_name='auto', 
                 categorical_feature='auto',
                 early_stopping_rounds: int=None, 
                 evals_result: Optional[float]=None, 
                 verbose_eval: int=100, 
                 keep_training_booster: bool=False, 
                 callbacks: list=None
                ):
        super(LGBM, self).__init__(in_chunk_len=in_chunk_len, 
                                   out_chunk_len=1, 
                                   skip_chunk_len=skip_chunk_len)
        if params is None:   #当用户未给出时参数时，给予params一个默认参数，
            params = {"boosting": "gbdt",
                      "objective": "regression", 
                      "metric": "mse", 
                      "learning_rate": 0.01,        
                      "lambda_l1": 0.25,            
                      "lambda_l2": 0.25,            
                      "num_leaves": 31, 
                      "max_depth": -1, 
                      "bagging_freq": 1,   
                      "bagging_fraction": 0.9, 
                      "feature_fraction": 0.8,
                      "min_data_in_leaf": 3,
                      "verbose": -1,
                      "num_threads": 1,
                      "seed": 28,
                      }
        # 数据格式参数
        self.in_chunk_len = in_chunk_len
        self.skip_chunk_len = skip_chunk_len
        self.out_chunk_len = out_chunk_len
        if out_chunk_len != 1:
            raise_log(ValueError("value of out_chunk_len is not 1, Please check your out_chunk_len value !"))
        self.sampling_stride = sampling_stride
        # 初始化lgbm参数
        self.params = params
        self.num_boost_round = num_boost_round
        self.valid_names = valid_names
        self.fobj = fobj
        self.feval = feval
        self.init_model = init_model
        self.feature_name = feature_name
        self.categorical_feature = categorical_feature
        self.early_stopping_rounds = early_stopping_rounds
        self.evals_result = evals_result
        self.verbose_eval = verbose_eval
        self.keep_training_booster = keep_training_booster
        self.callbacks = callbacks

    def fit(self, 
            train_data: Union[TSDataset, List[TSDataset]], 
            valid_data: Optional[TSDataset]=None
           ):
        """
        Interface of model training

        Args:
            train_data(TSDataset|List[TSDataset]): Data to be trained.
            valid_data(TSDataset, optional):  List of data to be evaluated during training.

        Return: self
        """
        # 转换数据格式 
        if isinstance(train_data, list):
            self._check_multi_tsdataset(train_data)
        if isinstance(valid_data, list):
            self._check_multi_tsdataset(valid_data)
        train_data = self._ts_to_lgbm_dataset(train_data, is_train=True)
        if len(train_data.data) < 1:
            raise_log(ValueError("length of train_data less than 1, Please check your train_data!"))
        # 判断验证集数据是否为None, 若为None则无需验证
        if valid_data is not None:
            valid_data = self._ts_to_lgbm_dataset(valid_data, is_train=True)
            if len(valid_data.data) < 1:  
                raise_log(ValueError("length of valid_data less than 1, Please check your valid_data!"))
        # 模型训练
        self.model = lgbm.train(self.params, train_set=train_data, valid_sets=valid_data, 
                                num_boost_round=self.num_boost_round, valid_names=self.valid_names, 
                                feature_name=self.feature_name, categorical_feature=self.categorical_feature, 
                                verbose_eval=self.verbose_eval, 
                                early_stopping_rounds=self.early_stopping_rounds, 
                                evals_result=self.evals_result, fobj=self.fobj, feval=self.feval, 
                                init_model=self.init_model, 
                                keep_training_booster=self.keep_training_booster, callbacks=self.callbacks
                                )
        return self

    @to_tsdataset(scenario="forecasting")
    def predict(self, 
                test_data: TSDataset) -> TSDataset:
        """
        Make a prediction.

        Args:
            test_data(TSDataset): Data source for prediction.

        Returns:
            TSDataset: prediction TSDataset.
        """
        # 调用adapter转换数据格式
        test_data = self._ts_to_lgbm_dataset(test_data, is_train=False)
        # 判断数据是否为None
        if len(test_data.data) < 1:
            raise_log(ValueError("length of test_data less than 1, Please check your test_data!"))
        # 模型预测
        predict = self.model.predict(test_data, num_iteration=-1, raw_score=False, pred_leaf=False, 
                                  pred_contrib=False, data_has_header=False, is_reshape=True, 
                                  pred_parameter=None)
        return predict

    def _ts_to_lgbm_dataset(self, data_ts, is_train=True):
        """
        Data format required for conversion into LGBM model.

        Args:
            data(TSDataset|List[TSDataset]): TSDataset type of data.
            is_train(bool): return lgbm.Dataset if is_train=True, return sample_x if is_train=False

        Returns
            lgbm.Dataset| sample_x

        """
        # 计算数据窗口
        if is_train:  #训练时不需要y
            self.time_window = None
        else:    #预测时，选择需要预测的数据列表长度，并返回每一条的predict结果
            boundary = len(data_ts.get_target().data) - 1
            boundary = boundary + self.skip_chunk_len + self.out_chunk_len
            self.time_window = (boundary, boundary)
        # adapter 
        if isinstance(data_ts, list):
            for data_ts_cur in data_ts:
                check_tsdataset(data_ts_cur)
        else:
            check_tsdataset(data_ts)
        adapter = DataAdapter()
        param = {
            # in = 0 代表 TSDataset被lag transform处理过
            "in_chunk_len": self.in_chunk_len,
            "skip_chunk_len": self.skip_chunk_len,
            "out_chunk_len": self.out_chunk_len,
            "sampling_stride": self.sampling_stride,
            "time_window": self.time_window
            }
        if isinstance(data_ts, TSDataset):
            data_ts = [data_ts]
        # concatenate data in list
        ml_ds = None
        for data in data_ts:
            son_ml_ds = adapter.to_ml_dataset(data, **param)
            if ml_ds is None:
                ml_ds = son_ml_ds
            else:
                ml_ds.samples = ml_ds.samples + son_ml_ds.samples
        ml_dataloader = adapter.to_ml_dataloader(ml_ds, batch_size=len(ml_ds))
        # data from ml_dataloader
        data = next(ml_dataloader)
        # concatenate data in train or test
        sample_x_keys = data.keys() - {"future_target"}
        if self.in_chunk_len < 1:
            sample_x_keys -= {"past_target"}
        product_keys = product(["numeric", "categorical"], ["known_cov", "observed_cov", "static_cov"])
        full_ordered_x_key_list =["past_target"] + [f"{t[1]}_{t[0]}" for t in product_keys]
        actual_ordered_x_key_list = []
        for k in full_ordered_x_key_list:
            if k in sample_x_keys:
                actual_ordered_x_key_list.append(k)
        reshaped_x_ndarray_list = []
        for k in actual_ordered_x_key_list:
            ndarray = data[k]
            # 3-dim -> 2-dim
            reshaped_ndarray = ndarray.reshape(ndarray.shape[0], ndarray.shape[1] * ndarray.shape[2])
            reshaped_x_ndarray_list.append(reshaped_ndarray)
        # Note: if a_ndarray.dtype = np.int64, b_ndarray.dtype = np.float32, then
        # np.hstack(tup=(a_ndarray, b_ndarray)).dtype will ALWAYS BE np.float32
        sample_x = np.hstack(tup=reshaped_x_ndarray_list)
        if is_train:
            sample_y = data['future_target'].flatten() #ravel
            lgbm_dataset = lgbm.Dataset(sample_x, sample_y, free_raw_data=False)
            return lgbm_dataset
        else:
            return sample_x 
