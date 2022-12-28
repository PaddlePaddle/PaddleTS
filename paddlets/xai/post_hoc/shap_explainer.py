# !/usr/bin/env python3
# -*- coding:utf-8 -*-

from typing import List, Optional, Union, Dict

import shap
import random
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import paddle
from paddle.io import DataLoader as PaddleDataLoader

from paddlets import TSDataset
from paddlets import Pipeline
from paddlets.logger import Logger, raise_if_not, raise_if, raise_log
from paddlets.models.forecasting import DeepARModel
from paddlets.models.forecasting.dl.paddle_base import PaddleBaseModel
from paddlets.xai.post_hoc import BaseExplainer
from paddlets.xai.post_hoc.data_wrapper import DatasetWrapper
from paddlets.xai.post_hoc.deep_paddle import PaddleDeep

logger = Logger(__name__)


class ShapExplainer(BaseExplainer):
    """
    Shap explainer. This class only (currently) supports regression model of forecasting task.
    It uses shap value to provide the contribution value of model input to model output.
    For shap, please see `https://github.com/slundberg/shap`.

    Args:
        model(PaddleBaseModel|Pipeline): A model object that supports `predict` function.
        background_data(TSDataset):  A TSDataset for training the shap explainer
        background_sample_number(int): number of instances sampled from the background_data
        shap_method(str): The shap method to apply. Optionally, {'kernel', 'deep'}. 
        task_type(str): Task type of the model. Only support the regression task.
        seed(int): random seed.
        use_paddleloader(bool): Only effective when the model is of type PaddleBaseModel.
        kwargs: Optionally, additional keyword arguments passed to `shap_method`.
    """
    _ShapMethod = {
        'kernel': shap.KernelExplainer,
        'deep': PaddleDeep,
    }
    
    def __init__(
                self,
                model: Optional[Union[PaddleBaseModel, Pipeline]],
                background_data: TSDataset,
                background_sample_number: Optional[int] = None,
                shap_method: str = 'kernel',
                task_type: str = 'regression',
                seed: int = 123,
                use_paddleloader: bool = False,
                **kwargs,
                ) -> None:
        self.background_sample_number = background_sample_number
        self.shap_method = shap_method
        
        raise_if(shap_method not in ['kernel', 'deep'], 'Only support kernel shap and deep shap!')
        raise_if_not(isinstance(background_data.freq, str), 'Only support timeindex data!')
        raise_if(len(background_data.get_target().columns) > 1, 'Only support univariate output!')
        
        self._model = model
        self.use_paddleloader = use_paddleloader
        
        if issubclass(type(model), Pipeline) and shap_method != 'deep':
            _model_obj = model._model
        elif issubclass(type(model), PaddleBaseModel):
            _model_obj = model
        else:
            raise_log(ValueError(f"The model type ({type(model)}) is not supported by %s explainer." % shap_method))
        
        raise_if(shap_method == 'deep' and issubclass(type(model), DeepARModel), "DeepAR is not supported by deep explainer.")
        # Judge whether it is probability prediction
        if hasattr(_model_obj, "_output_mode"):
            raise_if(_model_obj._output_mode == 'quantiles', 'Only support point prediction but not probability prediction!')
        
        # Base parameter
        self._in_chunk_len = _model_obj._in_chunk_len
        self._out_chunk_len = _model_obj._out_chunk_len
        self._skip_chunk_len = _model_obj._skip_chunk_len
        self._sampling_stride = _model_obj._sampling_stride if _model_obj._sampling_stride > 0 else 1

        # Data wrapper
        self.wrapper = DatasetWrapper(in_chunk_len=self._in_chunk_len, out_chunk_len=self._out_chunk_len, 
                                     skip_chunk_len=self._skip_chunk_len, sampling_stride=self._sampling_stride, 
                                     freq=background_data.freq)
        if shap_method == 'kernel':
            self.new_background_data = self._ts_to_df(background_data)
            sample_len = len(self.new_background_data)
        elif shap_method == 'deep':
            self.new_background_data = self._ts_to_tensor(background_data)
            sample_len = self.new_background_data['past_target'].shape[0]

        # Sampling background data
        if background_sample_number:
            raise_if(background_sample_number <= 0, \
                     'background_sample_number should be a positive integer.')
            background_sample_number = sample_len if sample_len < background_sample_number else background_sample_number
            if shap_method == 'kernel':
                self.new_background_data = self.new_background_data.sample(background_sample_number, random_state=seed)
            elif shap_method == 'deep':
                random.seed(seed)
                choice = random.sample(range(self.new_background_data['past_target'].shape[0]), background_sample_number)
                for k in self.new_background_data:
                    self.new_background_data[k] = self.new_background_data[k][choice]
                
        # Initializing explainer
        if shap_method == 'kernel':
            kwargs['keep_index'] = True
            if use_paddleloader and not issubclass(type(self._model), Pipeline):
                self.explainer = self._ShapMethod[shap_method](model=self._wrapper_paddle_predict, 
                                                               data=self.new_background_data, **kwargs)
            else:
                self.explainer = self._ShapMethod[shap_method](model=self._wrapper_predict, 
                                                               data=self.new_background_data, **kwargs)
        elif shap_method == 'deep':
            self.explainer = self._ShapMethod[shap_method](model=self._model, 
                                                           data=self.new_background_data)
        self.shap_value = None
        self.key_feature = background_data.columns
        self.deep_used_cols = self._get_feature_name(background_data)
        
    def _wrapper_paddle_predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        The prediction method based on Paddle loader.

        Args:
            df(pd.DataFrame): The data in pd.DataFrame format.

        Returns:
            np_res(np.ndarray): Prediction results.
        """
        paddle_ds = self.wrapper.dataframe_to_paddledsfromdf(df)

        data_loader = PaddleDataLoader(dataset=paddle_ds, batch_size=len(paddle_ds))
        df_res = self._model._predict(data_loader)
        np_res = np.squeeze(df_res, axis=2)
        return np_res  
    
    def _wrapper_predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        The commonly prediction method.

        Args:
            df(pd.DataFrame): The data in pd.DataFrame format.

        Returns:
            np_res(np.ndarray): Prediction results.
        """
        tss = self.wrapper.dataframe_to_ts(df)
        res = []
        for ts in tss:
            df_res = self._model.predict(ts)
            res.append(df_res.to_dataframe().values.transpose())
        np_res = np.concatenate(res, axis=0)

        return np_res
    
    def _ts_to_df(self, ts: TSDataset) -> pd.DataFrame:
        """
        Convert TSDataset to pd.dataframe.

        Args:
            ts(TSDataset): original data.

        Returns:
            dataframe object
        """
        return self.wrapper.dataset_to_dataframe(ts)
    
    def _ts_to_tensor(self, ts: TSDataset, is_explain: bool=False) -> Dict[str, paddle.Tensor]:
        """
        Convert TSDataset to Dict[str, paddle.Tensor].

        Args:
            ts(TSDataset): original data.
            is_explain(bool): whether to be used in function `self.explain`.

        Returns:
            Dict[str, paddle.Tensor]
        """
        _model = self._model
        
        _init_dataloaders = _model._init_fit_dataloaders if not is_explain else _model._init_predict_dataloader
        _samples = _init_dataloaders(ts)[0] if not is_explain else _init_dataloaders(ts)
        tensor_sample = {}
        for _sample in _samples:
            _sample, _ = _model._prepare_X_y(_sample)
            for k in _sample.keys():
                if k not in tensor_sample:
                    tensor_sample[k] = _sample[k]
                else:
                    tensor_sample[k] = paddle.concat([tensor_sample[k], _sample[k]], axis=0)
        return tensor_sample
    
    def explain(
                self, 
                foreground_data: TSDataset,
                nsamples: int = 100,
                sample_start_index: int = -1,
                sample_num: int = 1,
                **kwargs,
               ) -> np.ndarray:
        """
        Calculate the explanatory value of the test sample.
        
        Args:
            foreground_data(TSDataset): test data.
            nsamples(int): Number of times to re-evaluate the model when explaining each prediction. More samples
                           lead to lower variance estimates of the SHAP values. Only used in `shap_method=kernel`.
                           Default nsamples=100.
            sample_start_index(int): The sample start index of the test data. Default the latest sample.
            sample_num(int): The sample number of the test data.
            kwargs: Optionally, additional keyword arguments passed to `shap.explainer.shap_values`.
            
        Returns:
            np.ndarray object(out_chunk_len, samples, in_chunk_len + out_chunk_len(known_cov input), feature dims)
        """
        raise_if(nsamples < 1, 'nsamples should be a positive integer.')
        if self.shap_method == 'kernel':
            new_foreground_data = self._ts_to_df(foreground_data)
            sample_len = len(new_foreground_data)
        elif self.shap_method == 'deep':
            new_foreground_data = self._ts_to_tensor(foreground_data, is_explain=True)
            sample_len = new_foreground_data['past_target'].shape[0]
            
        raise_if((sample_start_index >= sample_len) or (sample_start_index < -sample_len), 
                 'sample_start_index should be less than the sample number of foreground_data!')
        sample_start_index = sample_start_index if sample_start_index >= 0 else \
                              sample_len + sample_start_index
        
        unique_cols = []
        
        if self.shap_method == 'kernel':
            new_foreground_data = new_foreground_data.iloc[sample_start_index: sample_start_index + sample_num, :]
            shap_value = self.explainer.shap_values(new_foreground_data, nsamples=nsamples, **kwargs)
            shap_value = np.asarray(shap_value)
            
            columns = new_foreground_data.columns
            unique_cols = list(set([col.rsplit(':', 1)[0] for col in columns]))
            
            #array shape(out_chunk_len, samples, in_chunk_len + out_chunk_len(only known exists), feature dims)
            shap_value_np = np.zeros((self._out_chunk_len, sample_len - sample_start_index, self._in_chunk_len + self._out_chunk_len, len(unique_cols)))
            for sample_index in range(sample_len - sample_start_index):
                shap_value_with_index = shap_value[:, sample_index, :]
                for col_index, col in enumerate(columns):
                    name, col = col.rsplit(':', 1)
                    lag_index = int(col.rsplit('_', 1)[1])
                    if lag_index > 0 and lag_index <= self._skip_chunk_len:
                        continue
                    lag_index = lag_index + self._in_chunk_len - 1 if lag_index <= 0 else lag_index - self._skip_chunk_len + self._in_chunk_len - 1
                    for out_index in range(self._out_chunk_len):
                        value = shap_value_with_index[out_index, col_index]
                        shap_value_np[out_index, sample_index, lag_index, unique_cols.index(name)] = value
                        
        elif self.shap_method == 'deep':
            for k in new_foreground_data.keys():
                new_foreground_data[k] = new_foreground_data[k][sample_start_index: sample_start_index + sample_num, :]

            shap_value = self.explainer.shap_values(new_foreground_data)
            self.shap_valuesss = shap_value
            for k, v in self.deep_used_cols.items():
                unique_cols.extend(v)
            shap_value_np = np.zeros((self._out_chunk_len, sample_len - sample_start_index, self._in_chunk_len + self._out_chunk_len, len(unique_cols)))
            for k, v in self.deep_used_cols.items():
                v_index = [unique_cols.index(v1) for v1 in v]
                shap_value_k = np.stack([tensor_dict[k] for tensor_dict in shap_value])
                shap_value_np[:, :, :shap_value_k.shape[2], v_index] += shap_value_k
            
        self.unique_cols = unique_cols
        self.out_cols = list(range(1, self._out_chunk_len + 1))
        self.in_cols = list(range(-self._in_chunk_len + 1, self._out_chunk_len + 1))
        self.new_foreground_data = new_foreground_data
        self.shap_value = shap_value_np
        return self.shap_value
    
    def get_explanation(
                self,
                out_chunk_index: int = 1,
                sample_index: int = 0,
                ) -> np.ndarray:
        """
        Get the explanatory output of a certain time point in the prediction length.
        
        Args:
            out_chunk_index(int): The certain time point in the prediction length.
            sample_index(int): The sample index of the explanatory value. Default the first sample.
            
        Returns:
            np.ndarray object(in_chunk_len + out_chunk_len(known_cov input), feature dims)
        """
        raise_if(out_chunk_index < 1, 'out_chunk_index should be a positive integer.')
        return self.shap_value[out_chunk_index - 1, sample_index]
    
    def plot(
            self, 
            method: Optional[Union[str, List[str]]] = None,
            sample_index: int = 0,
            **kwargs,
    ) -> None:
        """
        Display the shap value of different dimensions. Such as 'OI'(output time dimension vs input time dimension), 'OV'(output time dimension vs variable dimension), 'IV'(input time dimension vs variable dimension), 'I'(input time dimension), and 'V'(variable dimension).
        
        Args:
            method(str|List(str)): display method. Optional, {'OI', 'OV', 'IV', 'I', 'V'}.
            sample_index(int): The sample index of the explanatory value. Default the first sample.
            kwargs: other parameters.
            
        Returns:
            None
        """
        method = [method] if isinstance(method, str) else method
        
        _plot_method = {'OI': self._out_vs_in_plot,
                        'OV': self._out_vs_feature_plot,
                        'IV': self._in_vs_feature_plot,
                        'I': self._in_plot,
                        'V': self._feature_plot,
                       }
        
        shap_value_np = self.shap_value[:, sample_index, :, :]
        for key in method:
            _plot_method[key](shap_value_np, out_cols=self.out_cols, in_cols=self.in_cols, 
                              unique_cols=self.unique_cols, **kwargs)
        
    def force_plot(
                self,
                out_chunk_indice: Optional[Union[int, List[int]]] = 1,
                sample_index: int = 0,
                **kwargs,
                ) -> None:
        """
        Visualize the given SHAP values with an additive force layout.
        
        Args:
            out_chunk_indice(int): The certain time point in the prediction length.
            sample_index(int): The sample index of the explanatory value. Default the first sample.
            kwargs: Optionally, additional keyword arguments passed to `shap.force_plot`.
            
        Returns:
            None
        """
        out_chunk_indice = [out_chunk_indice] if isinstance(out_chunk_indice, int) else out_chunk_indice
        
        raise_if(min(out_chunk_indice) < 1, 'out_chunk_indice must be more than 0!')
        
        out_chunk_indice = [v - 1 for v in out_chunk_indice]
        feature_names = []
        for i in range(self.shap_value.shape[2]):
            for j in range(self.shap_value.shape[3]):
                feature_names.append('%s:%s_lag_%d' % (self.unique_cols[j], 
                                                       self.key_feature[self.unique_cols[j]].split('_')[0], 
                                                       i - self._in_chunk_len + 1))
        if self.shap_method == 'kernel':
            _sample = self.new_foreground_data.iloc[sample_index: sample_index + 1, :]
        elif self.shap_method == 'deep':
            values = []
            cols = []
            for k, tensor in self.new_foreground_data.items():
                tensor = tensor.numpy()
                for time_step in range(tensor.shape[1]):
                    for fea_index in range(tensor.shape[2]):
                        cols.append('%s:%s_lag_%d' % (self.deep_used_cols[k][fea_index], 
                                                      self.key_feature[self.deep_used_cols[k][fea_index]].split('_')[0], 
                                                      time_step - self._in_chunk_len + 1))
                        values.append(tensor[sample_index, time_step, fea_index])
            _sample = pd.DataFrame([values], columns=cols)
            
        kwargs['show'] = False
        kwargs['matplotlib'] = True
        # Plot the SHAP values
        for i, index in enumerate(out_chunk_indice):
            sv = self.shap_value[index, sample_index].reshape(1, -1)
            sv = sv[:, [feature_names.index(k) for k in _sample.columns]]
            shap.force_plot(self.explainer.expected_value[index], sv, _sample, **kwargs)
            plt.title('output_%d' % (index + 1))
            plt.show
        
    def summary_plot(
                self,
                out_chunk_indice: Optional[Union[int, List[int]]] = 1,
                sample_index: int = 0,
                **kwargs,
                ) -> None:
        """
        Create a SHAP feature importance based on previously interpreted samples.
        
        Args:
            out_chunk_indice(int): The certain time point in the prediction length.
            sample_index(int): The sample index of the explanatory value. Default the first sample.
            kwargs: Optionally, additional keyword arguments passed to `shap.summary_plot`.
            
        Returns:
            None
        """
        out_chunk_indice = [out_chunk_indice] if isinstance(out_chunk_indice, int) else out_chunk_indice
        
        raise_if(min(out_chunk_indice) < 1, 'out_chunk_indice must be more than 0!')
        
        out_chunk_indice = [v - 1 for v in out_chunk_indice]
        feature_names = []
        for i in range(self.shap_value.shape[2]):
            for j in range(self.shap_value.shape[3]):
                feature_names.append('%s:%s_lag_%d' % (self.unique_cols[j], 
                                                       self.key_feature[self.unique_cols[j]].split('_')[0], 
                                                       i - self._in_chunk_len + 1))
        if self.shap_method == 'kernel':
            _sample = self.new_foreground_data.iloc[sample_index: sample_index + 1, :]
        elif self.shap_method == 'deep':
            values = []
            cols = []
            for k, tensor in self.new_foreground_data.items():
                tensor = tensor.numpy()
                for time_step in range(tensor.shape[1]):
                    for fea_index in range(tensor.shape[2]):
                        cols.append('%s:%s_lag_%d' % (self.deep_used_cols[k][fea_index], 
                                                      self.key_feature[self.deep_used_cols[k][fea_index]].split('_')[0], 
                                                      time_step - self._in_chunk_len + 1))
                        values.append(tensor[sample_index, time_step, fea_index])
            _sample = pd.DataFrame([values], columns=cols)
        
        kwargs['show'] = False
        kwargs['plot_type'] = 'bar' if not 'plot_type' in kwargs else kwargs['plot_type']
        
        for index in out_chunk_indice:
            sv = self.shap_value[index, sample_index].reshape(1, -1)
            sv = sv[:, [feature_names.index(k) for k in _sample.columns]]
            
            plt.figure()
            shap.summary_plot(sv, _sample, **kwargs)
            plt.title('output_%d' % (index + 1))
            plt.show
            
    def _get_feature_name(self, ts: TSDataset) -> Dict[str, List[str]]:
        """
        Calculate feature names of paddle tensor.
        
        Args:
            ts(TSDataset): original data.
            
        Returns:
            Dict[str, List[str]]
        """
        past_target = ts.get_target()
        observed = ts.get_observed_cov()
        known = ts.get_known_cov()
        static = ts.get_static_cov()
        
        tensor_feature_name = defaultdict(list)
        if past_target:
            tensor_feature_name['past_target'].extend(past_target.columns)
        if observed:
            observed_df = observed.to_dataframe()
            observed_df_cat = observed_df.select_dtypes(np.integer)
            observed_df_num = observed_df.select_dtypes(np.floating)
            if not observed_df_cat.empty:
                tensor_feature_name['observed_cov_categorical'].extend(observed_df_cat.columns)
            if not observed_df_num.empty:
                tensor_feature_name['observed_cov_numeric'].extend(observed_df_num.columns)
        if known:
            known_df = known.to_dataframe()
            known_df_cat = known_df.select_dtypes(np.integer)
            known_df_num = known_df.select_dtypes(np.floating)
            if not known_df_cat.empty:
                tensor_feature_name['known_cov_categorical'].extend(known_df_cat.columns)
            if not known_df_num.empty:
                tensor_feature_name['known_cov_numeric'].extend(known_df_num.columns)
        if static:
            for k, v in static.items():
                if np.issubdtype(type(v), np.integer):
                    tensor_feature_name['static_cov_categorical'].append(k)
                if np.issubdtype(type(v), np.floating):
                    tensor_feature_name['static_cov_numeric'].append(k)
        return tensor_feature_name
