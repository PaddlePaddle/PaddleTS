# !/usr/bin/env python3
# -*- coding:utf-8 -*-

from typing import List, Optional, Union

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from paddle.io import DataLoader as PaddleDataLoader

from paddlets import TSDataset
from paddlets import Pipeline
from paddlets.logger import Logger, raise_if_not, raise_if, raise_log
from paddlets.models.forecasting.dl.paddle_base import PaddleBaseModel
from paddlets.xai.post_hoc import BaseExplainer
from paddlets.xai.post_hoc.data_wrapper import DatasetWrapper

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
        shap_method(str): Optionally, the shap method to apply. Now just support `kernel` method.
        task_type(str): Task type of the model. Now just support the regression task.
        seed(int): random seed.
        use_paddleloader(bool): Only effective when the model is of type PaddleBaseModel.
        kwargs: Optionally, additional keyword arguments passed to `shap_method`.
    """
    _ShapMethod = {
        'kernel': shap.KernelExplainer,
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
        if background_sample_number:
            raise_if(len(background_data.get_target()) < background_sample_number, \
                     'The length of background_data should be larger than background_sample_number.')
        raise_if(shap_method != 'kernel', 'Now just support kernel shap!')
        raise_if_not(isinstance(background_data.freq, str), 'Now just support timeindex data!')
        raise_if(len(background_data.get_target().columns) > 1, 'Now just support univariate output!')
        
        self._model = model
        self.use_paddleloader = use_paddleloader
        
        if issubclass(type(model), Pipeline):
            _model_obj = model._model
        elif issubclass(type(model), PaddleBaseModel):
            _model_obj = model
        else:
            raise_log(ValueError(f"The model type ({type(model)}) is not supported by the explainer."))

        # Judge whether it is probability prediction
        if hasattr(_model_obj, "_output_mode"):
            raise_if(_model_obj._output_mode == 'quantiles', 'Now just support point prediction but not probability prediction!')
        
        # Base parameter
        self._in_chunk_len = _model_obj._in_chunk_len
        self._out_chunk_len = _model_obj._out_chunk_len
        self._skip_chunk_len = _model_obj._skip_chunk_len
        self._sampling_stride = _model_obj._sampling_stride if _model_obj._sampling_stride > 0 else 1

        # Data wrapper
        self.wrapper = DatasetWrapper(in_chunk_len=self._in_chunk_len, out_chunk_len=self._out_chunk_len, 
                                     skip_chunk_len=self._skip_chunk_len, sampling_stride=self._sampling_stride, 
                                     freq=background_data.freq)
        self.df_background_data = self._ts_to_df(background_data)
        # Sampling background data
        if background_sample_number:
            self.df_background_data = self.df_background_data.sample(background_sample_number, random_state=seed)
        # Initializing explainer
        if use_paddleloader and not issubclass(type(self._model), Pipeline):
            self.explainer = self._ShapMethod[shap_method](model=self._wrapper_paddle_predict, 
                                                           data=self.df_background_data, **kwargs)
        else:
            self.explainer = self._ShapMethod[shap_method](model=self._wrapper_predict, 
                                                           data=self.df_background_data, **kwargs)
        self.shap_value = None
        
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
                           lead to lower variance estimates of the SHAP values. Default nsamples=100.
            sample_start_index(int): The sample start index of the test data. Default the latest sample.
            sample_num(int): The sample number of the test data.
            kwargs: Optionally, additional keyword arguments passed to `shap.explainer.shap_values`.
            
        Returns:
            np.ndarray object
        """
        raise_if(nsamples < 1, 'nsamples should be a positive integer.')
        raise_if(abs(sample_start_index) >= len(foreground_data.get_target()), 'abs(sample_start_index) should be less than len(foreground_data)!')
        
        df_foreground_data = self._ts_to_df(foreground_data)
        sample_start_index = sample_start_index if sample_start_index >= 0 else \
                              len(df_foreground_data) + sample_start_index
        
        self.df_foreground_data = df_foreground_data.iloc[sample_start_index: sample_start_index + sample_num, :]
        shap_value = self.explainer.shap_values(self.df_foreground_data, nsamples=nsamples, **kwargs)
        
        self.shap_value = np.asarray(shap_value)
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
            np.ndarray object
        """
        raise_if(out_chunk_index < 1, 'out_chunk_index should be a positive integer.')
        return self.shap_value[out_chunk_index - 1, sample_index, :]
    
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
        columns = self.df_foreground_data.columns
        unique_cols = list(set([col.rsplit(':', 1)[0] for col in columns]))
        
        _plot_method = {'OI': self._out_vs_in_plot,
                        'OV': self._out_vs_feature_plot,
                        'IV': self._in_vs_feature_plot,
                        'I': self._in_plot,
                        'V': self._feature_plot,
                       }
        
        shap_value_with_index = self.shap_value[:, sample_index, :]
        shap_value_np = np.zeros((self._out_chunk_len, self._in_chunk_len + self._out_chunk_len, len(unique_cols)))

        for col_index, col in enumerate(columns):
            name, col = col.rsplit(':', 1)
            lag_index = int(col.rsplit('_', 1)[1])
            if lag_index > 0 and lag_index <= self._skip_chunk_len:
                continue
            lag_index = lag_index + self._in_chunk_len - 1 if lag_index <= 0 else lag_index - self._skip_chunk_len + self._in_chunk_len - 1
            for out_index in range(self._out_chunk_len):
                value = shap_value_with_index[out_index, col_index]
                shap_value_np[out_index, lag_index, unique_cols.index(name)] = value

        out_cols = list(range(1, self._out_chunk_len + 1))
        in_cols = list(range(-self._in_chunk_len + 1, self._out_chunk_len + 1))
        
        for key in method:
            _plot_method[key](shap_value_np, out_cols=out_cols, in_cols=in_cols, 
                              unique_cols=unique_cols, **kwargs)
        
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
        sample_index = [sample_index] if isinstance(sample_index, int) else sample_index
        out_chunk_indice = [out_chunk_indice] if isinstance(out_chunk_indice, int) else out_chunk_indice
        
        raise_if(min(out_chunk_indice) < 1, 'out_chunk_indice must be more than 0!')
        
        out_chunk_indice = [v - 1 for v in out_chunk_indice]
        df_foreground_data = self.df_foreground_data.iloc[sample_index, :]
        
        shap_value = self.shap_value[out_chunk_indice, sample_index, :]
        kwargs['show'] = False
        kwargs['matplotlib'] = True
        # Plot the SHAP values
        for i, index in enumerate(out_chunk_indice):
            shap.force_plot(self.explainer.expected_value[index], shap_value[i], df_foreground_data, **kwargs)
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
        sample_index = [sample_index] if isinstance(sample_index, int) else sample_index
        out_chunk_indice = [out_chunk_indice] if isinstance(out_chunk_indice, int) else out_chunk_indice
        
        raise_if(min(out_chunk_indice) < 1, 'out_chunk_indice must be more than 0!')
        
        out_chunk_indice = [v - 1 for v in out_chunk_indice]
        df_foreground_data = self.df_foreground_data.iloc[sample_index, :]
        shap_value = self.shap_value[:, sample_index, :]
        
        kwargs['show'] = False
        kwargs['plot_type'] = 'bar' if not 'plot_type' in kwargs else kwargs['plot_type']
        
        for index in out_chunk_indice:
            plt.figure()
            shap.summary_plot(self.shap_value[index], df_foreground_data, **kwargs)
            plt.title('output_%d' % (index + 1))
            plt.show
            
