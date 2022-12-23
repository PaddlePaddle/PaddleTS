#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
This implementation is based on the article `Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting <https://arxiv.org/abs/1912.09363>`_.
"""

from typing import List, Dict, Any, Callable, Optional, Tuple
import copy
import math

import numpy as np
import pandas as pd
import paddle
from paddle import nn
import paddle.nn.functional as F
from paddle.optimizer import Optimizer

from paddlets.metrics.base import Metric
from paddlets.metrics import QuantileLoss
from paddlets.datasets import TSDataset
from paddlets.models.forecasting.dl.paddle_base_impl import PaddleBaseModelImpl
from paddlets.models.forecasting.dl.distributions import Likelihood, QuantileRegression
from paddlets.models.common.callbacks import Callback
from paddlets.logger import raise_if, raise_if_not, raise_log, Logger
from paddlets.models.forecasting.dl._tft import TemporalFusionTransformer

logger = Logger(__name__)


class TFTModel(PaddleBaseModelImpl):
    """
    Implementation of TFT model.

    Args:
        in_chunk_len(int): The size of the loopback window, i.e., the number of time steps feed to the model.
        out_chunk_len(int): The size of the forecasting horizon, i.e., the number of time steps output by the model.
        hidden_size(int, Optional): The number of features in the hidden state of the TFT module.
        lstm_layers_num(int, Optional): The number of LSTM layers.
        attention_heads_num(int, Optional): The number of heads of self-attention module.
        output_quantiles(List[float], Optional): The output quantiles of the model.
        dropout(float, Optional): The fraction of neurons that are dropped in all-but-last RNN layers.
        skip_chunk_len(int, Optional): Optional, the number of time steps between in_chunk and out_chunk for a single sample. The skip chunk is neither used as a feature (i.e. X) nor a label (i.e. Y) for a single sample. By default it will NOT skip any time steps.
        sampling_stride(int, optional): sampling intervals between two adjacent samples.
        loss_fn(Callable, Optional): loss function.
        optimizer_fn(Callable, Optional): optimizer algorithm.
        optimizer_params(Dict, Optional): optimizer parameters.
        callbacks(List[Callback], Optional): customized callback functions.
        batch_size(int, Optional): number of samples per batch.
        max_epochs(int, Optional): max epochs during training.
        verbose(int, Optional): verbosity mode.
        patience(int, Optional): number of epochs with no improvement after which learning rate wil be reduced.
        seed(int, Optional): global random seed.
    """

    def __init__(
        self,
        in_chunk_len: int,
        out_chunk_len: int,
        hidden_size: int = 64,
        lstm_layers_num: int = 1,
        attention_heads_num: int = 1,
        output_quantiles: List[float] = [0.1, 0.5, 0.9],
        dropout: float = 0.0,
        skip_chunk_len: int = 0,
        sampling_stride: int = 1,
        loss_fn: Callable[..., paddle.Tensor] = QuantileRegression().loss,
        optimizer_fn: Callable[..., Optimizer] = paddle.optimizer.Adam,
        optimizer_params: Dict[str, Any] = dict(learning_rate=1e-4),
        callbacks: List[Callback] = [],
        batch_size: int = 128,
        max_epochs: int = 10,
        verbose: int = 1,
        patience: int = 4,
        seed: int = 0
    ):
        self._hidden_size = hidden_size
        self._lstm_layers_num = lstm_layers_num
        self._attention_heads_num = attention_heads_num
        self._output_quantiles = sorted(output_quantiles)
        self._dropout = dropout
        self._output_mode = "quantiles"
        self._q_points = np.array(self._output_quantiles) * 100
        super(TFTModel, self).__init__(
            in_chunk_len=in_chunk_len,
            out_chunk_len=out_chunk_len,
            skip_chunk_len=skip_chunk_len,
            sampling_stride=sampling_stride,
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
            optimizer_params=optimizer_params,
            eval_metrics=[QuantileLoss(self._output_quantiles)],
            callbacks=callbacks,
            batch_size=batch_size,
            max_epochs=max_epochs,
            verbose=verbose,
            patience=patience,
            seed=seed,
        )
        
    def _check_params(self):
        """
        Parameter validity verification
        
        Check logic:

            batch_size: batch_size must be > 0.
            max_epochs: max_epochs must be > 0.
            verbose: verbose must be > 0.
            patience: patience must be >= 0.
            output_quantiles: each quantile should on [0, 1].
        """
        raise_if(self._batch_size <= 0, f"batch_size must be > 0, got {self._batch_size}.")
        raise_if(self._max_epochs <= 0, f"max_epochs must be > 0, got {self._max_epochs}.")
        raise_if(self._verbose <= 0, f"verbose must be > 0, got {self._verbose}.")
        raise_if(self._patience < 0, f"patience must be >= 0, got {self._patience}.")
        raise_if_not((np.array(self._output_quantiles) >= 0).all() and \
                 (np.array(self._output_quantiles) <= 1).all(),
                f"each quantile should on [0, 1], got {self._output_quantiles}.")
        if sorted(self._output_quantiles) != self._output_quantiles:
            logger.warning(f"output_quantiles should be sorted, got {self._output_quantiles}.")
            self._output_quantiles = sorted(self._output_quantiles)
            
    def _check_tsdataset(
        self,
        tsdataset: TSDataset
    ):
        """
        Rewrite _check_tsdataset to fit the specific model.
        """
        target_columns = tsdataset.get_target().dtypes.keys()
        for column, dtype in tsdataset.dtypes.items():
            if column in target_columns:
                raise_if_not(
                    np.issubdtype(dtype, np.floating),
                    f"TFT's target dtype only supports [float16, float32, float64]," \
                    f"but received {column}: {dtype}."
                )
            else:
                raise_if_not(
                    np.issubdtype(dtype, np.floating) or np.issubdtype(dtype, np.integer),
                    f"TFT's covariates' dtype only support float and integer," \
                    f"but received {column}: {dtype}."
                )
        raise_if_not(tsdataset.get_known_cov(), f"Known covariates are necessary to build TFT model.")
        super(TFTModel, self)._check_tsdataset(tsdataset)        

    def _update_fit_params(
        self,
        train_tsdataset: List[TSDataset],
        valid_tsdataset: Optional[List[TSDataset]] = None
    ) -> Dict[str, Any]:
        """
        Infer parameters by TSdataset automatically.

        Args:
            train_tsdataset(List[TSDataset]): list of train dataset
            valid_tsdataset(List[TSDataset], optional): list of validation dataset

        Returns:
            Dict[str, Any]: model parameters
        """
        df_list = []
        static_info = []
        # for meta info of all tsdatasets
        all_ts = train_tsdataset + valid_tsdataset if valid_tsdataset else train_tsdataset
        for ts in all_ts:
            static = ts.get_static_cov()
            df = ts.to_dataframe()
            if static:
                for col, val in static.items():
                    df[col] = val
            df_list.append(df)
        df_all = pd.concat(df_list)
        
        train_ts0 = train_tsdataset[0]
        target_dim = train_ts0.get_target().data.shape[1]
        self.target_cols = list(train_ts0.get_target().data.columns)
        # stat categorical variables' dict size
        # known info
        known_cat_size = []
        known_ts = train_ts0.get_known_cov()
        known_num_cols = []
        known_cat_cols = []
        if known_ts:
            known_dtypes = dict(known_ts.dtypes)
            for col in known_ts.columns:
                if np.issubdtype(known_dtypes[col], np.integer):
                    known_cat_size.append(len(df_all[col].unique()))
                    known_cat_cols.append(col)
                else:
                    known_num_cols.append(col)
        self.known_num_cols = known_num_cols
        self.known_cat_cols = known_cat_cols
        #observed info
        observed_cat_size = []
        observed_ts = train_ts0.get_observed_cov()
        observed_num_cols = []
        observed_cat_cols = []
        if observed_ts:
            observed_dtypes = dict(observed_ts.dtypes)
            for col in observed_ts.columns:
                if np.issubdtype(observed_dtypes[col], np.integer):
                    observed_cat_size.append(len(df_all[col].unique()))
                    observed_cat_cols.append(col)
                else:
                    observed_num_cols.append(col)
        self.observed_num_cols = observed_num_cols
        self.observed_cat_cols = observed_cat_cols
        # static info
        static_cat_size = []
        static_dic = train_ts0.get_static_cov()
        static_num_cols = []
        static_cat_cols = []
        if static_dic:
            for col, val in static_dic.items():
                if np.issubdtype(type(val), np.integer) or isinstance(val, int):
                    static_cat_size.append(len(df_all[col].unique()))
                    static_cat_cols.append(col)
                else:
                    static_num_cols.append(col)
        self.static_num_cols = static_num_cols
        self.static_cat_cols = static_cat_cols                
        fit_params = {
            "target_dim": target_dim,
            "known_num_dim": len(known_num_cols),
            "known_cat_dim": len(known_cat_size),
            "observed_num_dim": len(observed_num_cols),
            "observed_cat_dim": len(observed_cat_size),
            "static_num_dim": len(static_num_cols),
            "static_cat_dim": len(static_cat_size),
            "known_cat_size": known_cat_size,
            "observed_cat_size": observed_cat_size,
            "static_cat_size": static_cat_size,
            }
        return fit_params
                
    def _init_network(self) -> paddle.nn.Layer:
        """
        Init network.

        Returns:
            paddle.nn.Layer
        """
        return TemporalFusionTransformer(
            self._in_chunk_len,
            self._out_chunk_len,
            self._fit_params,
            self._hidden_size,
            self._lstm_layers_num,
            self._attention_heads_num,
            self._output_quantiles,
            self._dropout,
        )
    
    def predict_interpretable(
        self, 
        tsdataset: TSDataset
    ) -> Dict[str, np.ndarray]:
        """
        For interpretable use.
        
        Args:
            tsdataset(TSDataset): The TSDataset to be predict for interpretable results.
            
        Returns:
            results(Dict[str, np.ndarray]): The interpretable results.
        """
        self._network._interpretable_output = True
        self._network.eval()
        results = {}
        dataloader = self._init_predict_dataloader(tsdataset)
        for batch_nb, data in enumerate(dataloader):
            X, _ = self._prepare_X_y(data)
            output = self._network(X)
            for key in output:
                weights = output[key].numpy() if output[key] is not None else np.empty([0, 0])
                results.setdefault(key, [])
                results[key].append(weights)
        for key in results:
            results[key] = np.vstack(results[key])
        return results        
        
    def _predict(
        self, 
        dataloader: paddle.io.DataLoader,
    ) -> np.ndarray:
        """
        Predict function core logic.

        Args:
            dataloader(paddle.io.DataLoader): Data to be predicted.

        Returns:
            np.ndarray.
        """
        self._network._interpretable_output = False
        return super(TFTModel, self)._predict(dataloader)
        
