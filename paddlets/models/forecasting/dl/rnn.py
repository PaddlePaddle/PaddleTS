#!/usr/bin/env python3
# -*- coding: UTF-8 -*-


from typing import List, Dict, Any, Callable, Optional, Tuple
import collections

import pandas as pd
import numpy as np
import paddle
from paddle import nn
from paddle.optimizer import Optimizer
import paddle.nn.functional as F

from paddlets.datasets import TSDataset
from paddlets.models.forecasting.dl.paddle_base_impl import PaddleBaseModelImpl
from paddlets.models.common.callbacks import Callback
from paddlets.logger import raise_if_not, Logger

logger = Logger(__name__)

class _RNNBlock(nn.Layer):
    """
    RNN model implemented by Paddle
    
    Args:
        in_chunk_len(int): The size of the loopback window, i.e., the number of time steps feed to the model.
        out_chunk_len(int): The size of the forecasting horizon, i.e., the number of time steps output by the model.
        fit_params(dict): The dimensions and dict sizes of variables.
        rnn_type(str): The type of the specific paddle RNN module ("SimpleRNN", "GRU" or "LSTM").
        hidden_dim(int): The number of features in the hidden state `h` of the RNN module.
        embedding_dim(int): The size of each embedding vector.
        num_layers_recurrent(int): The number of recurrent layers.
        out_fcn_config(Optional[List]): A list containing the dimensions of the hidden layers of the fully connected NN.
        dropout(float): The fraction of neurons that are dropped in all-but-last RNN layers.
        pooling: Whether to use average pooling to aggregate embeddings, if False, concat each embedding.
    """
    def __init__(
        self,
        in_chunk_len: int,
        out_chunk_len: int,
        fit_params: dict,
        rnn_type: str,
        hidden_dim: int,
        embedding_dim: int,
        num_layers_recurrent: int = 1,
        out_fcn_config: Optional[List] = None,
        dropout: float = 0.0,
        pooling: bool = False,
    ):
        super().__init__()
        self.in_chunk_len = in_chunk_len
        self.out_chunk_len = out_chunk_len
        out_fcn_config = [] if out_fcn_config is None else out_fcn_config
        self._rnn_type = rnn_type
        self._embedding_dim = embedding_dim
        self._target_dim = fit_params["target_dim"]
        self._known_num_dim = fit_params["known_num_dim"]
        self._known_cat_dim = fit_params["known_cat_dim"]
        self._observed_num_dim = fit_params["observed_num_dim"]
        self._observed_cat_dim = fit_params["observed_cat_dim"]
        self._static_num_dim = fit_params["static_num_dim"]
        self._static_cat_dim = fit_params["static_cat_dim"]
        self._pooling = pooling
        self._num_size = self._target_dim + self._known_num_dim + self._observed_num_dim + self._static_num_dim
        if fit_params["known_cat_size"]:
            self._known_embedding = []
            for col, col_size in fit_params["known_cat_size"].items():
                self._known_embedding.append(nn.Embedding(col_size, embedding_dim))
        if fit_params["observed_cat_size"]:
            self._observed_embedding = []
            for col, col_size in fit_params["observed_cat_size"].items():
                self._observed_embedding.append(nn.Embedding(col_size, embedding_dim))
        if fit_params["static_cat_size"]:
            self._static_embedding = []
            for col, col_size in fit_params["static_cat_size"].items():
                self._static_embedding.append(nn.Embedding(col_size, embedding_dim))       
        if fit_params["known_cat_size"] or fit_params["observed_cat_size"] or fit_params["static_cat_size"]:
            if self._pooling:
                self._cat_size = self._embedding_dim
            else:
                self._cat_size = self._embedding_dim * \
                (len(fit_params["observed_cat_size"]) + \
                 len(fit_params["known_cat_size"]) + \
                 len(fit_params["static_cat_size"]))
        else:
            self._cat_size = 0
        self._input_size = self._num_size + self._cat_size
        self._rnn = getattr(nn, self._rnn_type)(self._input_size, hidden_dim, num_layers_recurrent, dropout=dropout)
        # Defining projection layer
        # The RNN module is followed by a fully connected network(FCN),
        # which maps the last hidden layer to the output of desired length
        last = hidden_dim
        feats = []
        for feature in out_fcn_config + [out_chunk_len * self._target_dim]:
            feats.append(nn.Linear(last, feature))
            last = feature
        self.fc = nn.Sequential(*feats)

    def forward(
            self,
            data: Dict[str, paddle.Tensor]
            ) -> paddle.Tensor:
        """
        Forward network.

        Args:
            data(Dict[str, paddle.Tensor]): a dict specifies all kinds of input data

        Returns:
            predictions: output of RNN model, with shape [batch_size, out_chunk_len, target_dim]
        """
        # process numeric feature
        feature = [data["past_target"]]
        if self._known_num_dim > 0:
            past_known_num = data["known_cov_numeric"][:, :self.in_chunk_len]
            feature.append(past_known_num)
        if self._observed_num_dim > 0:
            observed_num = data["observed_cov_numeric"]
            feature.append(observed_num)
        if self._static_num_dim > 0:
            static_num = data["static_cov_numeric"].tile([1, self.in_chunk_len, 1])
            feature.append(static_num)
        feature = paddle.concat(feature, axis=-1)
        
        # process categorical feature
        cat_feature = []
        if self._known_cat_dim > 0:
            past_known_cat = data["known_cov_categorical"][:, :self.in_chunk_len]
            for i in range(self._known_cat_dim):
                cat_feature.append(self._known_embedding[i](past_known_cat[..., i]))
        if self._observed_cat_dim > 0:
            observed_cat = data["observed_cov_categorical"]
            for i in range(self._observed_cat_dim):
                cat_feature.append(self._observed_embedding[i](observed_cat[..., i]))
        if self._static_cat_dim > 0:
            static_cat = data["static_cov_categorical"].tile([1, self.in_chunk_len, 1])
            for i in range(self._static_cat_dim):
                cat_feature.append(self._static_embedding[i](static_cat[..., i]))
        if cat_feature:
            if self._pooling:
                cat_feature = paddle.stack(cat_feature, axis = -1).mean(axis= -1)
            else:
                cat_feature = paddle.concat(cat_feature, axis=-1)
            x = paddle.concat([feature, cat_feature], axis=-1)
        else:
            x = feature
        # input x is of size: [batch_size, in_chunk_len, target_dim + observed_dim + known_dim]
        # `out` is the output of the model, with shape: [batch_size, time_steps, hidden_dim]
        # `hidden` is the final state of the model, with shape: [num_layers, batch_size, hidden_dim]
        out, hidden = self._rnn(x)

        #apply the FC network only on the last output point (at the final time step)
        if self._rnn_type == "LSTM":
            hidden = hidden[0] # for LSTM, hidden[0] shape: [num_layers, batch_size, hidden_dim]
        predictions = hidden[-1, :, :]
        predictions = self.fc(predictions)
        predictions = predictions.reshape([predictions.shape[0], self.out_chunk_len, self._target_dim])
        return predictions


class RNNBlockRegressor(PaddleBaseModelImpl):
    """
    Implementation of RNN Block model.

    Args:
        in_chunk_len(int): The size of the loopback window, i.e., the number of time steps feed to the model.
        out_chunk_len(int): The size of the forecasting horizon, i.e., the number of time steps output by the model.
        rnn_type_or_module(str, Optional): The type of the specific paddle RNN module ("SimpleRNN", "GRU" or "LSTM").
        fcn_out_config(List[int], Optional): A list containing the dimensions of the hidden layers of the fully connected NN.
        hidden_size(int, Optional): The number of features in the hidden state `h` of the RNN module.
        embedding_size(int, Optional): The size of each embedding vector.
        num_layers_recurrent(int, Optional): The number of recurrent layers.
        dropout(float, Optional): The fraction of neurons that are dropped in all-but-last RNN layers.
        pooling(bool, Optional): Whether to use average pooling to aggregate embeddings, if False, concat each embedding.
        skip_chunk_len(int, Optional): Optional, the number of time steps between in_chunk and out_chunk for a single sample. The skip chunk is neither used as a feature (i.e. X) nor a label (i.e. Y) for a single sample. By default it will NOT skip any time steps.
        sampling_stride(int, optional): sampling intervals between two adjacent samples.
        loss_fn(Callable, Optional): loss function.
        optimizer_fn(Callable, Optional): optimizer algorithm.
        optimizer_params(Dict, Optional): optimizer parameters.
        eval_metrics(List[str], Optional): evaluation metrics of model.
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
        rnn_type_or_module: str = "SimpleRNN",
        fcn_out_config: List[int] = None,
        hidden_size: int = 128,
        embedding_size: int = 128,
        num_layers_recurrent: int = 1,
        dropout: float = 0.0,
        pooling: bool = True,
        skip_chunk_len: int = 0,
        sampling_stride: int = 1,
        loss_fn: Callable[..., paddle.Tensor] = F.mse_loss,
        optimizer_fn: Callable[..., Optimizer] = paddle.optimizer.Adam,
        optimizer_params: Dict[str, Any] = dict(learning_rate=1e-4),
        eval_metrics: List[str] = [],
        callbacks: List[Callback] = [],
        batch_size: int = 128,
        max_epochs: int = 10,
        verbose: int = 1,
        patience: int = 4,
        seed: int = 0
    ):
        self._rnn_type_or_module = rnn_type_or_module
        self._fcn_out_config = fcn_out_config
        self._hidden_size = hidden_size
        self._embedding_size = embedding_size
        self._num_layers_recurrent = num_layers_recurrent
        self._pooling = pooling
        self._dropout = dropout

        #check parameters validation
        raise_if_not(
                self._rnn_type_or_module in {"SimpleRNN", "LSTM", "GRU"},
                "A valid RNN type should be specified, currently SimpleRNN, LSTM, and GRU are supported."
                )

        super(RNNBlockRegressor, self).__init__(
            in_chunk_len=in_chunk_len,
            out_chunk_len=out_chunk_len,
            skip_chunk_len=skip_chunk_len,
            sampling_stride=sampling_stride,
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
            optimizer_params=optimizer_params,
            eval_metrics=eval_metrics,
            callbacks=callbacks,
            batch_size=batch_size,
            max_epochs=max_epochs,
            verbose=verbose,
            patience=patience,
            seed=seed,
        )

    def _check_tsdataset(
        self,
        tsdataset: TSDataset
    ):
        """
        Rewrite _check_tsdataset to fit the specific model.
        For RNN, all data variables are expected to be float32.
        """
        target_columns = tsdataset.get_target().dtypes.keys()
        for column, dtype in tsdataset.dtypes.items():
            if column in target_columns:
                raise_if_not(
                    np.issubdtype(dtype, np.floating),
                    f"rnn's target dtype only supports [float16, float32, float64]," \
                    f"but received {column}: {dtype}."
                )
            else:
                raise_if_not(
                    np.issubdtype(dtype, np.floating) or np.issubdtype(dtype, np.integer),
                    f"rnn's covariates' dtype only support float and integer," \
                    f"but received {column}: {dtype}."
                )
        super(RNNBlockRegressor, self)._check_tsdataset(tsdataset)        

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
        train_ts0.sort_columns()
        target_dim = train_ts0.get_target().data.shape[1]
        # stat categorical variables' dict size
        # known info
        known_cat_size = collections.OrderedDict()
        known_ts = train_ts0.get_known_cov()
        known_num_cols = []
        if known_ts:
            known_dtypes = dict(known_ts.dtypes)
            for col in known_ts.columns:
                if np.issubdtype(known_dtypes[col], np.integer):
                    known_cat_size[col] = len(df_all[col].unique())
                else:
                    known_num_cols.append(col)
        #observed info
        observed_cat_size = collections.OrderedDict()
        observed_ts = train_ts0.get_observed_cov()
        observed_num_cols = []
        if observed_ts:
            observed_dtypes = dict(observed_ts.dtypes)
            for col in observed_ts.columns:
                if np.issubdtype(observed_dtypes[col], np.integer):
                    observed_cat_size[col] = len(df_all[col].unique())
                else:
                    observed_num_cols.append(col)
        # static info
        static_cat_size = collections.OrderedDict()
        static_dic = train_ts0.get_static_cov()
        static_num_cols = []
        if static_dic:
            for col, val in static_dic.items():
                if np.issubdtype(type(val), np.integer) or isinstance(val, int):
                    static_cat_size[col] = len(df_all[col].unique())
                else:
                    static_num_cols.append(col)
                
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
        return _RNNBlock(
            self._in_chunk_len,
            self._out_chunk_len,
            self._fit_params,
            self._rnn_type_or_module,
            self._hidden_size,
            self._embedding_size,
            self._num_layers_recurrent,
            self._fcn_out_config,
            self._dropout,
            self._pooling
        )
