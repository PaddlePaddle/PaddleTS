#!/usr/bin/env python3
# -*- coding: UTF-8 -*-


from typing import List, Dict, Any, Callable, Optional, Tuple

import numpy as np
import paddle
from paddle import nn
from paddle.optimizer import Optimizer
import paddle.nn.functional as F

from paddlets.datasets import TSDataset
from paddlets.models.dl.paddlepaddle.paddle_base_impl import PaddleBaseModelImpl
from paddlets.models.dl.paddlepaddle.callbacks import Callback
from paddlets.logger import raise_if_not, raise_log, Logger

logger = Logger(__name__)

class _RNNBlock(nn.Layer):
    """
    RNN model implemented by Paddle
    
    Args:
        in_chunk_len(int): The size of the loopback window, i.e., the number of time steps feed to the model.
        out_chunk_len(int): The size of the forecasting horizon, i.e., the number of time steps output by the model.
        target_dim(int): The numer of targets.
        known_cov_dim(int): The number of known covariate.
        observed_cov_dim(int): The number of observed covariate.
        rnn_type(str): The type of the specific paddle RNN module ("SimpleRNN", "GRU" or "LSTM").
        hidden_dim(int): The number of features in the hidden state `h` of the RNN module.
        num_layers_recurrent(int): The number of recurrent layers.
        out_fcn_config(Optional[List]): A list containing the dimensions of the hidden layers of the fully connected NN.
        dropout(float): The fraction of neurons that are dropped in all-but-last RNN layers.
    """
    def __init__(
        self,
        in_chunk_len: int,
        out_chunk_len: int,
        target_dim: int,
        known_cov_dim: int,
        observed_cov_dim: int,
        rnn_type: str,
        hidden_dim: int,
        num_layers_recurrent: int,
        out_fcn_config: Optional[List] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_chunk_len = in_chunk_len
        self.out_chunk_len = out_chunk_len

        #TODO: probability forecasting
        #self.likelihood = kwargs.get('likelihood', None)
        #self.nr_params = 1 if not self.likelihood else self.likelihood.num_parameters

        out_fcn_config = [] if out_fcn_config is None else out_fcn_config
        self._rnn_type = rnn_type
        self._target_dim = target_dim
        self._known_cov_dim = known_cov_dim
        self._observed_cov_dim = observed_cov_dim
        self._input_size = self._target_dim + self._known_cov_dim + self._observed_cov_dim
        self._rnn = getattr(nn, self._rnn_type)(self._input_size, hidden_dim, num_layers_recurrent, dropout=dropout)

        # Defining projection layer
        # The RNN module is followed by a fully connected network(FCN),
        # which maps the last hidden layer to the output of desired length
        last = hidden_dim
        feats = []
        for feature in out_fcn_config + [out_chunk_len * target_dim]:
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
        feature = [data["past_target"]]
        if self._known_cov_dim > 0:
            past_known = data["known_cov"][:, :self.in_chunk_len]
            feature.append(past_known)
        if self._observed_cov_dim > 0:
            observed = data["observed_cov"]
            feature.append(observed)

        x = paddle.concat(x=feature, axis=-1)

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
        num_layers_recurrent(int, Optional): The number of recurrent layers.
        dropout(float, Optional): The fraction of neurons that are dropped in all-but-last RNN layers.
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
        num_layers_recurrent: int = 1,
        dropout: float = 0.0,
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
        self._num_layers_recurrent = num_layers_recurrent
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
        for column, dtype in tsdataset.dtypes.items():
            raise_if_not(
                np.issubdtype(dtype, np.floating),
                f"rnn variables' dtype only supports [float16, float32, float64], " \
                f"but received {column}: {dtype}."
            )
        super(RNNBlockRegressor, self)._check_tsdataset(tsdataset)

    def _update_fit_params(
        self,
        train_tsdataset: TSDataset,
        valid_tsdataset: Optional[TSDataset] = None
    ) -> Dict[str, Any]:
        """
        Infer parameters by TSdataset automatically.

        Args:
            train_tsdataset(TSDataset): train dataset
            valid_tsdataset(TSDataset, optional): validation dataset

        Returns:
            Dict[str, Any]: model parameters
        """
        fit_params = {
                "target_dim": train_tsdataset.get_target().data.shape[1],
                "known_cov_dim": 0,
                "observed_cov_dim": 0
                }
        if train_tsdataset.get_known_cov() is not None:
            fit_params["known_cov_dim"] = train_tsdataset.get_known_cov().data.shape[1]
        if train_tsdataset.get_observed_cov() is not None:
            fit_params["observed_cov_dim"] = train_tsdataset.get_observed_cov().data.shape[1]
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
            self._fit_params["target_dim"],
            self._fit_params["known_cov_dim"],
            self._fit_params["observed_cov_dim"],
            self._rnn_type_or_module,
            self._hidden_size,
            self._num_layers_recurrent,
            self._fcn_out_config,
            self._dropout
        )

