#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from typing import List, Dict, Any, Callable, Optional 

from paddle.optimizer import Optimizer
import paddle.nn.functional as F
import numpy as np
import paddle

from paddlets.models.forecasting.dl.paddle_base_impl import PaddleBaseModelImpl
from paddlets.models.common.callbacks import Callback
from paddlets.logger import raise_if_not, Logger
from paddlets.datasets import TSDataset

PAST_TARGET = "past_target"


class _LSTNetModule(paddle.nn.Layer):
    """Network structure.

    Args:
        in_chunk_len(int): The size of the loopback window, i.e. the number of time steps feed to the model.
        out_chunk_len(int): The size of the forecasting horizon, i.e. the number of time steps output by the model.
        target_dim(int): The numer of targets.
        skip_size(int): Skip size for the skip RNN layer.
        channels(int): Number of channels for first layer Conv1D.
        kernel_size(int): Kernel size for first layer Conv1D.
        rnn_cell_type(str): Type of the RNN cell, Either GRU or LSTM.
        rnn_num_cells(int): Number of RNN cells for each layer.
        skip_rnn_cell_type(str): Type of the RNN cell for the skip layer, Either GRU or LSTM.
        skip_rnn_num_cells(int): Number of RNN cells for each layer for skip part.
        dropout_rate(float): Dropout regularization parameter.
        output_activation(str|None): The last activation to be used for output. 
            Accepts either None (default no activation), sigmoid or tanh.

    Attrubutes:
        _in_chunk_len(int): The size of the loopback window, i.e. the number of time steps feed to the model.
        _skip_size(int): Skip size for the skip RNN layer.
        _channels(int): Number of channels for first layer Conv1D.
        _rnn_num_cells(int): Number of RNN cells for each layer.
        _skip_rnn_num_cells(int): Number of RNN cells for each layer for skip part.
        _output_activation(str|None): The last activation to be used for output.
            Accepts either None (default no activation), sigmoid or tanh.
    """
    def __init__(
        self,
        in_chunk_len: int,
        out_chunk_len: int,
        target_dim: int,
        skip_size: int,
        channels: int,
        kernel_size: int,
        rnn_cell_type: str,
        rnn_num_cells: int,
        skip_rnn_cell_type: str,
        skip_rnn_num_cells: int,
        dropout_rate: float,
        output_activation: Optional[str] = None,
    ):
        super(_LSTNetModule, self).__init__()
        self._in_chunk_len = in_chunk_len
        self._channels = channels
        self._rnn_num_cells = rnn_num_cells
        self._skip_rnn_num_cells = skip_rnn_num_cells
        self._skip_size = skip_size
        self._output_activation = output_activation
        raise_if_not(
            channels > 0, 
            "`channels` must be a positive integer"
        )
        raise_if_not(
            rnn_cell_type in ("GRU", "LSTM"), 
            "`rnn_cell_type` must be either 'GRU' or 'LSTM'"
        )
        raise_if_not(
            skip_rnn_cell_type in ("GRU", "LSTM"), 
            "`skip_rnn_cell_type` must be either 'GRU' or 'LSTM'"
        )
        conv_out = in_chunk_len - kernel_size
        self._conv_skip = conv_out // skip_size
        raise_if_not(
            self._conv_skip > 0, 
            "conv1d output size must be greater than or equal to `skip_size`\n" \
            "Choose a smaller `kernel_size` or bigger `in_chunk_len`"
        )
        if output_activation is not None:
            raise_if_not(
                output_activation in ("sigmoid", "tanh"), 
                "`output_activation` must be either 'sigmoid' or 'tanh'"
            )

        self._cnn = paddle.nn.Conv1D(target_dim, channels, kernel_size, data_format="NLC")
        self._dropout = paddle.nn.Dropout(dropout_rate)

        rnn = {"LSTM": paddle.nn.LSTM, "GRU": paddle.nn.GRU}[rnn_cell_type]
        self._rnn = rnn(channels, rnn_num_cells)

        skip_rnn = {"LSTM": paddle.nn.LSTM, "GRU": paddle.nn.GRU}[skip_rnn_cell_type]
        self._skip_rnn = skip_rnn(channels, skip_rnn_num_cells)

        self._fc = paddle.nn.Linear(rnn_num_cells + skip_size * skip_rnn_num_cells, target_dim)
        self._ar_fc = paddle.nn.Linear(in_chunk_len, out_chunk_len)

    def forward(
        self,
        X: Dict[str, paddle.Tensor]
    ) -> paddle.Tensor:
        """Forward.

        Args:
            X(Dict[str, paddle.Tensor]): Dict of feature tensor.

        Returns:
            paddle.Tensor: Output of model.
        """
        # CNN
        cnn_out = self._cnn(X[PAST_TARGET]) # [B, T, C]
        cnn_out = F.relu(cnn_out) 
        cnn_out = self._dropout(cnn_out)

        # RNN
        _, rnn_out = self._rnn(cnn_out)                     
        rnn_out = (
            rnn_out[0] if isinstance(rnn_out, tuple) else rnn_out
        )
        rnn_out = self._dropout(rnn_out)          # [1, B, C] 
        rnn_out = paddle.squeeze(rnn_out, axis=0) # [B, C]

        # Skip-RNN
        skip_out = cnn_out[:, -self._conv_skip * self._skip_size:, :] # [B, T, C]
        skip_out = paddle.reshape(                                    # [B, conv_out // skip, skip, C]
            skip_out, 
            shape=[-1, self._conv_skip, self._skip_size, self._channels]
        )                                                                
        skip_out = paddle.transpose(skip_out, perm=[0, 2, 1, 3])                         # [B, skip, conv_out // skip, C]
        skip_out = paddle.reshape(skip_out, shape=[-1, self._conv_skip, self._channels]) # [B, conv_out // skip, C]
        _, skip_out = self._skip_rnn(skip_out)                                           # [1, B, C]
        skip_out = (
            skip_out[0] if isinstance(skip_out, tuple) else skip_out
        )
        skip_out = paddle.reshape(skip_out, shape=[-1, self._skip_size * self._skip_rnn_num_cells])
        skip_out = self._dropout(skip_out)
        res = self._fc(
            paddle.concat([rnn_out, skip_out], axis=1)
        )
        res = paddle.unsqueeze(res, axis=1)

        # Highway
        ar_in = X[PAST_TARGET][:, -self._in_chunk_len:, :]
        ar_in = paddle.transpose(ar_in, perm=[0, 2, 1])
        ar_out = self._ar_fc(ar_in)                       # [B, C, T]
        ar_out = paddle.transpose(ar_out, perm=[0, 2, 1]) # [B, T, C]
        out = ar_out + res
        if self._output_activation:
            out = (
                F.sigmoid(out) if self._output_activation == "sigmoid" else F.tanh(out)
            )
        return out


class LSTNetRegressor(PaddleBaseModelImpl):
    """LSTNet\[1\] is a time series forecasting model introduced in 2018. LSTNet uses the 
    Convolution Neural Network (CNN) and the Recurrent Neural Network (RNN) to extract short-term local 
    dependency patterns among variables and to discover long-term patterns for time series trends.

    \[1\] Lai G, et al. "Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks", `<https://arxiv.org/abs/1703.07015>`_

    Args:
        in_chunk_len(int): The size of the loopback window, i.e. the number of time steps feed to the model.
        out_chunk_len(int): The size of the forecasting horizon, i.e. the number of time steps output by the model.
        skip_chunk_len(int): Optional, the number of time steps between in_chunk and out_chunk for a single sample.
            The skip chunk is neither used as a feature (i.e. X) nor a label (i.e. Y) for a single sample.
            By default it will NOT skip any time steps.
        sampling_stride(int): Sampling intervals between two adjacent samples.
        loss_fn(Callable[..., paddle.Tensor]|None): Loss function.
        optimizer_fn(Callable[..., Optimizer]): Optimizer algorithm.
        optimizer_params(Dict[str, Any]): Optimizer parameters.
        eval_metrics(List[str]): Evaluation metrics of model.
        callbacks(List[Callback]): Customized callback functions.
        batch_size(int): Number of samples per batch.
        max_epochs(int): Max epochs during training.
        verbose(int): Verbosity mode.
        patience(int): Number of epochs to wait for improvement before terminating.
        seed(int|None): Global random seed.

        skip_size(int): Skip size for the skip RNN layer.
        channels(int): Number of channels for first layer Conv1D.
        kernel_size(int): Kernel size for first layer Conv1D.
        rnn_cell_type(str): Type of the RNN cell, Either GRU or LSTM.
        rnn_num_cells(int): Number of RNN cells for each layer.
        skip_rnn_cell_type(str): Type of the RNN cell for the skip layer, Either GRU or LSTM.
        skip_rnn_num_cells(int): Number of RNN cells for each layer for skip part.
        dropout_rate(float): Dropout regularization parameter.
        output_activation(str|None): The last activation to be used for output.
            Accepts either None (default no activation), sigmoid or tanh.

    Attributes:
        _in_chunk_len(int): The size of the loopback window, i.e. the number of time steps feed to the model.
        _out_chunk_len(int): The size of the forecasting horizon, i.e. the number of time steps output by the model.
        _skip_chunk_len(int): Optional, the number of time steps between in_chunk and out_chunk for a single sample.
            The skip chunk is neither used as a feature (i.e. X) nor a label (i.e. Y) for a single sample.
            By default it will NOT skip any time steps.
        _sampling_stride(int): Sampling intervals between two adjacent samples.
        _loss_fn(Callable[..., paddle.Tensor]): Loss function.
        _optimizer_fn(Callable[..., Optimizer]): Optimizer algorithm.
        _optimizer_params(Dict[str, Any]): Optimizer parameters.
        _eval_metrics(List[str]): Evaluation metrics of model.
        _callbacks(List[Callback]): Customized callback functions.
        _batch_size(int): Number of samples per batch.
        _max_epochs(int): Max epochs during training.
        _verbose(int): Verbosity mode.
        _patience(int): Number of epochs to wait for improvement before terminating.
        _seed(int|None): Global random seed.
        _stop_training(bool) Training status.
        _skip_size(int): Skip size for the skip RNN layer.
        _channels(int): Number of channels for first layer Conv1D.
        _kernel_size(int): Kernel size for first layer Conv1D.
        _rnn_cell_type(str): Type of the RNN cell, Either GRU or LSTM.
        _rnn_num_cells(int): Number of RNN cells for each layer.
        _skip_rnn_cell_type(str): Type of the RNN cell for the skip layer, Either GRU or LSTM.
        _skip_rnn_num_cells(int): Number of RNN cells for each layer for skip part.
        _dropout_rate(float): Dropout regularization parameter.
        _output_activation(str|None): The last activation to be used for output.
            Accepts either None (default no activation), sigmoid or tanh.
    """
    def __init__(
        self,
        in_chunk_len: int,
        out_chunk_len: int,
        skip_chunk_len: int = 0,
        sampling_stride: int = 1,
        loss_fn: Callable[..., paddle.Tensor] = F.mse_loss,
        optimizer_fn: Callable[..., Optimizer] = paddle.optimizer.Adam,
        optimizer_params: Dict[str, Any] = dict(learning_rate=1e-3),
        eval_metrics: List[str] = [], 
        callbacks: List[Callback] = [], 
        batch_size: int = 32,
        max_epochs: int = 100,
        verbose: int = 1,
        patience: int = 10,
        seed: Optional[int] = None,

        skip_size: int = 1,
        channels: int = 1,
        kernel_size: int = 3,
        rnn_cell_type: str = "GRU",
        rnn_num_cells: int = 10,
        skip_rnn_cell_type: str = "GRU",
        skip_rnn_num_cells: int = 10,
        dropout_rate: float = 0.2,
        output_activation: Optional[str] = None
    ):
        self._skip_size = skip_size
        self._channels = channels
        self._kernel_size = kernel_size
        self._rnn_cell_type = rnn_cell_type
        self._rnn_num_cells = rnn_num_cells
        self._skip_rnn_cell_type = skip_rnn_cell_type
        self._skip_rnn_num_cells = skip_rnn_num_cells
        self._dropout_rate = dropout_rate
        self._output_activation = output_activation
        super(LSTNetRegressor, self).__init__(
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
        """Ensure the robustness of input data (consistent feature order), at the same time,
            check whether the data types are compatible. If not, the processing logic is as follows:

            1> Integer: Convert to np.int64.

            2> Floating: Convert to np.float32.

            3> Missing value: Warning.

            4> Other: Illegal.

        Args:
            tsdataset(TSDataset): Data to be checked.
        """
        target_columns = tsdataset.get_target().dtypes.keys()
        for column, dtype in tsdataset.dtypes.items():
            if column in target_columns:
                raise_if_not(
                    np.issubdtype(dtype, np.floating),
                    f"lstnet's target dtype only supports [float16, float32, float64], " \
                    f"but received {column}: {dtype}."
                )
                continue
            raise_if_not(
                np.issubdtype(dtype, np.floating),
                f"lstnet's cov(observed or known) dtype currently only supports [float16, float32, float64], " \
                f"but received {column}: {dtype}."
            )
        super(LSTNetRegressor, self)._check_tsdataset(tsdataset)
        
    def _update_fit_params(
        self,
        train_tsdataset: List[TSDataset],
        valid_tsdataset: Optional[List[TSDataset]] = None
    ) -> Dict[str, Any]:
        """Infer parameters by TSdataset automatically.

        Args:
            train_tsdataset(List[TSDataset]): list of train dataset.
            valid_tsdataset(List[TSDataset]|None): list of validation dataset.

        Returns:
            Dict[str, Any]: model parameters.
        """
        target_dim = train_tsdataset[0].get_target().data.shape[1]
        fit_params = {
            "target_dim": target_dim
        }
        return fit_params
        
    def _init_network(self) -> paddle.nn.Layer:
        """Setup the network.

        Returns:
            paddle.nn.Layer.
        """
        return _LSTNetModule(
            in_chunk_len=self._in_chunk_len,
            out_chunk_len=self._out_chunk_len,
            target_dim=self._fit_params["target_dim"],
            skip_size=self._skip_size,
            channels=self._channels,
            kernel_size=self._kernel_size,
            rnn_cell_type=self._rnn_cell_type,
            rnn_num_cells=self._rnn_num_cells,
            skip_rnn_cell_type=self._skip_rnn_cell_type,
            skip_rnn_num_cells=self._skip_rnn_num_cells,
            dropout_rate=self._dropout_rate,
            output_activation=self._output_activation
        )
