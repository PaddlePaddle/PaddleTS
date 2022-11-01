#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from typing import List, Dict, Any, Callable, Optional
from functools import partial

from paddle.optimizer import Optimizer
import paddle.nn.functional as F
import numpy as np
import paddle

from paddlets.models.forecasting.dl.paddle_base_impl import PaddleBaseModelImpl
from paddlets.models.common.callbacks import Callback
from paddlets.logger import raise_if_not, Logger
from paddlets.datasets import TSDataset

logger = Logger(__name__)

PAST_TARGET = "past_target"


class _TemporalBlock(paddle.nn.Layer):
    """Paddle layer implementing a residual block.

    Args:
        in_channels(int): The number of channels in the input.
        out_channels(int): The number of filter. It is as same as the output feature map.
        kernel_size(int): The filter size.
        dilation(int): The dilation size.
        dropout_rate(float): Probability of setting units to zero.

    Attributes:
        _conv1(paddle.nn.Layer): 1D convolution Layer.
        _conv2(paddle.nn.Layer): 1D convolution Layer.
        _downsample(paddle.nn.Layer): 1D convolution Layer.
        _dropout(paddle.nn.Layer): Probability of setting units to zero.
        _padding(int): The size of zeros to be padded.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout_rate: float,
    ):
        super(_TemporalBlock, self).__init__()
        self._conv1 = paddle.nn.Conv1D(
            in_channels=in_channels, 
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation
        )
        self._conv2 = paddle.nn.Conv1D(
            in_channels=out_channels, 
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation
        )
        self._downsample = paddle.nn.Conv1D(
            in_channels, out_channels, 1
        ) if in_channels != out_channels else None
        self._conv1 = paddle.nn.utils.weight_norm(self._conv1)
        self._conv2 = paddle.nn.utils.weight_norm(self._conv2)
        self._dropout = paddle.nn.Dropout(dropout_rate)
        self._padding = dilation * (kernel_size - 1)
        
    def forward(
        self,
        X: paddle.Tensor
    ) -> paddle.Tensor:
        """Forward.

        Args:
            X(paddle.Tensor): Feature tensor.

        Returns:
            paddle.Tensor: Output of model
        """
        # In order to deal with the dimension mismatch during residual addition, 
        # use upsampling or downsampling to ensure that the input channel and output channel dimensions match.
        residual = (
            self._downsample(X) if self._downsample else X
        )
        
        # TCN is based on two principles:
        # 1> The convolution network produces an output of the same length as the input (by padding the input).
        # 2> No future information leakage.
        # The pad layer is used to pad data to ensure that future information is not used.
        out = F.pad(X, (self._padding, 0), data_format="NCL")
        out = F.relu(self._conv1(out))
        out = self._dropout(out)

        out = F.pad(out, (self._padding, 0), data_format="NCL")
        out = F.relu(self._conv2(out))
        out = self._dropout(out)
        out = out + residual
        return out


class _TCNModule(paddle.nn.Layer):
    """Paddle layer implementing TCN module.

    Args:
        in_chunk_len(int): The size of the loopback window, i.e. the number of time steps feed to the model.
        out_chunk_len(int): The size of the forecasting horizon, i.e. the number of time steps output by the model.
        target_dim(int): The numer of targets.
        hidden_config(List[int]): The config of channels.
        kernel_size(int): The filter size.
        dropout_rate(float): Probability of setting units to zero.

    Attrubutes:
        _temporal_layers(paddle.nn.LayerList): Dynamic graph LayerList.
    """
    def __init__(
        self,
        in_chunk_len: int,
        out_chunk_len: int,
        target_dim: int,
        hidden_config: List[int],
        kernel_size: int,
        dropout_rate: float,
    ):
        super(_TCNModule, self).__init__()
        self._out_chunk_len = out_chunk_len
        raise_if_not(
            1 < kernel_size <= in_chunk_len,
            f"The valid range of `kernel_size` is (1, in_chunk_len], " \
            f"got kernel_size:{kernel_size} <= 1 or kernel_size:{kernel_size} > in_chunk_len:{in_chunk_len}."
        )
        raise_if_not(
            out_chunk_len <= in_chunk_len,
            f"The `out_chunk_len` must be <= `in_chunk_len`, "
            f"got out_chunk_len:{out_chunk_len} > in_chunk_len:{in_chunk_len}."
        )

        if hidden_config is None:
            # If hidden_config is not passed, compute number of layers needed for full history coverage.
            num_layers = np.ceil(
                np.log2((in_chunk_len - 1) / (kernel_size - 1) / 2 + 1)
            )
            hidden_config = [target_dim] * (int(num_layers) - 1)

        else:
            # If hidden_config is passed, compute the receptive field.
            num_layers = len(hidden_config) + 1
            receptive_filed = 1 + 2 * (kernel_size - 1) * (2 ** num_layers - 1)
            if receptive_filed > in_chunk_len:
                logger.warning("The receptive field of TCN exceeds the in_chunk_len.")

        raise_if_not(
            np.any(np.array(hidden_config) > 0),
            f"hidden_config must be > 0, got {hidden_config}."
        )
        
        channels, temporal_layers = [target_dim] + hidden_config + [target_dim], []
        for k, (in_channel, out_channel) in \
            enumerate(zip(channels[:-1], channels[1:])):
            temporal_layer = _TemporalBlock(
                in_channels=in_channel, 
                out_channels=out_channel, 
                kernel_size=kernel_size,
                dilation=(2 ** k),
                dropout_rate=dropout_rate,
            )
            temporal_layers.append(temporal_layer)
        self._temporal_layers = paddle.nn.Sequential(*temporal_layers)

    def forward(
        self,
        X: Dict[str, paddle.Tensor]
    ) -> paddle.Tensor:
        """Forward.

        Args:
            X(Dict[str, paddle.Tensor]): Dict of feature tensor.

        Returns:
            paddle.Tensor: Output of model
        """
        out = X[PAST_TARGET]
        out = paddle.transpose(out, perm=[0, 2, 1])
        out = self._temporal_layers(out)
        out = paddle.transpose(out, perm=[0, 2, 1])
        out = out[:, -self._out_chunk_len:, :]
        return out


class TCNRegressor(PaddleBaseModelImpl):
    """Temporal Convolution Net\[1\].

    \[1\] Bai S, et al. "An empirical evaluation of generic convolutional and recurrent networks for sequence modeling", 
    `<https://arxiv.org/pdf/1803.01271>`_

    Args:
        in_chunk_len(int): The size of the loopback window, i.e. the number of time steps feed to the model.
        out_chunk_len(int): The size of the forecasting horizon, i.e. the number of time steps output by the model.
        skip_chunk_len(int): Optional, the number of time steps between in_chunk and out_chunk for a single sample.
            The skip chunk is neither used as a feature (i.e. X) nor a label (i.e. Y) for a single sample.
            By default it will NOT skip any time steps.
        sampling_stride(int): Sampling intervals between two adjacent samples.
        loss_fn(Callable[..., paddle.Tensor]): Loss function.
        optimizer_fn(Callable[..., Optimizer]): Optimizer algorithm.
        optimizer_params(Dict[str, Any]): Optimizer parameters.
        eval_metrics(List[str]): Evaluation metrics of model.
        callbacks(List[Callback]): Customized callback functions.
        batch_size(int): Number of samples per batch.
        max_epochs(int): Max epochs during training.
        verbose(int): Verbosity mode.
        patience(int): Number of epochs to wait for improvement before terminating.
        seed(int|None): Global random seed.

        hidden_config(List[int]|None): Hidden layer configuration.
        kernel_size(int): The filter size.
        dropout_rate(float): Probability of setting units to zero.

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

        _hidden_config(List[int]|None): Hidden layer configuration.
        _kernel_size(int): The filter size.
        _dropout_rate(float): Probability of setting units to zero.
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

        hidden_config: List[int] = None,
        kernel_size: int = 3,
        dropout_rate: float = 0.2,
    ):
        self._hidden_config = hidden_config
        self._kernel_size = kernel_size
        self._dropout_rate = dropout_rate
        super(TCNRegressor, self).__init__(
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
                    f"tcn's target dtype only supports [float16, float32, float64], " \
                    f"but received {column}: {dtype}."
                )
                continue
            raise_if_not(
                np.issubdtype(dtype, np.floating),
                f"tcn's cov(observed or known) dtype currently only supports [float16, float32, float64], " \
                f"but received {column}: {dtype}."
            )
        super(TCNRegressor, self)._check_tsdataset(tsdataset)
        
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
            Dict[str, Any]: model parameters
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
        return _TCNModule(
            in_chunk_len=self._in_chunk_len,
            out_chunk_len=self._out_chunk_len,
            target_dim=self._fit_params["target_dim"],
            hidden_config=self._hidden_config,
            kernel_size=self._kernel_size,
            dropout_rate=self._dropout_rate,
        )
