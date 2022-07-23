#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from typing import List, Dict, Any, Callable, Optional
from functools import partial

from paddle.optimizer import Optimizer
import paddle.nn.functional as F
import numpy as np
import paddle

from paddlets.models.dl.paddlepaddle.paddle_base_impl import PaddleBaseModelImpl
from paddlets.models.dl.paddlepaddle.callbacks import Callback
from paddlets.logger import raise_if_not, Logger
from paddlets.datasets import TSDataset

logger = Logger(__name__)


class _Chomp1d(paddle.nn.Layer):
    """Auxiliary Causal convolution layer.

    TCN is based on two principles: 
        1> The convolution network produces an output of the same length as the input (by padding the input).
        2> No future information leakage. 
    The Chomp1d layer is used to slice the padding data to ensure that future information is not used.

    Args:
        chomp_size(int): Slice length.

    Attributes:
        _chomp_size(int): Slice length.

    """
    def __init__(self, chomp_size: int):
        super(_Chomp1d, self).__init__()
        self._chomp_size = chomp_size

    def forward(
        self, 
        X: paddle.Tensor
    ) -> paddle.Tensor:
        """Forward.

        Args:
            X(paddle.Tensor): Feature tensor.

        Returns:
            paddle.Tensor:  Output of Layer.
        """
        return X[:, :-self._chomp_size, :]


class _TemporalBlock(paddle.nn.Layer):
    """Paddle layer implementing a residual block.

    Args:
        in_chunk_len(int): The size of the loopback window, i.e. the number of time steps feed to the model.
        out_chunk_len(int): The size of the forecasting horizon, i.e. the number of time steps output by the model.
        kernel_size(int): The filter size.
        padding(int): The size of zeros to be padded.
        dilation(int): The dilation size.
        dropout_rate(float): Probability of setting units to zero.

    Attributes:
        _nn(paddle.nn.LayerList): Dynamic graph LayerList.
        _downsample(paddle.nn.Layer): Dynamic graph Layer.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        dilation: int,
        dropout_rate: float,
    ):
        super(_TemporalBlock, self).__init__()
        Conv1D = partial(paddle.nn.Conv1D, data_format="NLC")
        conv1 = paddle.nn.utils.weight_norm(
            Conv1D(
                in_channels, 
                out_channels, 
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
            )
        )
        conv2 = paddle.nn.utils.weight_norm(
            Conv1D(
                out_channels, 
                out_channels, 
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
            )
        )
        self._nn = paddle.nn.Sequential(
            conv1, _Chomp1d(padding), paddle.nn.ReLU(), paddle.nn.Dropout(dropout_rate),
            conv2, _Chomp1d(padding), paddle.nn.ReLU(), paddle.nn.Dropout(dropout_rate),
        )
        self._downsample = (
            Conv1D(in_channels, out_channels, 1) if in_channels != out_channels else None
        )

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
        res = (
            self._downsample(X) if self._downsample else X
        )
        return self._nn(X) + res


class _TCNBlock(paddle.nn.Layer):
    """Paddle layer implementing TCN block.

    Args:
        in_chunk_len(int): The size of the loopback window, i.e. the number of time steps feed to the model.
        out_chunk_len(int): The size of the forecasting horizon, i.e. the number of time steps output by the model.
        target_dim(int): The numer of targets.
        hidden_config(List[int]): The config of channels.
        kernel_size(int): The filter size.
        dropout_rate(float): Probability of setting units to zero.

    Attrubutes:
        _nn(paddle.nn.LayerList): Dynamic graph LayerList.
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
        super(_TCNBlock, self).__init__()
        self._out_chunk_len = out_chunk_len

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
        raise_if_not(
            0 < kernel_size <= in_chunk_len,
            f"The valid range of `kernel_size` is (0, in_chunk_len], " \
            f"got kernel_size:{kernel_size} <= 0 or kernel_size:{kernel_size} > in_chunk_len:{in_chunk_len}."
        )
        raise_if_not(
            out_chunk_len <= in_chunk_len,
            f"The `out_chunk_len` must be <= `in_chunk_len`, "
            f"got out_chunk_len:{out_chunk_len} > in_chunk_len:{in_chunk_len}."
        )
        channels, layers = [target_dim] + hidden_config + [target_dim], []
        for k, (in_channel, out_channel) in enumerate(zip(channels[:-1], channels[1:])):
            dilation = 2 ** k
            layers.append(
                _TemporalBlock(
                    in_channel, 
                    out_channel, 
                    kernel_size=kernel_size,
                    padding=(kernel_size - 1) * dilation,
                    dilation=dilation,
                    dropout_rate=dropout_rate
                )
            )
        self._nn = paddle.nn.Sequential(*layers)

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
        out = self._nn(X["past_target"])
        return out[:, -self._out_chunk_len:, :]


class TCNRegressor(PaddleBaseModelImpl):
    """Temporal Convolution Net.

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
            check whether the data types are compatible. If not, the processing logic is as follows.

        Processing logic:

            1> Integer: Convert to np.int64.

            2> Floating: Convert to np.float32.

            3> Missing value: Warning.

            4> Other: Illegal.

        Args:
            tsdataset(TSDataset): Data to be checked.
        """
        for column, dtype in tsdataset.get_target().dtypes.items():
            raise_if_not(
                np.issubdtype(dtype, np.floating),
                f"tcn's target dtype only supports [float16, float32, float64], " \
                f"but received {column}: {dtype}."
            )
        super(TCNRegressor, self)._check_tsdataset(tsdataset)
        
    def _update_fit_params(
        self,
        train_tsdataset: TSDataset,
        valid_tsdataset: Optional[TSDataset] = None
    ) -> Dict[str, Any]:
        """Infer parameters by TSdataset automatically.

        Args:
            train_tsdataset(TSDataset): train dataset.
            valid_tsdataset(TSDataset|None): validation dataset.
        
        Returns:
            Dict[str, Any]: model parameters
        """
        fit_params = {
            "target_dim": train_tsdataset.get_target().data.shape[1]
        }
        return fit_params
        
    def _init_network(self) -> paddle.nn.Layer:
        """Setup the network.

        Returns:
            paddle.nn.Layer.
        """
        return _TCNBlock(
            self._in_chunk_len,
            self._out_chunk_len,
            self._fit_params["target_dim"],
            self._hidden_config,
            self._kernel_size,
            self._dropout_rate,
        )

