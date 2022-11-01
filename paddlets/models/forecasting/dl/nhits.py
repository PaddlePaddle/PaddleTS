#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
This implementation is based on the article `N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting <https://arxiv.org/abs/2201.12886>`_ .
"""

from enum import Enum
from typing import List, Dict, Any, Callable, Optional, NewType, Tuple, Union

import numpy as np
import paddle
from paddle import nn
from paddle.optimizer import Optimizer
import paddle.nn.functional as F

from paddlets.datasets import TSDataset
from paddlets.models.forecasting.dl.paddle_base_impl import PaddleBaseModelImpl
from paddlets.models.common.callbacks import Callback
from paddlets.logger import raise_if, raise_if_not, Logger

logger = Logger(__name__)


ACTIVATIONS = [
    "ReLU",
    "PReLU",
    "ELU",
    "Softplus",
    "Tanh",
    "SELU",
    "LeakyReLU",
    "Sigmoid",
    "GELU",
]

class _Block(nn.Layer):
    """
    Block implementation, basic component of NHiTS.
    the block produces forecast and backcast of the current block

    Args:
        in_chunk_len: The length of the input sequence fed to the model.
        out_chunk_len: The length of the forecast of the model.
        in_chunk_len_flat: The length of the flattened input sequence(produced by concatenating past_target, known_cov, observed_cov) fed to the model.
        target_dim: The dimension of target.
        known_cov_dim(int): The number of known covariates.
        observed_cov_dim(int): The number of observed covariates.
        num_layers: The number of fully connected layers preceding the final forking layers.
        layer_width: The number of neurons that make up each fully connected layer.
        pooling_kernel_size: The kernel size for the initial pooling layer.
        n_freq_downsample: The factor by which to downsample time at the output (before interpolating).
        batch_norm: Whether to use batch norm.
        dropout: Dropout probability.
        activation: The activation function of encoder/decoder intermediate layer.
        MaxPool1d: Whether to use MaxPool1d pooling, False uses AvgPool1d.
    """
    def __init__(
        self,
        in_chunk_len: int,
        out_chunk_len: int,
        in_chunk_len_flat: int,
        target_dim: int,
        known_cov_dim: int,
        observed_cov_dim: int,
        num_layers: int,
        layer_width: int,
        pooling_kernel_size: int,
        n_freq_downsample: int,
        batch_norm: bool,
        dropout: float,
        activation: str,
        MaxPool1d: bool,
    ):
        super().__init__()
        self._in_chunk_len = in_chunk_len
        self._out_chunk_len = out_chunk_len
        self._target_dim = target_dim
        
        raise_if_not(
            activation in ACTIVATIONS, f"'{activation}' is not in {ACTIVATIONS}"
        )
        
        self._activation = getattr(nn, activation)()
        n_theta_backcast = max(in_chunk_len // n_freq_downsample * target_dim, 1) # multi-past_target input for number of base points
        n_theta_forecast = max(out_chunk_len // n_freq_downsample * target_dim, 1) # multi-future_target output for number of base points
        
        # pooling layer
        pool1d = nn.MaxPool1D if MaxPool1d else nn.AvgPool1D
        
        self.pooling_layer = pool1d(
            kernel_size=pooling_kernel_size,
            stride=pooling_kernel_size,
            ceil_mode=True,
        )
        # layer widths
        in_len = int(np.ceil(in_chunk_len / pooling_kernel_size)) * (target_dim + known_cov_dim + observed_cov_dim) +\
        int(np.ceil(out_chunk_len / pooling_kernel_size)) * known_cov_dim
        
        layer_widths = [in_len] + [layer_width] * num_layers
        # FC layers
        layers = []
        for i in range(num_layers):
            layers.append(
                nn.Linear(
                    in_features=layer_widths[i],
                    out_features=layer_widths[i + 1],
                )
            )
            layers.append(self._activation)

            if batch_norm:
                layers.append(nn.BatchNorm1D(num_features=layer_widths[i + 1]))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
        self.layers = nn.Sequential(*layers)
        # Fully connected layer producing forecast/backcast expansion coeffcients (waveform generator parameters).
        # The coefficients are emitted for each parameter of the likelihood for the forecast.
        self.backcast_linear_layer = nn.Linear(
            in_features=layer_width, out_features=n_theta_backcast
        )
        self.forecast_linear_layer = nn.Linear(
            in_features=layer_width, out_features=n_theta_forecast
        )

    def forward(
            self, 
            backcast: paddle.Tensor,
            known_cov: paddle.Tensor,
            observed_cov: paddle.Tensor
            ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """
        forward block.
        
        Args:
            backcast: past target, shape: [batch_size, in_chunk_len, target_dim]
            known_cov: known covariates, shape: [batch_size, in_chunk_len + target_length, known_cov_dim]
            observed_cov: observed covariates, shape: [batch_size, in_chunk_len, observed_cov_dim]

        Returns:
            x_hat: approximation of backcast on specific frequency, shape [batch_size, in_chunk_len, target_dim]
            y_hat: tensor containing the forward forecast of the block, shape [batch_size, out_chunk_len, target_dim]
        """
        # compose feature x
        batch_size = backcast.shape[0]
        # concat backcast, known_cov, observed_cov if any;
        past_feature = [backcast]
        future_feature = None
        if known_cov is not None:
            past_feature.append(known_cov[:, :self._in_chunk_len, :])
            future_feature = known_cov[:, self._in_chunk_len:, :].transpose(perm=[0, 2, 1])
        if observed_cov is not None:
            past_feature.append(observed_cov)
        past_feature = paddle.concat(x=past_feature, axis=2).transpose(perm=[0, 2, 1]) # (N,C,L)
        # pooling layer
        x = self.pooling_layer(past_feature).reshape([batch_size, -1])
        if future_feature is not None:
            x_ = self.pooling_layer(future_feature).reshape([batch_size, -1])
            x = paddle.concat([x, x_], axis=1)

        # fully connected layer stack
        x = self.layers(x)

        # forked linear layers producing waveform generator parameters
        theta_backcast = self.backcast_linear_layer(x) # in_chunk_len * target_dim
        theta_forecast = self.forecast_linear_layer(x) # out_chunk_len * target_dim

        # set the expansion coefs in last dimension for the forecasts
        theta_forecast = theta_forecast.reshape((batch_size, self._target_dim, -1))

        # set the expansion coefs in last dimension for the backcasts
        theta_backcast = theta_backcast.reshape((batch_size, self._target_dim, -1))

        # interpolate both backcast and forecast from the theta_backcast and theta_forecast
        x_hat = F.interpolate(
            theta_backcast, size=[self._in_chunk_len], mode="linear", data_format='NCW'
        )
        y_hat = F.interpolate(
            theta_forecast, size=[self._out_chunk_len], mode="linear", data_format='NCW'
        )
        x_hat = paddle.transpose(x_hat, perm=[0, 2, 1])
        y_hat = paddle.transpose(y_hat, perm=[0, 2, 1])
        return x_hat, y_hat


class _Stack(nn.Layer):
    """
    Stack implementation of the NHiTS architecture, comprises multiple basic blocks.

    Args:
        in_chunk_len: The length of input sequence fed to the model.
        out_chunk_len: The length of the forecast of the model.
        in_chunk_len_flat: The length of the flattened input sequence(produced by concatenating past_target, known_cov, observed_cov) fed to the model.
        num_blocks: The number of blocks making up this stack.
        num_layers: The number of fully connected layers preceding the final forking layers in each block.
        layer_width: The number of neurons that make up each fully connected layer in each block.
        target_dim: The dimension of target.
        known_cov_dim(int): The number of known covariates.
        observed_cov_dim(int): The number of observed covariates.
        pooling_kernel_size: The kernel size for the initial pooling layer.
        n_freq_downsample: The factor by which to downsample time at the output (before interpolating).
        batch_norm: Whether to use batch norm.
        dropout: Dropout probability.
        activation: The activation function of encoder/decoder intermediate layer.
        MaxPool1d: Whether to use MaxPool1d pooling, False uses AvgPool1d.
    """
    def __init__(
        self,
        in_chunk_len: int,
        out_chunk_len: int,
        in_chunk_len_flat: int,
        num_blocks: int,
        num_layers: int,
        layer_width: int,
        target_dim: int,
        known_cov_dim: int,
        observed_cov_dim: int,
        pooling_kernel_sizes: Tuple[int],
        n_freq_downsample: Tuple[int],
        batch_norm: bool,
        dropout: float,
        activation: str,
        MaxPool1d: bool,
    ):
        super().__init__()
        self.in_chunk_len = in_chunk_len
        self.out_chunk_len = out_chunk_len
        self._target_dim = target_dim

        # TODO: leave option to share weights across blocks?
        self._blocks_list = [
            _Block(
                in_chunk_len,
                out_chunk_len,
                in_chunk_len_flat,
                target_dim,
                known_cov_dim,
                observed_cov_dim,
                num_layers,
                layer_width,
                pooling_kernel_sizes[i],
                n_freq_downsample[i],
                batch_norm=(
                    batch_norm and i == 0
                ),  # batch norm only on first block of first stack
                dropout=dropout,
                activation=activation,
                MaxPool1d=MaxPool1d,
            )
            for i in range(num_blocks)
        ]
        self._blocks = nn.LayerList(self._blocks_list)
        
    def forward(
            self,
            backcast: paddle.Tensor,
            known_cov: paddle.Tensor,
            observed_cov: paddle.Tensor
            ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """
        forward stack.

        Args:
            backcast(paddle.Tensor): past target, shape: [batch_size, in_chunk_len, target_dim].
            past_covariate(paddle.Tensor): past known covariate, shape: [batch_size, in_chunk_len, covariate_dim].
            future_covariate(paddle.Tensor): future known covariate, shape: [batch_size, out_chunk_len, covariate_dim].

        Returns:
            stack_residual: residual tensor of backcast, shape [batch_size, in_chunk_len, target_dim].
            stack_forecast tensor containing the forward forecast of the stack, shape [batch_size, out_chunk_len].
        """
        #init stack_forecast as paddle.zeros
        stack_forecast = paddle.zeros(
            shape=(backcast.shape[0], self.out_chunk_len, self._target_dim),
            dtype=backcast.dtype,
        )
        for block in self._blocks_list:
            # pass input through block
            x_hat, y_hat = block(backcast, known_cov, observed_cov)
            # add block forecast to stack forecast
            stack_forecast = stack_forecast + y_hat
            # subtract backcast from input to produce residual

            backcast = backcast - x_hat
        stack_residual = backcast
        return stack_residual, stack_forecast


class _NHiTSModule(nn.Layer):
    """
    Implementation of NHiTS, cover multi-targets, known_covariates, observed_covariates.
    """
    def __init__(
        self,
        in_chunk_len: int,
        out_chunk_len: int,
        target_dim: int,
        known_cov_dim: int,
        observed_cov_dim: int,
        num_stacks: int,
        num_blocks: int,
        num_layers: int,
        layer_widths: List[int],
        pooling_kernel_sizes: Optional[Tuple[Tuple[int]]],
        n_freq_downsample: Optional[Tuple[Tuple[int]]],
        batch_norm: bool,
        dropout: float,
        activation: str,
        MaxPool1d: bool,
    ):
        """
        Args:
            in_chunk_len(int): The length of the input sequence fed to the model.
            out_chunk_len(int): The length of the forecast of the model.
            target_dim(int): The number of targets to be forecasted.
            known_cov_dim(int): The number of known covariates.
            observed_cov_dim(int): The number of observed covariates.
            num_stacks(int): The number of stacks that make up the whole model.
            num_blocks(int): The number of blocks making up every stack.
            num_layers(int): The number of fully connected layers preceding the final forking layers in each block of every stack.
            layer_widths(List[int]): Determines the number of neurons that make up each fully connected layer in each block of every stack. It must have a length equal to `num_stacks` and every entry in that list corresponds to the layer width of the corresponding stack. If an integer is passed, every stack will have blocks with FC layers of the same width.
            pooling_kernel_size(Tuple[Tuple[int]], option): The kernel size for the initial pooling layer.
            n_freq_downsample(Tuple[Tuple[int]], option): The factor by which to downsample time at the output (before interpolating).
            batch_norm(bool): Whether to use batch norm.
            dropout(float): Dropout probability.
            activation(str): The activation function of encoder/decoder intermediate layer.
            MaxPool1d(bool): Whether to use MaxPool1d pooling, False uses AvgPool1d.
        """
        super().__init__()
        self._known_cov_dim = known_cov_dim
        self._observed_cov_dim = observed_cov_dim
        self._target_dim = target_dim
        self._target_length = out_chunk_len
        input_dim = target_dim + known_cov_dim + observed_cov_dim
        self._in_chunk_len_multi = in_chunk_len * input_dim + out_chunk_len * known_cov_dim
        self._pooling_kernel_sizes, self._n_freq_downsample = self._check_pooling_downsampling(
            pooling_kernel_sizes,
            n_freq_downsample,
            in_chunk_len,
            out_chunk_len,
            num_blocks,
            num_stacks
        )
        self._stacks_list = [
            _Stack(
                in_chunk_len,
                out_chunk_len,
                self._in_chunk_len_multi,
                num_blocks,
                num_layers,
                layer_widths[i],
                target_dim,
                known_cov_dim,
                observed_cov_dim,
                self._pooling_kernel_sizes[i],
                self._n_freq_downsample[i],
                batch_norm=(batch_norm and i == 0), # batch norm only on the first block of the first stack
                dropout=dropout,
                activation=activation,
                MaxPool1d=MaxPool1d,
            )
            for i in range(num_stacks)
        ]

        self._stacks = nn.LayerList(self._stacks_list)
        self._stacks_list[-1]._blocks[-1].backcast_linear_layer.stop_gradient = True

    def _check_pooling_downsampling(
        self,
        pooling_kernel_sizes: Optional[Tuple[Tuple[int]]],
        n_freq_downsample: Optional[Tuple[Tuple[int]]],
        in_len: int,
        out_len: int,
        num_blocks: int,
        num_stacks: int
    ):
        """
        check validation of pooling kernel sizes and n_freq_downsample if user set,
        or compute the best values automatically.

        Args:
            pooling_kernel_sizes: The kernel size for the initial pooling layer.
            n_freq_downsample: The factor by which to downsample time at the output (before interpolating).
            in_len: The length of the input sequence.
            out_len: The length of the forecast.
            num_blocks: The number of blocks making up every stack.
            num_stacks: The number of stacks that make up the whole model.

        Returns:
            pooling_kernel_sizes: valid pooling_kernel_sizes.
            n_freq_downsample: valid n_freq_downsample.
        """
        def _check_sizes(tup, name):
            raise_if_not(
                len(tup) == num_stacks,
                f"the length of {name} must match the number of stacks.",
            )
            raise_if_not(
                all([len(i) == num_blocks for i in tup]),
                f"the length of each tuple in {name} must be `num_blocks={num_blocks}`",
            )

        if pooling_kernel_sizes is None:
            # make stacks handle different frequencies
            # go from in_len/2 to 1 in num_stacks steps:
            max_v = max(in_len // 2, 1)
            pooling_kernel_sizes = tuple(
                (max(int(v), 1),) * num_blocks
                for v in max_v // np.geomspace(1, max_v, num_stacks)
            )
        else:
            # check provided pooling format
            _check_sizes(pooling_kernel_sizes, "`pooling_kernel_sizes`")

        if n_freq_downsample is None:
            # go from out_len/2 to 1 in num_stacks steps:
            max_v = max(out_len // 2, 1)
            n_freq_downsample = tuple(
                (max(int(v), 1),) * num_blocks
                for v in max_v // np.geomspace(1, max_v, num_stacks)
            )
        else:
            # check provided downsample format
            _check_sizes(n_freq_downsample, "`n_freq_downsample`")

            # check that last value is 1
            raise_if_not(
                n_freq_downsample[-1][-1] == 1,
                "the downsampling coefficient of the last block of the last stack must be 1 "
                + "(i.e., `n_freq_downsample[-1][-1]`).",
            )
        return pooling_kernel_sizes, n_freq_downsample

    def forward(
            self,
            data: Dict[str, paddle.Tensor]
            ) -> paddle.Tensor:
        """
        forward NHiTS network.

        Args:
            data(Dict[str, paddle.Tensor]): a dict specifies all kinds of input data

        Returns:
            forecast(padle.Tensor): Tensor containing the output of the NHiTS model.
        """
        backcast = data["past_target"]
        known_cov = data["known_cov_numeric"] if self._known_cov_dim > 0 else None
        observed_cov = data["observed_cov_numeric"] if self._observed_cov_dim > 0 else None
        # init forecast tensor
        forecast = paddle.zeros(
            shape = (backcast.shape[0], self._target_length, self._target_dim))
        for stack_index, stack in enumerate(self._stacks_list):
            # compute stack output
            stack_residual, stack_forecast = stack(backcast, known_cov, observed_cov)
            # accumulate stack_forecast to final output
            forecast = forecast + stack_forecast
            # set current stack residual as input for next stack
            backcast = stack_residual

        return forecast


class NHiTSModel(PaddleBaseModelImpl):
    """
    Implementation of NHiTS model

    Args:
        in_chunk_len(int): The size of the loopback window, i.e., the number of time steps feed to the model.
        out_chunk_len(int): The size of the forecasting horizon, i.e., the number of time steps output by the model.
        num_stacks: The number of stacks that make up the whole model.
        num_blocks: The number of blocks making up every stack.
        num_layers: The number of fully connected layers preceding the final forking layers in each block of every stack.
        layer_widths: Determines the number of neurons that make up each fully connected layer in each block of every stack. If a list is passed, it must have a length equal to `num_stacks` and every entry in that list corresponds to the layer width of the corresponding stack. If an integer is passed, every stack will have blocks with FC layers of the same width.
        pooling_kernel_size(Tuple[Tuple[int]], option): The kernel size for the initial pooling layer.
        n_freq_downsample(Tuple[Tuple[int]], option): The factor by which to downsample time at the output (before interpolating).
        batch_norm(bool): Whether to use batch normalization.
        dropout(float): Dropout probability.
        activation(str): The activation function of encoder/decoder intermediate layer.
        MaxPool1d(bool): Whether to use MaxPool1d pooling, False uses AvgPool1d.
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
        num_stacks: int = 3,
        num_blocks: int = 3,
        num_layers: int = 2,
        layer_widths: Union[int, List[int]] = 512,
        pooling_kernel_sizes: Optional[Tuple[Tuple[int]]] = None,
        n_freq_downsample: Optional[Tuple[Tuple[int]]] = None,
        batch_norm: bool = False,
        dropout: float = 0.1,
        activation: str = "ReLU",
        MaxPool1d: bool = True,
        skip_chunk_len: int = 0,
        sampling_stride: int = 1,
        loss_fn: Callable[..., paddle.Tensor] = F.mse_loss,
        optimizer_fn: Callable[..., Optimizer] = paddle.optimizer.Adam,
        optimizer_params: Dict[str, Any] = dict(learning_rate=1e-4), 
        eval_metrics: List[str] = [], 
        callbacks: List[Callback] = [], 
        batch_size: int = 256,
        max_epochs: int = 10,
        verbose: int = 1,
        patience: int = 4,
        seed: int = 0
    ):
        self._num_stacks = num_stacks
        self._num_blocks = num_blocks
        self._num_layers = num_layers
        self._layer_widths = layer_widths
        self._pooling_kernel_sizes = pooling_kernel_sizes
        self._n_freq_downsample = n_freq_downsample

        self._activation = activation
        self._MaxPool1d = MaxPool1d
        self._dropout = dropout
        self._batch_norm = batch_norm

        if isinstance(self._layer_widths, int):
            self._layer_widths = [self._layer_widths] * self._num_stacks

        super(NHiTSModel, self).__init__(
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

    def _check_params(self):
        """
        check validation of parameters
        """
        raise_if(
            isinstance(self._layer_widths, list) and len(self._layer_widths) != self._num_stacks,
            "Stack number should be equal to the length of the List: layer_widths."
        )
        super(NHiTSModel, self)._check_params()

    def _check_tsdataset(
        self,
        tsdataset: TSDataset
        ):
        """-
        Rewrite _check_tsdataset to fit the specific model.
        For NHiTS, all data variables are expected to be float32.
        """
        for column, dtype in tsdataset.dtypes.items():
            raise_if_not(
                np.issubdtype(dtype, np.floating),
                f"nhits variables' dtype only supports [float16, float32, float64], " \
                f"but received {column}: {dtype}."
            )
        super(NHiTSModel, self)._check_tsdataset(tsdataset)

    def _update_fit_params(
        self,
        train_tsdataset: List[TSDataset],
        valid_tsdataset: Optional[List[TSDataset]] = None
    ) -> Dict[str, Any]:
        """
        Infer parameters by TSDataset automatically.

        Args:
            train_tsdataseet(List[TSDataset]): list of train dataset
            valid_tsdataset(List[TSDataset], optional): list of validation dataset
        Returns:
            Dict[str, Any]: model parameters
        """
        train_ts0 = train_tsdataset[0]
        fit_params = {
                "target_dim": train_ts0.get_target().data.shape[1],
                "known_cov_dim": 0,
                "observed_cov_dim": 0
                }
        if train_ts0.get_known_cov() is not None:
            fit_params["known_cov_dim"] = train_ts0.get_known_cov().data.shape[1]
        if train_ts0.get_observed_cov() is not None:
            fit_params["observed_cov_dim"] = train_ts0.get_observed_cov().data.shape[1]
        return fit_params

    def _init_network(self) -> paddle.nn.Layer:
        """
        init network

        Returns:
            paddle.nn.Layer
        """
        
        return _NHiTSModule(
            self._in_chunk_len,
            self._out_chunk_len,
            self._fit_params["target_dim"],
            self._fit_params["known_cov_dim"],
            self._fit_params["observed_cov_dim"],
            self._num_stacks,
            self._num_blocks,
            self._num_layers,
            self._layer_widths,
            self._pooling_kernel_sizes,
            self._n_freq_downsample,
            self._batch_norm,
            self._dropout,
            self._activation,
            self._MaxPool1d,
        )
