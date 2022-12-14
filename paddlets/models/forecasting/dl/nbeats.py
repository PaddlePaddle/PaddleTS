#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
This implementation is based on the article `N-BEATS: Neural basis expansion analysis for interpretable time series forecasting <https://arxiv.org/pdf/1905.10437.pdf>`_ .

Base model features
    Basic architecture: A network with hierarchical stacking, bi-directional residual connection and interpretable generator.

    Hierarchical stacking: The Design of multi-stacks with multi-blocks in each is for different kinds of information extraction, ie, trend, seasonality, etc..

    Bi-directional residual cascade: The backward residual connection is for computing of the residual signal, and pass the residual to the next layer; the forward residual connection is for accumulating  all layers' forecasts to the final output.

Updated features
    Multi-target: support multi-target modelling.

    Covariates: support known covariates(future known covariates) and observed covariates(future unknown covariates).

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
from paddlets.models.forecasting.dl.revin import revin_norm
from paddlets.models.common.callbacks import Callback
from paddlets.logger import raise_if, raise_if_not, raise_log, Logger

logger = Logger(__name__)


class _GType(Enum):
    GENERIC = 1
    TREND = 2
    SEASONALITY = 3


GTypes = NewType("GTypes", _GType)


class _TrendGenerator(nn.Layer):
    """
    Trend generator, implemented by polynomial function.

    Args:
        expansion_coefficient_dim: max degree of polynomial
        target_length: forecasting horizon
    """
    def __init__(
            self, 
            expansion_coefficient_dim: int,
            target_length: int,
            ):
        super().__init__()
        # basis is a vector of series of power functions of t, ie, basis = [1, t, t^2, ... t^p], where t refers to time point.
        # size: (expansion_coefficient_dim, target_length)
        self._basis = paddle.stack(
            [
                (paddle.arange(target_length) / target_length) ** i
                for i in range(expansion_coefficient_dim)
            ],
            axis=1,
        ).T

    def forward(
            self,
            x: paddle.Tensor,
            ) -> paddle.Tensor:
        """
        Forward trend generator.

        Args:
            x: A tensor of the output of paddle.nn.Linear, the fitted result of the coefficients of the polynomial, ie, x = [theta_0, theta_1, theta_2, ..., theta_p].

        Returns:
            The value of the polynomial: x * basis, the fitted result of trend.
        """
        return paddle.matmul(x, self._basis)


class _SeasonalityGenerator(nn.Layer):
    """
    Seasonality generator, implemented by harmonic function.

    Args:
        target_length: forecasting horizon.
    """
    def __init__(
            self,
            target_length: int,
            ):
        super().__init__()
        half_minus_one = int(target_length / 2 - 1)
        cos_vectors = [
            paddle.cos(paddle.arange(target_length) / target_length * 2 * np.pi * i)
            for i in range(1, half_minus_one + 1)
        ]
        sin_vectors = [
            paddle.sin(paddle.arange(target_length) / target_length * 2 * np.pi * i)
            for i in range(1, half_minus_one + 1)
        ]
        # basis is a vector composed of a series of sin/cos functions, ie, basis = [1, cos(2*pi*t), cos(2*pi*2t),... sin(2*pi*t), sin(2*pi*2t), ...].
        # size: (2 * int(target_length / 2 - 1) + 1, target_length)
        self._basis = paddle.stack(
            [paddle.ones([target_length])] + cos_vectors + sin_vectors, axis=1
        ).T

    def forward(
            self,
            x: paddle.Tensor
            ) -> paddle.Tensor:
        """
        Forward seasonality generator.

        Args:
            x: A tensor of the output of paddle.nn.Linear, the fitted result of the coefficient of harmonic function.

        Returns:
            The value of the harmonic function: x * basis, as the fitted result of seasonality.
        """
        return paddle.matmul(x, self._basis)


class _Block(nn.Layer):
    """
    Block implementation, basic component of N-Beats.The blocks produce outputs of size (target_length, target_dim); i.e."one vector per dimension". The parameters are predicted only for forecast outputs. Backcast outputs are in the original "domain".

    Args:
        num_layers: The number of fully connected layers preceding the final forking layers.
        layer_width: The number of neurons that make up each fully connected layer.
        expansion_coefficient_dim: The dimensionality of the waveform generator parameters,
            also known as expansion coefficients, used in the generic architecture and the
            trend module of the interpretable architecture, where it determines the degree
            of the polynomial basis.
        backcast_length: The length of the input sequence fed to the model.
        in_chunk_len: The length of the flatten input sequence fed to the model.
        target_length: The length of the forecast of the model.
        target_dim: The dimension of target
        g_type: The type of function that is implemented by the waveform generator.
    """
    def __init__(
        self,
        num_layers: int,
        layer_width: int,
        expansion_coefficient_dim: int,
        backcast_length: int,
        in_chunk_len: int,
        target_length: int,
        target_dim: int,
        g_type: GTypes,
    ):
        super().__init__()
        self._target_dim = target_dim
        self._relu = nn.ReLU()
        # fully connected stack before fork
        self._linear_layer_stack_list = [nn.Linear(in_chunk_len, layer_width)]
        self._linear_layer_stack_list += [
            nn.Linear(layer_width, layer_width) for _ in range(num_layers - 1)
        ]
        self._fc_stack = nn.LayerList(self._linear_layer_stack_list)

        # Fully connected layer producing forecast/backcast expansion coeffcients (waveform generator parameters).
        # The coefficients are emitted for each parameter of the likelihood.
        if g_type == _GType.SEASONALITY:
            self._backcast_linear_layer = nn.Linear(
                layer_width,  (2 * int(backcast_length / 2 - 1) + 1) * target_dim
            )
            self._forecast_linear_layer = nn.Linear( 
                layer_width, (2 * int(target_length / 2 - 1) + 1) * target_dim
            )
        else:
            self._backcast_linear_layer = nn.Linear(
                layer_width, expansion_coefficient_dim * target_dim
            )
            self._forecast_linear_layer = nn.Linear( 
                layer_width, expansion_coefficient_dim * target_dim
            )

        # waveform generator functions
        if g_type == _GType.GENERIC:
            self._backcast_g = nn.Linear(expansion_coefficient_dim, backcast_length)
            self._forecast_g = nn.Linear(expansion_coefficient_dim, target_length)

        elif g_type == _GType.TREND:
            self._backcast_g = _TrendGenerator(expansion_coefficient_dim, backcast_length)
            self._forecast_g = _TrendGenerator(expansion_coefficient_dim, target_length)

        elif g_type == _GType.SEASONALITY:
            self._backcast_g = _SeasonalityGenerator(backcast_length)
            self._forecast_g = _SeasonalityGenerator(target_length)
        else:
            raise ValueError("g_type not supported")

    def forward(
            self,
            backcast: paddle.Tensor,
            known_cov: paddle.Tensor,
            observed_cov: paddle.Tensor
            ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """
        Forward block.

        Args:
            backcast: past target, shape: [batch_size, in_chunk_len, target_dim]
            known_cov: known covariates, shape: [batch_size, in_chunk_len + target_length, known_cov_dim]
            observed_cov: observed covariates, shape: [batch_size, in_chunk_len, observed_cov_dim]

        Returns:
            x_hat: approximation of backcast, shape [batch_size, in_chunk_len, target_dim]
            y_hat: tensor containing the forward forecast of the block, shape [batch_size, out_chunk_len, target_dim]
        """
        batch_size = paddle.shape(backcast)[0]
        # concat backcast, known_cov, observed_cov if any
        feature = [backcast.reshape((batch_size, -1))]
        if known_cov is not None:
            feature.append(known_cov.reshape((batch_size, -1)))
        if observed_cov is not None:
            feature.append(observed_cov.reshape((batch_size, -1)))

        x = paddle.concat(x=feature, axis=1)
        # fully connected layer stack
        for n, layer in enumerate(self._linear_layer_stack_list):
            x = self._relu(layer(x))

        # forked linear layers producing waveform generator parameters
        theta_backcast = self._backcast_linear_layer(x)
        theta_forecast = self._forecast_linear_layer(x)

        # set the expansion coefs in last dimension for the forecasts
        theta_forecast = theta_forecast.reshape([batch_size, self._target_dim, -1])
        theta_backcast = theta_backcast.reshape([batch_size, self._target_dim, -1])

        # waveform generator applications (project the expansion coefs onto basis vectors)
        x_hat = self._backcast_g(theta_backcast)
        y_hat = self._forecast_g(theta_forecast)

        # to keep waveform, cannot use reshape here
        y_hat = paddle.transpose(y_hat, [0, 2, 1])
        x_hat = paddle.transpose(x_hat, [0, 2, 1])

        return x_hat, y_hat


class _Stack(nn.Layer):
    """
    Stack implementation of the N-BEATS architecture, comprises multiple basic blocks.

    Args:
        num_blocks: The number of blocks making up this stack.
        num_layers: The number of fully connected layers preceding the final forking layers in each block.
        layer_width: The number of neurons that make up each fully connected layer in each block.
        expansion_coefficient_dim: The dimensionality of the waveform generator parameters, also known as expansion coefficients.
        backcast_length: The length of the input sequence fed to the model.
        in_chunk_len: The length of the flatten input sequence fed to the model.
        target_length: The length of the forecast of the model.
        target_dim: The dimension of target.
        g_type: The function that is implemented by the waveform generators in each block.
    """
    def __init__(
        self,
        num_blocks: int,
        num_layers: int,
        layer_width: int,
        expansion_coefficient_dim: int,
        backcast_length: int,
        in_chunk_len: int,
        target_length: int,
        target_dim: int,
        g_type: GTypes,
    ):
        super().__init__()
        self._target_length = target_length
        self._target_dim = target_dim

        if g_type == _GType.GENERIC:
            # different block instance for generic stack
            self._blocks_list = [
                _Block(
                    num_layers,
                    layer_width,
                    expansion_coefficient_dim,
                    backcast_length,
                    in_chunk_len,
                    target_length,
                    target_dim,
                    g_type,
                )
                for _ in range(num_blocks)
            ]
        else:
            # same block instance is used for weight sharing
            interpretable_block = _Block(
                num_layers,
                layer_width,
                expansion_coefficient_dim,
                backcast_length,
                in_chunk_len,
                target_length,
                target_dim,
                g_type,
            )
            self._blocks_list = [interpretable_block] * num_blocks
        self._blocks = nn.LayerList(self._blocks_list)

    def forward(
            self,
            backcast: paddle.Tensor,
            known_cov: paddle.Tensor,
            observed_cov: paddle.Tensor
            ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """
        Forward stack.

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
            shape=(backcast.shape[0], self._target_length, self._target_dim),
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


class _NBEATSModule(nn.Layer):
    """
    NBeats implementation, upgrade the origin version(univariate, no covarites) to more general version(multi-targets, known covariates, observed covariates).
    """
    def __init__(
        self,
        # general params:
        in_chunk_len: int,
        out_chunk_len: int,
        target_dim: int,
        known_cov_dim: int,
        observed_cov_dim: int,
        generic_architecture: bool,
        num_stacks: int,
        num_blocks: Union[int, List[int]],
        num_layers: int,
        layer_widths: Union[int, List[int]],
        expansion_coefficient_dim: int,
        trend_polynomial_degree: int,
    ):
        """
        Args:
            in_chunk_len(int): The length of the input sequence fed to the model.
            out_chunk_len(int): The length of the forecast of the model.
            target_dim(int): The number of targets to be forecasted.
            known_cov_dim(int): The number of known covariates.
            observed_cov_dim(int): The number of observed covariates.
            generic_architecture(bool): Boolean value indicating whether the generic architecture of N-BEATS is used. If not, the interpretable architecture outlined in the paper (consisting of one trend and one seasonality stack with appropriate waveform generator functions).
            num_stacks(int): The number of stacks that make up the whole model. Only used if `generic_architecture` is set to `True`.
            num_blocks(Union[int, List[int]]): The number of blocks making up each stack. If a list is passed, it must have a length equal to `num_stacks` and every entry in that list corresponds to the corresponding stack. If an integer is passed, every stack will have the same number of blocks.
            num_layers(int): The number of fully connected layers preceding the final forking layers in each block of every stack. Only used if `generic_architecture` is set to `True`.
            layer_widths(Union[int, List[int]]): Determines the number of neurons that make up each fully connected layer in each block of every stack. If a list is passed, it must have a length equal to `num_stacks` and every entry in that list corresponds to the layer width of the corresponding stack. If an integer is passed, every stack will have blocks with FC layers of the same width.
            expansion_coefficient_dim(int): The dimensionality of the waveform generator parameters, also known as expansion coefficients. Only used if `generic_architecture` is set to `True`.
            trend_polynomial_degree(int): The degree of the polynomial used as waveform generator in trend stacks. Only used if `generic_architecture` is set to `False`.
        """
        super().__init__()
        self._in_chunk_len = in_chunk_len
        self._out_chunk_len = out_chunk_len
        self._target_dim = target_dim
        self._known_cov_dim = known_cov_dim
        self._observed_cov_dim = observed_cov_dim
        self._input_dim = target_dim + known_cov_dim + observed_cov_dim
        self._in_chunk_len_multi = self._in_chunk_len * self._input_dim + \
        self._out_chunk_len * self._known_cov_dim
        self._target_length = self._out_chunk_len
        # generic architecture
        if generic_architecture:
            self._stacks_list = [
                _Stack(
                    num_blocks[i],
                    num_layers,
                    layer_widths[i],
                    expansion_coefficient_dim,
                    self._in_chunk_len,
                    self._in_chunk_len_multi,
                    self._target_length,
                    self._target_dim,
                    _GType.GENERIC,
                )
                for i in range(num_stacks)
            ]
        # interpretable architecture
        else:
            num_stacks = 2
            trend_stack = _Stack(
                num_blocks[0],
                num_layers,
                layer_widths[0],
                trend_polynomial_degree + 1,
                self._in_chunk_len,
                self._in_chunk_len_multi,
                self._target_length,
                self._target_dim,
                _GType.TREND,
            )
            seasonality_stack = _Stack(
                num_blocks[1],
                num_layers,
                layer_widths[1],
                -1, # no need to set in seasonality stack
                self._in_chunk_len,
                self._in_chunk_len_multi,
                self._target_length,
                self._target_dim,
                _GType.SEASONALITY,
            )
            self._stacks_list = [trend_stack, seasonality_stack]

        self._stacks = nn.LayerList(self._stacks_list)

        # setting the last backcast "branch" to be not trainable (without next block/stack, it doesn't need to be
        # backpropagated). Removing this lines would cause logtensorboard to crash, since no gradient is stored
        # on this params (the last block backcast is not part of the final output of the net).
        self._stacks_list[-1]._blocks[-1]._backcast_linear_layer.stop_gradient = True
        self._stacks_list[-1]._blocks[-1]._backcast_g.stop_gradient = True

    def forward(
            self,
            data: Dict[str, paddle.Tensor]
            ) -> paddle.Tensor:
        """
        Forward NBeats network.

        Args:
            data(Dict[str, paddle.Tensor]): a dict specifies all kinds of input data

        Returns:
            forecast(padle.Tensor): Tensor containing the output of the NBEATS model.
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


class NBEATSModel(PaddleBaseModelImpl):
    """
    Implementation of NBeats model.

    Args:
        in_chunk_len(int): The size of the loopback window, i.e., the number of time steps feed to the model.
        out_chunk_len(int): The size of the forecasting horizon, i.e., the number of time steps output by the model.
        generic_architecture(bool, Optional): Boolean value indicating whether the generic architecture of N-BEATS is used. If not, the interpretable architecture outlined in the paper (consisting of one trend and one seasonality stack with appropriate waveform generator functions).
        num_stacks(int, Optional): The number of stacks that make up the whole model. Only used if `generic_architecture` is set to `True`.
        num_blocks(Union[int, List[int]], Optional): The number of blocks making up each stack. If a list is passed, it must have a length equal to `num_stacks` and every entry in that list corresponds to the corresponding stack. If an integer is passed, every stack will have the same number of blocks.
        num_layers(int, Optional): The number of fully connected layers preceding the final forking layers in each block of every stack. Only used if `generic_architecture` is set to `True`.
        layer_widths(Union[int, List[int]], Optional): Determines the number of neurons that make up each fully connected layer in each block of every stack. If a list is passed, it must have a length equal to `num_stacks` and every entry in that list corresponds to the layer width of the corresponding stack. If an integer is passed, every stack will have blocks with FC layers of the same width.
        expansion_coefficient_dim(int, Optional): The dimensionality of the waveform generator parameters, also known as expansion coefficients. Only used if `generic_architecture` is set to `True`.
        trend_polynomial_degree(int, Optional): The degree of the polynomial used as waveform generator in trend stacks. Only used if `generic_architecture` is set to `False`.
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
        generic_architecture: bool = True,
        num_stacks: int = 2,
        num_blocks: Union[int, List[int]] = 3,
        num_layers: int = 4,
        layer_widths: Union[int, List[int]] = 128,
        expansion_coefficient_dim: int = 128,
        trend_polynomial_degree: int = 4,
        skip_chunk_len: int = 0,
        sampling_stride: int = 1,
        loss_fn: Callable[..., paddle.Tensor] = F.mse_loss,
        optimizer_fn: Callable[..., Optimizer] = paddle.optimizer.Adam,
        optimizer_params: Dict[str, Any] = dict(learning_rate=1e-4), 
        use_revin: bool = False,
        revin_params: Dict[str, Any] = dict(eps=1e-5, affine=True),
        eval_metrics: List[str] = [], 
        callbacks: List[Callback] = [], 
        batch_size: int = 32,
        max_epochs: int = 10,
        verbose: int = 1,
        patience: int = 10,
        seed: int = 0 
    ):
        self._generic_architecture = generic_architecture
        self._num_stacks = num_stacks
        self._num_blocks = num_blocks
        self._num_layers = num_layers
        self._layer_widths = layer_widths
        self._expansion_coefficient_dim = expansion_coefficient_dim
        self._trend_polynomial_degree = trend_polynomial_degree
        self._use_revin = use_revin
        self._revin_params = revin_params
        # If not using general architecture, for interpretable purpose, number of stacks is forced to be 2, for trend and seasonality implementation
        if not self._generic_architecture:
            self._num_stacks = 2
        if isinstance(self._num_blocks, int):
            self._num_blocks = [self._num_blocks] * self._num_stacks
        if isinstance(self._layer_widths, int):
            self._layer_widths = [self._layer_widths] * self._num_stacks

        super(NBEATSModel, self).__init__(
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
        Check validation of parameters.
        """
        raise_if(
            isinstance(self._num_blocks, list) and len(self._num_blocks) != self._num_stacks,
            "Stack number should be equal to the length of the List: num_blocks."
        )
        raise_if(
            isinstance(self._layer_widths, list) and len(self._layer_widths) != self._num_stacks,
            "Stack number should be equal to the length of the List: layer_widths."
        )
        super(NBEATSModel, self)._check_params()

    def _check_tsdataset(
        self,
        tsdataset: TSDataset
    ):
        """ 
        Rewrite _check_tsdataset to fit the specific model.
        For NBeats, all data variables are expected to be float32.
        """
        for column, dtype in tsdataset.dtypes.items():
            raise_if_not(
                np.issubdtype(dtype, np.floating),
                f"nbeats variables' dtype only supports [float16, float32, float64], " \
                f"but received {column}: {dtype}."
            )
        super(NBEATSModel, self)._check_tsdataset(tsdataset)

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
        fit_params = {
                "target_dim": train_tsdataset[0].get_target().data.shape[1],
                "known_cov_dim": 0,
                "observed_cov_dim": 0
                }
        if train_tsdataset[0].get_known_cov() is not None:
            fit_params["known_cov_dim"] = train_tsdataset[0].get_known_cov().data.shape[1]
        if train_tsdataset[0].get_observed_cov() is not None:
            fit_params["observed_cov_dim"] = train_tsdataset[0].get_observed_cov().data.shape[1]
        return fit_params

    @revin_norm
    def _init_network(self) -> paddle.nn.Layer:
        """
        Init network.

        Returns:
            paddle.nn.Layer
        """
        return _NBEATSModule(
            self._in_chunk_len,
            self._out_chunk_len,
            self._fit_params["target_dim"],
            self._fit_params["known_cov_dim"],
            self._fit_params["observed_cov_dim"],
            self._generic_architecture,
            self._num_stacks,
            self._num_blocks,
            self._num_layers,
            self._layer_widths,
            self._expansion_coefficient_dim,
            self._trend_polynomial_degree
        )
