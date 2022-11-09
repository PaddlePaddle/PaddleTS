#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from typing import List, Tuple

import paddle.nn.functional as F
import numpy as np
import paddle

from paddlets.models.representation.dl._cost.mask import (
    generate_binomial_mask,
    generate_true_mask,
    paddle_mask_fill
)


class SamePadConv(paddle.nn.Layer):
    """Paddle layer implementing Same Padding Convolution layer.

    Args:
        in_channels(int): The number of channels in the input series.
        out_channels(int): The number of channels in the output series. 
        kernel_size(int): The filter size.
        dilation(int): The dilation size.

    Attributes:
        _conv(paddle.nn.Layer): The conv1d layer.
        _remove(int): The tag whether to delete the last time step.
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int, 
        dilation: int, 
    ):
        super(SamePadConv, self).__init__()
        receptive_field = (kernel_size - 1) * dilation + 1
        padding = receptive_field // 2
        k = np.sqrt(1. / (in_channels * kernel_size))
        weight_attr = bias_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Uniform(-k, k)
        )
        self._conv = paddle.nn.Conv1D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            weight_attr=weight_attr,
            bias_attr=bias_attr,
        )
        self._remove = (
            1 if receptive_field % 2 == 0 else 0
        )

    def forward(self, X: paddle.Tensor) -> paddle.Tensor:
        """Forward.

        Args:
            X(paddle.Tensor): Feature tensor.

        Returns:
            paddle.Tensor:  Output of Layer.
        """
        out = self._conv(X)
        if self._remove > 0:
            out = out[:, :, :-self._remove]
        return out


class ConvLayer(paddle.nn.Layer):
    """Paddle layer implementing ConvLayer.
    
    Args:
        in_channels(int): The number of channels in the input series.
        out_channels(int): The number of channels in the output series. 
        kernel_size(int): The filter size.
        dilation(int): The dilation size.
        final(bool): The tag whether it is an output layer.
    
    Attributes:
        _conv1(paddle.nn.Layer): The conv1d layer.
        _conv2(paddle.nn.Layer): The conv1d layer.
        _out_proj(paddle.nn.Layer): The output projection layer.

    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        final: bool
    ):
        super(ConvLayer, self).__init__()
        self._conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation)
        self._conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation)
        k = np.sqrt(1. / (in_channels * 1))
        weight_attr = bias_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Uniform(-k, k)
        )
        self._out_proj = (
            paddle.nn.Conv1D(
                in_channels, out_channels, 1, 
                weight_attr=weight_attr, 
                bias_attr=bias_attr
            ) if (in_channels != out_channels or final) else None
        )

    def forward(self, X: paddle.Tensor) -> paddle.Tensor:
        """Forward.

        Args:
            X(paddle.Tensor): Feature tensor.

        Returns:
            paddle.Tensor: Output of Layer.
        """
        residual = (
            self._out_proj(X) if self._out_proj else X
        )
        out = F.gelu(X)
        out = self._conv1(out)
        out = F.gelu(out)
        out = self._conv2(out)
        return residual + out


class DilatedConvLayer(paddle.nn.Layer):
    """Paddle layer implementing DilatedConvLayer.
    
    Args:
        in_channels(int): The number of channels in the input series.
        out_channels(int): The number of channels in the output series. 
        hidden_channels(int): The number of channels in the hidden layer.
        kernel_size(int): The filter size.
        num_layers(int): The number of `ConvLayer` to be stacked.

    Attributes:
        _conv_layers(paddle.nn.Layer): A stacked LayerList containing `ConvLayer`.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        kernel_size: int,
        num_layers: int
    ):
        super(DilatedConvLayer, self).__init__()
        conv_layers = [] 
        channels = [in_channels] + [hidden_channels] * num_layers + [out_channels]
        for k, (in_channels, out_channels) in \
            enumerate(zip(channels[:-1], channels[1:])):
            conv_layer = ConvLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=2 ** k,
                final=(k == num_layers)
            )
            conv_layers.append(conv_layer)
        self._conv_layers = paddle.nn.Sequential(*conv_layers)

    def forward(self, X: paddle.Tensor) -> paddle.Tensor:
        """Forward.

        Args:
            X(paddle.Tensor): Feature tensor.

        Returns:
            paddle.Tensor: Output of Layer.
        """
        return self._conv_layers(X)


class TFDLayer(paddle.nn.Layer):
    """Paddle layer implementing trend feature disentangler.

    Args:
        in_channels(int): The number of channels in the input series.
        out_channels(int): The number of channels in the output series. 
        kernels(List): A list containing a set of convolution kernel size.
    
    Attributes:
        _tfd(paddle.nn.LayerList): A list containing a set of convolution layer.
        _kernels(List): A list containing a set of convolution kernel size.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernels: List[int],
    ):
        super(TFDLayer, self).__init__()
        conv_layers = [
            paddle.nn.Conv1D(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=k,
            ) for k in kernels
        ]
        self._tfd = paddle.nn.LayerList(conv_layers)
        self._kernels = kernels

    def forward(self, X):
        """Forward.
        
        Args:
            X(paddle.Tensor): The output of CoST's feature extractor.

        Returns:
            paddle.Tensor: Output of Layer.
        """
        trends = []
        for idx, layer in enumerate(self._tfd):
            if self._kernels[idx] != 1:
                padding = (self._kernels[idx] - 1)
                out = F.pad(
                    X, (padding, 0), data_format="NCL"
                )
                trends.append(layer(out))
                continue
            trends.append(layer(X))
        trends = paddle.stack(trends, axis=0)
        trends = paddle.mean(trends, axis=0)
        return trends


class SFDLayer(paddle.nn.Layer):
    """Paddle layer implementing seasonal feature disentangler.

    Args:
        in_channels(int): The number of channels in the input series.
        out_channels(int): The number of channels in the output series. 
        seq_len(int): The size of time series segment
    
    Attributes:
        _weight(paddle.nn.Tensor): The trainable parameter.
        _bias(paddle.nn.Tensor): The trainable parameter.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        seq_len: int
    ):
        super(SFDLayer, self).__init__()
        freqs = (seq_len // 2 + 1)
        from paddle.nn.initializer import KaimingUniform
        weight_shape = [freqs, in_channels, out_channels, 2]
        bias_shape = [freqs, out_channels, 2]
        weight_attr = paddle.ParamAttr(initializer=KaimingUniform())
        self._weight = self.create_parameter(weight_shape, weight_attr)
        self._bias = self.create_parameter(bias_shape, is_bias=True)

    def forward(self, X):
        """Forward.
        
        Args:
            X(paddle.Tensor): The output of CoST's feature extractor.

        Returns:
            paddle.Tensor: Output of Layer.
        """
        batch_size, _, seq_len = X.shape
        out = paddle.fft.rfft(X, axis=-1)           # [B, C, L // 2 + 1]
        out = paddle.transpose(out, perm=[2, 0, 1]) # [L // 2 + 1, B, C]
        out = paddle.matmul(out, paddle.as_complex(self._weight))
        out = paddle.transpose(out, perm=[1, 0, 2])
        out = out + paddle.as_complex(self._bias)
        out = paddle.transpose(out, perm=[0, 2, 1])
        out = paddle.fft.irfft(out, seq_len, axis=-1)
        return out


class TSEncoder(paddle.nn.Layer):
    """Paddle layer implementing TSEncoder.

    Args:
        in_channels(int): The number of channels in the input series.
        out_channels(int): The number of channels in the output series. 
        hidden_channels(int): The number of channels in the hidden layer.
        num_layers(int): The number of `ConvLayer` to be stacked.
        seq_len(int): The time series length.

    Attributes:
        _in_proj(paddle.nn.Layer): The input projection layer.
        _repr_dropout(paddle.nn.Layer): The dropout layer.
        _dilated_conv(paddle.nn.Layer): A stacked LayerList containing `ConvLayer`.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        num_layers: int,
        seq_len: int
    ):
        super(TSEncoder, self).__init__()
        k = np.sqrt(1. / in_channels)
        weight_attr = bias_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Uniform(-k, k)
        )
        self._in_proj = paddle.nn.Linear(
            in_channels, hidden_channels,
            weight_attr=weight_attr,
            bias_attr=bias_attr,
        )
        self._repr_dropout = paddle.nn.Dropout(0.1)
        self._dilated_conv = DilatedConvLayer(
            in_channels=hidden_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            kernel_size=3
        )

        kernels = [1, 2, 4, 8, 16, 32, 64, 128]
        self._tfd = TFDLayer(out_channels, out_channels // 2, kernels)
        self._sfd = SFDLayer(out_channels, out_channels // 2, seq_len)

    def forward(
        self, 
        X: paddle.Tensor, 
        mask: str,
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """Forward. 
        Args:
            X(paddle.Tensor): Feature tensor.
            mask(str): The mask type, ["binomial", "all_true"] is optional.

        Returns:
            paddle.Tensor: The trend of time series. 
            paddle.Tensor: The season of time series. 
        """
        batch_size, seq_len, _ = X.shape
        nan_mask = paddle.any(paddle.isnan(X), -1)
        X = paddle_mask_fill(X, nan_mask, 0)

        out = self._in_proj(X)
        if mask == "binomial":
            mask = generate_binomial_mask(batch_size, seq_len)
        elif mask == "all_true":
            mask = generate_true_mask(batch_size, seq_len)
        
        mask &= (~nan_mask)
        out = paddle_mask_fill(out, ~mask, 0)

        out = paddle.transpose(out, perm=[0, 2, 1])
        out = self._dilated_conv(out)
        trend = self._tfd(out)
        season = self._sfd(out)
        trend = paddle.transpose(trend, perm=[0, 2, 1])
        season = paddle.transpose(season, perm=[0, 2, 1])
        return trend, self._repr_dropout(season)
