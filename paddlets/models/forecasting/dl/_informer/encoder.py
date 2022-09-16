#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from typing import Optional, Dict, Any

import paddle

from paddlets.models.forecasting.dl._informer.attention import ProbSparseAttention


class ConvLayer(paddle.nn.Layer):
    """Paddle layer implementing conv layer.

    Args:
        d_model(int): The expected feature size for the input/output of the informer's encoder/decoder.

    Attributes:
        _config(Dict[str, Any]): Dict of parameter setting.
        _maxpool(paddle.nn.Layer): 1D max pooling.
        _norm(paddle.nn.Layer): Batch normalization.
        _activation(paddle.nn.Layer): ELU Activation.
        _downconv(paddle.nn.Layer): 1D convolution.
    """
    def __init__(self, d_model: int):
        self._config = locals()
        self._config.pop("self")
        self._config.pop("__class__")
        super(ConvLayer, self).__init__()
        self._maxpool = paddle.nn.MaxPool1D(3, 2, 1)
        self._norm = paddle.nn.BatchNorm1D(d_model)
        self._activation = paddle.nn.ELU()
        self._downconv = paddle.nn.Conv1D(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            padding_mode="circular"
        )

    def forward(self, src: paddle.Tensor) -> paddle.Tensor:
        """Forward.

        Args:
            src(paddle.Tensor): Feature tensor.

        Returns:
            paddle.Tensor: Output of Layer.
        """
        out = paddle.transpose(src, perm=[0, 2, 1])
        out = self._downconv(out)
        out = self._norm(out)
        out = self._activation(out)
        out = self._maxpool(out)
        out = paddle.transpose(out, perm=[0, 2, 1])
        return out


class InformerEncoderLayer(paddle.nn.Layer):
    """Paddle layer implementing informer encoder layer.

    Args:
        d_model(int): The expected feature size for the input/output of the informer's encoder/decoder.
        num_heads(int): The number of heads in multi-head attention.
        ffn_channels(int): The Number of channels for Conv1D of FFN layer.
        dropout_rate(float): The dropout probability used on attention 
            weights to drop some attention targets.

    Attributes:
        _config(Dict[str, Any]): Dict of parameter setting.
        _attn(paddle.nn.Layer): Probability sparse attention layer.
        _conv1(paddle.nn.Layer): Conv1D of FFT layer.
        _conv2paddle.nn.Layer): Conv1D of FFT layer.
        _norm(paddle.nn.Layer): Batch normalization.
        _dropout(paddle.nn.Layer): The dropout layer.
        _activation(paddle.nn.Layer): ELU Activation.
    """
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        ffn_channels: int,
        activation: str,
        dropout_rate: float
    ):
        self._config = locals()
        self._config.pop("self")
        self._config.pop("__class__")
        super(InformerEncoderLayer, self).__init__()
        self._attn = ProbSparseAttention(d_model, d_model, d_model, num_heads, dropout_rate)
        self._conv1 = paddle.nn.Conv1D(d_model, ffn_channels, 1)
        self._conv2 = paddle.nn.Conv1D(ffn_channels, d_model, 1)
        self._norm = paddle.nn.LayerNorm(d_model)
        self._dropout = paddle.nn.Dropout(dropout_rate)
        self._activation = (
            paddle.nn.GELU() if activation == "gelu" else paddle.nn.ReLU()
        )

    def forward(
        self, 
        src: paddle.Tensor, 
        src_mask: Optional[paddle.Tensor] = None
    ) -> paddle.Tensor:
        """Forward.

        Args:
            src(paddle.Tensor): The input of Informer encoder.
            src_mask(paddle.Tensor|None): A tensor used in multi-head attention 
                to prevent attention to some unwanted positions.
        
        Returns:
            paddle.Tensor: Output of layer. 
        """
        # src: [batch_size, in_chunk_len, d_model]
        out = residual = self._attn(src, src, src, src_mask)
        out = paddle.transpose(out, perm=[0, 2, 1])
        out = self._conv1(out)
        out = self._activation(out)
        out = self._dropout(out)

        out = self._conv2(out)
        out = paddle.transpose(out, perm=[0, 2, 1])
        out = self._dropout(out)
        return self._norm(out + residual)


class InformerEncoder(paddle.nn.Layer):
    """Paddle layer implementing informer encoder.

    Args:
        encoder_layer(paddle.nn.Layer): An instance of the `InformerEncoderLayer`. It
            would be used as the first layer, and the other layers would be created
            according to the configurations of it.
        conv_layer(paddle.nn.Layer): An instance of the `ConvLayer`. It
            would be used as the first layer, and the other layers would be created
            according to the configurations of it.
        num_layers(int): The number of InformerEncoderLayer/ConvLayer layers to be stacked.

    Attributes:
        _encoder_layers(paddle.nn.LayerList): A stacked LayerList containing InformerEncoderLayer.
        _conv_layers(paddle.nn.LayerList): A stacked LayerList containing ConvLayer.
    """
    def __init__(
        self,
        encoder_layer: paddle.nn.Layer,
        conv_layer: paddle.nn.Layer,
        num_layers: int,
    ):
        super(InformerEncoder, self).__init__()
        self._encoder_layers = paddle.nn.LayerList(
            [type(encoder_layer)(**encoder_layer._config) for _ in range(num_layers)]
        )
        self._conv_layers = paddle.nn.LayerList(
            [type(conv_layer)(**conv_layer._config) for _ in range(num_layers - 1)]
        )

    def forward(
        self, 
        src: paddle.Tensor, 
        src_mask: Optional[paddle.Tensor] = None
    ) -> paddle.Tensor:
        """Forward.

        Args:
            src(paddle.Tensor): The input of Informer encoder.
            src_mask(paddle.Tensor|None): A tensor used in multi-head attention 
                to prevent attention to some unwanted positions.
        
        Returns:
            paddle.Tensor: Output of layer. 
        """
        # src [batch_size, in_chunk_len, d_model]
        for enc_layer, conv_layer in zip(self._encoder_layers[:-1], self._conv_layers):
            src = enc_layer(src, src_mask)
            src = conv_layer(src)
        src = self._encoder_layers[-1](src, src_mask)
        return src
