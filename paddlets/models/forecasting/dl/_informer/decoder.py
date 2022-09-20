#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from typing import Optional

import paddle

from paddlets.models.forecasting.dl._informer.attention import ProbSparseAttention
from paddlets.models.forecasting.dl._informer.attention import CrossAttention


class InformerDecoderLayer(paddle.nn.Layer):
    """Paddle layer implementing informer decoder layer.

    Args:
        d_model(int): The expected feature size for the input/output of the informer's encoder/decoder.
        num_heads(int): The number of heads in multi-head attention.
        ffn_channels(int): The Number of channels for Conv1D of FFN layer.
        dropout_rate(float): The dropout probability used on attention
            weights to drop some attention targets.

    Attributes:
        _config(Dict[str, Any]): Dict of parameter setting.
        _selfAttn(paddle.nn.Layer): Prob sparse attention layer.
        _crossAttn(paddle.nn.Layer): Cross attention layer.
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
        dropout_rate: float
    ):
        self._config = locals()
        self._config.pop("self")
        self._config.pop("__class__")
        super(InformerDecoderLayer, self).__init__()
        self._selfAttn = ProbSparseAttention(d_model, d_model, d_model, num_heads, dropout_rate)
        self._crossAttn = CrossAttention(d_model, d_model, d_model, num_heads, dropout_rate)
        self._conv1 = paddle.nn.Conv1D(d_model, ffn_channels, 1)
        self._conv2 = paddle.nn.Conv1D(ffn_channels, d_model, 1)
        self._norm = paddle.nn.LayerNorm(d_model)
        self._dropout = paddle.nn.Dropout(dropout_rate)
        self._activation = paddle.nn.GELU()

    def forward(
        self,
        tgt: paddle.Tensor, 
        memory: paddle.Tensor,
        tgt_mask: Optional[paddle.Tensor] = None,
        memory_mask: Optional[paddle.Tensor] = None
    ) -> paddle.Tensor:
        """Forward.

        Args:
            tgt(paddle.Tensor): The output of Informer decoder.
            memory(paddle.Tensor): The output of Informer encoder.
            tgt_mask(paddle.Tensor|None): A tensor used in multi-head attention
                to prevents attention to some unwanted positions.
            memory_mask(paddle.Tensor|None): A tensor used in decoder-encoder
                cross attention to prevents attention to some unwanted positions.

        Returns:
            paddle.Tensor: Output of layer.
        """
        out = self._selfAttn(tgt, tgt, tgt, tgt_mask)
        out = residual = self._crossAttn(out, memory, memory, memory_mask)

        out = paddle.transpose(out, perm=[0, 2, 1])
        out = self._conv1(out)
        out = self._activation(out)
        out = self._dropout(out)

        out = self._conv2(out)
        out = paddle.transpose(out, perm=[0, 2, 1])
        out = self._dropout(out)
        return self._norm(residual + out)


class InformerDecoder(paddle.nn.Layer):
    """Paddle layer implementing informer decoder.

    Args:
        decoder_layer(paddle.nn.Layer): An instance of the `InformerDecoderLayer`. It
            would be used as the first layer, and the other layers would be created
            according to the configurations of it.
        num_layers(int): The number of InformerDecoderLayer layers to be stacked.

    Attributes:
        _decoder_layers(paddle.nn.LayerList): A stacked LayerList containing InformerDecoderLayer.
    """
    def __init__(
        self,
        decoder_layer: paddle.nn.Layer,
        num_layers: int,
    ):
        super(InformerDecoder, self).__init__()
        self._decoder_layers = paddle.nn.LayerList(
            [type(decoder_layer)(**decoder_layer._config) for _ in range(num_layers)]
        )

    def forward(
        self,
        tgt: paddle.Tensor, 
        memory: paddle.Tensor,
        tgt_mask: Optional[paddle.Tensor] = None,
        memory_mask: Optional[paddle.Tensor] = None
    ) -> paddle.Tensor:
        """Forward.

        Args:
            tgt(paddle.Tensor): The input of Informer decoder.
            memory(paddle.Tensor): The output of Informer encoder.
            tgt_mask(paddle.Tensor|None): A tensor used in multi-head attention
                to prevents attention to some unwanted positions.
            memory_mask(paddle.Tensor|None): A tensor used in decoder-encoder
                cross attention to prevents attention to some unwanted positions.

        Returns:
            paddle.Tensor: Output of layer. 
        """
        for dec_layer in self._decoder_layers:
            tgt = dec_layer(tgt, memory, tgt_mask, memory_mask)
        return tgt
