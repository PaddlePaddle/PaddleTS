#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from typing import List, Dict, Any, Callable, Optional

import paddle.nn.functional as F
import paddle


class EncoderLayer(paddle.nn.Layer):
    """
    EncoderLayer in anomaly transformer.
    
    Args:
        attention(Callable[..., paddle.Tensor]): The attention in encoderlayer.
        d_model(int): The expected feature size for the input of the anomaly transformer.
        d_ff(int): The Number of channels for FFN layers.
        dropout(float): Dropout regularization parameter.
        activation(Callable[..., paddle.Tensor]): The activation function for the EncoderLayer, defalut: F.gelu.
        
    Attributes:
        _nn(paddle.nn.Sequential): Dynamic graph LayerList. 
    """
    def __init__(
        self, 
        attention: Callable[..., paddle.Tensor], 
        d_model: int, 
        d_ff: int = None, 
        dropout: float = 0.1, 
        activation: Callable[..., paddle.Tensor] = F.gelu
    ):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = paddle.nn.Conv1D(in_channels=d_model, out_channels=d_ff, kernel_size=1, data_format="NLC")
        self.conv2 = paddle.nn.Conv1D(in_channels=d_ff, out_channels=d_model, kernel_size=1, data_format="NLC")
        self.norm1 = paddle.nn.LayerNorm(d_model)
        self.norm2 = paddle.nn.LayerNorm(d_model)
        self.dropout = paddle.nn.Dropout(dropout)
        self.activation = activation

    def forward(
        self, 
        x, 
        attn_mask = None
    )-> paddle.Tensor:
        """ Encoder Forward.

        Args: 
            x(paddle.Tensor): Dict of feature tensor.
            attn_mask(Callable[..., paddle.Tensor]): Whether to use mask in encoder.
        
        Returns:
            paddle.Tensor: Output of EncoderLayer.
        """   
        new_x, attn, mask, sigma = self.attention(
            x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y)))
        y = self.conv2(y)
        y = self.dropout(y)
        return self.norm2(x + y), attn, mask, sigma


class Encoder(paddle.nn.Layer):
    """
    Encoder layers in anomaly transformer.
    
    Args:
        attn_layers(Callable[..., paddle.Tensor]): Dict of feature tensor.
        norm_layer(Callable[..., paddle.Tensor]): Layernorm in encoder for attention layer.
    
    Attributes:
        _nn(paddle.nn.Sequential): Dynamic graph LayerList. 
    """
    def __init__(
        self, 
        attn_layers, 
        norm_layer = None
    ):
        super(Encoder, self).__init__()
        self.attn_layers = paddle.nn.LayerList(attn_layers)
        self.norm = norm_layer

    def forward(
        self, 
        x: paddle.Tensor, 
        attn_mask: Callable[..., paddle.Tensor] = None,
    ) -> paddle.Tensor:
        """Encoder Forward.

        Args: 
            x(paddle.Tensor): The input of Encoder.
            attn_mask(Callable[..., paddle.Tensor]): Whether to use mask in ecoder.

        Returns:
            paddle.Tensor: Output of model.
        """ 
        # x [B, L, D]
        series_list = []
        prior_list = []
        sigma_list = []
        for attn_layer in self.attn_layers:
            x, series, prior, sigma = attn_layer(x, attn_mask=attn_mask)
            series_list.append(series)
            prior_list.append(prior)
            sigma_list.append(sigma)
        if self.norm is not None:
            x = self.norm(x)
        return x, series_list, prior_list, sigma_list
    