#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from typing import List, Dict, Any, Callable, Optional

import paddle.nn.functional as F
import paddle
import math


class PositionalEmbedding(paddle.nn.Layer):
    """
    Compute the positional encodings once in log space.
    
    Args:
        d_model(int): The expected feature size for the input of the anomaly transformer.
        max_len(int): The dimensionality of the computed positional encoding array.
        
    Attributes:
        _nn(paddle.nn.Sequential): Dynamic graph LayerList.
    """
    def __init__(
        self, 
        d_model: int, 
        max_len: int = 5000
    ):
        super(PositionalEmbedding, self).__init__()
        pe = paddle.zeros(shape=[max_len, d_model], dtype='float32')
        position = paddle.arange(0, max_len, dtype='float32').unsqueeze(1)
        div_term = (paddle.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = paddle.sin(position * div_term)
        pe[:, 1::2] = paddle.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(
        self, 
        x: paddle.Tensor,
    ) -> paddle.Tensor:
        """PositionalEmbedding Forward.
        
        Args:
            x(paddle.Tensor): Input tensor.
        
        Returns:
            paddle.Tensor: Output of PositionalEmbedding.
        """
        return self.pe[:, :x.shape[1]]


class TokenEmbedding(paddle.nn.Layer):
    """
    Fills the input Tensor with values according to the method described in Delving deep into rectifiers: 
        Surpassing human-level performance on ImageNet classification - He, K. et al. (2015), using a normal distribution. 
        
    Args:
        c_in(int): The Number of channels for Conv1D.
        d_model(int): The expected feature size for the input of the anomaly transformer.
        
    Attributes:
        _nn(paddle.nn.Sequential): Dynamic graph LayerList.
    """
    def __init__(
        self, 
        c_in: int, 
        d_model: int,
    ):
        super(TokenEmbedding, self).__init__()
        padding = 1
        self.tokenConv = paddle.nn.Conv1D(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular',
                                   weight_attr=paddle.nn.initializer.KaimingNormal(),
                                   data_format="NLC",
                                  )

    def forward(
        self, 
        x: paddle.Tensor,
    )-> paddle.Tensor:
        """TokenEmbedding Forward.
        
        Args:
           x(paddle.Tensor): Input tensor. 
           
        Returns:
            paddle.Tensor: Output of TokenEmbedding.
        """
        out = self.tokenConv(x)
        return out


class DataEmbedding(paddle.nn.Layer):
    """
    data embedding = PositionalEmbedding + TokenEmbedding.
    
    Args:
        c_in(int):The Number of channels for embedding.
        d_model(int): The expected feature size for the input of the anomaly transformer.
        dropout(int): Dropout regularization parameter.
        
    Attributes:
        _nn(paddle.nn.Sequential): Dynamic graph LayerList.
    """
    def __init__(
        self, 
        c_in: int, 
        d_model: int, 
        dropout: int = 0.0
    ):
        super(DataEmbedding, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = paddle.nn.Dropout(p=dropout)

    def forward(
        self, 
        x: paddle.Tensor,
    ) -> paddle.Tensor:
        """DataEmbedding Forward.
        
        Args:
            x(paddle.Tensor): Input tensor.
        
        Returns:
            paddle.Tensor: Output of DataEmbedding.
        
        """
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)
