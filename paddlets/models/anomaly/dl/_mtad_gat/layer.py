#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from typing import Tuple

import paddle.nn.functional as F
import numpy as np
import paddle

from paddlets.logger import raise_if_not


class ConvLayer(paddle.nn.Layer):
    """Convolution layer to extract features.

    Args:
        feature_dim(int): The number of features.
        kernel_size(int): Kernel size for Conv1D.

    Attributes:
        _pad(paddle.nn.Layer): The pad layer.
        _conv(paddle.nn.Layer): The conv layer.
        _relu(paddle.nn.Layer): The relu layer.
    """
    def __init__(
        self,
        feature_dim: int,
        kernel_size: int = 7
    ):
        super(ConvLayer, self).__init__()
        self._pad = paddle.nn.Pad1D((kernel_size - 1) // 2, mode="constant")
        self._conv = paddle.nn.Conv1D(in_channels=feature_dim, out_channels=feature_dim, kernel_size=kernel_size)
        self._relu = paddle.nn.ReLU()
        
    def forward(self, x):
        """Forward
        
        Args:
            x(paddle.Tensor): The input data.
            
        Returns:
            paddle.Tensor: Output of conv layer.
        """
        x = paddle.transpose(x, perm=[0, 2, 1])
        x = self._pad(x)
        x = self._relu(self._conv(x))
                
        return paddle.transpose(x, perm=[0, 2, 1])
            

class GRULayer(paddle.nn.Layer):
    """GRU layer.

    Args:
        input_size(int): The input size
        hidden_size(int): The hidden size.
        num_layers(int): The number of layer.
        dropout(float): Dropout regularization parameter.

    Attributes:
        _dropout(float): Dropout regularization parameter.
        _gru(paddle.nn.Layer): The gru layer.
    """

    def __init__(
        self, 
        input_size: int, 
        hidden_size: int,
        num_layers: int,
        dropout: float
    ):
        super(GRULayer, self).__init__()
        self._dropout = 0.0 if num_layers == 1 else dropout
        self._gru = paddle.nn.GRU(input_size, hidden_size, num_layers=num_layers, dropout=self._dropout)

    def forward(self, x):
        """Forward
        
        Args:
            x(paddle.Tensor): The input data.
            
        Returns:
            out(paddle.Tensor): Output of grulayer. 
            h(paddle.Tensor): final_states.
        """
        out, h = self._gru(x)
        return out, h
