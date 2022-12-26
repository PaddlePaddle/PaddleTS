#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from typing import Tuple

import paddle.nn.functional as F
import numpy as np
import paddle

from paddlets.models.anomaly.dl._mtad_gat.layer import ConvLayer, GRULayer


class Reconstruction(paddle.nn.Layer):
    """Reconstruction based Model.

    Args:
        in_chunk_len(int): The size of the loopback window, i.e. the number of time steps feed to the model.
        feature_dim(int): The number of features.
        hidden_size(int): The hidden size.
        out_dim(int): The number of output features.
        num_layers(int): The number of layer.
        dropout(float): Dropout regularization parameter.

    Attributes:
        _in_chunk_len(int): The size of the loopback window, i.e. the number of time steps feed to the model.
        _decoder(paddle.nn.Layer): The gru decoder layer.
        _fc(paddle.nn.Layer): The fc layer.
    """

    def __init__(
        self, 
        in_chunk_len: int,
        feature_dim: int,
        hidden_size: int,
        out_dim: int,
        num_layers: int,
        dropout: float
    ):
        super(Reconstruction, self).__init__()
        self._in_chunk_len = in_chunk_len
        self._decoder = GRULayer(feature_dim, hidden_size, num_layers, dropout)
        self._fc = paddle.nn.Linear(hidden_size, out_dim)

    def forward(self, x):
        """Forward
        
        Args:
            x(paddle.Tensor): The input data.
            
        Returns:
            paddle.Tensor): Output of Reconstruction. 
        """
        # x will be last hidden state of the GRU layer
        h_end = paddle.repeat_interleave(x, repeats=self._in_chunk_len, axis=1)
        h_end = h_end.reshape((x.shape[0], self._in_chunk_len, -1))

        decoder_out, _ = self._decoder(h_end)      
        out = self._fc(decoder_out)
        
        return out
            

class Forecasting(paddle.nn.Layer):
    """Forecasting based Model.

    Args:
        feature_dim(int): The number of features.
        hidden_size(int): The hidden size.
        out_dim(int): The number of output features.
        num_layers(int): The number of layer.
        dropout(float): Dropout regularization parameter.

    Attributes:
        _layers(paddle.nn.Sequential): Dynamic graph LayerList.
        _dropout(paddle.nn.Dropout): The dropout layer.
        _relu(paddle.nn.RelU): The relu layer.
    """

    def __init__(
        self, 
        feature_dim: int, 
        hidden_size: int, 
        out_dim: int, 
        num_layers: int, 
        dropout: float
    ):
        super(Forecasting, self).__init__()
        layers = [paddle.nn.Linear(feature_dim, hidden_size)]
        for _ in range(num_layers - 1):
            layers.append(paddle.nn.Linear(hidden_size, hidden_size))
        layers.append(paddle.nn.Linear(hidden_size, out_dim))
        self._layers = paddle.nn.LayerList(layers)
        self._dropout = paddle.nn.Dropout(dropout)
        self._relu = paddle.nn.ReLU()
        
    def forward(self, x):
        """Forward
        
        Args:
            x(paddle.Tensor): The input data.
            
        Returns:
            paddle.Tensor): Output of Forecasting. 
        """
        
        for i in range(len(self._layers) - 1):
            x = self._relu(self._layers[i](x))
            x = self._dropout(x)
        return self._layers[-1](x)
