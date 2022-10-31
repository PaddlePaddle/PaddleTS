#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from typing import List, Dict, Any, Callable, Optional

from paddle.optimizer import Optimizer
import paddle.nn.functional as F
import numpy as np
import paddle

class MLP(paddle.nn.Layer):
    """MLP Network structure used in the encoder and decoder
    
    Args:
        input_dim(int): The size of the loopback window, i.e. the number of time steps feed to the model.
        feature_dim(int): The numer of feature.
        hidden_config(List(int)): The ith element represents the number of neurons in the ith hidden layer.
        activation(Callable[..., paddle.Tensor]): The activation function for the hidden layers.
        last_layer_activation(Callable[..., paddle.Tensor]): The activation function for the last layer.
        dropout_rate(float): Dropout regularization parameter.
        use_bn(bool): Whether to use batch normalization.
        use_drop(bool): Whether to use dropout.
    
    Attributes:
        _nn(paddle.nn.Sequential): Dynamic graph LayerList.
    """
    def __init__(
        self,
        input_dim: int,
        feature_dim: int,
        hidden_config: List[int],
        activation: Callable[..., paddle.Tensor],
        last_layer_activation: Callable[..., paddle.Tensor],
        dropout_rate: float = 0.5,
        use_bn: bool = True,
        use_drop: bool = True,
    ):
        super(MLP, self).__init__()
        dims, layers = [input_dim] + hidden_config, []
        for i in range(1, len(dims)):
            layers.append(paddle.nn.Linear(dims[i - 1], dims[i]))
            if use_bn:
                layers.append(paddle.nn.BatchNorm1D(feature_dim))
            if i < len(dims) - 1:
                layers.append(activation())
                if use_drop:
                    layers.append(paddle.nn.Dropout(dropout_rate))
            else:
                layers.append(last_layer_activation())
        self._nn = paddle.nn.Sequential(*layers)
        
    def forward(self, x):
        return self._nn(x)

    
class CNN(paddle.nn.Layer):
    """CNN Network structure used in the encoder and decoder
    
    Args:
        input_dim(int): The size of the loopback window, i.e. the number of time steps feed to the model.
        hidden_config(List(int)): The ith element represents the number of neurons in the ith hidden layer.
        activation(Callable[..., paddle.Tensor]): The activation function for the hidden layers.
        last_layer_activation(Callable[..., paddle.Tensor]): The activation function for the last layer.
        kernel_size(int): Kernel size for Conv1D.
        use_drop(bool): Whether to use dropout.
        dropout_rate(float): Dropout regularization parameter.
        use_bn(bool): Whether to use batch normalization.
        is_encoder(bool): Encoder or Decoder.
        data_format(str): Specify the input data format.N is the batch size, C is the number of channels and L is the characteristic length. 
    
    Attributes:
        _nn(paddle.nn.Sequential): Dynamic graph LayerList.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_config: List[int],
        activation: Callable[..., paddle.Tensor],
        last_layer_activation: Callable[..., paddle.Tensor],
        kernel_size: int,
        dropout_rate: float = 0.5,
        use_bn: bool = True,
        is_encoder: bool = True,
        use_drop: bool = True,
        data_format: str = 'NCL',
    ):
        super(CNN, self).__init__()
        dims, layers = [input_dim] + hidden_config, []
        for i in range(1, len(dims)):
            if is_encoder:
                layers.append(paddle.nn.Conv1D(dims[i - 1], dims[i], kernel_size, data_format=data_format))
            else:
                layers.append(paddle.nn.Conv1DTranspose(dims[i - 1], dims[i], kernel_size, data_format=data_format))
            if use_bn:
                layers.append(paddle.nn.BatchNorm1D(dims[i], data_format=data_format))
            if i < len(dims) - 1:
                layers.append(activation())
                if use_drop:
                    layers.append(paddle.nn.Dropout(dropout_rate))
            else:
                layers.append(last_layer_activation())
        self._nn = paddle.nn.Sequential(*layers)
        
    def forward(self, x):
        return self._nn(x)

    
class LSTM(paddle.nn.Layer):
    """LSTM Network structure used in the encoder and decoder
    
    Args:
        input_dim(int): The size of the loopback window, i.e. the number of time steps feed to the model.
        hidden_config(List(int)): The ith element represents the number of neurons in the ith hidden layer.
        activation(Callable[..., paddle.Tensor]): The activation function for the hidden layers.
        last_layer_activation(Callable[..., paddle.Tensor]): The activation function for the last layer.
        dropout_rate(float): Dropout regularization parameter.
        use_drop(bool): Whether to use dropout.
        num_layers(int): layers of LSTM.
        direction(str): the direction of LSTM.
        
    Attributes:
        _nn(paddle.nn.Sequential): Dynamic graph LayerList.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_config: List[int],
        activation: Callable[..., paddle.Tensor],
        last_layer_activation: Callable[..., paddle.Tensor],
        dropout_rate: float = 0,
        use_drop: bool = True,
        num_layers: int = 1,
        direction: str = 'forward',
        ):
        super(LSTM, self).__init__()
        dims, layers = [input_dim] + hidden_config, []
        for i in range(1, len(dims)):
            layers.append(paddle.nn.LSTM(dims[i - 1], dims[i], 
                          num_layers=num_layers, dropout=dropout_rate, direction=direction))
            if i < len(dims) - 1:
                layers.append(activation())
                if use_drop:
                    layers.append(paddle.nn.Dropout(dropout_rate))
            else:
                layers.append(last_layer_activation())
        self._nn = paddle.nn.Sequential(*layers)
        
    def forward(self, x):
        return self._nn(x)
