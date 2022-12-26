#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from typing import Tuple
from typing import List, Dict, Any, Callable, Optional

import paddle.nn.functional as F
import numpy as np
import paddle

class FeatOrTempAttention(paddle.nn.Layer):
    """Feature/Temporal Graph Attention Layer.

    Args:
        feature_dim(int): The number of features.
        in_chunk_len(int): The size of the loopback window, i.e. the number of time steps feed to the model.
        dropout(float): The percentage of nodes to dropout.
        alpha(float): The negative slope used in the LeakyReLU activation function.
        embed_dim(None|int): The embedding dimension (output dimension of linear transformation).
        use_gatv2(bool): Whether to use the modified attention mechanism of GATv2 instead of standard GAT.
        use_bias(bool): whether to include a bias term in the attention layer.
        name(str): Feature or Temporal Graph.

    Attributes:
        _feature_dim(int): The number of features/nodes.
        _in_chunk_len(int): The size of the loopback window, i.e. the number of time steps feed to the model.
        _dropout(float): The percentage of nodes to dropout.
        _alpha(float): The negative slope used in the LeakyReLU activation function.
        _embed_dim(None|int): The embedding dimension (output dimension of linear transformation).
        _use_gatv2(bool): Whether to use the modified attention mechanism of GATv2 instead of standard GAT.
        _use_bias(bool): whether to include a bias term in the attention layer.
        _name(str): Feature or Temporal Graph.
        _nodes_num(int): Number of nodes in a graph.
        _lin(paddle.nn.Layer): The linear transformation layer.
        _att(paddle.Parameter): The attention parameter.
        _bias(paddle.Parameter): The bias parameter.
        _leakyrelu(paddle.nn.Layer): The LeakyReLU activation layer.
        _sigmoid(paddle.nn.Layer): The Sigmoid layer.
       
    """
    def __init__(
        self,
        feature_dim: int,
        in_chunk_len: int,
        dropout: float,
        alpha: int,
        embed_dim: Optional[int] = None,
        use_gatv2: bool = True,
        use_bias: bool = True,
        name: str = 'feature'
    ):
        super(FeatOrTempAttention, self).__init__()
        if name == "temporal":
            feature_dim, in_chunk_len = in_chunk_len, feature_dim
        self._feature_dim = feature_dim
        self._in_chunk_len = in_chunk_len
        self._dropout = dropout
        self._alpha = alpha
        self._embed_dim = embed_dim if embed_dim is not None else in_chunk_len
        self._use_gatv2 = use_gatv2
        self._use_bias = use_bias
        self._name = name
        self._nodes_num = feature_dim
        
        # Because linear transformation is done after concatenation in GATv2
        if self._use_gatv2:
            self._embed_dim *= 2
            lin_input_dim = 2 * in_chunk_len
            att_input_dim = self._embed_dim
        else:
            lin_input_dim = in_chunk_len
            att_input_dim = 2 * self._embed_dim
        
        self._lin = paddle.nn.Linear(lin_input_dim, self._embed_dim)
        param = paddle.empty(shape=[att_input_dim, 1])
        self._att = paddle.create_parameter(shape=param.shape, dtype=str(param.numpy().dtype), \
                                         default_initializer=paddle.nn.initializer.XavierUniform())
        if self._use_bias:
            bias_param = paddle.empty(shape=[feature_dim, feature_dim])
            self._bias = paddle.create_parameter(shape=bias_param.shape, dtype=str(bias_param.numpy().dtype), \
                                         default_initializer=paddle.nn.initializer.Assign(bias_param))
            
        self._leakyrelu = paddle.nn.LeakyReLU(alpha)
        self._sigmoid = paddle.nn.Sigmoid()
        
    def _prepare_attention_input(
        self, 
        v: paddle.Tensor
    )-> paddle.Tensor:
        """Preparing the feature/temporal attention mechanism. Creating matrix with all possible combinations of concatenations of node.
        if feature graph attention, each node consists of all values of that node within the in_chunk_len:
            v1 || v1,
            ...
            v1 || vK,
            v2 || v1,
            ...
            v2 || vK,
            ...
            vK || v1,
            ...
            vK || vK,
        if temporal graph attention, each node consists all features at the same time:
            (f1, f2..)_t1 || (f1, f2..)_t1
            (f1, f2..)_t1 || (f1, f2..)_t2
            ...
            (f1, f2..)_tn || (f1, f2..)_t1
            (f1, f2..)_tn || (f1, f2..)_t2
            ...
            
        Args:
            v(paddle.Tensor): The data for prepare graph attention input.
        
        Returns:
            p_v(paddle.Tensor): The input of graph attention.

        """
        K = self._nodes_num
        # Left-side of the matrix
        blocks_repeating = paddle.repeat_interleave(v, repeats=K, axis=1)
        # Right-side of the matrix
        blocks_alternating = paddle.tile(v, repeat_times=(1, K, 1))
        # [batch_size, feature_dim*feature_dim/in_chunk_len*in_chunk_len, 2*in_chunk_len/2*feature_dim]
        combined = paddle.concat([blocks_repeating, blocks_alternating], axis=2)
        
        if self._use_gatv2:
            return paddle.reshape(combined, (combined.shape[0], K, K, 2 * self._in_chunk_len))
        else:
            return paddle.reshape(combined, (combined.shape[0], K, K, 2 * self._embed_dim))
                 
    def forward(
        self, 
        x
    ) -> paddle.Tensor:
        """Feature extraction based on graph attention network

        Args:
            x(paddle.Tensor): The input data.

        Returns:
            paddle.Tensor: Output of Layer.
            
        """
        # x: [batch_size, in_chunk_len, feature_dim]
        # For temporal attention a node is represented as all feature values at a specific timestamp
        if self._name == "feature":
            # For feature attention a node is represented as the values of a particular feature across all timestamps
            x = paddle.transpose(x, perm=[0, 2, 1])
        # 'Dynamic' GAT attention
        # Proposed by Brody et. al., 2021 (https://arxiv.org/pdf/2105.14491.pdf)
        # Linear transformation applied after concatenation and attention layer applied after leakyrelu
        if self._use_gatv2:
            #[batch_size, feature_dim/in_chunk_len, feature_dim/in_chunk_len, 2*feature_dim/2*in_chunk_len]
            att_input = self._prepare_attention_input(x)
            #[batch_size, feature_dim/in_chunk_len, feature_dim/in_chunk_len, embed_dim]
            att_input = self._leakyrelu(self._lin(att_input))
            #[batch_size, feature_dim/in_chunk_len, feature_dim/in_chunk_len, 1]
            e = paddle.matmul(att_input, self._att).squeeze(3)   
        # Original GAT attention
        else:
            #[batch_size, feature_dim/in_chunk_len, feature_dim/in_chunk_len, embed_dim]
            wx = self._lin(x)
            #[batch_size, feature_dim/in_chunk_len, feature_dim/in_chunk_len, 2 *embed_dim]
            att_input = self._prepare_attention_input(wx)
            #[batch_size, feature_dim/in_chunk_len, feature_dim/in_chunk_len, 1]
            e = self._leakyrelu(paddle.matmul(att_input, self._att)).squeeze(3)
        if self._use_bias:
            e += self._bias
        
        # Attention weights
        attention = paddle.nn.Softmax(axis=2)(e)
        dropout = paddle.nn.Dropout(p=self._dropout)
        attention = dropout(attention)
        
        # Computing new node features using the attention
        h = self._sigmoid(paddle.matmul(attention, x))
        
        if self._name == "feature":
            return paddle.transpose(h, perm=[0, 2, 1])
        else:
            return h
