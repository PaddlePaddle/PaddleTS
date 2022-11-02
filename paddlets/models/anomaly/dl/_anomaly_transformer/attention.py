#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from typing import List, Dict, Any, Callable, Optional

from math import sqrt
import paddle.nn.functional as F
import numpy as np
import paddle
import math


class TriangularCausalMask():
    """
    Triangular Causal Mask.
    
    Args:
        batch_size(int): Number of samples per batch.
        length(int): Length of samples per data.

    Attributes:
        _nn(paddle.nn.Sequential): Dynamic graph LayerList.  
    """
    def __init__(
        self, 
        batch_size: int, 
        length: int
    ):
        mask_shape = [batch_size, 1, length, length]
        with paddle.no_grad():
            self._mask = paddle.triu(paddle.ones(shape=mask_shape, dtype="bool"), diagonal=1)

    @property
    def mask(
        self
    ):
        return self._mask


class AnomalyAttention(paddle.nn.Layer):
    """ 
    Anomaly Attention: 
        For the prior-association, a learnable Gaussian kernel to calculate the prior with the relative temporal distance.
        For the series-association branch is to learn the associations from raw series.
        
    Args:
        win_size(int): The size of the loopback window, i.e. the number of time steps feed to the model.
        mask_flag(bool): Whether to use attn_mask.
        scale(int|None): It can scale the dot products.
        attention_dropout(float): Dropout regularization parameter.
        output_attention(bool): Whether to output series, prior and sigma.
        
    Attributes:
        _nn(paddle.nn.Sequential): Dynamic graph LayerList.    
    """
    def __init__(
        self, 
        win_size: int, 
        mask_flag: bool = True, 
        scale: bool = None, 
        attention_dropout: float = 0.0, 
        output_attention: bool = False,
    ):
        super(AnomalyAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = paddle.nn.Dropout(attention_dropout)
        self.distances = paddle.zeros((win_size, win_size))
        for i in range(win_size):
            for j in range(win_size):
                self.distances[i][j] = abs(i - j)

    def forward(
        self, 
        queries: paddle.Tensor, 
        keys: paddle.Tensor, 
        values: paddle.Tensor, 
        sigma: paddle.Tensor, 
        attn_mask: Callable[..., paddle.Tensor],
    ) -> paddle.Tensor:
        """
        The prior-association result from Gaussian kernel branch. the series-association result from self attention branch.

        Args: 
            queries(paddle.Tensor): The query projection layer. 
            keys(paddle.Tensor): The key projection layer.
            values(paddle.Tensor): The value projection layer.
            sigma(paddle.Tensor): A learnable scale parameter for the Gaussian kernel, 
                making ther prior-associations adapt the various time series patterns.
            attn_mask(Callable[..., paddle.Tensor]|None): Whether to use mask in ecoder.
            
        Returns:
            V(paddle.Tensor): Output of AnomalyAttention.
            series(paddle.Tensor): The series-association from Gaussian kernel branch.
            prior(paddle.Tensor): The prior-association from self attention.
            sigma(paddle.Tensor): A learnable scale parameter for the Gaussian kernel.
        """
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)
        scores = paddle.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        attn = scale * scores
        series = self.dropout(F.softmax(attn, axis=-1))
        V = paddle.einsum("bhls,bshd->blhd", series, values)
        window_size = attn.shape[-1]
        sigma = paddle.transpose(sigma, perm=[0, 2, 1])  # B L H ->  B H L
        sigma = paddle.nn.functional.sigmoid(sigma * 5) + 1e-5
        sigma = paddle.pow(paddle.to_tensor(3.), sigma)  # - 1
        sigma = paddle.tile(sigma.unsqueeze(-1), repeat_times=[1, 1, 1, window_size])  # B H L L
        prior = paddle.tile(self.distances.unsqueeze(0).unsqueeze(0), [sigma.shape[0], sigma.shape[1], 1, 1])
        prior = 1.0 / (math.sqrt(2 * math.pi) * sigma) * paddle.exp(-prior ** 2 / 2 / (sigma ** 2))
        if self.output_attention:
            return (V, series, prior, sigma)
        else:
            return (V, None)

        
class AttentionLayer(paddle.nn.Layer):
    """
    AttentionLayer for anomaly transformer.
    
    Args:
        attention(Callable[..., paddle.Tensor]): Attention layers in anomaly transformer.
        d_model(int): The expected feature size for the input of the anomaly transformer.
        n_heads(int): The number of heads in multi-head attention.
        d_keys(int): The feature size in key.
        d_values(int): The feature size in value.
    
    Attributes:
        _nn(paddle.nn.Sequential): Dynamic graph LayerList. 
    """
    def __init__(
        self, 
        attention: Callable[..., paddle.Tensor], 
        d_model: int, 
        n_heads: int, 
        d_keys: int = None,
        d_values: int = None
    ):
        super(AttentionLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.norm = paddle.nn.LayerNorm(d_model)
        self.inner_attention = attention
        self.query_projection = paddle.nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = paddle.nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = paddle.nn.Linear(d_model, d_values * n_heads)
        self.sigma_projection = paddle.nn.Linear(d_model, n_heads)
        self.out_projection = paddle.nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(
        self, 
        queries: paddle.Tensor,
        keys: paddle.Tensor,
        values: paddle.Tensor,
        attn_mask: Callable[..., paddle.Tensor],
    )-> paddle.Tensor:
        """ The series-association and the prior-association forward.
        
        Args: 
            queries(paddle.Tensor): The query projection layer tensor. 
            keys(paddle.Tensor): The key projection layer tensor.
            values(paddle.Tensor): The value projection layer tensor.
            sigma(paddle.Tensor): A learnable scale parameter for the Gaussian kernel.
            attn_mask(Callable[..., paddle.Tensor]): Whether to use mask in ecoder.
        
        Returns:
            self.out_projection(out)(paddle.Tensor): pred of model.
            series(paddle.Tensor): The series-association output tensor.
            prior(paddle.Tensor): The prior-association output tensor.
            sigma(paddle.Tensor): A learnable scale parameter for the Gaussian kernel.
        """
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        x = queries
        queries = paddle.reshape(self.query_projection(queries), [B, L, H, -1])
        keys = paddle.reshape(self.key_projection(keys), [B, S, H, -1])
        values =  paddle.reshape(self.value_projection(values), [B, S, H, -1])
        sigma = paddle.reshape(self.sigma_projection(x), [B, L, H])
        out, series, prior, sigma = self.inner_attention(queries, keys, values, sigma, attn_mask)
        out = paddle.reshape(out, [B, L, -1])
        return self.out_projection(out), series, prior, sigma