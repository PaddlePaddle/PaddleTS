#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from typing import Tuple

import paddle.nn.functional as F
import numpy as np
import paddle

from paddlets.logger import raise_if_not


class CrossAttention(paddle.nn.Layer):
    """Paddle layer implementing cross attention.

    Args:
        embed_dim(int): The expected feature size in the input and output.
        kdim(int): The feature size in key. 
        vdim(int): The feature size in value. 
        num_heads(int): The number of heads in multi-head attention.
        dropout_rate(float): The dropout probability used on attention
            weights to drop some attention targets.

    Attributes:
        _embed_dim(int): The expected feature size in the input and output.
        _kdim(int): The feature size in key.
        _vdim(int): The feature size in value.
        _num_heads(int): The number of heads in multi-head attention.
        _dropout(paddle.nn.Layer): The dropout layer.
        _norm(paddle.nn.Layer): The layer norm layer.
        _q_proj(paddle.nn.Layer): The query projection layer.
        _k_proj(paddle.nn.Layer): The key projection layer.
        _v_proj(paddle.nn.Layer): The value projection layer.
        _out_proj(paddle.nn.Layer): The output projection layer.
    """
    def __init__(
        self,
        embed_dim: int,
        kdim: int,
        vdim: int,
        num_heads: int,
        dropout_rate: float = 0.1,
    ):
        super(CrossAttention, self).__init__()
        self._embed_dim = embed_dim
        self._kdim = kdim
        self._vdim = vdim
        self._num_heads = num_heads
        self._head_dim = embed_dim // num_heads
        self._dropout = paddle.nn.Dropout(dropout_rate)
        self._norm = paddle.nn.LayerNorm(embed_dim)
        raise_if_not(
            self._head_dim * num_heads == embed_dim,
            "embed_dim must be divisible by num_heads."
        )
        self._q_proj = paddle.nn.Linear(embed_dim, embed_dim)
        self._k_proj = paddle.nn.Linear(kdim, embed_dim)
        self._v_proj = paddle.nn.Linear(vdim, embed_dim)
        self._out_proj = paddle.nn.Linear(embed_dim, embed_dim)

    def _prepare_qkv(
        self, 
        query: paddle.Tensor, 
        key: paddle.Tensor, 
        value: paddle.Tensor
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """Prapares linearly projected queries, keys and values for usage of subsequent
        multiple parallel attention. then splits heads (reshape and transpose) to get keys 
        and values from different representation subspaces. The results are used as key-values pairs 
        for subsequent multiple parallel attention.

        Args:
            query(paddle.Tensor): The queries for multi-head attention.
            key(paddle.Tensor): The keys for multi-head attention.
            value(paddle.Tensor): The values for multi-head attention.

        Returns:
            q(paddle.Tensor): Linear projected query.
            k(paddle.Tensor): Linear projected key.
            v(paddle.Tensor): Linear projected value.
        """
        q = self._q_proj(query)
        k = self._k_proj(key)
        v = self._v_proj(value)
        q = paddle.reshape(q, shape=[0, 0, self._num_heads, self._head_dim])
        k = paddle.reshape(k, shape=[0, 0, self._num_heads, self._head_dim])
        v = paddle.reshape(v, shape=[0, 0, self._num_heads, self._head_dim])
        q = paddle.transpose(q, perm=[0, 2, 1, 3])
        k = paddle.transpose(k, perm=[0, 2, 1, 3])
        v = paddle.transpose(v, perm=[0, 2, 1, 3])
        return q, k, v

    def forward(
        self, 
        query: paddle.Tensor, 
        key: paddle.Tensor, 
        value: paddle.Tensor, 
        attn_mask: paddle.Tensor
    ) -> paddle.Tensor:
        """Applies multi-head attention to map queries and a set of key-value pairs to outputs.

        Args:
            query(paddle.Tensor): The queries for multi-head attention.
            key(paddle.Tensor): The keys for multi-head attention.
            value(paddle.Tensor): The values for multi-head attention.
            attn_mask(paddle.Tensor): A tensor used in multi-head attention 
                to prevent attention to some unwanted positions.

        Returns:
            paddle.Tensor: Output of Layer.
            
        """
        batch_size = query.shape[0]
        # q/k/v: [batch_size, in_chunk_len, d_model] -> 
        # [batch_size, n_heads, in_chunk_len, head_dim]
        q, k, v = self._prepare_qkv(query, key, value)
       
        # scores : [batch_size, n_heads, in_chunk_len, in_chunk_len]
        scores = paddle.matmul(
            q * np.sqrt(self._head_dim),
            paddle.transpose(k, perm=[0, 1, 3, 2])
        )
        
        if attn_mask is not None:
            # attn_mask: [in_chunk_len, in_chunk_len] ->
            # [batch_size, n_heads, in_chunk_len, in_chunk_len]
            attn_mask = paddle.tile(
                attn_mask[None, None, :, :]
                [batch_size, self._num_heads, 1, 1]
            )
            scores = scores + attn_mask
        
        # context: [batch_size, n_heads, in_chunk_len, head_dim]
        attn = F.softmax(scores, axis=-1)
        context = paddle.matmul(attn, v)
        
        # combine heads 
        # context: [batch_size, n_heads, in_chunk_len, head_dim] ->
        # [batch_size, in_chunk_len, d_model]
        context = paddle.transpose(context, perm=[0, 2, 1, 3])
        context = paddle.reshape(context, shape=[0, 0, self._num_heads * self._head_dim])

        # project to output
        out = self._out_proj(context)
        out = self._dropout(out)
        return self._norm(out + query)
        

class ProbSparseAttention(paddle.nn.Layer):
    """Paddle layer implementing probability sparse attention.

    Args:
        embed_dim(int): The expected feature size in the input and output.
        kdim(int): The feature size in key.
        vdim(int): The feature size in value.
        num_heads(int): The number of heads in multi-head attention.
        dropout_rate(float): The dropout probability used on attention
            weights to drop some attention targets.

    Attributes:
        _embed_dim(int): The expected feature size in the input and output.
        _kdim(int): The feature size in key.
        _vdim(int): The feature size in value.
        _num_heads(int): The number of heads in multi-head attention.
        _dropout(paddle.nn.Layer): The dropout layer.
        _norm(paddle.nn.Layer): The layer norm layer.
        _q_proj(paddle.nn.Layer): The query projection layer.
        _k_proj(paddle.nn.Layer): The key projection layer.
        _v_proj(paddle.nn.Layer): The value projection layer.
        _out_proj(paddle.nn.Layer): The output projection layer.
    """
    def __init__(
        self,
        embed_dim: int,
        kdim: int,
        vdim: int,
        num_heads: int,
        dropout_rate: float = 0.1,
    ):
        super(ProbSparseAttention, self).__init__()
        self._embed_dim = embed_dim
        self._num_heads = num_heads
        self._kdim = kdim
        self._vdim = vdim
        self._head_dim = embed_dim // num_heads
        self._dropout = paddle.nn.Dropout(dropout_rate)
        self._norm = paddle.nn.LayerNorm(embed_dim)
        raise_if_not(
            self._head_dim * num_heads == embed_dim,
            "embed_dim must be divisible by num_heads."
        )
        self._q_proj = paddle.nn.Linear(embed_dim, embed_dim)
        self._k_proj = paddle.nn.Linear(kdim, embed_dim)
        self._v_proj = paddle.nn.Linear(vdim, embed_dim)
        self._out_proj = paddle.nn.Linear(embed_dim, embed_dim)
    
    def _prepare_qkv(
        self,
        query: paddle.Tensor,
        key: paddle.Tensor,
        value: paddle.Tensor
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """Prapares linearly projected queries, keys and values for usage of subsequent
        multiple parallel attention. then splits heads (reshape and transpose) to get keys
        and values from different representation subspaces. The results are used as key-values pairs
        for subsequent multiple parallel attention.

        Args:
            query(paddle.Tensor): The queries for multi-head attention.
            key(paddle.Tensor): The keys for multi-head attention.
            value(paddle.Tensor): The values for multi-head attention.

        Returns:
            q(paddle.Tensor): Linearly projected query.
            k(paddle.Tensor): Linearly projected key.
            v(paddle.Tensor): Linearly projected value.
        """
        q = self._q_proj(query)
        k = self._k_proj(key)
        v = self._v_proj(value)
        q = paddle.reshape(q, shape=[0, 0, self._num_heads, self._head_dim])
        k = paddle.reshape(k, shape=[0, 0, self._num_heads, self._head_dim])
        v = paddle.reshape(v, shape=[0, 0, self._num_heads, self._head_dim])
        q = paddle.transpose(q, perm=[0, 2, 1, 3])
        k = paddle.transpose(k, perm=[0, 2, 1, 3])
        v = paddle.transpose(v, perm=[0, 2, 1, 3])
        return q, k, v

    def _compute_prob_qk(
        self, 
        query: paddle.Tensor, 
        key: paddle.Tensor, 
        sample_k: int, 
        n_top: int
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """Select top n queries according to algorithm `ProbSparse self-attention`.

        Args:
            query(paddle.Tensor): The queries for multi-head attention.
            key(paddle.Tensor): The keys for multi-head attention.
            sample_k(int): The number of samples of key.
            n_top(int): The number of top query for sparisty measurement.

        Returns:
            Q_sample(paddle.Tensor): The top n query with sparisty measurement.
            M_top_index(paddle.Tensor): The index of top n query.

        """
        B, H, L_K, _ = key.shape
        _, _, L_Q, _ = query.shape

        # K_expand: [batch_size, n_heads, input_chunk_len, input_chunk_len, head_dim]
        K_expand = paddle.tile(
            key[:, :, None, :, :],
            [1, 1, L_Q, 1, 1]
        )
        
        # randomly select `sample_k` keys
        sampled_idx = paddle.tile(
            paddle.randint(0, L_K, [L_Q, sample_k])[None, None, :, :, None],
            [B, H, 1, 1, 1]
        )
        K_sample = paddle.take_along_axis(K_expand, sampled_idx, axis=-2)
        
        # compute the Q_K score after sampling
        Q_K_sample = paddle.matmul(
            # [batch_size, n_heads, input_chunk_len, 1, head_dim]
            query[:, :, :, None, :], 
            # [batch_size, n_heads, input_chunk_len, head_dim, input_chunk_len]
            paddle.transpose(K_sample, perm=[0, 1, 2, 4, 3]) 
        )
        Q_K_sample = paddle.squeeze(Q_K_sample, axis=-2)

        # find the `n_top` query with sparisty measurement
        M = paddle.max(Q_K_sample, axis=-1) - paddle.mean(Q_K_sample, axis=-1)
        M_top_index = paddle.topk(M, n_top, sorted=False)[1]

        Q_sample = paddle.take_along_axis(query, M_top_index[:, :, :, None], axis=-2)
        return Q_sample, M_top_index
    
    def _get_initial_context(
        self, 
        value: paddle.Tensor, 
        L_Q: int, 
        attn_mask: paddle.Tensor
    ) -> paddle.Tensor:
        """According to algorithm `ProbSparse self-attention`, 
        use `mean/cumsum` of value to approximate the original context.

        Args:
            value(paddle.Tensor): The values for multi-head attention.
            L_Q(paddle.Tensor): The size of the loopback window, 
                i.e. the number of time steps feed to the model.
            attn_mask(paddle.Tensor): A tensor used in multi-head attention
                to prevent attention to some unwanted positions.

        Returns:
            paddle.Tensor: Output of Layer.

        """
        if attn_mask is not None:
            return  paddle.cumsum(value, axis=-2)
        context = paddle.tile(
            paddle.mean(value, axis=-2)[:, :, None, :],
            [1, 1, L_Q, 1]
        )
        return context

    def forward(
        self, 
        query: paddle.Tensor, 
        key: paddle.Tensor, 
        value: paddle.Tensor, 
        attn_mask: paddle.Tensor
    ) -> paddle.Tensor:
        """Applies prob sparse attention to map queries 
        and a set of key-value pairs to outputs.

        Args:
            query(paddle.Tensor): The queries for multi-head attention.
            key(paddle.Tensor): The keys for multi-head attention.
            value(paddle.Tensor): The values for multi-head attention.
            attn_mask(paddle.Tensor): A tensor used in multi-head attention
                to prevent attention to some unwanted positions.

        Returns:
            paddle.Tensor: Output of Layer.
        """
        L_K, L_Q, batch_size = key.shape[1], query.shape[1], query.shape[0]
        # set the number of samples of query and key
        u_k = max(min(int(5 * np.log(L_K)), L_Q), 1)
        u_q = max(min(int(5 * np.log(L_Q)), L_Q), 1)
        
        # q/k/v: [batch_size, in_chunk_len, d_model] ->
        # [batch_size, n_heads, in_chunk_len, head_dim]
        q, k, v = self._prepare_qkv(query, key, value)
        
        # Select u_q queries according to algorithm `ProbSparse self-attention`.
        Q_sample, index = self._compute_prob_qk(q, k, sample_k=u_k, n_top=u_q)

        # use the reduced Q_sample to calculate Q_K
        scores = paddle.matmul(
            Q_sample * np.sqrt(self._head_dim), 
            paddle.transpose(k, perm=[0, 1, 3, 2])
        )
        if attn_mask is not None:
            # attn_mask: [in_chunk_len, in_chunk_len] ->
            # [batch_size, n_heads, in_chunk_len, in_chunk_len]
            attn_mask = paddle.tile(
                attn_mask[None, None, :, :]
                [batch_size, self._num_heads, 1, 1]
            )
            attn_mask = paddle.take_along_axis(attn_mask, index[:, :, :, None], axis=-2)
            scores = scores + attn_mask

        # values: [batch_size, n_heads, in_chunk_len, head_dim]
        attn = F.softmax(scores, axis=-1)
        values = paddle.matmul(attn, v)

        context = self._get_initial_context(v, L_Q, attn_mask)
        context = paddle.put_along_axis(context, index[:, :, :, None], values, axis=-2)

        # combine heads
        # context: [batch_size, n_heads, in_chunk_len, head_dim] ->
        # [batch_size, in_chunk_len, d_model]
        context = paddle.transpose(context, perm=[0, 2, 1, 3])
        context = paddle.reshape(context, shape=[0, 0, self._num_heads * self._head_dim])

        # project to output
        out = self._out_proj(context)
        out = self._dropout(out)
        return self._norm(out + query)
