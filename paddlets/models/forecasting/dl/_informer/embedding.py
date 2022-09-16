#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import numpy as np
import paddle


class PositionalEmbedding(paddle.nn.Layer):
    """Paddle layer implementing positional embedding.

    Args:
        d_model(int): The expected feature size for the input/output of the informer's encoder/decoder.
        max_len(int): The dimensionality of the computed positional encoding array.

    Attributes:
        _position_embedding(paddle.Tensor): positional encoding as buffer into the layer.
    """
    def __init__(self, d_model: int, max_len: int):
        super(PositionalEmbedding, self).__init__()
        # The calculation formula of the positional encodeing is as follows.
        # PE(pos, 2i) = sin(pos / 1e4 ** (2i / d_model)).
        # PE(pos, 2i + 1) = cos(pos / 1e4 ** (2i / d_model)).
        # Where: 
        #   d_model: The expected feature size for the input/output of the transformer's encoder/decoder.
        #   pos: a position in the input sequence. 
        #   2i/2i + 1: odd/even index of d_model.
        position_embedding = paddle.zeros((max_len, d_model))
        position = paddle.unsqueeze(
            paddle.arange(0, max_len, dtype="float32"), axis=1
        )
        div_term = paddle.exp(
            paddle.arange(0, d_model, 2, dtype="float32") * (-1. * np.log2(1e4) / d_model)
        )
        position_embedding[:, 0::2] = paddle.sin(position * div_term)
        position_embedding[:, 1::2] = paddle.cos(position * div_term)
        self.register_buffer("_position_embedding", position_embedding)

    def forward(self, X: paddle.Tensor) -> paddle.Tensor:
        """Forward.

        Args:
            X(paddle.Tensor): The input of Informer encoder/decoder.

        Returns:
            paddle.Tensor: Output of layer.
        """
        return self._position_embedding[:X.shape[1], :]


class TimeFeatureEmbedding(paddle.nn.Layer):
    """Paddle layer implementing time feature embedding.

    Args:
        d_stamp(int): The input/output sequence’s timestamp size.
        d_model(int): The expected feature size for the input/output of the informer's encoder/decoder.

    Attributes:
        _timefeat_embedding(paddle.nn.Layer): time feature embedding.
    """
    def __init__(self, d_stamp: int, d_model: int):
        super(TimeFeatureEmbedding, self).__init__()
        self._timefeat_embedding = paddle.nn.Linear(d_stamp, d_model)

    def forward(self, X: paddle.Tensor) -> paddle.Tensor:
        """Forward.

        Args:
            X(paddle.Tensor): The input of Informer encoder/decoder.

        Returns:
            paddle.Tensor: Output of layer.
        """
        return self._timefeat_embedding(X)


class TokenEmbedding(paddle.nn.Layer):
    """Paddle layer implementing token embedding, To align the dimension(position_embedding), 
    we project the scalar context into d_model-dim vector with 1-D convolutional filters.

    Args:
        target_dim(int): The numer of targets.
        d_model(int): The expected feature size for the input/output of the informer's encoder/decoder.

    Attributes:
        _token_embedding(paddle.nn.Layer): token embedding.
    """
    def __init__(self, target_dim: int, d_model: int):
        super(TokenEmbedding, self).__init__()
        _weight_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.KaimingNormal()
        )
        self._token_embedding = paddle.nn.Conv1D(
            in_channels=target_dim, 
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
            weight_attr=_weight_attr
        )

    def forward(self, X: paddle.Tensor) -> paddle.Tensor:
        """Forward.

        Args:
            X(paddle.Tensor): The input of Informer encoder/decoder.
        
        Returns:
            paddle.Tensor: Output of layer.
        """
        out = paddle.transpose(X, perm=[0, 2, 1])
        out = self._token_embedding(out)
        out = paddle.transpose(out, perm=[0, 2, 1])
        return out


class MixedEmbedding(paddle.nn.Layer):
    """Paddle layer implementing data(position, token) embedding.

    Args:
        target_dim(int): The numer of targets.
        d_model(int): The expected feature size for the input/output of the informer's encoder/decoder.
        max_len(int): The dimensionality of the computed positional encoding array.
        dropout_rate(float): Fraction of neurons affected by Dropout.

    Attributes:
        _token_embedding(paddle.nn.Layer): token embedding.
        _position_embedding(paddle.Tensor): position embedding.
    """
    def __init__(self,
        target_dim: int,
        d_model: int,
        max_len: int,
        dropout_rate: float = 0.1
    ):
        super(MixedEmbedding, self).__init__()
        self._token_embedding = TokenEmbedding(target_dim, d_model)
        self._position_embedding = PositionalEmbedding(d_model, max_len)
        self._dropout = paddle.nn.Dropout(dropout_rate)
    
    def forward(self, X: paddle.Tensor) -> paddle.Tensor:
        """Forward.
        
        Args:
            X(paddle.Tensor): The input of Informer encoder/decoder.
        
        Returns:
            paddle.Tensor: Output of Layer.
        """
        out = self._token_embedding(X) + self._position_embedding(X)
        return self._dropout(out)
