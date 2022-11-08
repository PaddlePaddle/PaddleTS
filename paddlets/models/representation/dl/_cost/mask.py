#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import numpy as np
import paddle


def generate_true_mask(
    batch_size: int, 
    seq_len: int
) -> paddle.Tensor:
    """Generate masks that are all `true`

    Args:
        batch_size(int): The number of samples per batch.
        seq_len(int): The sequence length.

    Returns:
        paddle.Tensor: Output of function.
    """
    mask = np.full((batch_size, seq_len), True)
    return paddle.to_tensor(mask, dtype="bool")


def generate_binomial_mask(
    batch_size: int,
    seq_len: int,
    p: float = 0.5
) -> paddle.Tensor:
    """Mask generation by Bernoulli distribution.

    Args:
        batch_size(int): The number of samples per batch.
        seq_len(int): The sequence length.
        p(float): The parameter of the distribution.

    Returns:
        paddle.Tensor: Output of function.
    """
    mask = np.random.binomial(1, p, (batch_size, seq_len))
    return paddle.to_tensor(mask, dtype="bool")


def paddle_mask_fill(
    tensor: paddle.Tensor,
    mask: paddle.Tensor,
    value: float
) -> paddle.Tensor:
    """Fills elements of tensor with value where mask is True.

    Args:
        tensor(paddle.Tensor): The tensor to be masked.
        mask(paddle.Tensor): The boolean mask.
        value(float): The value to fill in with.

    Returns:
        paddle.Tensor: Output of function.
    """
    mask = paddle.expand_as(mask[:, :, None], tensor)
    cache = paddle.full(tensor.shape, value, tensor.dtype)
    return paddle.where(mask, cache, tensor)
