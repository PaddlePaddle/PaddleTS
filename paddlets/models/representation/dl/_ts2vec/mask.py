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


def generate_false_mask(
    batch_size: int, 
    seq_len: int
) -> paddle.Tensor:
    """Generate masks that are all `false`

    Args:
        batch_size(int): The number of samples per batch.
        seq_len(int): The sequence length.

    Returns:
        paddle.Tensor: Output of function.
    """
    mask = np.full((batch_size, seq_len), False)
    return paddle.to_tensor(mask, dtype="bool")


def generate_last_mask(
    batch_size: int, 
    seq_len: int
) -> paddle.Tensor:
    """Gernerate all `true` masks except the last time step.

    Args:
        batch_size(int): The number of samples per batch.
        seq_len(int): The sequence length.

    Returns:
        paddle.Tensor: Output of function.
    """
    mask = generate_true_mask(batch_size, seq_len)
    mask[:, -1] = False
    return mask


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


def generate_continuous_mask(
    batch_size: int,
    seq_len: int,
    mask_num: int = 5,
    mask_ratio: float = 0.1
) -> paddle.Tensor:
    """Generate continuous mask.

    Args:
        batch_size(int): The number of samples per batch.
        seq_len(int): The sequence length.
        mask_num(int): The number of masked segments.
        mask_ratio(float): The length of each masked segment.

    Returns:
        paddle.Tensor: Output of function.
    """
    mask = generate_true_mask(batch_size, seq_len)
    mask_num = max(min(mask_num, seq_len // 2), 1)
    mask_len = max(int(mask_ratio * seq_len), 1)
    for row in range(batch_size):
        for _ in range(mask_num):
            start = np.random.randint(seq_len - mask_len + 1)
            mask[row, start: start + mask_len] = False
    return mask


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
