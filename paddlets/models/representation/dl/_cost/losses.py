#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from typing import Tuple

import paddle.nn.functional as F
import paddle


def time_contrastive_loss(
    anchor: paddle.Tensor,
    pos: paddle.Tensor,
    neg: paddle.Tensor,
    temperature: float
) -> paddle.Tensor:
    """The time domain contrastive loss.

    Args:
        anchor(paddle.Tensor): The representation obtained from anchor sample.
        pos(paddle.Tensor): The representation obtained from positive sample.
        neg(paddle.Tensor): The representation obtained from negative sample.
        temperature(float): The temperature coefficient. 
 
    Returns:
        paddle.Tensor: The loss value.
    """
    pos = paddle.sum(anchor * pos, 1, keepdim=True)
    neg = paddle.matmul(anchor, neg)
    logits = paddle.concat([pos, neg], -1)
    logits = logits / temperature
    label = paddle.zeros(anchor.shape[:1], dtype="int64")
    loss = F.cross_entropy(logits, label)
    return loss


def frequency_contrastive_loss(
    repr1: paddle.Tensor,
    repr2: paddle.Tensor
) -> paddle.Tensor:
    """The frequency domain contrastive loss.

    Args:
        repr1(paddle.Tensor): The representation obtained from augmented sample1.
        repr2(paddle.Tensor): The representation obtained from augmented sample2.
    
    Returns:
        paddle.Tensor: The loss value.
    """
    batch_size, seq_len = repr1.shape[:2]
    repr = paddle.concat([repr1, repr2], axis=0)      # [2 * batch_size, seq_len, d_model]
    repr = paddle.transpose(repr, perm=[1, 0, 2])     # [seq_len, 2 * batch_size, d_model]
    sim = paddle.matmul(repr, repr, transpose_y=True) # [seq_len, 2 * batch_size, 2 * batch_size]
    logits = paddle.tril(sim, diagonal=-1)[:, :, :-1] # [seq_len, 2 * batch_size, 2 * batch_size - 1]
    logits += paddle.triu(sim, diagonal=1)[:, :, 1:] 
    logits = -1. * F.log_softmax(logits, axis=-1)
    loss = paddle.mean(logits[:, :batch_size, batch_size - 1: 2 * batch_size - 1])
    loss += paddle.mean(logits[:, batch_size: 2 * batch_size, :batch_size])
    return loss / 2.


def convert_coefficient(
    tensor: paddle.Tensor
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    """Compute amplitude and phase in frequency domain.

    Args:
        tensor(paddle.Tensor): The complex tensor.

    Returns:
        paddle.Tensor: The amplitude.
        paddle.Tensor: The phase.
    """
    amp = paddle.sqrt(
        paddle.square(paddle.real(tensor)) + paddle.square(paddle.imag(tensor))
    )
    phase = paddle.atan2(paddle.imag(tensor), paddle.real(tensor))
    return amp, phase
