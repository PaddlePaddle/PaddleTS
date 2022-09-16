#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import paddle.nn.functional as F
import paddle


def instance_contrastive_loss(
    repr1: paddle.Tensor,
    repr2: paddle.Tensor
) -> paddle.Tensor:
    """The instance contrastive loss.
    
    Args:
        repr1(paddle.Tensor): The representation obtained from augmented sample1.
        repr2(paddle.Tensor): The representation obtained from augmented sample2.
    
    Returns:
        paddle.Tensor: The loss value.
    """
    batch_size, seq_len = repr1.shape[:2]
    if batch_size == 1:
        return paddle.zeros([1])

    repr = paddle.concat([repr1, repr2], axis=0)      # [2 * batch_size, seq_len, d_model]
    repr = paddle.transpose(repr, perm=[1, 0, 2])     # [seq_len, 2 * batch_size, d_model]
    sim = paddle.matmul(repr, repr, transpose_y=True) # [seq_len, 2 * batch_size, 2 * batch_size]
    logits = paddle.tril(sim, diagonal=-1)[:, :, :-1] # [seq_len, 2 * batch_size, 2 * batch_size - 1]
    logits += paddle.triu(sim, diagonal=1)[:, :, 1:] 
    logits = -1. * F.log_softmax(logits, axis=-1)

    loss = paddle.mean(logits[:, :batch_size, batch_size - 1: 2 * batch_size - 1])
    loss += paddle.mean(logits[:, batch_size: 2 * batch_size, :batch_size])
    return loss / 2.


def temporal_contrastive_loss(
    repr1: paddle.Tensor,
    repr2: paddle.Tensor
) -> paddle.Tensor:
    """The temporal contrastive loss.

    Args:
        repr1(paddle.Tensor): The representation obtained from augmented sample1.
        repr2(paddle.Tensor): The representation obtained from augmented sample2.
    
    Returns:
        paddle.Tensor: The loss value.
    """
    batch_size, seq_len = repr1.shape[:2]
    if seq_len == 1:
        return paddle.zeros([1])

    repr = paddle.concat([repr1, repr2], axis=1)      # [batch_size, 2 * seq_len, d_model]
    sim = paddle.matmul(repr, repr, transpose_y=True) # [seq_len, 2 * seq_len, 2 * seq_len]
    logits = paddle.tril(sim, diagonal=-1)[:, :, :-1] # [batch_size, 2 * seq_len, 2 * seq_len - 1]
    logits += paddle.triu(sim, diagonal=1)[:, :, 1:]
    logits = -1. * F.log_softmax(logits, axis=-1)

    loss = paddle.mean(logits[:, :seq_len, seq_len - 1: 2 * seq_len - 1])
    loss += paddle.mean(logits[:, seq_len: 2 * seq_len, :seq_len])
    return loss / 2.


def hierarchical_contrastive_loss(
    repr1: paddle.Tensor, 
    repr2: paddle.Tensor,
    alpha: float = 0.5, 
    temporal_unit: int = 0
) -> paddle.Tensor:
    """The hierarchical contrastive loss (contains instance contrastive loss and temporal contrastive loss).

    Args:
        repr1(paddle.Tensor): The representation obtained from augmented sample1.
        repr2(paddle.Tensor): The representation obtained from augmented sample2.
        alpha(float): The adjustment factor.
        temporal_unit(int): The minimum unit to perform temporal contrast.
    
    Returns:
        paddle.Tensor: The loss value.
    """
    loss, d = paddle.zeros([1]), 0
    while repr1.shape[1] > 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(repr1, repr2)

        if d >= temporal_unit and 1 - alpha != 0:
            loss += (1 - alpha) * temporal_contrastive_loss(repr1, repr2)
        
        repr1 = paddle.transpose(repr1, perm=[0, 2, 1])
        repr2 = paddle.transpose(repr2, perm=[0, 2, 1])
        repr1 = F.max_pool1d(repr1, kernel_size=2)
        repr2 = F.max_pool1d(repr2, kernel_size=2)
        repr1 = paddle.transpose(repr1, perm=[0, 2, 1])
        repr2 = paddle.transpose(repr2, perm=[0, 2, 1])
        d += 1

    if repr1.shape[1] == 1 and alpha != 0:
        loss += alpha * instance_contrastive_loss(repr1, repr2)
        d += 1
    return loss / d
