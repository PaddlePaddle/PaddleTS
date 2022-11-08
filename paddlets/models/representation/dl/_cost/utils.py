#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from typing import List, Dict, Tuple

import paddle.nn.functional as F
import numpy as np
import paddle

COVS = ["observed_cov_numeric", "known_cov_numeric"]
PAST_TARGET = "past_target"


def create_cost_inputs(
    X: Dict[str, paddle.Tensor]
) -> paddle.Tensor:
    """`TSDataset` stores time series in the (batch_size, seq_len, target_dim) format.
    Convert it into the shape of (batch_size, seq_len, target_dim + cov_dim)
    as the input of the model.

    Args:
        X(Dict[str, paddle.Tensor]): Dict of feature tensor.

    Returns:
         paddle.Tensor: The inputs of the model.
    """
    feats = [
        X[col] for col in [PAST_TARGET] + COVS if col in X
    ]
    feats = paddle.concat(feats, axis=-1)
    return feats


def create_contrastive_inputs(
    tensor: paddle.Tensor,
) -> paddle.Tensor:
    """CoST uses data augmentations as interventions on the error and learn invariant representations 
    of trend and season via constrastive learning. Since it is impossible to generate all possible variations of errors, 
    CoST selects three typical augmentations: scale, shift and jitter, which can simulate a large and diverse set of errors, 
    beneficial for learning better representations.

    Args:
        tensor(paddle.Tensor): The raw input series.

    Returns:
        paddle.Tensor: The augmented contexts.
    """
    if np.random.random() < 0.5:
        tensor = tensor * (paddle.randn(tensor.shape[-1:]) * 0.5 + 1)
    if np.random.random() < 0.5:
        tensor = tensor + (paddle.randn(tensor.shape[-1:]) * 0.5)
    if np.random.random() < 0.5:
        tensor = tensor + (paddle.randn(tensor.shape) * 0.5)
    return tensor


def centerize_effective_series(
    tensor: paddle.Tensor
) -> paddle.Tensor:
    """In order to ensure that the sampling falls in the effective area as much as possible,    
    the series needs to be centerized. i.e. [nan, nan, nan, 1, 1, nan] -> [nan, nan, 1, 1, nan, nan].
    
    Args:
        tensor(paddle.Tensor): The series to be centerized.

    Returns
        paddle.Tensor: The centerized series.
    """
    tensor = tensor.numpy()
    batch_size, seq_len = tensor.shape[:2]
    forward_nan_mask = np.isnan(tensor).all(axis=-1)
    backward_nan_mask = np.isnan(tensor[:, ::-1]).all(axis=-1)
    first_valid_index = np.argmax(~forward_nan_mask, axis=1)
    last_valid_index = np.argmax(~backward_nan_mask, axis=1)
    offset = (first_valid_index + last_valid_index) // 2 - first_valid_index
    rows, column_indices = np.ogrid[:batch_size, :seq_len]
    offset[offset < 0] += seq_len
    column_indices = column_indices - offset[:, None]
    return paddle.to_tensor(tensor[rows, column_indices])


def custom_collate_fn(samples: list):
    """In order to align with the paper of CoST, a customized data organization is required.

    Args:
        samples(list): The raw sample list.

    Returns:
        List[Dict[str, np.ndarray]]: The reorganized sample list.
    """
    def _padding_series_with_equal_length(arr, target_len, axis):
        """padding.
        """
        pad_size = target_length - arr.shape[axis]
        if pad_size <= 0:
            return arr
        npad = [(0, 0)] * arr.ndim
        npad[axis] = (0, pad_size)
        return np.pad(arr, pad_width=npad, mode="constant", constant_values=np.nan)

    from collections import defaultdict
    COLS = [col for col in [PAST_TARGET] + COVS if col in samples[0]]
    segment_size = len(samples[0][PAST_TARGET])
    sample_dict, chains = defaultdict(list), []
    for col in COLS:
        for sample in samples:
            sample_dict[col].append(sample[col])
        cache = np.vstack(sample_dict[col])
        mask = (~np.isnan(cache).any(axis=-1).ravel())
        cache = cache[mask]

        sections = len(cache) // segment_size
        if sections >= 2:
            arrs = np.array_split(cache, sections, axis=0)
            target_length = arrs[0].shape[0]
            for index in range(len(arrs)):
                arrs[index] = _padding_series_with_equal_length(
                    arrs[index], target_length, axis=0
                )
        else:
            arrs = [cache]
        chains.append(arrs)
    return [dict(zip(COLS, item)) for item in zip(*chains)]
