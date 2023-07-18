#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from typing import Dict, Tuple

import paddle.nn.functional as F
import numpy as np
import paddle

COVS = ["observed_cov_numeric", "known_cov_numeric"]
PAST_TARGET = "past_target"


def create_ts2vec_inputs(X: Dict[str, paddle.Tensor]) -> paddle.Tensor:
    """`TSDataset` stores time series in the (batch_size, seq_len, target_dim) format.
    Convert it into the shape of (batch_size, seq_len, target_dim + cov_dim)
    as the input of the model.

    Args:
        X(Dict[str, paddle.Tensor]): Dict of feature tensor.

    Returns:
         paddle.Tensor: The inputs of the model.
    """
    feats = [X[col] for col in [PAST_TARGET] + COVS if col in X]
    feats = paddle.concat(feats, axis=-1)
    return feats


def create_contrastive_inputs(
        tensor: paddle.Tensor,
        temporal_unit: int, ) -> Tuple[paddle.Tensor, paddle.Tensor, int]:
    """TS2Vec guarantees the contextual consistency, which treats the representations at 
    the same timestamp in two augmented contexts as positive pairs. A context is generated by 
    random cropping on the input time series. 

    Args:
        tensor(paddle.Tensor): The raw input series.
        temporal_unit(int): The minimum unit to perform temporal contrast. When training on a very long sequence, 
            this param helps to reduce the cost of time and memory.

    Returns:
        Tuple[paddle.Tensor, paddle.Tensor, int]: The augmented contexts.
    """

    def _slice(tensor: paddle.Tensor, start: np.ndarray,
               offset: int) -> paddle.Tensor:
        """Slice into section.
        """
        index = start[:, None] + np.arange(offset)
        return tensor[np.arange(index.shape[0])[:, None], index]

    seq_len = tensor.shape[1]
    overlap_len = np.random.randint(2**(temporal_unit + 1), seq_len + 1)
    series2_left_boundary = np.random.randint(seq_len - overlap_len + 1)
    series1_right_boundary = series2_left_boundary + overlap_len
    series1_left_boundary = np.random.randint(series2_left_boundary + 1)
    series2_right_boundary = np.random.randint(series1_right_boundary,
                                               seq_len + 1)
    turbulence = np.random.randint(-series1_left_boundary,
                                   seq_len - series2_right_boundary + 1,
                                   tensor.shape[0])
    aug1 = _slice(tensor, turbulence + series1_left_boundary,
                  series1_right_boundary - series1_left_boundary)
    aug2 = _slice(tensor, turbulence + series2_left_boundary,
                  series2_right_boundary - series2_left_boundary)
    return aug1, aug2, overlap_len


def instance_level_encoding(tensor: paddle.Tensor) -> paddle.Tensor:
    """Compute instance-level representations.

    Args:
        tensor(paddle.Tensor): The raw input series.

    Returns:
        paddle.Tensor: The instance-level representations.
    """
    out = paddle.transpose(tensor, perm=[0, 2, 1])
    out = F.max_pool1d(out, kernel_size=out.shape[-1])
    out = paddle.transpose(out, perm=[0, 2, 1])
    return out


def multiscale_encoding(tensor: paddle.Tensor, ) -> paddle.Tensor:
    """Compute multi-scale representations(pooling with different window sizes).

    Args:
        tensor: The Raw input series.

    Returns:
        paddle.Tensor: The multi-scale representations.
    """
    out, p, reprs = tensor, 0, []
    while (1 << p) + 1 < out.shape[1]:
        aggregate_out = F.max_pool1d(
            paddle.transpose(
                out, perm=[0, 2, 1]),
            kernel_size=(1 << (p + 1)) + 1,
            padding=(1 << p))
        aggregate_out = paddle.transpose(aggregate_out, perm=[0, 2, 1])
        reprs.append(aggregate_out[:, -1:, :])
        p += 1
    out = paddle.concat(reprs, axis=-1)
    return out


def centerize_effective_series(tensor: paddle.Tensor) -> paddle.Tensor:
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
    """In order to align with the paper of TS2Vec, a customized data organization is required.

    Args:
        samples(list): The raw sample list.

    Returns:
        list: The reorganized sample list.
    """

    def _padding_series_with_equal_length(arr, target_len, axis):
        """padding.
        """
        pad_size = target_length - arr.shape[axis]
        if pad_size <= 0:
            return arr
        npad = [(0, 0)] * arr.ndim
        npad[axis] = (0, pad_size)
        return np.pad(arr,
                      pad_width=npad,
                      mode="constant",
                      constant_values=np.nan)

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
                    arrs[index], target_length, axis=0)
        else:
            arrs = [cache]
        chains.append(arrs)
    return [dict(zip(COLS, item)) for item in zip(*chains)]
