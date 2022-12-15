#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File Name:     revin.py
# Author:        Yangrun
# Created Time:  2022-12-07  17:40
# Last Modified: <none>-<none>

from typing import Dict
import functools
import numpy as np
import paddle
from paddle import nn
from paddlets.logger import raise_if_not, Logger
from paddlets.models import BaseModel

logger = Logger(__name__)


class RevinWrapper(nn.Layer):
    def __init__(self, base_net: nn.Layer, num_features: int, eps=1e-5, affine=True):
        """
        The paddlepaddle implementation of Reversible Instance Normalization for
        Accurate Time-Series Forecasting against Distribution Shift (RevIN)

        The RevIN (Reversible Instance Normalization) is a simple yet effective normalization method
        to mitigating the negaitve effects of temporal distribution shift problem.

        This method has been proved to be effective in improving the performance of various existing models.

        @inproceedings{kim2021reversible,
            title={Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift},
            author={Kim, Taesung and Kim, Jinhee and Tae, Yunwon and Park, Cheonbok and Choi, Jang-Ho and Choo, Jaegul},
            booktitle = {International Conference on Learning Representations},
            year= {2021},
            url= {https://openreview.net/forum?id=cGDAkQo1C0p}
            }

        This code is based on the official Pytorch Implementation (https://github.com/ts-kim/RevIN)


        Args:
        base_net(nn.Layer): the base network
        num_features (int): the number of features or channels
        eps (float): a value added for numerical stability
        affine (float): if True, RevIN has learnable affine parameters
        """

        super(RevinWrapper, self).__init__()
        self._base_net = base_net
        self.num_features = num_features
        eps_buffer = paddle.to_tensor(np.array([eps]).astype(np.float32))
        self.register_buffer("eps", eps_buffer, persistable=True)
        self.affine = affine
        if affine:
            self._init_params()

    def _init_params(self):
        affine_weight = self.create_parameter([self.num_features],
                                              default_initializer=nn.initializer.Constant(value=1.0),
                                              dtype="float32")
        self.add_parameter("affine_weight", affine_weight)
        affine_bias = self.create_parameter([self.num_features],
                                            default_initializer=nn.initializer.Constant(value=0.0),
                                            dtype="float32")
        self.add_parameter("affine_bias", affine_bias)

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = paddle.mean(x, axis=dim2reduce, keepdim=True).detach()
        self.stdev = paddle.sqrt(paddle.var(x, axis=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        x = x - self.affine_bias
        x = x / (self.affine_weight + self.eps*self.eps)
        if self.affine:
            x = x * self.stdev
            x = x + self.mean
        return x

    def forward(self, data: Dict[str, paddle.Tensor]):
        """
        The past_target is first normalizated by the revin and the fed into the base model
        Args:
        data(Dict[str, paddle.Tensor]): a dict specifies all kinds of input data
        Returns:
            predictions: output of RNN model, with shape [batch_size, out_chunk_len, target_dim]
        """
        past_target = data['past_target']
        self._get_statistics(past_target)
        data['past_target'] = self._normalize(past_target)
        out = self._base_net(data)
        raise_if_not(
                isinstance(out, paddle.Tensor),
                'RevIN only support forcasting schema'
                )
        out = self._denormalize(out)
        return out


def revin_norm(func):
    @functools.wraps(func)
    def wrapper(obj: BaseModel, *args, **kwargs):
        """
        The core logic. The base model is been wrappered by the RevinWrapper.
        """
        model = func(obj, *args, **kwargs)
        if obj._use_revin:
            logger.warning("Using reversible instance normalization (revin) to remove and restore the statistical information of a time-series instance")
            model = RevinWrapper(model, obj._fit_params['target_dim'], **obj._revin_params)
        return model
    return wrapper
