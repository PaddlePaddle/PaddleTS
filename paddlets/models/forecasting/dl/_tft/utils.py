#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
Implements util classes of TFT model.
"""

import paddle
from paddle import nn


class TimeDistributed(nn.Layer):
    """
    This module can wrap any given module and stacks the time dimension with the batch dimension of the inputs
    before applying the module.

    Args:
        module(nn.Layer): The wrapped module.
        batch_first(bool): A boolean indicating whether the batch dimension is expected to be the first dimension of the input or not.
        return_reshaped(bool): A boolean indicating whether to return the output in the corresponding original shape or not.
    """

    def __init__(
        self, 
        module: nn.Layer, 
        batch_first: bool = True, 
        return_reshaped: bool = True
    ):
        super(TimeDistributed, self).__init__()
        self.module: nn.Layer = module  # the wrapped module
        self.batch_first: bool = batch_first  # indicates the dimensions order of the sequential data.
        self.return_reshaped: bool = return_reshaped

    def forward(
        self, 
        x: paddle.Tensor
    ) -> paddle.Tensor:
        """
        The forward computation of time distributed layer.
        
        Args:
            x(paddle.Tensor): The input tensor.
            
        Returns:
            y(paddle.Tensor): The result tensor.
        """

        # in case the incoming tensor is a two-dimensional tensor - inferring temporal information is not involved,
        # and simply apply the module
        if len(paddle.shape(x)) <= 2:
            return self.module(x)

        # Squash samples and time-steps into a single axis
        x_reshape = x.reshape([-1, x.shape[-1]]) # (samples * time-steps, input_size)
        # apply the module on each time-step separately
        y = self.module(x_reshape)

        # reshaping the module output as sequential tensor (if required)
        if self.return_reshaped:
            if self.batch_first:
                #y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, time-steps, output_size)
                y = y.reshape([paddle.shape(x)[0], -1, y.shape[-1]])
            else:
                y = y.reshape([-1, x.shape[1], y.shape[-1]])  # (time-steps, samples, output_size)
        return y


class NullTransform(nn.Layer):
    """
    Define the null transformation operation for the case that input data is empty tensor.
    """
    def __init__(self):
        super(NullTransform, self).__init__()

    @staticmethod
    def forward(empty_input: paddle.Tensor) -> list:
        """
        In case input data is empty tensor, return an empty list.
        
        Args:
            empty_input(paddle.Tensor): The empty tensor input.
            
        Returns:
            list: The empty list of Null transform.
        """
        return []

