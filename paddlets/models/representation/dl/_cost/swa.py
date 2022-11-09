#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from typing import Callable, Optional
from copy import deepcopy

import paddle


class AveragedModel(paddle.nn.Layer):
    """Paddle layer implementing averaged model for Stochastic Weight Averaging (SWA).

    Args:
        network(paddle.nn.Layer): The network to use with SWA.
        avg_fn(Callable[..., paddle.Tensor]|None): The averaging function used to update parameters.

    Attributes:
        _network(paddle.nn.Layer): The network to use with SWA.
        _avg_fn(Callable[..., paddle.Tensor]): The averaging function used to update parameters.
    """
    def __init__(
        self, 
        network: paddle.nn.Layer,
        avg_fn: Optional[Callable[..., paddle.Tensor]] = None,
    ):
        super(AveragedModel, self).__init__()
        self._network = deepcopy(network)
        self.register_buffer("_num_averaged", paddle.to_tensor(0))

        if avg_fn is None:
            def default_avg_fn(averaged_model_params, model_params, num_averaged):
                return averaged_model_params * 0.999 + model_params * (1 - 0.999)

        self._avg_fn = default_avg_fn

    def forward(self, *args, **kwargs) -> paddle.Tensor:
        """Forward.

        Returns:
            paddle.Tensor: Output of model.
        """
        return self._network(*args, **kwargs)

    def update_parameters(self, network):
        """Synchronize parameters.
        Args:
            network(paddle.nn.Layer): The network parameters referenced 
                when synchronizing parameters.
        """
        src_params = self._network.parameters()
        tgt_params = network.parameters()
        for p_swa, p_model in zip(src_params, tgt_params):
            if self._num_averaged == 0:
                p_swa.detach().set_value(p_model.detach())
            else:
                p_swa.detach().set_value(
                    self._avg_fn(p_swa.detach(), p_model.detach(), self._num_averaged)
                )
        self._num_averaged += 1
