#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from typing import List, Dict, Any, Callable, Optional, Tuple
from abc import ABC, abstractmethod

import paddle
from paddle import nn

from paddle.distribution import Distribution
from paddle.distribution import Normal


class Likelihood(ABC):
    """
    Abstract class for a distributional regression model.
    
    Args:
        mode(str): The default value is "distribution" for probability distributional regression, for quantile regression, set "quantiles".
    """
    def __init__(
        self,
        mode: str="distribution"
    ):
        self.mode = mode
    
    @abstractmethod
    def output_to_params(
        self,
        model_output: paddle.Tensor,
    ) -> paddle.Tensor:
        """
        Rescale model output to distribution params, to be implemented in subclasses.

        Args:
            model_output(paddle.Tensor): The output of model.

        Returns:
            paddle.Tensor: The distribution parameters respect to subclass' distribution.
        """
        pass
    
    @abstractmethod
    def params_to_distr(
        self,
        distr_params: paddle.Tensor,
    ) -> Distribution:
        """
        Construct a distribution class by distribution parameters, to be implemented in subclasses.

        Args:
            distr_params(paddle.Tensor): The parameters of distribution.

        Returns:
            Distribution: The distribution instance defined in paddle.
        """
        pass
    
    def sample(
        self,
        model_output: paddle.Tensor,
        num_samples: int = 1,
    ) -> paddle.Tensor:
        """
        Samples a prediction from the model output：

        1> output to distribution parameters;
        
        2> distribution parameters to distribution;
        
        3> sample by distribution

        Args:
            model_output(paddle.Tensor): The output of model.
            num_samples(int): The number of samples to be sampled.
        Returns:
            paddle.Tensor: The samples of distribution.
            
        """
        distr_params = self.output_to_params(model_output)
        distr = self.params_to_distr(distr_params)
        return distr.sample([num_samples])
    
    @abstractmethod
    def get_mean(
        self,
        distr_params: paddle.Tensor
    ) -> paddle.Tensor:
        """
        Compute mean by distribution params, to be implemented in subclasses.
        
        Args:
            distr_params: The params of distribution.

        Returns:
            paddle.Tensor: The mean of distribution.
        """
        pass
    
    @property
    @abstractmethod
    def num_params(self) -> int:
        """
        Returns the number of parameters that define the probability distribution for one single target value.
        
        Returns:
            int: The number of parameters.
        """
        pass
    
    def loss(
        self, 
        distr_params: paddle.Tensor, 
        target: paddle.Tensor
    ) -> paddle.Tensor:
        """
        Compute NLL loss by predicted distrbution parameters and ground truth target.

        This is the basic way to compute the NLL loss. It can be overwritten by likelihood for which paddle proposes a numerically better NLL loss.
        
        Args:
            distr_params: The parameters of distribution.
            target: The ground truth of the sample.

        Returns:
            paddle.Tensor: The loss computed by distribution parameters and ground truth.
        """
        distr = self.params_to_distr(distr_params)
        losses = -distr.log_prob(target)
        return losses.mean()
        

class GaussianLikelihood(Likelihood):
    """
    Univariate Gaussian distribution.
    """
    def __init__(self):
        super(GaussianLikelihood, self).__init__(mode="distribution")
        self.rescale = nn.Softplus()
        
    def output_to_params(
        self,
        model_output: paddle.Tensor,
    ) -> paddle.Tensor:
        """
        Use softplus to rescale sigma parameter as it should be positive.

        Args:
            model_output(paddle.Tensor): The output of model.

        Returns:
            paddle.Tensor: The Gaussian distribution parameters(mu and sigma).
        """
        mu = model_output[..., 0].unsqueeze(-1)
        sigma = self.rescale(model_output[..., 1]).unsqueeze(-1)
        return paddle.concat([mu, sigma], axis=-1)
        
    def params_to_distr(
        self,
        distr_params: paddle.Tensor,
        ) -> Distribution:
        """
        Construct Normal instance by parameters: mu and sigma.

        Args:
            distr_params(paddle.Tensor): Tensor of mu and sigma.

        Returns:
            Distribution: The Gaussian instance defined in paddle.
        """
        mu, sigma = distr_params[..., 0], distr_params[..., 1]
        return Normal(mu, sigma)
    
    def get_mean(
        self,
        distr_params: paddle.Tensor
    ) -> paddle.Tensor:
        """
        Return mean of the distribution.
        
        Args:
            distr_params: Tensor of parameters mu and sigma.

        Returns:
            paddle.Tensor: Mean of Gaussian distribution.

        """
        return distr_params[..., 0]
        
    @property
    def num_params(self) -> int:
        """
        For Gaussian, the number of parameters is 2.

        Returns:
            int: The number of parameters of Gaussian distribution.
        """
        return 2
