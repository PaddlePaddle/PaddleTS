# !/usr/bin/env python3
# -*- coding:utf-8 -*-

import paddle
from paddle.distribution import Normal
import unittest
from unittest import TestCase
from paddlets.models.forecasting.dl.distributions import GaussianLikelihood


class TestDistribution(TestCase):
    def setUp(self):
        """unittest function
        """
        super().setUp()

    def test_normal(self):
        """unittest function
        """
        batch_size = 32
        seq_len = 24
        target_dim = 4
        likelihood = GaussianLikelihood()

        # case0: num_params
        num_params = likelihood.num_params
        self.assertEqual(num_params, 2)

        # case1: output to params
        model_output = paddle.rand([batch_size, seq_len, target_dim, num_params])
        params = likelihood.output_to_params(model_output)
        self.assertEqual(params.shape, [batch_size, seq_len, target_dim, num_params])
        
        # case2: params to distr
        distr = likelihood.params_to_distr(params)
        self.assertEqual(type(distr), Normal)

        # case3: sample
        num_samples = 100
        samples = likelihood.sample(model_output, num_samples)
        self.assertEqual(samples.shape, [num_samples, batch_size, seq_len, target_dim])

        # case4: get mean
        mean = likelihood.get_mean(params)
        self.assertEqual(mean.shape, [batch_size, seq_len, target_dim])

        # case5: loss
        l = likelihood.loss(params, paddle.rand([batch_size, seq_len, target_dim]))
        self.assertEqual((l.dtype, l.shape), (paddle.float32, [1]))

if __name__ == "__main__":
    unittest.main()
