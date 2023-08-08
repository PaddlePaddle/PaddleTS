import paddle
from paddle import nn


class RevIN(paddle.nn.Layer):

    def __init__(self, num_features: int, eps=1e-05, affine=True,
        subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        affine_weight = self.create_parameter(
            [self.num_features],
            default_initializer=nn.initializer.Constant(value=1.0),
            dtype="float32")
        self.add_parameter("affine_weight", affine_weight)
        affine_bias = self.create_parameter(
            [self.num_features],
            default_initializer=nn.initializer.Constant(value=0.0),
            dtype="float32")
        self.add_parameter("affine_bias", affine_bias)

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, (-1), :].unsqueeze(axis=1)
        else:
            self.mean = paddle.mean(x=x, axis=dim2reduce, keepdim=True).detach(
                )
        self.stdev = paddle.sqrt(x=paddle.var(x=x, axis=dim2reduce, keepdim
            =True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x
