import paddle
import math

__all__ = ['Transpose', 'get_activation_fn', 'moving_avg', 'series_decomp',
    'PositionalEncoding', 'SinCosPosEncoding', 'Coord2dPosEncoding',
    'Coord1dPosEncoding', 'positional_encoding']


class Transpose(paddle.nn.Layer):

    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose([*self.dims])
        else:
            return x.transpose([*self.dims])


def get_activation_fn(activation):
    if callable(activation):
        return activation()
    elif activation.lower() == 'relu':
        return paddle.nn.ReLU()
    elif activation.lower() == 'gelu':
        return paddle.nn.GELU()
    raise ValueError(
        f'{activation} is not available. You can use "relu", "gelu", or a callable'
        )


class moving_avg(paddle.nn.Layer):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = paddle.nn.AvgPool1d(kernel_size=kernel_size, stride=
            stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].tile(repeat_times=[1, (self.kernel_size - 1) //
            2, 1])
        end = x[:, -1:, :].tile(repeat_times=[1, (self.kernel_size - 1) // 
            2, 1])
        x = paddle.concat(x=[front, x, end], axis=1)
        x = self.avg(x.transpose(perm=[0, 2, 1]))
        x = x.transpose(perm=[0, 2, 1])
        return x


class series_decomp(paddle.nn.Layer):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


def PositionalEncoding(q_len, d_model, normalize=True):
    pe = paddle.zeros(shape=[q_len, d_model])
    position = paddle.arange(start=0, end=q_len).unsqueeze(axis=1)
    div_term = paddle.exp(x=paddle.arange(start=0, end=d_model, step=2) * -
        (math.log(10000.0) / d_model))
    pe[:, 0::2] = paddle.sin(x=position * div_term)
    pe[:, 1::2] = paddle.cos(x=position * div_term)
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
    return pe


SinCosPosEncoding = PositionalEncoding


def Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True,
    eps=0.001, verbose=False):
    x = 0.5 if exponential else 1
    i = 0
    for i in range(100):
        cpe = 2 * paddle.linspace(start=0, stop=1, num=q_len).reshape(-1, 1
            ) ** x * paddle.linspace(start=0, stop=1, num=d_model).reshape(
            1, -1) ** x - 1
        #pv(f'{i:4.0f}  {x:5.3f}  {cpe.mean():+6.3f}', verbose)
        if abs(cpe.mean()) <= eps:
            break
        elif cpe.mean() > eps:
            x += 0.001
        else:
            x -= 0.001
        i += 1
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe


def Coord1dPosEncoding(q_len, exponential=False, normalize=True):
    cpe = 2 * paddle.linspace(start=0, stop=1, num=q_len).reshape(-1, 1) ** (
        0.5 if exponential else 1) - 1
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe


def positional_encoding(pe, learn_pe, q_len, d_model):
    if pe == None:
        W_pos = paddle.empty(shape=(q_len, d_model))
        paddle.nn.initializer.Uniform(W_pos, -0.02, 0.02)
        learn_pe = False
    elif pe == 'zero':
        W_pos = paddle.empty(shape=(q_len, 1))
        paddle.nn.initializer.Uniform(W_pos, -0.02, 0.02)
    elif pe == 'zeros':
        # W_pos = paddle.empty(shape=(q_len, d_model))
        # paddle.nn.initializer.Uniform(W_pos, -0.02, 0.02)
        W_pos = paddle.uniform(shape=[q_len, d_model], min=-0.02, max=0.02)
    elif pe == 'normal' or pe == 'gauss':
        W_pos = paddle.zeros(shape=(q_len, 1))
        paddle.nn.initializer.Uniform(W_pos, mean=0.0, std=0.1)
    elif pe == 'uniform':
        W_pos = paddle.zeros(shape=(q_len, 1))
        paddle.nn.initializer.Uniform(W_pos, a=0.0, b=0.1)
    elif pe == 'lin1d':
        W_pos = Coord1dPosEncoding(q_len, exponential=False, normalize=True)
    elif pe == 'exp1d':
        W_pos = Coord1dPosEncoding(q_len, exponential=True, normalize=True)
    elif pe == 'lin2d':
        W_pos = Coord2dPosEncoding(q_len, d_model, exponential=False,
            normalize=True)
    elif pe == 'exp2d':
        W_pos = Coord2dPosEncoding(q_len, d_model, exponential=True,
            normalize=True)
    elif pe == 'sincos':
        W_pos = PositionalEncoding(q_len, d_model, normalize=True)
    else:
        raise ValueError(
            f"{pe} is not a valid pe (positional encoder. Available types: 'gauss'=='normal',         'zeros', 'zero', uniform', 'lin1d', 'exp1d', 'lin2d', 'exp2d', 'sincos', None.)"
            )
    W_pos = paddle.create_parameter(shape=W_pos.shape,default_initializer=paddle.nn.initializer.Assign(W_pos), dtype=str(W_pos.numpy().dtype))
    W_pos.stop_gradient = learn_pe
    return W_pos

