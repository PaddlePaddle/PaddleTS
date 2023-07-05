import paddle
import math
from paddlets.utils import param_init


class PositionalEmbedding(paddle.nn.Layer):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = paddle.zeros(shape=[max_len, d_model]).astype(dtype='float32')
        pe.stop_gradient = True
        position = paddle.arange(start=0, end=max_len).astype(dtype='float32'
            ).unsqueeze(axis=1)
        div_term = (paddle.arange(start=0, end=d_model, step=2).astype(
            dtype='float32') * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = paddle.sin(x=position * div_term)
        pe[:, 1::2] = paddle.cos(x=position * div_term)
        pe = pe.unsqueeze(axis=0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.shape[1]]


class TokenEmbedding(paddle.nn.Layer):

    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if paddle.__version__ >= '1.5.0' else 2
        self.tokenConv = paddle.nn.Conv1D(in_channels=c_in, out_channels=
            d_model, kernel_size=3, padding=padding, padding_mode=
            'circular', bias_attr=False)
        self.init_weight()
        
    def init_weight(self):
        for m in self.sublayers():
            if isinstance(m, paddle.nn.Conv1D):
                param_init.kaiming_normal_init(m.weight,
                    nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.transpose(perm=[0, 2, 1]))
        perm_0 = list(range(x.ndim))
        perm_0[1] = 2
        perm_0[2] = 1
        x = x.transpose(perm=perm_0)
        return x


class FixedEmbedding(paddle.nn.Layer):

    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()
        w = paddle.zeros(shape=[c_in, d_model]).astype(dtype='float32')
        w.require_grad = False
        position = paddle.arange(start=0, end=c_in).astype(dtype='float32'
            ).unsqueeze(axis=1)
        div_term = (paddle.arange(start=0, end=d_model, step=2).astype(
            dtype='float32') * -(math.log(10000.0) / d_model)).exp()
        w[:, 0::2] = paddle.sin(x=position * div_term)
        w[:, 1::2] = paddle.cos(x=position * div_term)
        self.emb = paddle.nn.Embedding(c_in, d_model)
        self.emb.weight = paddle.nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(paddle.nn.Layer):

    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()
        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13
        Embed = FixedEmbedding if embed_type == 'fixed' else paddle.nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.astype(dtype='int64')
        minute_x = self.minute_embed(x[:, :, (4)]) if hasattr(self,
            'minute_embed') else 0.0
        hour_x = self.hour_embed(x[:, :, (3)])
        weekday_x = self.weekday_embed(x[:, :, (2)])
        day_x = self.day_embed(x[:, :, (1)])
        month_x = self.month_embed(x[:, :, (0)])
        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(paddle.nn.Layer):

    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()
        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3,
            'b': 3}
        d_inp = freq_map[freq]
        self.embed = paddle.nn.Linear(in_features=d_inp, out_features=
            d_model, bias_attr=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(paddle.nn.Layer):

    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1
        ):
        super(DataEmbedding, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model,
            embed_type=embed_type, freq=freq
            ) if embed_type != 'timeF' else TimeFeatureEmbedding(d_model=
            d_model, embed_type=embed_type, freq=freq)
        self.dropout = paddle.nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.temporal_embedding(x_mark
            ) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbedding_wo_pos(paddle.nn.Layer):

    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1
        ):
        super(DataEmbedding_wo_pos, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model,
            embed_type=embed_type, freq=freq
            ) if embed_type != 'timeF' else TimeFeatureEmbedding(d_model=
            d_model, embed_type=embed_type, freq=freq)
        self.dropout = paddle.nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)
