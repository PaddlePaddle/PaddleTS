import paddle
import paddle.nn.functional as F
from einops import rearrange, repeat
import numpy as np
from math import sqrt


class FullAttention(paddle.nn.Layer):
    """
    The Attention operation
    """

    def __init__(self, scale=None, attention_dropout=0.1):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.dropout = paddle.nn.Dropout(p=attention_dropout)

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / sqrt(E)
        scores = paddle.einsum('blhe,bshe->bhls', queries, keys)
        A = self.dropout(F.softmax(scale * scores, axis=-1))
        V = paddle.einsum('bhls,bshd->blhd', A, values)
        return V


class AttentionLayer(paddle.nn.Layer):
    """
    The Multi-head Self-Attention (MSA) Layer
    """

    def __init__(self, d_model, n_heads, d_keys=None, d_values=None, mix=
        True, dropout=0.1):
        super(AttentionLayer, self).__init__()
        d_keys = d_keys or d_model // n_heads
        d_values = d_values or d_model // n_heads
        self.inner_attention = FullAttention(scale=None, attention_dropout=
            dropout)
        self.query_projection = paddle.nn.Linear(in_features=d_model,
            out_features=d_keys * n_heads)
        self.key_projection = paddle.nn.Linear(in_features=d_model,
            out_features=d_keys * n_heads)
        self.value_projection = paddle.nn.Linear(in_features=d_model,
            out_features=d_values * n_heads)
        self.out_projection = paddle.nn.Linear(in_features=d_values *
            n_heads, out_features=d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        
        queries = self.query_projection(queries).reshape([B, L, H, -1])
        keys = self.key_projection(keys).reshape([B, S, H, -1])
        values = self.value_projection(values).reshape([B, S, H, -1])
        out = self.inner_attention(queries, keys, values)
        if self.mix:
            x = out
            perm_0 = list(range(x.ndim))
            perm_0[2] = 1
            perm_0[1] = 2
            out = x.transpose(perm=perm_0)
        
        out = out.reshape([B, L, -1])
        return self.out_projection(out)


class TwoStageAttentionLayer(paddle.nn.Layer):
    """
    The Two Stage Attention (TSA) Layer
    input/output shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
    """

    def __init__(self, seg_num, factor, d_model, n_heads, d_ff=None,
        dropout=0.1):
        super(TwoStageAttentionLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.time_attention = AttentionLayer(d_model, n_heads, dropout=dropout)
        self.dim_sender = AttentionLayer(d_model, n_heads, dropout=dropout)
        self.dim_receiver = AttentionLayer(d_model, n_heads, dropout=dropout)
        x = paddle.randn(shape=[seg_num, factor, d_model])
        self.router = paddle.create_parameter(shape=[seg_num, factor, d_model],
                                    dtype=str(x.numpy().dtype),
                                    default_initializer=paddle.nn.initializer.Assign(x))
        self.dropout = paddle.nn.Dropout(p=dropout)
        self.norm1 = paddle.nn.LayerNorm(normalized_shape=d_model, epsilon=
            1e-05, weight_attr=None, bias_attr=None)
        self.norm2 = paddle.nn.LayerNorm(normalized_shape=d_model, epsilon=
            1e-05, weight_attr=None, bias_attr=None)
        self.norm3 = paddle.nn.LayerNorm(normalized_shape=d_model, epsilon=
            1e-05, weight_attr=None, bias_attr=None)
        self.norm4 = paddle.nn.LayerNorm(normalized_shape=d_model, epsilon=
            1e-05, weight_attr=None, bias_attr=None)
        self.MLP1 = paddle.nn.Sequential(paddle.nn.Linear(in_features=
            d_model, out_features=d_ff), paddle.nn.GELU(), paddle.nn.Linear
            (in_features=d_ff, out_features=d_model))
        self.MLP2 = paddle.nn.Sequential(paddle.nn.Linear(in_features=
            d_model, out_features=d_ff), paddle.nn.GELU(), paddle.nn.Linear
            (in_features=d_ff, out_features=d_model))

    def forward(self, x):
        batch = x.shape[0]
        time_in = rearrange(x,
            'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')
        time_enc = self.time_attention(time_in, time_in, time_in)
        dim_in = time_in + self.dropout(time_enc)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        dim_in = self.norm2(dim_in)
        dim_send = rearrange(dim_in,
            '(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model', b=batch)
        batch_router = repeat(self.router,
            'seg_num factor d_model -> (repeat seg_num) factor d_model',
            repeat=batch)
        dim_buffer = self.dim_sender(batch_router, dim_send, dim_send)
        dim_receive = self.dim_receiver(dim_send, dim_buffer, dim_buffer)
        dim_enc = dim_send + self.dropout(dim_receive)
        dim_enc = self.norm3(dim_enc)
        dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
        dim_enc = self.norm4(dim_enc)
        final_out = rearrange(dim_enc,
            '(b seg_num) ts_d d_model -> b ts_d seg_num d_model', b=batch)
        return final_out
