import paddle
import paddle.nn.functional as F
import numpy as np
from math import sqrt
from paddlets.models.forecasting.dl._nonstationary.masking import TriangularCausalMask, ProbMask


def masked_fill(x, mask, value):
    y = paddle.full(x.shape, value, x.dtype)
    return paddle.where(mask, y, x)


class DSAttention(paddle.nn.Layer):
    """De-stationary Attention"""

    def __init__(self, mask_flag=True, factor=5, scale=None,
        attention_dropout=0.1, output_attention=False):
        super(DSAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = paddle.nn.Dropout(p=attention_dropout)
        #self.softmax = paddle.nn.Softmax()

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / sqrt(E)
        tau = 1.0 if tau is None else tau.unsqueeze(axis=1).unsqueeze(axis=1)
        delta = 0.0 if delta is None else delta.unsqueeze(axis=1).unsqueeze(
            axis=1)
        scores = paddle.einsum('blhe,bshe->bhls', queries, keys) * tau + delta
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L)

            scores = masked_fill(scores, attn_mask.mask, -np.inf)
        A = self.dropout(F.softmax(scale * scores, axis=-1))
        V = paddle.einsum('bhls,bshd->blhd', A, values)
        if self.output_attention:
            return V, A
        else:
            return V, None


class DSProbAttention(paddle.nn.Layer):
    """De-stationary ProbAttention for Informer"""

    def __init__(self, mask_flag=True, factor=5, scale=None,
        attention_dropout=0.1, output_attention=False):
        super(DSProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = paddle.nn.Dropout(p=attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape
        K_expand = K.unsqueeze(axis=-3).expand(shape=[B, H, L_Q, L_K, E])
        index_sample = paddle.randint(low=L_K, high=(L_Q, sample_k))
        K_sample = K_expand[:, :, (paddle.arange(start=L_Q).unsqueeze(axis=
            1)), (index_sample), :]
        x = K_sample
        perm_0 = list(range(x.ndim))
        perm_0[-2] = -1
        perm_0[-1] = -2
        Q_K_sample = paddle.matmul(x=Q.unsqueeze(axis=-2), y=x.transpose(
            perm=perm_0)).squeeze()
        M = Q_K_sample.max(axis=-1)[0] - paddle.divide(x=Q_K_sample.sum(
            axis=-1), y=L_K)
        M_top = M.topk(k=n_top, sorted=False)[1]
        Q_reduce = Q[(paddle.arange(start=B)[:, (None), (None)]), (paddle.
            arange(start=H)[(None), :, (None)]), (M_top), :]
        x = K
        perm_1 = list(range(x.ndim))
        perm_1[-2] = -1
        perm_1[-1] = -2
        Q_K = paddle.matmul(x=Q_reduce, y=x.transpose(perm=perm_1))
        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            V_sum = V.mean(axis=-2)
            contex = V_sum.unsqueeze(axis=-2).expand(shape=[B, H, L_Q,
                V_sum.shape[-1]]).clone()
        else:
            assert L_Q == L_V
            contex = V.cumsum(axis=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape
        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        attn = paddle.softmax(scores, dim=-1)
        context_in[(paddle.arange(start=B)[:, (None), (None)]), (paddle.
            arange(start=H)[(None), :, (None)]), (index), :] = paddle.matmul(x
            =attn, y=V).astype(dtype=context_in.dtype)
        if self.output_attention:
            
            attns = (paddle.ones(shape=[B, H, L_V, L_V]) / L_V).astype(dtype
                =attn.dtype)
            attns[(paddle.arange(start=B)[:, (None), (None)]), (paddle.
                arange(start=H)[(None), :, (None)]), (index), :] = attn
            return context_in, attns
        else:
            return context_in, None

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape
        x = queries
        perm_2 = list(range(x.ndim))
        perm_2[2] = 1
        perm_2[1] = 2
        queries = x.transpose(perm=perm_2)
        x = keys
        perm_3 = list(range(x.ndim))
        perm_3[2] = 1
        perm_3[1] = 2
        keys = x.transpose(perm=perm_3)
        x = values
        perm_4 = list(range(x.ndim))
        perm_4[2] = 1
        perm_4[1] = 2
        values = x.transpose(perm=perm_4)
        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()
        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part,
            n_top=u)
        tau = 1.0 if tau is None else tau.unsqueeze(axis=1).unsqueeze(axis=1)
        delta = 0.0 if delta is None else delta.unsqueeze(axis=1).unsqueeze(
            axis=1)
        scores_top = scores_top * tau + delta
        scale = self.scale or 1.0 / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        context = self._get_initial_context(values, L_Q)
        context, attn = self._update_context(context, values, scores_top,
            index, L_Q, attn_mask)
        return context, attn


class AttentionLayer(paddle.nn.Layer):

    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None
        ):
        super(AttentionLayer, self).__init__()
        d_keys = d_keys or d_model // n_heads
        d_values = d_values or d_model // n_heads
        self.inner_attention = attention
        self.query_projection = paddle.nn.Linear(in_features=d_model,
            out_features=d_keys * n_heads)
        self.key_projection = paddle.nn.Linear(in_features=d_model,
            out_features=d_keys * n_heads)
        self.value_projection = paddle.nn.Linear(in_features=d_model,
            out_features=d_values * n_heads)
        self.out_projection = paddle.nn.Linear(in_features=d_values *
            n_heads, out_features=d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        queries = self.query_projection(queries).reshape([B, L, H, -1])
        keys = self.key_projection(keys).reshape([B, S, H, -1])
        values = self.value_projection(values).reshape([B, S, H, -1])
        out, attn = self.inner_attention(queries, keys, values, attn_mask,
            tau, delta)
        out = out.reshape([B, L, -1])
        return self.out_projection(out), attn
