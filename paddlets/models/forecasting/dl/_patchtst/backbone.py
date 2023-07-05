import paddle
from typing import Callable, Optional
import numpy as np
from paddlets.models.forecasting.dl._patchtst.layer import positional_encoding, Transpose, get_activation_fn
from paddlets.models.forecasting.dl._patchtst.revin import RevIN


def unfold(tensor, dimension, size, step=1):
    assert dimension < len(tensor.shape), 'dimension must be less than tensor dimensions'
    assert tensor.shape[dimension] >= size, 'size should not be greater than the dimension of tensor'
    
    slices = []
    for i in range(0, tensor.shape[dimension] - size + 1, step):
        start = [0] * len(tensor.shape)
        end = list(tensor.shape)
        start[dimension] = i
        end[dimension] = i + size
        axes = list(range(len(start)))
        slice = paddle.slice(tensor, axes, start, end)
        slices.append(slice)

    unfolded_tensor = paddle.stack(slices, axis=dimension)

    return unfolded_tensor

class PatchTST_backbone(paddle.nn.Layer):

    def __init__(self, c_in: int, context_window: int, target_window: int,
        patch_len: int, stride: int, max_seq_len: Optional[int]=1024,
        n_layers: int=3, d_model=128, n_heads=16, d_k: Optional[int]=None,
        d_v: Optional[int]=None, d_ff: int=256, norm: str='BatchNorm',
        attn_dropout: float=0.0, dropout: float=0.0, act: str='gelu',
        key_padding_mask: bool='auto', padding_var: Optional[int]=None,
        attn_mask: Optional[paddle.Tensor]=None, res_attention: bool=True,
        pre_norm: bool=False, store_attn: bool=False, pe: str='zeros',
        learn_pe: bool=True, fc_dropout: float=0.0, head_dropout=0,
        padding_patch=None, pretrain_head: bool=False, head_type='flatten',
        individual=False, revin=True, affine=True, subtract_last=False,
        verbose: bool=False, **kwargs):
        super().__init__()
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(c_in, affine=affine, subtract_last=
                subtract_last)
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((context_window - patch_len) / stride + 1)
        if padding_patch == 'end':
            self.padding_patch_layer = paddle.nn.Pad1D(padding=(0, stride),mode = "replicate")
            patch_num += 1
        self.backbone = TSTiEncoder(c_in, patch_num=patch_num, patch_len=
            patch_len, max_seq_len=max_seq_len, n_layers=n_layers, d_model=
            d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
            attn_dropout=attn_dropout, dropout=dropout, act=act,
            key_padding_mask=key_padding_mask, padding_var=padding_var,
            attn_mask=attn_mask, res_attention=res_attention, pre_norm=
            pre_norm, store_attn=store_attn, pe=pe, learn_pe=learn_pe,
            verbose=verbose, **kwargs)
        self.head_nf = d_model * patch_num
        self.n_vars = c_in
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.individual = individual
        if self.pretrain_head:
            self.head = self.create_pretrain_head(self.head_nf, c_in,
                fc_dropout)
        elif head_type == 'flatten':
            self.head = Flatten_Head(self.individual, self.n_vars, self.
                head_nf, target_window, head_dropout=head_dropout)

    def forward(self, z):
        if self.revin:
            z = z.transpose(perm=[0, 2, 1])
            z = self.revin_layer(z, 'norm')
            z = z.transpose(perm=[0, 2, 1])
        
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)

        z = unfold(z, dimension=-1, size=self.patch_len, step=self.stride)
        z = self.backbone(z)
        z = self.head(z)
        if self.revin:
            z = z.transpose(perm=[0, 2, 1])
            z = self.revin_layer(z, 'denorm')
            z = z.transpose(perm=[0, 2, 1])
        return z

    def create_pretrain_head(self, head_nf, vars, dropout):
        return paddle.nn.Sequential(paddle.nn.Dropout(p=dropout), paddle.nn
            .Conv1D(in_channels=head_nf, out_channels=vars, kernel_size=1))


class Flatten_Head(paddle.nn.Layer):

    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.individual = individual
        self.n_vars = n_vars
        if self.individual:
            self.linears = paddle.nn.LayerList()
            self.dropouts = paddle.nn.LayerList()
            self.flattens = paddle.nn.LayerList()
            for i in range(self.n_vars):
                self.flattens.append(paddle.nn.Flatten(start_axis=-2))
                self.linears.append(paddle.nn.Linear(in_features=nf,
                    out_features=target_window))
                self.dropouts.append(paddle.nn.Dropout(p=head_dropout))
        else:
            self.flatten = paddle.nn.Flatten(start_axis=-2)
            self.linear = paddle.nn.Linear(in_features=nf, out_features=
                target_window)
            self.dropout = paddle.nn.Dropout(p=head_dropout)

    def forward(self, x):
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, (i), :, :])
                z = self.linears[i](z)
                z = self.dropouts[i](z)
                x_out.append(z)
            x = paddle.stack(x=x_out, axis=1)
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x


class TSTiEncoder(paddle.nn.Layer):

    def __init__(self, c_in, patch_num, patch_len, max_seq_len=1024,
        n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None, d_ff=256,
        norm='BatchNorm', attn_dropout=0.0, dropout=0.0, act='gelu',
        store_attn=False, key_padding_mask='auto', padding_var=None,
        attn_mask=None, res_attention=True, pre_norm=False, pe='zeros',
        learn_pe=True, verbose=False, **kwargs):
        super().__init__()
        self.patch_num = patch_num
        self.patch_len = patch_len
        q_len = patch_num
        self.W_P = paddle.nn.Linear(in_features=patch_len, out_features=d_model
            )
        self.seq_len = q_len
        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)
        self.dropout = paddle.nn.Dropout(p=dropout)
        self.encoder = TSTEncoder(q_len, d_model, n_heads, d_k=d_k, d_v=d_v,
            d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=
            dropout, pre_norm=pre_norm, activation=act, res_attention=
            res_attention, n_layers=n_layers, store_attn=store_attn)

    def forward(self, x) ->paddle.Tensor:
        n_vars = x.shape[1]
        x = x.transpose(perm=[0, 1, 3, 2])
        x = self.W_P(x)
        u = paddle.reshape(x=x, shape=(x.shape[0] * x.shape[1], x.shape[2],
            x.shape[3]))
        u = self.dropout(u + self.W_pos)
        z = self.encoder(u)
        z = paddle.reshape(x=z, shape=(-1, n_vars, z.shape[-2], z.shape[-1]))
        z = z.transpose(perm=[0, 1, 3, 2])
        return z


class TSTEncoder(paddle.nn.Layer):

    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=
        None, norm='BatchNorm', attn_dropout=0.0, dropout=0.0, activation=
        'gelu', res_attention=False, n_layers=1, pre_norm=False, store_attn
        =False):
        super().__init__()
        self.layers = paddle.nn.LayerList(sublayers=[TSTEncoderLayer(q_len,
            d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=
            norm, attn_dropout=attn_dropout, dropout=dropout, activation=
            activation, res_attention=res_attention, pre_norm=pre_norm,
            store_attn=store_attn) for i in range(n_layers)])
        self.res_attention = res_attention

    def forward(self, src: paddle.Tensor, key_padding_mask: Optional[paddle
        .Tensor]=None, attn_mask: Optional[paddle.Tensor]=None):
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers:
                output, scores = mod(output, prev=scores, key_padding_mask=
                    key_padding_mask, attn_mask=attn_mask)
            return output
        else:
            for mod in self.layers:
                output = mod(output, key_padding_mask=key_padding_mask,
                    attn_mask=attn_mask)
            return output


class TSTEncoderLayer(paddle.nn.Layer):

    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=
        256, store_attn=False, norm='BatchNorm', attn_dropout=0, dropout=
        0.0, bias=True, activation='gelu', res_attention=False, pre_norm=False
        ):
        super().__init__()
        assert not d_model % n_heads, f'd_model ({d_model}) must be divisible by n_heads ({n_heads})'
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v,
            attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=
            res_attention)
        self.dropout_attn = paddle.nn.Dropout(p=dropout)
        if 'batch' in norm.lower():
            self.norm_attn = paddle.nn.Sequential(Transpose(0,2, 1), paddle.
                nn.BatchNorm1D(num_features=d_model, momentum=1 - 0.1,
                epsilon=1e-05, weight_attr=None, bias_attr=None,
                use_global_stats=True), Transpose(0,2, 1))
        else:
            self.norm_attn = paddle.nn.LayerNorm(normalized_shape=d_model,
                epsilon=1e-05, weight_attr=None, bias_attr=None)
        self.ff = paddle.nn.Sequential(paddle.nn.Linear(in_features=d_model,
            out_features=d_ff, bias_attr=bias), get_activation_fn(
            activation), paddle.nn.Dropout(p=dropout), paddle.nn.Linear(
            in_features=d_ff, out_features=d_model, bias_attr=bias))
        self.dropout_ffn = paddle.nn.Dropout(p=dropout)
        if 'batch' in norm.lower():
            self.norm_ffn = paddle.nn.Sequential(Transpose(0, 2, 1), paddle.nn
                .BatchNorm1D(num_features=d_model, momentum=1 - 0.1,
                epsilon=1e-05, weight_attr=None, bias_attr=None,
                use_global_stats=True), Transpose(0, 2, 1 ))
        else:
            self.norm_ffn = paddle.nn.LayerNorm(normalized_shape=d_model,
                epsilon=1e-05, weight_attr=None, bias_attr=None)
        self.pre_norm = pre_norm
        self.store_attn = store_attn

    def forward(self, src: paddle.Tensor, prev: Optional[paddle.Tensor]=
        None, key_padding_mask: Optional[paddle.Tensor]=None, attn_mask:
        Optional[paddle.Tensor]=None) ->paddle.Tensor:
        if self.pre_norm:
            src = self.norm_attn(src)
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev,
                key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            src2, attn = self.self_attn(src, src, src, key_padding_mask=
                key_padding_mask, attn_mask=attn_mask)
        if self.store_attn:
            self.attn = attn
        src = src + self.dropout_attn(src2)
        if not self.pre_norm:
            src = self.norm_attn(src)
        if self.pre_norm:
            src = self.norm_ffn(src)
        src2 = self.ff(src)
        src = src + self.dropout_ffn(src2)
        if not self.pre_norm:
            src = self.norm_ffn(src)
        if self.res_attention:
            return src, scores
        else:
            return src


class _MultiheadAttention(paddle.nn.Layer):

    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=
        False, attn_dropout=0.0, proj_dropout=0.0, qkv_bias=True, lsa=False):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v
        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v
        self.W_Q = paddle.nn.Linear(in_features=d_model, out_features=d_k *
            n_heads, bias_attr=qkv_bias)
        self.W_K = paddle.nn.Linear(in_features=d_model, out_features=d_k *
            n_heads, bias_attr=qkv_bias)
        self.W_V = paddle.nn.Linear(in_features=d_model, out_features=d_v *
            n_heads, bias_attr=qkv_bias)
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads,
            attn_dropout=attn_dropout, res_attention=self.res_attention,
            lsa=lsa)
        self.to_out = paddle.nn.Sequential(paddle.nn.Linear(in_features=
            n_heads * d_v, out_features=d_model), paddle.nn.Dropout(p=
            proj_dropout))

    def forward(self, Q: paddle.Tensor, K: Optional[paddle.Tensor]=None, V:
        Optional[paddle.Tensor]=None, prev: Optional[paddle.Tensor]=None,
        key_padding_mask: Optional[paddle.Tensor]=None, attn_mask: Optional
        [paddle.Tensor]=None):
        bs = Q.shape[0]
        if K is None:
            K = Q
        if V is None:
            V = Q
        
        q_s = self.W_Q(Q).reshape([bs, -1, self.n_heads, self.d_k]).transpose([0,2,1,3])
        k_s = self.W_K(K).reshape([bs, -1, self.n_heads, self.d_k]).transpose(perm
            =[0, 2, 3, 1])
        v_s = self.W_V(V).reshape([bs, -1, self.n_heads, self.d_v]).transpose([0,2,1,3])
        # perm_1 = list(range(x.ndim))
        # perm_1[1] = 2
        # perm_1[2] = 1
        # v_s = x.transpose(perm=perm_1)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s,
                prev=prev, key_padding_mask=key_padding_mask, attn_mask=
                attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s,
                key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        x = output
        perm_2 = list(range(x.ndim))
        perm_2[1] = 2
        perm_2[2] = 1
        
        output = x.transpose(perm=perm_2).reshape([bs, -1, self.n_heads * self.d_v])
        output = self.to_out(output)
        if self.res_attention:
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights


class _ScaledDotProductAttention(paddle.nn.Layer):
    """Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0.0, res_attention=
        False, lsa=False):
        super().__init__()
        self.attn_dropout = paddle.nn.Dropout(p=attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        """Class Method """
        x = paddle.to_tensor(head_dim ** - 0.5)
        self.scale = paddle.create_parameter(shape=x.shape,
                        dtype=str(x.numpy().dtype),
                        default_initializer=paddle.nn.initializer.Assign(x))
        self.scale.stop_gradient = lsa
        self.lsa = lsa

    def forward(self, q: paddle.Tensor, k: paddle.Tensor, v: paddle.Tensor,
        prev: Optional[paddle.Tensor]=None, key_padding_mask: Optional[
        paddle.Tensor]=None, attn_mask: Optional[paddle.Tensor]=None):
        """
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        """
        attn_scores = paddle.matmul(x=q, y=k) * self.scale
        if prev is not None:
            attn_scores = attn_scores + prev
        if attn_mask is not None:
            if attn_mask.dtype == 'bool':
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask
        if key_padding_mask is not None:
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(axis=1).
                unsqueeze(axis=2), -np.inf)
        attn_weights = paddle.nn.functional.softmax(x=attn_scores, axis=-1)
        attn_weights = self.attn_dropout(attn_weights)
        output = paddle.matmul(x=attn_weights, y=v)
        if self.res_attention:
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights
