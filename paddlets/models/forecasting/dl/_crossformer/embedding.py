import paddle
from einops import rearrange


class DSW_embedding(paddle.nn.Layer):

    def __init__(self, seg_len, d_model):
        super(DSW_embedding, self).__init__()
        self.seg_len = seg_len
        self.linear = paddle.nn.Linear(in_features=seg_len, out_features=
            d_model)

    def forward(self, x):
        batch, ts_len, ts_dim = x.shape
        x_segment = rearrange(x,
            'b (seg_num seg_len) d -> (b d seg_num) seg_len', seg_len=self.
            seg_len)
        x_embed = self.linear(x_segment)
        x_embed = rearrange(x_embed,
            '(b d seg_num) d_model -> b d seg_num d_model', b=batch, d=ts_dim)
        return x_embed
