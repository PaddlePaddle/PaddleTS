import paddle
import paddle.nn as nn
from paddlets.models.base_model._timesnet.inception import Inception_Block_V1


def FFT_for_Period(x, k=2):
    xf = paddle.fft.rfft(x=x, axis=1)  # 时间序列频率项
    frequency_list = xf.abs().mean(axis=0).mean(axis=-1)  # 频率值的平均
    frequency_list[0] = 0
    _, top_list = paddle.topk(x=frequency_list, k=k)  # 幅度最大的前k个
    top_list = top_list.detach().cast('int32')
    period = x.shape[1] // top_list.cpu().numpy()  # 长度/频率=周期
    return period, xf.abs().mean(axis=-1).index_select(
        index=top_list, axis=1)  # 幅值最大的几个频率项


class TimesBlock(nn.Layer):
    def __init__(
            self,
            in_chunk_len: int,
            out_chunk_len: int,
            d_model: int,
            d_ff: int=32,
            top_k: int=5,
            num_kernels: int=6, ):
        super(TimesBlock, self).__init__()
        self.seq_len = in_chunk_len
        self.pred_len = out_chunk_len
        self.k = top_k
        self.conv = nn.Sequential(
            Inception_Block_V1(
                d_model, d_ff, num_kernels=num_kernels),
            nn.GELU(),
            Inception_Block_V1(
                d_ff, d_model, num_kernels=num_kernels))

    def forward(self, x):
        B, T, N = x.shape
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            if (self.seq_len + self.pred_len) % period != 0:
                length = ((self.seq_len + self.pred_len) // period + 1) * period
                padding = paddle.zeros(shape=[
                    x.shape[0], length - (self.seq_len + self.pred_len),
                    x.shape[2]
                ])
                out = paddle.concat(x=[x, padding], axis=1)
            else:
                length = self.seq_len + self.pred_len
                out = x

            out = out.reshape([B, length // period, period, N]).transpose(
                perm=[0, 3, 1, 2])
            out = self.conv(out)
            out = out.transpose(perm=[0, 2, 3, 1]).reshape([B, -1, N])
            res.append(out[:, :self.seq_len + self.pred_len, :])
        res = paddle.stack(x=res, axis=-1)
        period_weight = nn.functional.softmax(x=period_weight, axis=1)
        period_weight = period_weight.unsqueeze(axis=1).unsqueeze(axis=1).tile(
            repeat_times=[1, T, N, 1])

        res = paddle.sum(x=res * period_weight, axis=-1)
        res = res + x
        return res
