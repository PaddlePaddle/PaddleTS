import paddle


class TriangularCausalMask:

    def __init__(self, B, L):
        mask_shape = [B, 1, L, L]
        with paddle.no_grad():
            self._mask = paddle.triu(x=paddle.ones(shape=mask_shape, dtype=
                'bool'), diagonal=1)

    @property
    def mask(self):
        return self._mask


class ProbMask:

    def __init__(self, B, H, L, index, scores):
        _mask = paddle.ones(shape=[L, scores.shape[-1]], dtype='bool').triu(1)
        _mask_ex = _mask[(None), (None), :].expand(shape=[B, H, L, scores.
            shape[-1]])
        indicator = _mask_ex[(paddle.arange(start=B)[:, (None), (None)]), (
            paddle.arange(start=H)[(None), :, (None)]), (index), :]
        self._mask = indicator.view(scores.shape)

    @property
    def mask(self):
        return self._mask
