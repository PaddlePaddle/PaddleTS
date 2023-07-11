import paddle

from paddlets.utils import param_init


class Inception_Block_V1(paddle.nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_kernels=6,
                 init_weight=True):
        """
        The inception block, which combines the result of multiple kernel to fusion feature with different receptive field.

        in_channels: The in channels of conv layer.
        out_channels: the out channels of conv layer.
        num_kernels: number of conv layers used in a inception block.
        """
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(
                paddle.nn.Conv2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=2 * i + 1,
                    padding=i))
        self.kernels = paddle.nn.LayerList(sublayers=kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.sublayers():
            if isinstance(m, paddle.nn.Conv2D):
                param_init.kaiming_normal_init(
                    m.weight,  # misaligned mode='fan_out'
                    nonlinearity='relu')
                if m.bias is not None:
                    param_init.constant_init(m.bias, value=0.0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = paddle.stack(x=res_list, axis=-1).mean(axis=-1)
        return res
