#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from typing import List, Dict, Any, Callable, Optional

from paddle.optimizer import Optimizer
import paddle.nn.functional as F
import numpy as np
import paddle

from paddlets.models.classify.dl.paddle_base import PaddleBaseClassifier
from paddlets.models.common.callbacks import Callback
from paddlets.datasets import TSDataset
from paddlets.logger import raise_if,raise_if_not

ACTIVATIONS = [
    "ReLU",
    "PReLU",
    "ELU",
    "Softmax",
    "Softplus",
    "Tanh",
    "SELU",
    "LeakyReLU",
    "Sigmoid",
    "GELU",
]

class _InceptionModule(paddle.nn.Layer):
    """
    InceptionModule.
    
    Args:
        input_len: Input tensor length.
        conv_out_size: Output length for conv and pooling layer.
        kernel_size: Kenel size for conv layers.
        activation: The activation function.
        use_bottleneck: If use bottleneck layer or not, Set to True by default.
    """
    def __init__(self,
                 input_len:int,
                 conv_out_size:int,
                 kernel_size:int = 40,
                 activation:Callable = paddle.nn.ReLU(),
                 use_bottleneck:bool = True):

        super().__init__()

        # compute kernel size
        kernel_size = [kernel_size // (2 ** i) for i in range(3)]
        kernel_size = [k if k % 2 != 0 else k - 1 for k in kernel_size]  # ensure odd ks

        # init bottleneck layer
        use_bottleneck = use_bottleneck if input_len > 1 else False
        self.bottleneck = paddle.nn.Conv1D(input_len, conv_out_size, 1, bias_attr=False,
                                           padding="SAME") if use_bottleneck else None

        # init conv layers
        self.convs = paddle.nn.LayerList([
            paddle.nn.Conv1D(conv_out_size if use_bottleneck else input_len, conv_out_size, k, bias_attr=False,
                             padding="SAME") for k in kernel_size])

        #init pooling layer
        self.maxconvpool = paddle.nn.Sequential(*[paddle.nn.MaxPool1D(3, stride=1, padding="SAME"),
                                                  paddle.nn.Conv1D(input_len, conv_out_size, 1, bias_attr=False)])

        #init BN layer
        self.bn = paddle.nn.BatchNorm1D(conv_out_size * 4)
        self.act = activation

    def forward(self, 
                x:paddle.Tensor) -> paddle.Tensor:
        """
        InceptionModule forward function.
        
        Args:
            x(paddle.Tensor): input data in format of paddle.Tensor.
        Returns:
            paddle.Tensor: The output of the InceptionModule.
        """

        input_tensor = x

        if self.bottleneck:
            x = self.bottleneck(input_tensor)

        x = paddle.concat([conv(x) for conv in self.convs] + [self.maxconvpool(input_tensor)],axis=1)
        return self.act(self.bn(x))

class _InceptionBlock(paddle.nn.Layer):
    """
    InceptionBlock.
    
    Args:
        input_len: Input tensor length.
        output_len: Output tensor length.
        kernel_size: Kenel size for conv layers.
        depth: The depth of InceptionBlock.
        activation: The activation function.
        use_residual: If add residuals between Inception modules.
        use_bottleneck: If use bottleneck layer or not, Set to True by default.
    """
    def __init__(self, 
                 input_len,
                 output_len=128,
                 kernel_size=40,
                 depth=6,
                 activation=paddle.nn.ReLU(),
                 use_residual=True,
                 use_bottleneck=True):
        super().__init__()
        self.residual, self.depth = use_residual, depth
        self.inception, self.shortcut = paddle.nn.LayerList(), paddle.nn.LayerList()
        for d in range(depth):
            self.inception.append(_InceptionModule(input_len if d == 0 else output_len ,int(output_len/4),kernel_size=kernel_size,activation=activation,use_bottleneck=use_bottleneck))
            if self.residual and d % 3 == 2: 
                n_in, n_out = input_len if d == 2 else output_len , output_len 
                self.shortcut.append(paddle.nn.BatchNorm1D(n_in) if n_in == n_out else paddle.nn.Conv1D(n_in, n_out, 1,padding="SAME"))
        self.act = paddle.nn.ReLU()
        
    def forward(self, x):
        """
        forward function.
        
        Args:
            x(paddle.Tensor): input data in format of paddle.Tensor.
        Returns:
            paddle.Tensor: The output of the InceptionBlock.
        """
        res = x
        for d, l in enumerate(range(self.depth)):
            x = self.inception[d](x)
            if self.residual and d % 3 == 2: res = x = self.act(x.add(self.shortcut[d//3](res)))
        return x

class _InceptionTime(paddle.nn.Layer):
    """
    InceptionBlock.
    
    Args:
        channel_in: Input tensor length.
        channel_out: Output tensor length.
        kernel_size: Kenel size for conv layers.
        block_out_size: The output size of InceptionBlock.
        block_depth: The depth of InceptionBlock.
        activation: The activation function.
        use_residual: If add residuals between Inception modules.
        use_bottleneck: If use bottleneck layer or not, Set to True by default.
    """
    def __init__(self,
                 channel_in,
                 channel_out,
                 kernel_size=41,
                 block_out_size=128,
                 block_depth=6,
                 activation=paddle.nn.ReLU(),
                 use_residual=True,
                 use_bottleneck=True):
        super().__init__()
        self.inceptionblock = _InceptionBlock(channel_in, block_out_size, kernel_size, block_depth,
                                             activation, use_residual, use_bottleneck)
        self.gap = paddle.nn.AdaptiveAvgPool1D(1)
        self.flatten = paddle.nn.Flatten()
        self.fc = paddle.nn.Linear(block_out_size, channel_out)

    def forward(self, x):
        """
        The main logic of InceptionTime.
        
        Args:
            x(Dict[str, paddle.Tensor]): A dict specifies all kinds of input data.
        Returns:
            output(paddle.Tensor): The output of the model.
        """
        x = x['features']
        x = paddle.transpose(x, perm=[0, 2, 1])
        x = self.inceptionblock(x)
        x = self.gap(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

class InceptionTimeClassifier(PaddleBaseClassifier):
    """InceptionTime\[1\] is a time series Classification model introduced in 2019.
    InceptionTime an ensemble of deep Convolutional Neural Network (CNN) models, inspired by the Inception-v4 architecture.

    \[1\] Hassan I.F, et al. "InceptionTime: Finding AlexNet for Time Series
    Classification", `<https://arxiv.org/pdf/1909.04939v3.pdf>`_

    Args:
        optimizer_fn(Callable[..., Optimizer]): Optimizer algorithm.
        optimizer_params(Dict[str, Any]): Optimizer parameters.
        eval_metrics(List[str]): Evaluation metrics of model.
        callbacks(List[Callback]): Customized callback functions.
        batch_size(int): Number of samples per batch.
        max_epochs(int): Max epochs during training.
        verbose(int): Verbosity mode.
        patience(int): Number of epochs to wait for improvement before terminating.
        seed(int|None): Global random seed.

        activation(str): Activation function,set to "ReLU" by defalut.
        kernel_size(int): Kernel size for inception module, set to 40 by default.
        block_out_size(int): Output size for inception block, set to 128 by default.
        block_depth(int): Depth for inception block, set to 6 by default.
        use_bottleneck(bool):If add residuals between Inception modules.
        use_residual(bool): If use bottleneck layer or not, Set to True by default.
    """

    def __init__(
            self,
            loss_fn: Callable[..., paddle.Tensor] = F.mse_loss,
            optimizer_fn: Callable[..., Optimizer] = paddle.optimizer.Adam,
            optimizer_params: Dict[str, Any] = dict(learning_rate=1e-3),
            eval_metrics: List[str] = [],
            callbacks: List[Callback] = [],
            batch_size: int = 32,
            max_epochs: int = 100,
            verbose: int = 1,
            patience: int = 10,
            seed: Optional[int] = None,

            activation: str = "ReLU",
            kernel_size=40,
            block_out_size=128,
            block_depth=6,
            use_bottleneck=True,
            use_residual=True
    ):
        raise_if_not(
            activation in ACTIVATIONS, f"'{activation}' is not in {ACTIVATIONS}"
        )

        self._activation = getattr(paddle.nn, activation)()
        self._kernel_size = kernel_size
        self._block_out_size = block_out_size
        self._block_depth = block_depth
        self.use_bottleneck = use_bottleneck
        self.use_residual = use_residual

        super(InceptionTimeClassifier, self).__init__(
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
            optimizer_params=optimizer_params,
            eval_metrics=eval_metrics,
            callbacks=callbacks,
            batch_size=batch_size,
            max_epochs=max_epochs,
            verbose=verbose,
            patience=patience,
            seed=seed,
        )

    def _check_params(self):
        """
        Parameter validity verification.
        """
        raise_if_not(isinstance(self._kernel_size, int), "kernel_size should be in type of int")
        raise_if_not(isinstance(self._block_out_size, int), "block_out_size should be in type of int")
        raise_if_not(isinstance(self._block_depth, int), "block_depth should be in type of int")
        raise_if_not(isinstance(self.use_bottleneck, bool), "use_bottlneck should be in type of bool")
        raise_if_not(isinstance(self.use_residual, bool), "use_residual should be in type of bool")

        raise_if_not(self._block_depth > 0, "block_depth should be larger than 0 ")
        raise_if_not(self._kernel_size > 0, "kernel_size should be larger than 0 ")
        raise_if_not(self._block_out_size % 4 == 0,
                     "block_out_size should be a multiple of 4, because inception moudle has 4 banches")
        super()._check_params()

    def _update_fit_params(
            self,
            train_tsdatasets: List[TSDataset],
            train_labels: np.ndarray,
            valid_tsdatasets: List[TSDataset],
            valid_labels: np.ndarray
    ) -> Dict[str, Any]:
        """
        Infer parameters by TSdataset automatically.

        Args:
            train_tsdataset(TSDataset): Train set.
            train_labels:(np.ndarray) : The train data class labels
            valid_tsdataset(TSDataset|None): Eval set, used for early stopping.
            valid_labels:(np.ndarray) : The valid data class labels
        Returns:
            Dict[str, Any]: model parameters.
        """
        fit_params = {
            "feature_dim": train_tsdatasets[0].get_target().data.shape[1],
            "input_lens": train_tsdatasets[0].get_target().data.shape[0]
        }
        return fit_params

    def _init_network(self) -> paddle.nn.Layer:
        """
        Setup the network.

        Returns:
            paddle.nn.Layer.
        """

        return _InceptionTime(
            self._fit_params["feature_dim"],
            self._n_classes,
            self._kernel_size,
            self._block_out_size,
            self._block_depth,
            self._activation,
            self.use_residual,
            self.use_bottleneck
        )
