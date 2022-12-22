#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from collections import OrderedDict
from typing import Optional, Dict, Any, Callable, List, Tuple

import paddle
import numpy as np

from paddlets import TSDataset
from paddlets.logger import Logger, raise_if, raise_if_not, raise_log
from paddlets.models.forecasting.dl.paddle_base_impl import PaddleBaseModelImpl
from paddlets.metrics.base import Metric
from paddlets.metrics.metrics import MetricContainer
from paddlets.models.common.callbacks import Callback

logger = Logger(__name__)


class _Splitter(paddle.nn.Layer):
    """
    Time series split module, split raw sequence to even and odd sub-sequences.
    """

    def __init__(self):
        super(_Splitter, self).__init__()

    def _even(self, x: paddle.Tensor) -> paddle.Tensor:
        """
        Down-sample and get the sub-sequence from input sequence x, where each element in sub-sequence is sampled
        from input x's even index.

        Args:
            x(paddle.Tensor): Input Sequence to be down-sampled.

        Returns:
            paddle.Tensor: down-sampled sub-sequence.
        """
        return x[:, ::2, :]

    def _odd(self, x: paddle.Tensor):
        """
        Down-sample and get the sub-sequence from input sequence x, where each element in sub-sequence is sampled
        from input x's odd index.

        Args:
            x(paddle.Tensor): Input Sequence to be down-sampled.

        Returns:
            paddle.Tensor: down-sampled sub-sequence.
        """
        return x[:, 1::2, :]

    def forward(self, x: paddle.Tensor) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """
        Down-sample and build the even / odd sub-sequences from input sequence.

        Args:
            x(paddle.Tensor): input sequence to be down-sampled.

        Returns:
            Tuple[paddle.Tensor, paddle.Tensor]: A two-element tuple, where 1-st element is even sub-sequence, 2-nd
                element is odd sub-sequence.
        """
        return self._even(x), self._odd(x)


class _Interactor(paddle.nn.Layer):
    """
    Interactive Learning module, perform convolutional and interactive process on even / odd sub-sequences.

    Args:
        in_planes(int): The size of the loopback window, i.e., the number of time steps feed to the model.
        kernel_size(int): kernel size for Conv1D layer.
        dropout_rate(float): dropout regularization parameter.
        num_group(int): group number for Conv1D layer groups parameter.
        hidden_size(int): The number of features in hidden state.
    """

    def __init__(
        self,
        in_planes: int,
        kernel_size: int,
        dropout_rate: float,
        num_group: int,
        hidden_size: int
    ):
        super(_Interactor, self).__init__()

        self._in_planes = in_planes
        self._kernel_size = kernel_size
        self._dropout_rate = dropout_rate
        self._hidden_size = hidden_size
        self._num_group = num_group

        self._dilation = 1
        self._split_layer = _Splitter()

        # psi(ψ) / phi(Φ) / eta(η) / rho(ρ)
        self._psi = self._build_single_internal_module()
        self._phi = self._build_single_internal_module()
        self._eta = self._build_single_internal_module()
        self._rho = self._build_single_internal_module()

    def forward(self, x: paddle.Tensor) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """
        Perform convolutional and interactive process on even / odd sub-sequences, return processed sub-sequences.

        Args:
            x(paddle.Tensor): Input sequence to be forward processed.

        Returns(Tuple[paddle.Tensor, paddle.Tensor]): A 2-element tuple, where 1-st element is processed even
            sub-sequence, 2-nd element is processed odd sub-sequence.
        """
        # (Batch_size, in_chunk_len, in_dim)
        x_even, x_odd = self._split_layer(x)

        # (Batch_size, in_dim, in_chunk_len)
        x_even = paddle.transpose(x_even, perm=[0, 2, 1])
        x_odd = paddle.transpose(x_odd, perm=[0, 2, 1])

        x_scaled_even = paddle.multiply(x=x_even, y=paddle.exp(self._phi(x_odd)))
        x_scaled_odd = paddle.multiply(x=x_odd, y=paddle.exp(self._psi(x_even)))

        x_even_update = x_scaled_even + self._eta(x_scaled_odd)
        x_odd_update = x_scaled_odd - self._rho(x_scaled_even)

        # Return shape: (Batch_size, in_chunk_len, in_dim), (Batch_size, in_chunk_len, in_dim)
        return paddle.transpose(x_even_update, perm=[0, 2, 1]), paddle.transpose(x_odd_update, perm=[0, 2, 1])

    def _build_single_internal_module(self) -> paddle.nn.Sequential:
        """
        Build an internal forward process module for current SCINet's interactive learning module. Each interactive
        learning module contains 4 such internal sequential modules. Each of this internal sequential module
        contains 5 submodules. The 1-st replication padding submodule is to keep the border shrunk caused by the
        convolution operation. The 2-nd 1-D convolutional layer with self.kernel_size is to extend the input Channel
        C to self.hidden_size * C. Followed by the 3-rd LeakyReLU and 4-th Dropout layer, the 5-th 1-D convolutional
        layer is to recover the channel self.hidden_size * C to C.

        Returns:
            paddle.nn.Sequential: Built internal module for current SCINet's interactive learning module.
        """
        if self._kernel_size % 2 == 0:
            # by default: stride = 1
            pad_l = self._dilation * (self._kernel_size - 2) // 2 + 1
            pad_r = self._dilation * self._kernel_size // 2 + 1
        else:
            # we fix the kernel size of the second layer as 3.
            pad_l = self._dilation * (self._kernel_size - 1) // 2 + 1
            pad_r = self._dilation * (self._kernel_size - 1) // 2 + 1

        prev_size = 1
        layers = [
            paddle.nn.Pad1D((pad_l, pad_r)),
            paddle.nn.Conv1D(
                in_channels=self._in_planes * prev_size,
                out_channels=self._in_planes * self._hidden_size,
                kernel_size=self._kernel_size,
                dilation=self._dilation,
                stride=1,
                groups=self._num_group
            ),
            paddle.nn.LeakyReLU(negative_slope=0.01),
            paddle.nn.Dropout(p=self._dropout_rate),
            paddle.nn.Conv1D(
                in_channels=self._in_planes * self._hidden_size,
                out_channels=self._in_planes,
                kernel_size=3,
                stride=1,
                groups=self._num_group
            ),
            paddle.nn.Tanh()
        ]
        return paddle.nn.Sequential(*layers)


class _SCINetTree(paddle.nn.Layer):
    """
    SCINet encode binary tree.

    Args:
        in_planes(int): The size of the loopback window, i.e., the number of time steps feed to the model.
        current_level(int): tree level for current child tree. 0 means leaf node, max_tree_level means root node.
        kernel_size(int): kernel size for Conv1D layer.
        dropout_rate(float): dropout regularization parameter.
        num_group(int): group number for Conv1D layer groups parameter.
        hidden_size(int): The number of features in hidden state.
    """

    def __init__(
        self,
        in_planes: int,
        current_level: int,
        kernel_size: int,
        dropout_rate: float,
        num_group: int,
        hidden_size: int
    ):
        super(_SCINetTree, self).__init__()
        self._current_level = current_level

        self._workingblock = _Interactor(
            in_planes=in_planes,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate,
            num_group=num_group,
            hidden_size=hidden_size
        )

        if self._current_level != 0:
            self._scinet_tree_even = _SCINetTree(
                in_planes=in_planes,
                current_level=current_level - 1,
                kernel_size=kernel_size,
                dropout_rate=dropout_rate,
                num_group=num_group,
                hidden_size=hidden_size
            )

            self._scinet_tree_odd = _SCINetTree(
                in_planes=in_planes,
                current_level=current_level - 1,
                kernel_size=kernel_size,
                dropout_rate=dropout_rate,
                num_group=num_group,
                hidden_size=hidden_size
            )

    def _concat_and_realign(self, even, odd) -> paddle.Tensor:
        """
        Internal method, concatenate input even and odd sub-sequences and realign the timesteps to original order.

        Args:
            even(paddle.Tensor): Input even sub-sequence to be concatenated, e.g., [0, 2, 4, 6]
            odd(paddle.Tensor): Input odd sub-sequence to be concatenated, e.g., [1, 3, 5]

        Returns:
            paddle.Tensor: concatenated new sequence with timestep order realigned, e.g., [0, 1, 2, 3, 4, 5, 6].
        """
        # T_seq_len, Batch, Dim
        even = paddle.transpose(even, perm=[1, 0, 2])
        odd = paddle.transpose(odd, perm=[1, 0, 2])
        even_len = even.shape[0]
        odd_len = odd.shape[0]
        min_len = min((odd_len, even_len))
        all_time_steps = []
        for i in range(min_len):
            all_time_steps.append(even[i].unsqueeze(0))
            all_time_steps.append(odd[i].unsqueeze(0))
        if odd_len < even_len:
            # Given:
            # full seq = [0, 1, 2, 3, 4]
            # Thus:
            # even = [0, 2, 4]
            # odd = [1, 3]
            # thus odd_len (2) < even_len (3)
            # Note that even_len < odd_len will NEVER occur, because index always start with an even number (zero).
            all_time_steps.append(even[-1].unsqueeze(0))
        concat_all_time_steps = paddle.concat(all_time_steps, axis=0)
        # Batch, T_seq_len, Dim
        return paddle.transpose(concat_all_time_steps, perm=[1, 0, 2])

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """
        Forward compute function for SCINet encode tree network.

        Args:
            x(paddle.Tensor): input tensor to be forward computed.

        Returns:
            paddle.Tensor: forward processed tensor output.
        """
        x_even_update, x_odd_update = self._workingblock(x)
        # reorder odd/even sub-sequences recursively.
        if self._current_level == 0:
            # leaf nodes.
            return self._concat_and_realign(x_even_update, x_odd_update)
        else:
            # non-leaf nodes, call zip_up recursively until reach leaf nodes.
            return self._concat_and_realign(self._scinet_tree_even(x_even_update), self._scinet_tree_odd(x_odd_update))


class _StackedSCINetModule(paddle.nn.Layer):
    """
    Stacked SCINet, contains one or more SCINet(s).

    Args:
        in_chunk_len(int): The size of the loopback window, i.e., the number of time steps feed to the model.
        out_chunk_len(int): The size of the forecasting horizon, i.e. the number of time steps output by the model.
        in_dim(int): input feature dimensions, contain both target dim + (possibly be None) known cov dim +
            (possibly be None) observed cov dim.
        num_stack(int): stack number in Stacked SCINet.
        num_level(int): scinet tree level.
        num_decoder_layer(int): decoder layer number.
        concat_len(int): length to concat per stack.
        kernel_size(int): kernel size for Conv1D layer.
        dropout_rate(float): dropout regularization parameter.
        num_group(int): group number for Conv1D layer groups parameter.
        hidden_size(int): The number of features in hidden state.
    """

    def __init__(
        self,
        in_chunk_len: int,
        out_chunk_len: int,
        in_dim: int,
        num_stack: int = 1,
        num_level: int = 3,
        num_decoder_layer: int = 1,
        concat_len: int = 0,
        kernel_size: int = 5,
        dropout_rate: float = 0.5,
        num_group: int = 1,
        hidden_size: int = 1
    ):
        super(_StackedSCINetModule, self).__init__()

        self._in_chunk_len = in_chunk_len
        self._out_chunk_len = out_chunk_len
        self._in_dim = in_dim
        self._hidden_size = hidden_size
        self._num_stack = num_stack
        self._num_level = num_level
        self._num_decoder_layer = num_decoder_layer
        self._concat_len = concat_len
        self._num_group = num_group
        self._kernel_size = kernel_size
        self._dropout_rate = dropout_rate

        div_num = 6
        self._overlap_len = self._in_chunk_len // 4
        self._div_len = self._in_chunk_len // div_num

        self._encoder1 = _SCINetTree(
            in_planes=self._in_dim,
            current_level=self._num_level - 1,
            kernel_size=self._kernel_size,
            dropout_rate=self._dropout_rate,
            num_group=self._num_group,
            hidden_size=self._hidden_size
        )

        # only implement two stacks at most.
        if self._num_stack == 2:
            self._encoder2 = _SCINetTree(
                in_planes=self._in_dim,
                current_level=self._num_level - 1,
                kernel_size=self._kernel_size,
                dropout_rate=self._dropout_rate,
                num_group=self._num_group,
                hidden_size=self._hidden_size
            )

        self._decoder1 = paddle.nn.Conv1D(
            in_channels=self._in_chunk_len,
            out_channels=self._out_chunk_len,
            kernel_size=1,
            stride=1
        )

        self._div_projection = paddle.nn.LayerList()
        if self._num_decoder_layer > 1:
            self._decoder1 = paddle.nn.Linear(self._in_chunk_len, self._out_chunk_len)
            for layer_idx in range(self._num_decoder_layer - 1):
                div_projection = paddle.nn.LayerList()
                for i in range(div_num):
                    lens = min(i * self._div_len + self._overlap_len, self._in_chunk_len) - i * self._div_len
                    div_projection.append(paddle.nn.Linear(lens, self._div_len))
                self._div_projection.append(div_projection)

        if self._num_stack == 2:
            if self._concat_len > 0:
                in_channels = self._concat_len + self._out_chunk_len
            else:
                in_channels = self._in_chunk_len + self._out_chunk_len
            self._decoder2 = paddle.nn.Conv1D(
                in_channels=in_channels,
                out_channels=self._out_chunk_len,
                kernel_size=1
            )

    def forward(self, x_dict: Dict[str, paddle.Tensor]) -> Tuple[paddle.Tensor, Optional[paddle.Tensor]]:
        """
        Stacked SCINet network.

        Args:
            x_dict(Dict[str, paddle.Tensor]): key-value formatted samples, built by data_adapter.

        Returns:
            Tuple[paddle.Tensor, Optional[paddle.Tensor]]: forward processed output with two-element tuple shape, where
                1-st element is the final output from the last stack, the 2-nd element is the intermediate output from
                previous stack.
        """
        # [target_features, observed_features, past_known_features]
        x = x_dict["past_target"]
        if "observed_cov_numeric" in x_dict.keys():
            x = paddle.concat(x=(x, x_dict["observed_cov_numeric"]), axis=2)
        if "known_cov_numeric" in x_dict.keys():
            # x_dict["known_cov_numeric"].shape = (batch_size, in_chunk_len + out_chunk_len, known_dim)
            # past_known.shape = (batch_size, in_chunk_len, known_dim)
            past_known = x_dict["known_cov_numeric"][:, :self._in_chunk_len, ]
            x = paddle.concat(x=(x, past_known), axis=2)

        # the first stack
        res1 = x
        x = self._encoder1(x)
        x += res1
        if self._num_decoder_layer == 1:
            # (B, in_chunk_len, D) -> decoder -> (B, out_chunk_len, D)
            x = self._decoder1(x)
        else:
            x = paddle.transpose(x, perm=[0, 2, 1])
            for div_projection in self._div_projection:
                output = paddle.zeros(x.shape, dtype=x.dtype)
                for i, div_layer in enumerate(div_projection):
                    div_x = x[:, :, i * self._div_len:min(i * self._div_len + self._overlap_len, self._in_chunk_len)]
                    output[:, :, i * self._div_len:(i + 1) * self._div_len] = div_layer(div_x)
                x = output
            x = self._decoder1(x)
            x = paddle.transpose(x, perm=[0, 2, 1])

        if self._num_stack == 1:
            return x, None

        # self._num_stack == 2
        mid_output = x
        if self._concat_len > 0:
            x = paddle.concat(x=(res1[:, -self._concat_len:, :], x), axis=1)
        else:
            # (B, out_chunk_len, D) -> concat -> (B, in_chunk_len + out_chunk_len, D)
            x = paddle.concat(x=(res1, x), axis=1)

        # the second stack
        res2 = x
        x = self._encoder2(x)
        x += res2
        # (B, in_chunk_len + out_chunk_len, D) -> decoder2 -> (B, out_chunk_len, D)
        x = self._decoder2(x)
        return x, mid_output


class SCINetModel(PaddleBaseModelImpl):
    """
    DownSampled Convolutional Interactive Network (SCINet) for time series forcasting.
    Refers to `SCINet <https://arxiv.org/pdf/2106.09305.pdf>`_ .

    Args:
        in_chunk_len(int): The size of the loopback window, i.e. the number of time steps feed to the model.
        out_chunk_len(int): The size of the forecasting horizon, i.e. the number of time steps output by the model.
        skip_chunk_len(int): Optional, the number of time steps between in_chunk and out_chunk for a single sample.
            The skip chunk is neither used as a feature (i.e. X) nor a label (i.e. Y) for a single sample.
            By default, it will NOT skip any time steps.
        sampling_stride(int): Time steps to stride over the i-th sample and (i+1)-th sample. More precisely,
            let `t` be the time index of target time series, `t[i]` be the start time of the i-th sample,
            `t[i+1]` be the start time of the (i+1)-th sample, thus `sampling_stride` is equal to `t[i+1] - t[i]`.
        loss_fn(Callable[..., paddle.Tensor]|None): Loss function.
        optimizer_fn(Callable[..., Optimizer]): Optimizer algorithm.
        optimizer_params(Dict[str, Any], optional): Optimizer parameters.
        eval_metrics(List[str]|List[Metric], optional): Evaluation metrics of model.
        callbacks(List[Callback], optional): Customized callback functions.
        batch_size(int): Number of samples per batch.
        max_epochs(int): Max training epochs.
        verbose(int): Verbosity mode.
        patience(int): Number of epochs to wait for improvement before terminating.
        seed(int|None): Global random seed.
        num_stack(int): stack number in Stacked SCINet.
        num_level(int): scinet tree level.
        num_decoder_layer(int): decoder layer number.
        concat_len(int): length to concat per stack.
        kernel_size(int): kernel size for Conv1D layer.
        dropout_rate(float): dropout regularization parameter.
        num_group(int): group number for Conv1D layer groups parameter.
        hidden_size(int): The number of features in hidden state for SCINet Interactor module.
    """
    def __init__(
        self,
        in_chunk_len: int,
        out_chunk_len: int,
        skip_chunk_len: int = 0,
        sampling_stride: int = 1,
        loss_fn: Callable[..., paddle.Tensor] = paddle.nn.functional.mse_loss,
        optimizer_fn: Callable[..., paddle.optimizer.Optimizer] = paddle.optimizer.Adam,
        optimizer_params: Optional[Dict[str, Any]] = None,
        eval_metrics: Optional[List[str]] = None,
        callbacks: Optional[List[Callback]] = None,
        batch_size: int = 8,
        max_epochs: int = 100,
        verbose: int = 1,
        patience: int = 10,
        seed: Optional[int] = None,
        num_stack: int = 1,
        num_level: int = 3,
        num_decoder_layer: int = 1,
        concat_len: int = 0,
        kernel_size: int = 5,
        dropout_rate: float = 0.5,
        num_group: int = 1,
        hidden_size: int = 1
    ):
        super(SCINetModel, self).__init__(
            in_chunk_len=in_chunk_len,
            out_chunk_len=out_chunk_len,
            skip_chunk_len=skip_chunk_len,
            sampling_stride=sampling_stride,
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
            optimizer_params=optimizer_params,
            eval_metrics=eval_metrics,
            callbacks=callbacks,
            batch_size=batch_size,
            max_epochs=max_epochs,
            verbose=verbose,
            patience=patience,
            seed=seed
        )
        if self._optimizer_params is None:
            self._optimizer_params = dict(learning_rate=1e-3)
        if self._eval_metrics is None:
            self._eval_metrics = list()
        if self._callbacks is None:
            self._callbacks = list()

        self._hidden_size = hidden_size
        self._num_stack = num_stack
        self._num_level = num_level
        self._num_decoder_layer = num_decoder_layer
        self._concat_len = concat_len
        self._num_group = num_group
        self._kernel_size = kernel_size
        self._dropout_rate = dropout_rate

        raise_if_not(0 < num_stack <= 2, f"The number of stack can only be 1 or 2, got {num_stack}.")
        # limit the recursion depth.
        raise_if(
            self._in_chunk_len % (np.power(2, self._num_level)) != 0,
            f"in_chunk_len % 2**num_level must != 0. Actual " +
            f"in_chunk_len = {self._in_chunk_len}, " +
            f"num_level = {self._num_level}, " +
            f"in_chunk_len % 2**num_level = ({self._in_chunk_len % (np.power(2, self._num_level))})."
        )

    def _check_tsdataset(self, tsdataset: TSDataset) -> None:
        """
        SCINet only allows float32 type variables, any int-like variables are not allowed.

        Args:
            tsdataset(TSDataset): input TSDataset to be checked.
        """
        for column, dtype in tsdataset.dtypes.items():
            raise_if_not(
                np.issubdtype(dtype, np.floating),
                f"scinet variables' dtype only supports [float16, float32, float64], "
                f"but received {column}: {dtype}."
            )
        super(SCINetModel, self)._check_tsdataset(tsdataset)

    def _update_fit_params(
        self,
        train_tsdataset: List[TSDataset],
        valid_tsdataset: Optional[List[TSDataset]] = None
    ) -> Dict[str, Any]:
        """
        Build fit parameters.

        Args:
            train_tsdataset(List[TSDataset]): list of train dataset.
            valid_tsdataset(List[TSDataset]|None): list of validation dataset.

        Returns:
            Dict[str, Any]: model parameters
        """
        tsdataset = train_tsdataset[0]
        target_dim = tsdataset.target.data.shape[1]
        # observed_num_dim = observed cov numeric dimension
        observed_num_dim = 0
        if tsdataset.observed_cov is not None:
            observed_num_dim = tsdataset.observed_cov.data.shape[1]
        # known_num_dim = known cov numeric dimension
        known_num_dim = 0
        if tsdataset.known_cov is not None:
            known_num_dim = tsdataset.known_cov.data.shape[1]

        # Because self._check_tsdataset already guarantee that only numeric features are supported, so
        # here the below dims only contain numeric dim, but not contain categorical dims.
        fit_params = {
            "target_dim": target_dim,
            "observed_num_dim": observed_num_dim,
            "known_num_dim": known_num_dim
        }
        return fit_params

    def _init_network(self) -> paddle.nn.Layer:
        """
        Initialize the SCINet network.

        Returns:
            paddle.nn.Layer: Initialized network.
        """
        in_dim = self._fit_params["target_dim"] + \
            self._fit_params["observed_num_dim"] + \
            self._fit_params["known_num_dim"]
        return _StackedSCINetModule(
            in_chunk_len=self._in_chunk_len,
            out_chunk_len=self._out_chunk_len,
            in_dim=in_dim,
            num_stack=self._num_stack,
            num_level=self._num_level,
            num_decoder_layer=self._num_decoder_layer,
            concat_len=self._concat_len,
            kernel_size=self._kernel_size,
            dropout_rate=self._dropout_rate,
            num_group=self._num_group,
            hidden_size=self._hidden_size
        )

    def _init_metrics(self, eval_names: List[str]) -> Tuple[List[Metric], List[str], Dict[str, MetricContainer]]:
        """
        Set attributes relative to the metrics.

        Args:
            eval_names(List[str]): List of eval set names.

        Returns:
            List[Metric]: List of metric instance.
            List[str]: List of metric names.
            Dict[str, MetricContainer]: Dict of metric container.
        """
        metrics = self._eval_metrics
        metric_container_dict = OrderedDict()
        for name in eval_names:
            metric_container_dict.update({name: MetricContainer(metrics, prefix=f"{name}_")})
            if self._num_stack == 2:
                # add metric container for SCINet mid predict output.
                metric_container_dict.update({f"{name}_mid": MetricContainer(metrics, prefix=f"{name}_mid_")})
        metrics, metrics_names = [], []
        for _, metric_container in metric_container_dict.items():
            metrics.extend(metric_container._metrics)
            metrics_names.extend(metric_container._names)
        return metrics, metrics_names, metric_container_dict

    def _compute_loss(
        self,
        y_score: Tuple[paddle.Tensor, Optional[paddle.Tensor]],
        y_true: paddle.Tensor
    ) -> paddle.Tensor:
        """
        Internal method, compute loss for current batch.

        Args:
            y_score(paddle.Tensor): predicted y for current batch.
            y_true(paddle.Tensor): ground truth for current batch.

        Returns:
            paddle.Tensor: computed loss for current batch.
        """
        # y_score = tuple, where tuple[0] is final pred res, tuple[1] is intermediate pred res from previous stack.
        y_pred, mid_pred = y_score
        loss = self._loss_fn(y_pred[:, :, :self._fit_params["target_dim"]], y_true)
        # pre-check already guarantee that num_stack can only be either 1 or 2.
        if self._num_stack == 1:
            # not contain mid output
            return loss
        if self._num_stack == 2:
            # contain mid output
            mid_loss = self._loss_fn(mid_pred[:, :, :self._fit_params["target_dim"]], y_true)
            return loss + mid_loss
        raise_log(exception=ValueError(f"num_stack ({self._num_stack}) must be either 1 or 2."), logger=logger)

    def _predict(self, dataloader: paddle.io.DataLoader) -> np.ndarray:
        """
        Predict function core logic.

        SCINet will return a tuple of tensors = (pred, mid_pred), where mid_pred is the intermediate output
        from previous stack, the mid_pred is used for computing loss, so here only need to append tuple[0] and discard
        tuple[1].

        Args:
            dataloader(paddle.io.DataLoader): Data to be predicted.

        Returns:
            np.ndarray: predicted ndarray matrix.
        """
        self._network.eval()
        results = []
        for batch_nb, data in enumerate(dataloader):
            x, _ = self._prepare_X_y(data)
            # output.shape = (out_chunk_len, in_dim)
            output, _ = self._network(x)
            predictions = output.numpy()
            results.append(predictions)
        # results.shape = (batch_size, out_chunk_len, in_dim)
        results = np.vstack(results)
        # Note: the pre-logic already guarantee that the feature layout is as follows (most left is target, middle is
        # observed cov, most right is known cov):
        # x =       [target_features, observed_features, known_features]
        # results = [target_pred_res, observed_pred_res, known_pred_res]
        # As we only predict target, but not predict co-variates, so here we only cut and return target predict res.
        return results[:, :, :self._fit_params["target_dim"]]

    def _predict_epoch(self, name: str, loader: paddle.io.DataLoader) -> None:
        """
        Predict an epoch and update metrics.

        Args:
            name(str): Name of the validation set.
            loader(paddle.io.DataLoader): DataLoader with validation set.
        """
        self._network.eval()

        y_true_list = []
        y_pred_list = []
        mid_pred_list = []
        for batch_idx, data in enumerate(loader):
            # y_true_batch.shape = (batch_size, out_chunk_len, target_dim)
            X, y_true_batch = self._prepare_X_y(data)
            # y_pred_batch.shape = (batch_size, out_chunk_len, in_dim)
            # mid_pred_batch.shape = (batch_size, out_chunk_len, in_dim)
            y_pred_batch, mid_pred_batch = self._predict_batch(X)

            y_true_list.append(y_true_batch)
            y_pred_list.append(y_pred_batch)
            mid_pred_list.append(mid_pred_batch)

        y_true = np.vstack(y_true_list)
        # y_true.shape[2] = target_dim + observed_cov_col_num
        # y_pred.shape[2] mid_pred.shape[2] = target_dim
        y_pred = np.vstack(y_pred_list)[:, :, :self._fit_params["target_dim"]]
        y_pred_metrics_logs = self._metric_container_dict[name](y_true, y_pred)
        self._history._epoch_metrics.update(y_pred_metrics_logs)

        if self._num_stack == 2:
            mid_pred = np.vstack(mid_pred_list)[:, :, :self._fit_params["target_dim"]]
            mid_pred_metrics_logs = self._metric_container_dict[f"{name}_mid"](y_true, mid_pred)
            self._history._epoch_metrics.update(mid_pred_metrics_logs)

        self._network.train()

    def _predict_batch(self, X: paddle.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict one batch of data.

        SCINet will return tuple(pred, mid_pred), where mid_pred is used for computing loss. Here only need

        Args:
            X(paddle.Tensor): Feature tensor.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Prediction results, where tuple[0] is final pred result, tuple[1] is
            intermediate (previous stack) pred result.
        """
        y_pred, mid_pred = self._network(X)
        if self._num_stack == 1:
            # mid_pred is None when num_stack == 1, thus cannot call .numpy() for a NoneType Object.
            return y_pred.numpy(), mid_pred
        return y_pred.numpy(), mid_pred.numpy()
