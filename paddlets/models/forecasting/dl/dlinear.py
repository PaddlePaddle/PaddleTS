#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle.nn import AvgPool1D
from paddle.optimizer import Optimizer

from paddlets.datasets import TSDataset
from paddlets.logger import Logger, raise_if, raise_if_not
from paddlets.models.common.callbacks import Callback
from paddlets.models.forecasting.dl.paddle_base_impl import PaddleBaseModelImpl

logger = Logger(__name__)

PAST_TARGET = "past_target"
OBSERVED_COV = "observed_cov_numeric"
KNOWN_COV = "known_cov_numeric"


class _SeriesDecomp(paddle.nn.Layer):
    """
    Series decomposition block.
    """

    def __init__(self, kernel_size: int):
        """
        Args:
            kernel_size(int): The size of the kernel for the moving average.
        """
        super(_SeriesDecomp, self).__init__()
        self.moving_avg = AvgPool1D(kernel_size, stride=1, padding="same")

    def forward(self, x: paddle.Tensor) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """
        Args:
            x(paddle.Tensor): Input tensor of the decomposition block.

        Returns:
            trend(padle.Tensor): Tensor containing the trend component by a moving average.
            seasonal(padle.Tensor): Tensor containing the the remainder (seasonal) component.
        """
        trend = paddle.transpose(x, perm=[0, 2, 1])
        trend = self.moving_avg(trend)
        trend = paddle.transpose(trend, perm=[0, 2, 1])
        seasonal = x - trend
        return trend, seasonal


class _NNBlock(paddle.nn.Layer):
    """
    Linear layer(s) block.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        hidden_config: List[int],
        use_bn: bool,
    ):
        """
        Args:
            dim_in(int): The number of input units of the (first) linear layer.
            dim_out(int): The number of output units of the (last) linear layer.
            hidden_config(List[int]): The ith element represents the number of neurons in the ith hidden layer. There is no hidden layer when it is empty.
            use_bn(bool): Whether to use batch normalization. Only used if `hidden_config` is not empty.
        """
        super(_NNBlock, self).__init__()
        layers = []
        dims = [dim_in] + hidden_config + [dim_out]
        for i in range(len(dims) - 2):
            layers.append(paddle.nn.Linear(dims[i], dims[i + 1]))
            if use_bn:
                layers.append(paddle.nn.BatchNorm1D(1))
            layers.append(paddle.nn.ReLU())
        layers.append(paddle.nn.Linear(dims[-2], dims[-1]))
        self._nn = paddle.nn.Sequential(*layers)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """
        Args:
            x(paddle.Tensor]: input tensor of the nn block.

        Returns:
            out(padle.Tensor): output tensor of the nn block.
        """
        return self._nn(x)


class _DLinearModel(paddle.nn.Layer):
    """
    Network structure of DLinear model, cover multi-targets, known_covariates, observed_covariates.
    """

    def __init__(
        self,
        in_chunk_len: int,
        out_chunk_len: int,
        kernel_size: int,
        target_dim: int,
        known_cov_dim: int,
        observed_cov_dim: int,
        hidden_config: List[int],
        use_bn: bool,
    ):
        """
        Args:
            in_chunk_len(int): The length of the input sequence fed to the model.
            out_chunk_len(int): The length of the forecast of the model.
            kernel_size(int): The size of the kernel for the moving average.
            target_dim(int): The number of targets to be forecasted.
            known_cov_dim(int): The number of known covariates.
            observed_cov_dim(int): The number of observed covariates.
            hidden_config(List[int]): The ith element represents the number of neurons in the ith hidden layer. There is no hidden layer when it is empty.
            use_bn(bool): Whether to use batch normalization. Only used if `hidden_config` is not empty.
        """
        super(_DLinearModel, self).__init__()
        self._target_dim = target_dim
        self._known_cov_dim = known_cov_dim
        self._observed_cov_dim = observed_cov_dim

        input_dim = target_dim + known_cov_dim + observed_cov_dim
        in_chunk_len_multi = in_chunk_len * input_dim + out_chunk_len * known_cov_dim
        out_chunk_len_multi = out_chunk_len * target_dim

        self._decompsition = _SeriesDecomp(kernel_size)
        # Unlike MLP, `hidden_config` is empty by default, so there can be no hidden layer.
        self._seasonal_nn = _NNBlock(
            dim_in=in_chunk_len_multi,
            dim_out=out_chunk_len_multi,
            hidden_config=hidden_config,
            use_bn=use_bn,
        )
        self._trend_nn = _NNBlock(
            dim_in=in_chunk_len_multi,
            dim_out=out_chunk_len_multi,
            hidden_config=hidden_config,
            use_bn=use_bn,
        )

    def forward(self, data: Dict[str, paddle.Tensor]) -> paddle.Tensor:
        """
        Args:
            data(Dict[str, paddle.Tensor]): a dict specifies all kinds of input data

        Returns:
            out(padle.Tensor): tensor containing the output of the DLinear model.
        """
        backcast = data[PAST_TARGET]
        known_cov = data[KNOWN_COV] if self._known_cov_dim > 0 else None
        observed_cov = data[OBSERVED_COV] if self._observed_cov_dim > 0 else None
        batch_size = paddle.shape(backcast)[0]

        # decomposition
        trend, seasonal = self._decompsition(backcast)

        # concat backcast, known_cov, observed_cov if any
        trend_feature = [trend.reshape((batch_size, 1, -1))]
        seasonal_feature = [seasonal.reshape((batch_size, 1, -1))]
        if known_cov is not None:
            known_cov_flat = known_cov.reshape((batch_size, 1, -1))
            trend_feature.append(known_cov_flat)
            seasonal_feature.append(known_cov_flat)
        if observed_cov is not None:
            observed_cov_flat = observed_cov.reshape((batch_size, 1, -1))
            trend_feature.append(observed_cov_flat)
            seasonal_feature.append(observed_cov_flat)
        trend_out = paddle.concat(x=trend_feature, axis=2)
        seasonal_out = paddle.concat(x=seasonal_feature, axis=2)

        # forward
        trend_out = self._trend_nn(trend_out)
        seasonal_out = self._seasonal_nn(seasonal_out)
        out = trend_out + seasonal_out
        out = out.reshape([batch_size, -1, self._target_dim])

        return out


class DLinearModel(PaddleBaseModelImpl):
    """
    Implementation of DLinear model, cover multi-targets, known_covariates, observed_covariates.
    """

    def __init__(
        self,
        in_chunk_len: int,
        out_chunk_len: int,
        kernel_size: int,
        skip_chunk_len: int = 0,
        sampling_stride: int = 1,
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
        hidden_config: List[int] = [],
        use_bn: bool = False,
    ):
        """
        Args:
            in_chunk_len(int): The size of the loopback window, i.e., the number of time steps feed to the model.
            out_chunk_len(int): The size of the forecasting horizon, i.e., the number of time steps output by the model.
            kernel_size(int): The size of the kernel for the moving average.
            skip_chunk_len(int, Optional): Optional, the number of time steps between in_chunk and out_chunk for a single sample. The skip chunk is neither used as a feature (i.e. X) nor a label (i.e. Y) for a single sample. By default it will NOT skip any time steps.
            sampling_stride(int, optional): sampling intervals between two adjacent samples.
            loss_fn(Callable, Optional): loss function.
            optimizer_fn(Callable, Optional): optimizer algorithm.
            optimizer_params(Dict, Optional): optimizer parameters.
            eval_metrics(List[str], Optional): evaluation metrics of model.
            callbacks(List[Callback], Optional): customized callback functions.
            batch_size(int, Optional): number of samples per batch.
            max_epochs(int, Optional): max epochs during training.
            verbose(int, Optional): verbosity mode.
            patience(int, Optional): number of epochs with no improvement after which learning rate wil be reduced.
            seed(int, Optional): global random seed.
            hidden_config(List[int], Optional): The ith element represents the number of neurons in the ith hidden layer. There is no hidden layer when it is empty by default.
            use_bn(bool, Optional): Whether to use batch normalization. Only used if `hidden_config` is not empty.
        """
        self._kernel_size = kernel_size
        self._hidden_config = hidden_config
        self._use_bn = use_bn
        super(DLinearModel, self).__init__(
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
            seed=seed,
        )

    def _check_tsdataset(self, tsdataset: TSDataset):
        """
        Rewrite _check_tsdataset to fit the specific model.
        For DLinear, all data variables are expected to be float.
        """
        for column, dtype in tsdataset.dtypes.items():
            raise_if_not(
                np.issubdtype(dtype, np.floating),
                f"DLinear variables' dtype only supports [float16, float32, float64], "
                f"but received {column}: {dtype}.",
            )
        super(DLinearModel, self)._check_tsdataset(tsdataset)

    def _update_fit_params(
        self,
        train_tsdataset: List[TSDataset],
        valid_tsdataset: Optional[List[TSDataset]] = None,
    ) -> Dict[str, Any]:
        """
        Infer parameters by TSDataset automatically.

        Args:
            train_tsdataseet(List[TSDataset]): list of train dataset
            valid_tsdataset(List[TSDataset], optional): list of validation dataset

        Returns:
            Dict[str, Any]: model parameters
        """
        fit_params = {
            "target_dim": train_tsdataset[0].get_target().data.shape[1],
            "known_cov_dim": 0,
            "observed_cov_dim": 0,
        }
        if train_tsdataset[0].get_known_cov() is not None:
            fit_params["known_cov_dim"] = (
                train_tsdataset[0].get_known_cov().data.shape[1]
            )
        if train_tsdataset[0].get_observed_cov() is not None:
            fit_params["observed_cov_dim"] = (
                train_tsdataset[0].get_observed_cov().data.shape[1]
            )
        return fit_params

    def _init_network(self) -> paddle.nn.Layer:
        """
        Init network.

        Returns:
            paddle.nn.Layer
        """
        return _DLinearModel(
            self._in_chunk_len,
            self._out_chunk_len,
            self._kernel_size,
            self._fit_params["target_dim"],
            self._fit_params["known_cov_dim"],
            self._fit_params["observed_cov_dim"],
            self._hidden_config,
            self._use_bn,
        )
