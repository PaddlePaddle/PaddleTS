#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from typing import List, Dict, Any, Callable, Optional

from paddle.optimizer import Optimizer
import paddle.nn.functional as F
import numpy as np
import paddle

from paddlets.models.forecasting.dl.paddle_base_impl import PaddleBaseModelImpl
from paddlets.models.common.callbacks import Callback
from paddlets.datasets import TSDataset
from paddlets.logger import raise_if, raise_if_not

PAST_TARGET = "past_target"


class _MLPModule(paddle.nn.Layer):
    """Network structure.

    Args:
        in_chunk_len(int): The size of the loopback window, i.e. the number of time steps feed to the model.
        out_chunk_len(int): The size of the forecasting horizon, i.e. the number of time steps output by the model.
        target_dim(int): The numer of targets.
        hidden_config(List[int]): The ith element represents the number of neurons in the ith hidden layer.
        use_bn(bool): Whether to use batch normalization.
        
    Attributes:
        _nn(paddle.nn.Sequential): Dynamic graph LayerList.
    """
    def __init__(
        self,
        in_chunk_dim: int,
        out_chunk_dim: int,
        target_dim: int,
        hidden_config: List[int],
        use_bn: bool = False
    ):
        super(_MLPModule, self).__init__()
        raise_if(
            np.any(np.array(hidden_config) <= 0),
            f"hidden_config must be > 0, got {hidden_config}."
        )       
        dimensions, layers = [in_chunk_dim] + hidden_config, []
        for in_dim, out_dim in zip(dimensions[:-1], dimensions[1:]):
            layers.append(paddle.nn.Linear(in_dim, out_dim))
            if use_bn: layers.append(paddle.nn.BatchNorm1D(target_dim))
            layers.append(paddle.nn.ReLU())
        layers.append(
            paddle.nn.Linear(dimensions[-1], out_chunk_dim)
        )
        self._nn = paddle.nn.Sequential(*layers)

    def forward(
        self, 
        X: Dict[str, paddle.Tensor]
    ) -> paddle.Tensor:
        """Forward.

        Args: 
            X(paddle.Tensor): Dict of feature tensor.

        Returns:
            paddle.Tensor: Output of model.
        """
        out = X[PAST_TARGET]
        out = paddle.transpose(out, perm=[0, 2, 1])
        out = self._nn(out)
        out = paddle.transpose(out, perm=[0, 2, 1])
        return out


class MLPRegressor(PaddleBaseModelImpl):
    """Multilayer Perceptron.

    Args:
        in_chunk_len(int): The size of the loopback window, i.e. the number of time steps feed to the model.
        out_chunk_len(int): The size of the forecasting horizon, i.e. the number of time steps output by the model.
        skip_chunk_len(int): Optional, the number of time steps between in_chunk and out_chunk for a single sample.
            The skip chunk is neither used as a feature (i.e. X) nor a label (i.e. Y) for a single sample.
            By default it will NOT skip any time steps.
        sampling_stride(int): Sampling intervals between two adjacent samples.
        loss_fn(Callable[..., paddle.Tensor]): Loss function.
        optimizer_fn(Callable[..., Optimizer]): Optimizer algorithm.
        optimizer_params(Dict[str, Any]): Optimizer parameters.
        eval_metrics(List[str]): Evaluation metrics of model.
        callbacks(List[Callback]): Customized callback functions.
        batch_size(int): Number of samples per batch.
        max_epochs(int): Max epochs during training.
        verbose(int): Verbosity mode.
        patience(int): Number of epochs to wait for improvement before terminating.
        seed(int|None): Global random seed.

        hidden_config(List[int]|None): The ith element represents the number of neurons in the ith hidden layer.
        use_bn(bool): Whether to use batch normalization.

    Attributes:
        _in_chunk_len(int): The size of the loopback window, i.e. the number of time steps feed to the model.
        _out_chunk_len(int): The size of the forecasting horizon, i.e. the number of time steps output by the model.
        _skip_chunk_len(int): Optional, the number of time steps between in_chunk and out_chunk for a single sample.
            The skip chunk is neither used as a feature (i.e. X) nor a label (i.e. Y) for a single sample.
            By default it will NOT skip any time steps.
        _sampling_stride(int): Sampling intervals between two adjacent samples.
        _loss_fn(Callable[..., paddle.Tensor]): Loss function.
        _optimizer_fn(Callable[..., Optimizer]): Optimizer algorithm.
        _optimizer_params(Dict[str, Any]): Optimizer parameters.
        _eval_metrics(List[str]): Evaluation metrics of model.
        _callbacks(List[Callback]): Customized callback functions.
        _batch_size(int): Number of samples per batch.
        _max_epochs(int): Max epochs during training.
        _verbose(int): Verbosity mode.
        _patience(int): Number of epochs to wait for improvement before terminating.
        _seed(int|None): Global random seed.
        _stop_training(bool) Training status.
        _hidden_config(List[int]|None): The ith element represents the number of neurons in the ith hidden layer.
        _use_bn(bool): Whether to use batch normalization.
    """
    def __init__(
        self,
        in_chunk_len: int,
        out_chunk_len: int,
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

        hidden_config: List[int] = None,
        use_bn: bool = False
    ):
        self._hidden_config = (
            hidden_config if hidden_config else [100]
        )
        self._use_bn = use_bn
        super(MLPRegressor, self).__init__(
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

    def _check_tsdataset(
        self,
        tsdataset: TSDataset
    ):
        """Ensure the robustness of input data (consistent feature order), at the same time,
            check whether the data types are compatible. If not, the processing logic is as follows:

            1> Integer: Convert to np.int64.

            2> Floating: Convert to np.float32.

            3> Missing value: Warning.

            4> Other: Illegal.

        Args:
            tsdataset(TSDataset): Data to be checked.
        """
        target_columns = tsdataset.get_target().dtypes.keys()
        for column, dtype in tsdataset.dtypes.items():
            if column in target_columns:
                raise_if_not(
                    np.issubdtype(dtype, np.floating),
                    f"mlp's target dtype only supports [float16, float32, float64], " \
                    f"but received {column}: {dtype}."
                )
                continue
            raise_if_not(
                np.issubdtype(dtype, np.floating),
                f"mlp's cov(observed or known) dtype currently only supports [float16, float32, float64], " \
                f"but received {column}: {dtype}."
            )
        super(MLPRegressor, self)._check_tsdataset(tsdataset)
        
    def _update_fit_params(
        self,
        train_tsdataset: List[TSDataset],
        valid_tsdataset: Optional[List[TSDataset]] = None
    ) -> Dict[str, Any]:
        """Infer parameters by TSdataset automatically.

        Args:
            train_tsdataset(List[TSDataset]): list of train dataset.
            valid_tsdataset(List[TSDataset]|None): list of validation dataset.

        Returns:
            Dict[str, Any]: model parameters.
        """
        target_dim = train_tsdataset[0].get_target().data.shape[1]
        fit_params = {
            "target_dim": target_dim
        }
        return fit_params
        
    def _init_network(self) -> paddle.nn.Layer:
        """Setup the network.

        Returns:
            paddle.nn.Layer.
        """
        return _MLPModule(
            in_chunk_dim=self._in_chunk_len,
            out_chunk_dim=self._out_chunk_len,
            target_dim=self._fit_params["target_dim"],
            hidden_config=self._hidden_config,
            use_bn=self._use_bn
        )
