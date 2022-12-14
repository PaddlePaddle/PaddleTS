#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from typing import List, Dict, Any, Callable, Optional, Tuple

from paddle.optimizer import Optimizer
import paddle.nn.functional as F
import numpy as np
import paddle

from paddlets.models.forecasting.dl._informer import MixedEmbedding, Informer
from paddlets.models.forecasting.dl.paddle_base_impl import PaddleBaseModelImpl
from paddlets.models.forecasting.dl.revin import revin_norm
from paddlets.models.common.callbacks import Callback
from paddlets.logger import raise_if_not
from paddlets.datasets import TSDataset

PAST_TARGET = "past_target"


class _InformerModule(paddle.nn.Layer):
    """Paddle layer implementing informer module.

    Args:
        in_chunk_len(int): The size of the loopback window, i.e. the number of time steps feed to the model.
        out_chunk_len(int): The size of the forecasting horizon, i.e. the number of time steps output by the model.
        start_token_len(int): The start token size of the forecasting horizon.
        target_dim(int): The numer of targets.
        d_model(int): The expected feature size for the input/output of informer's encoder/decoder.
        nhead(int): The number of heads in the multi-head attention mechanism.
        ffn_channels(int): The Number of channels for Conv1D of FFN layer.
        num_encoder_layers(int): The number of encoder layers in the encoder.
        num_decoder_layers(int): The number of decoder layers in the decoder.
        activation(str): The activation function of encoder/decoder intermediate layer, 
            ["relu", "gelu"] is optional.
        dropout_rate(float): Fraction of neurons affected by Dropout.

    Attributes:
        _out_chunk_len(int): The size of the forecasting horizon, i.e. the number of time steps output by the model.
        _start_token_len(int): The start token size of the forecasting horizon.
        _target_dim(int): The numer of targets.
        _src_embedding(paddle.nn.Layer): A data(position + token) embedding.
        _tgt_embedding(paddle.nn.Layer): A data(position + token) embedding.
        _informer(paddle.nn.Layer): A Informer model composed of an instance of `InformerEncoder` 
            and an instance `InformerDecoder`
        _out_proj(paddle.nn.Layer): The projection layer.
    """
    def __init__(
        self,
        in_chunk_len: int,
        out_chunk_len: int,
        start_token_len: int,
        target_dim: int,
        d_model: int,
        nhead: int,
        ffn_channels: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        activation: str,
        dropout_rate: float,
    ):
        super(_InformerModule, self).__init__()
        self._in_chunk_len = in_chunk_len
        self._out_chunk_len = out_chunk_len
        self._start_token_len = start_token_len
        self._target_dim = target_dim 
        raise_if_not(
            in_chunk_len >= start_token_len,
            f"`in_chunk_len` must be greater than or equal to `start_token_len`\n" \
            f"Choose a smaller `start_token_len` or bigger `in_chunk_len`."
        )

        # Encoding step.
        #   1> Adding relative position/timfeat/token information to the input sequence.
        self._src_embedding = MixedEmbedding(target_dim, d_model, in_chunk_len, dropout_rate)
        self._tgt_embedding = MixedEmbedding(target_dim, d_model, start_token_len + out_chunk_len, dropout_rate)

        # Informer(interact features using prob_sparse_attention and cross_attention) step.
        #   1> Interact src sequence features using prob_sparse_attention.
        #   2> Interact tgt sequence features using prob_sparse_attention.
        #   3> Interact encoded sequence with decoded sequence using cross_attention mechanism.
        self._informer = Informer(
            d_model=d_model,
            nhead=nhead,
            ffn_channels=ffn_channels,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            activation=activation,
            dropout_rate=dropout_rate,
        )

        # Projection step.
        self._out_proj = paddle.nn.Linear(d_model, target_dim)

    def _create_informer_inputs(
        self, 
        X: Dict[str, paddle.Tensor]
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """`TSDataset` stores time series in the (batch_size, in_chunk_len, target_dim) format.
            Take [X[batch_size, -out_chunk_len:, target_dim], paddle.zeros([batch_size, -out_chunk_len:, target_dim])] 
            as input to decoder.

        Args:
            X(Dict[str, paddle.Tensor]): Dict of feature tensor.
        
        Returns:
            Tuple[paddle.Tensor, paddle.Tensor]: The inputs for the encoder and decoder
        """
        src = X[PAST_TARGET]
        batch_size, _, d_model = src.shape
        tgt = src[:, self._in_chunk_len - self._start_token_len:, :]
        padding = paddle.zeros([batch_size, self._out_chunk_len, d_model])
        tgt = paddle.concat([tgt, padding], axis=1)
        return src, tgt

    def forward(
        self,
        X: Dict[str, paddle.Tensor]
    ) -> paddle.Tensor:
        """Forward.

        Args:
            X(Dict[str, paddle.Tensor]): Dict of feature tensor.
        
        Returns:
            paddle.Tensor: Output of model.
        """
        # Here we create `src` and `tgt`, 
        # the inputs for the encoder and decoder side of the informer architecture.
        src, tgt = self._create_informer_inputs(X)
        src = self._src_embedding(src)
        tgt = self._tgt_embedding(tgt)
        out = self._informer(src, tgt)
        out = self._out_proj(out)
        # Since the decoder output contains information of start token,
        # we need to truncate the last out_chunk_len time step as the final prediction result.
        out = out[:, -self._out_chunk_len:, :]
        return out


class InformerModel(PaddleBaseModelImpl):
    """Informer\[1\] is a state-of-the-art deep learning model introduced in 2021. 
    It is an encoder-decoder architecture whose core feature is the `prob sparse attention` mechanism, 
    which achieves the O(LlogL) time complexity and O(LlogL) memory usage on dependency alignments.

    \[1\] Zhou H, et al. "Informer: Beyond efficient transformer for long sequence time-series forecasting", `<https://arxiv.org/abs/2012.07436>`_

    Args:
        in_chunk_len(int): The size of the loopback window, i.e. the number of time steps feed to the model.
        out_chunk_len(int): The size of the forecasting horizon, i.e. the number of time steps output by the model.
        start_token_len(int): The start token size of the forecasting horizon.
        skip_chunk_len(int): Optional, the number of time steps between in_chunk and out_chunk for a single sample.
            The skip chunk is neither used as a feature (i.e. X) nor a label (i.e. Y) for a single sample.
            By default it will NOT skip any time steps.
        sampling_stride(int): Sampling intervals between two adjacent samples.
        loss_fn(Callable[..., paddle.Tensor]|None): Loss function.
        optimizer_fn(Callable[..., Optimizer]): Optimizer algorithm.
        optimizer_params(Dict[str, Any]): Optimizer parameters.
        eval_metrics(List[str]): Evaluation metrics of model.
        callbacks(List[Callback]): Customized callback functions.
        batch_size(int): Number of samples per batch.
        max_epochs(int): Max epochs during training.
        verbose(int): Verbosity mode.
        patience(int): Number of epochs to wait for improvement before terminating.
        seed(int|None): Global random seed.

        d_model(int): The expected feature size for the input/output of the informer's encoder/decoder.
        nhead(int): The number of heads in the multi-head attention mechanism.
        ffn_channels(int): The Number of channels for Conv1D of FFN layer.
        num_encoder_layers(int): The number of encoder layers in the encoder.
        num_decoder_layers(int): The number of decoder layers in the decoder.
        activation(str): The activation function of encoder/decoder intermediate layer, 
            ["relu", "gelu"] is optional.
        dropout_rate(float): Fraction of neurons affected by Dropout.

    Attributes:
        _in_chunk_len(int): The size of the loopback window, i.e. the number of time steps feed to the model.
        _out_chunk_len(int): The size of the forecasting horizon, i.e. the number of time steps output by the model.
        _start_token_len(int): The start token size of the forecasting horizon.
        _skip_chunk_len(int): Optional, the number of time steps between in_chunk and out_chunk for a single sample.
            The skip chunk is neither used as a feature (i.e. X) nor a label (i.e. Y) for a single sample.
            By default it will NOT skip any time steps.
        _sampling_stride(int): Sampling intervals between two adjacent samples.
        _loss_fn(Callable[..., paddle.Tensor]|None): Loss function.
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
        
        _d_model(int): The expected feature size for the input/output of the informer's encoder/decoder.
        _nhead(int): The number of heads in the multi-head attention mechanism.
        _num_encoder_layers(int): The number of encoder layers in the encoder.
        _num_decoder_layers(int): The number of decoder layers in the decoder.
        _activation(str): The activation function of encoder/decoder intermediate layer.
            ["relu", "gelu"] is optional.
        _dropout_rate(float): Fraction of neurons affected by Dropout.
    """
    def __init__(
        self,
        in_chunk_len: int,
        out_chunk_len: int,
        start_token_len: int = 0,
        skip_chunk_len: int = 0,
        sampling_stride: int = 1,
        loss_fn: Callable[..., paddle.Tensor] = F.mse_loss,
        optimizer_fn: Callable[..., Optimizer] = paddle.optimizer.Adam,
        optimizer_params: Dict[str, Any] = dict(learning_rate=1e-3), 
        eval_metrics: List[str] = [], 
        callbacks: List[Callback] = [], 
        batch_size: int = 128,
        max_epochs: int = 10,
        verbose: int = 1,
        patience: int = 4,
        seed: Optional[int] = None,
        
        d_model: int = 512,
        nhead: int = 8,
        ffn_channels: int = 2048,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 1,
        activation: str = "relu",
        dropout_rate: float = 0.1,
        use_revin: bool = False,
        revin_params: Dict[str, Any] = dict(eps=1e-5, affine=True),
    ):
        self._start_token_len = start_token_len
        self._d_model = d_model
        self._nhead = nhead
        self._ffn_channels = ffn_channels
        self._num_encoder_layers = num_encoder_layers
        self._num_decoder_layers = num_decoder_layers
        self._activation = activation
        self._dropout_rate = dropout_rate
        self._use_revin = use_revin
        self._revin_params = revin_params

        super(InformerModel, self).__init__(
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
                    f"informer's target dtype only supports [float16, float32, float64], " \
                    f"but received {column}: {dtype}."
                )
                continue
            raise_if_not(
                np.issubdtype(dtype, np.floating),
                f"informer's cov(observed or known) dtype currently only supports [float16, float32, float64], " \
                f"but received {column}: {dtype}."
            )
        super(InformerModel, self)._check_tsdataset(tsdataset)
        
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
        
    @revin_norm
    def _init_network(self) -> paddle.nn.Layer:
        """Setup the network.

        Returns:
            paddle.nn.Layer
        """
        return _InformerModule(
            in_chunk_len=self._in_chunk_len,
            out_chunk_len=self._out_chunk_len,
            start_token_len=self._start_token_len,
            target_dim=self._fit_params["target_dim"],
            d_model=self._d_model,
            nhead=self._nhead,
            ffn_channels=self._ffn_channels,
            num_encoder_layers=self._num_encoder_layers,
            num_decoder_layers=self._num_decoder_layers,
            activation=self._activation,
            dropout_rate=self._dropout_rate,
        )
