#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from typing import List, Dict, Any, Callable, Optional, Tuple

from paddle.optimizer import Optimizer
import paddle.nn.functional as F
import numpy as np
import paddle

from paddlets.models.forecasting.dl.paddle_base_impl import PaddleBaseModelImpl
from paddlets.models.common.callbacks import Callback
from paddlets.logger import raise_if_not
from paddlets.datasets import TSDataset

COVS = ["observed_cov_numeric", "known_cov_numeric"]
PAST_TARGET = "past_target"


class _PositionalEncoding(paddle.nn.Layer):
    """Paddle layer implementing positional encoding.

    Args:
        d_model(int): The expected feature size for the input/output of the transformer's encoder/decoder.
        max_len(int): The dimensionality of the computed positional encoding array.
        dropout(float): Fraction of neurons affected by Dropout.

    Attributes:
        _dropout(paddle.nn.Layer): Fraction of neurons affected by Dropout.
        _pe(paddle.nn.Tensor): positional encoding as buffer into the layer.
    """
    def __init__(
        self,
        d_model: int,
        max_len: int,
        dropout_rate,
    ):
        super(_PositionalEncoding, self).__init__()
        self._dropout = paddle.nn.Dropout(dropout_rate)

        # The calculation formula of the positional encodeing is as follows.
        # PE(pos, 2i) = sin(pos / 1e4 ** (2i / d_model)).
        # PE(pos, 2i + 1) = cos(pos / 1e4 ** (2i / d_model)).
        # Where: 
        #   d_model: The expected feature size for the input/output of the transformer's encoder/decoder.
        #   pos: a position in the input sequence. 
        #   2i/2i + 1: odd/even index of d_model.
        pe = paddle.zeros((max_len, d_model))
        position = paddle.unsqueeze(
            paddle.arange(0, max_len, dtype="float32"), axis=1
        )
        div_term = paddle.exp(
            paddle.arange(0, d_model, 2, dtype="float32") * (-1. * np.log2(1e4) / d_model)
        )
        pe[:, 0::2] = paddle.sin(position * div_term)
        pe[:, 1::2] = paddle.cos(position * div_term)
        self.register_buffer("_pe", pe)

    def forward(
        self, 
        X: paddle.Tensor
    ) -> paddle.Tensor:
        """Forward.

        Args:
            X(paddle.Tensor): Feature tensor.
                Tensor containing the embedded time series.
                X of shape `(batch_size, in_chunk_len, d_model)`
        
        Returns:
            paddle.Tensor: Output of Layer.
                Tensor containing the embedded time series enhanced with positional encoding.
                Output of shape `(batch_size, input_size, d_model)`
        """
        out = X + self._pe[: X.shape[1], :]
        return self._dropout(out)


class _TransformerModule(paddle.nn.Layer):
    """Paddle layer implementing Transformer module.

    Args:
        in_chunk_len(int): The size of the loopback window, i.e. the number of time steps feed to the model.
        out_chunk_len(int): The size of the forecasting horizon, i.e. the number of time steps output by the model.
        target_dim(int): The numer of targets.
        input_dim(int): The number of channels in the input series.
        d_model(int): The expected feature size for the input/output of transformer's encoder/decoder.
        nhead(int): The number of heads in the multi-head attention mechanism.
        num_encoder_layers(int): The number of encoder layers in the encoder.
        num_decoder_layers(int): The number of decoder layers in the decoder.
        dim_feedforward(int): The dimension of the feedforward network model.
        activation(str): The activation function of encoder/decoder intermediate layer. ["relu", "gelu"] is optional.
        dropout_rate(float): Fraction of neurons affected by Dropout.
        custom_encoder(paddle.nn.Layer|None): A custom user-provided encoder module for the transformer.
        custom_decoder(paddle.nn.Layer|None): A custom user-provided decoder module for the transformer.

    Attributes:
        _in_chunk_len(int): The size of the loopback window, i.e. the number of time steps feed to the model.
        _out_chunk_len(int): The size of the forecasting horizon, i.e. the number of time steps output by the model.
        _target_dim(int): The numer of targets.
        _input_dim(int): The number of channels in the input series.
        _encoder(paddle.nn.Layer): The encoder.
        _positional_encoding(paddle.nn.Layer): The positional encoding.
        _activation(str): The activation function of encoder/decoder intermediate layer.
        _transformer(paddle.nn.Layer): Transformer is a state-of-the-art deep learning model.
        _decoder(paddle.nn.Layer): The decoder projection layer.
    """
    def __init__(
        self,
        in_chunk_len: int,
        out_chunk_len: int,
        target_dim: int,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        activation: str,
        dropout_rate: float,
        custom_encoder: Optional[paddle.nn.Layer] = None,
        custom_decoder: Optional[paddle.nn.Layer] = None,
    ):
        super(_TransformerModule, self).__init__()
        self._in_chunk_len = in_chunk_len
        self._out_chunk_len = out_chunk_len
        self._target_dim = target_dim 
        self._input_dim = input_dim

        # Encoding step.
        #   1> Mapping the target_dim to d_model with a linear layer.
        #   2> Adding relative position information to the input sequence using PositionalEncoding.
        self._encoder = paddle.nn.Linear(input_dim, d_model)
        self._positional_encoding = _PositionalEncoding(
            d_model, in_chunk_len, dropout_rate
        )

        # Transformer(interact features using self-attention) step.
        #   1> Interact input sequence features using self-attention
        #   2> Interact encoded sequence with decoded sequence using attention mechanism.
        #   3> Note that the length of the decoding sequence here is 1, 
        #       so there is no self-attention between the decoding sequene.
        self._transformer = paddle.nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
            activation=activation,
            custom_encoder=custom_encoder,
            custom_decoder=custom_decoder,
        )

        # Decoding step.
        # Since the length of the decoded sequence is 1 (with shape [batch_size, 1, d_model]), 
        # we need to use linear to map it to out_chunk_len(with shape [batch_size, out_chunk_len, target_dim]) 
        # to get the final prediction result.
        self._decoder = paddle.nn.Linear(
            d_model, out_chunk_len * target_dim
        )

    def _create_transformer_inputs(
        self, 
        X: Dict[str, paddle.Tensor]
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """`TSDataset` stores time series in the (batch_size, in_chunk_len, target_dim) format.
            Take X[batch_size, -1:, target_dim] as input to decoder.

        Args:
            X(Dict[str, paddle.Tensor]): Dict of feature tensor.
        
        Returns:
            Tuple[paddle.Tensor, paddle.Tensor]: The inputs for the encoder and decoder
        """
        covs = [
            X[cov][:, :self._in_chunk_len, :] for cov in COVS if cov in X
        ]
        feats = [X[PAST_TARGET]] + covs
        src = paddle.concat(feats, axis=-1)
        tgt = src[:, -1:, :]
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
        # Here we create `src` and `tgt`, the inputs for the encoder and decoder
        # side of the Transformer architecture
        src, tgt = self._create_transformer_inputs(X)

        # "np.sqrt(input_dim)" is a normalization factor
        # see section 3.2.1 in 'Attention is All you Need' by Vaswani et al. (2017)
        src = self._encoder(src) * np.sqrt(self._input_dim)
        src = self._positional_encoding(src)
        
        tgt = self._encoder(tgt) * np.sqrt(self._input_dim)
        tgt = self._positional_encoding(tgt)

        out = self._transformer(src, tgt)
        out = self._decoder(out)
        # Here we change the data format
        # from (batch_size, 1, out_chunk_len * target_dim)
        # to (batch_size, out_chunk_len, target_dim)
        out = paddle.reshape(out[:, 0, :], shape=[-1, self._out_chunk_len, self._target_dim])
        return out


class TransformerModel(PaddleBaseModelImpl):
    """Transformer\[1\] is a state-of-the-art deep learning model introduced in 2017. 
    It is an encoder-decoder architecture whose core feature is the `multi-head attention` mechanism, 
    which is able to draw intra-dependencies within the input vector and within the output vector (`self-attention`)
    as well as inter-dependencies between input and output vectors (`encoder-decoder attention`).

    \[1\] Vaswani A, et al. "Attention Is All You Need", `<https://arxiv.org/abs/1706.03762>`_

    Args:
        in_chunk_len(int): The size of the loopback window, i.e. the number of time steps feed to the model.
        out_chunk_len(int): The size of the forecasting horizon, i.e. the number of time steps output by the model.
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

        d_model(int): The expected feature size for the input/output of the transformer's encoder/decoder.
        nhead(int): The number of heads in the multi-head attention mechanism.
        num_encoder_layers(int): The number of encoder layers in the encoder.
        num_decoder_layers(int): The number of decoder layers in the decoder.
        dim_feedforward(int): The dimension of the feedforward network model.
        activation(str): The activation function of encoder/decoder intermediate layer, ["relu", "gelu"] is optional.
        dropout_rate(float): Fraction of neurons affected by Dropout.
        custom_encoder(paddle.nn.Layer|None): A custom user-provided encoder module for the transformer.
        custom_decoder(paddle.nn.Layer|None): A custom user-provided decoder module for the transformer.

    Attributes:
        _in_chunk_len(int): The size of the loopback window, i.e. the number of time steps feed to the model.
        _out_chunk_len(int): The size of the forecasting horizon, i.e. the number of time steps output by the model.
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

        _d_model(int): The expected feature size for the input/output of the transformer's encoder/decoder.
        _nhead(int): The number of heads in the multi-head attention mechanism.
        _num_encoder_layers(int): The number of encoder layers in the encoder.
        _num_decoder_layers(int): The number of decoder layers in the decoder.
        _dim_feedforward(int): The dimension of the feedforward network model.
        _activation(str): The activation function of encoder/decoder intermediate layer. ["relu", "gelu"] is optional.
        _dropout_rate(float): Fraction of neurons affected by Dropout.
        _custom_encoder(paddle.nn.Layer|None): A custom user-provided encoder module for the transformer.
        _custom_decoder(paddle.nn.Layer|None): A custom user-provided decoder module for the transformer.
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
        batch_size: int = 128,
        max_epochs: int = 10,
        verbose: int = 1,
        patience: int = 4,
        seed: Optional[int] = None,

        d_model: int = 8,
        nhead: int = 4,
        num_encoder_layers: int = 1,
        num_decoder_layers: int = 1,
        dim_feedforward: int = 64,
        activation: str = "relu",
        dropout_rate: float = 0.1,
        custom_encoder: Optional[paddle.nn.Layer] = None,
        custom_decoder: Optional[paddle.nn.Layer] = None,
    ):
        self._d_model = d_model
        self._nhead = nhead
        self._num_encoder_layers = num_encoder_layers
        self._num_decoder_layers = num_decoder_layers
        self._dim_feedforward = dim_feedforward
        self._activation = activation
        self._dropout_rate = dropout_rate
        self._custom_encoder = custom_encoder
        self._custom_decoder = custom_decoder
        super(TransformerModel, self).__init__(
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
                    f"transformer's target dtype only supports [float16, float32, float64], " \
                    f"but received {column}: {dtype}."
                )
                continue
            raise_if_not(
                np.issubdtype(dtype, np.floating),
                f"transformer's cov(observed or known) dtype currently only supports [float16, float32, float64], " \
                f"but received {column}: {dtype}."
            )
        super(TransformerModel, self)._check_tsdataset(tsdataset)
        
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
        input_dim = target_dim = train_tsdataset[0].get_target().data.shape[1]
        if train_tsdataset[0].get_observed_cov():
            input_dim += train_tsdataset[0].get_observed_cov().data.shape[1]
        if train_tsdataset[0].get_known_cov():
            input_dim += train_tsdataset[0].get_known_cov().data.shape[1]
        fit_params = {
            "target_dim": target_dim,
            "input_dim": input_dim
        }
        return fit_params
        
    def _init_network(self) -> paddle.nn.Layer:
        """Setup the network.

        Returns:
            paddle.nn.Layer
        """
        return _TransformerModule(
            in_chunk_len=self._in_chunk_len,
            out_chunk_len=self._out_chunk_len,
            target_dim=self._fit_params["target_dim"],
            input_dim=self._fit_params["input_dim"],
            d_model=self._d_model,
            nhead=self._nhead,
            num_encoder_layers=self._num_encoder_layers,
            num_decoder_layers=self._num_decoder_layers,
            dim_feedforward=self._dim_feedforward,
            activation=self._activation,
            dropout_rate=self._dropout_rate,
            custom_encoder=self._custom_encoder,
            custom_decoder=self._custom_decoder
        )
