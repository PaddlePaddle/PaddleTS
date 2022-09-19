#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from typing import Optional

import paddle

from paddlets.models.forecasting.dl._informer.encoder import InformerEncoderLayer
from paddlets.models.forecasting.dl._informer.decoder import InformerDecoderLayer
from paddlets.models.forecasting.dl._informer.encoder import InformerEncoder
from paddlets.models.forecasting.dl._informer.decoder import InformerDecoder
from paddlets.models.forecasting.dl._informer.encoder import ConvLayer


class Informer(paddle.nn.Layer):
    """A Informer model composed of an instance of `InformerEncoder` and an instance of 
    `InformerDecoder`. While the embedding layer and output layer are not included.

    Args:
        d_model(int): The expected feature size for the input/output of the informer's encoder/decoder.
        nheads(int): The number of heads in multi-head attention.
        ffn_channels(int): The Number of channels for Conv1D of FFN layer.
        num_encoder_layers(int): The number of encoder layers in the encoder.
        num_decoder_layers(int): The number of decoder layers in the decoder.
        dropout_rate(float): The dropout probability used on attention
            weights to drop some attention targets.

    Attributes:
        _encoder(paddle.nn.Layer): A encoder module for the informer.
        _decoder(paddle.nn.Layer): A decoder module for the informer.

    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        ffn_channels: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        activation: str,
        dropout_rate: float = 0.1,
    ):
        super(Informer, self).__init__() 
        conv_layer = ConvLayer(d_model)
        encoder_layer = InformerEncoderLayer(d_model, nhead, ffn_channels, activation, dropout_rate)
        decoder_layer = InformerDecoderLayer(d_model, nhead, ffn_channels, dropout_rate)
        self._encoder = InformerEncoder(encoder_layer, conv_layer, num_encoder_layers)
        self._decoder = InformerDecoder(decoder_layer, num_decoder_layers)

    def forward(
        self,
        src: paddle.Tensor,
        tgt: paddle.Tensor,
        src_mask: Optional[paddle.Tensor] = None, 
        tgt_mask: Optional[paddle.Tensor] = None,
        memory_mask: Optional[paddle.Tensor] = None
    ) -> paddle.Tensor:
        """Forward.
        Args:
            src(paddle.Tensor): The input of Informer encoder.
            tgt(paddle.Tensor): The input of Informer decoder.
            src_mask(paddle.Tensor|None): A tensor used in multi-head attention 
                to prevents attention to some unwanted positions.
            tgt_mask(paddle.Tensor|None): A tensor used in multi-head attention
                to prevents attention to some unwanted positions.

        Returns:
            paddle.Tensor: Output of model.
        """
        memory = self._encoder(src, src_mask)
        out = self._decoder(tgt, memory, tgt_mask, memory_mask)
        return out
