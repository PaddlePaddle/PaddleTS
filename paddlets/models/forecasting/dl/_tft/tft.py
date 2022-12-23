#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
This implementation is TFT model based on the article `Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting <https://arxiv.org/abs/1912.09363>`_.
"""

from typing import List, Dict, Optional, Tuple
import copy

import numpy as np
import pandas as pd
import paddle
from paddle import nn
import paddle.nn.functional as F

from paddlets.models.forecasting.dl._tft import GateAddNorm
from paddlets.models.forecasting.dl._tft import GatedResidualNetwork
from paddlets.models.forecasting.dl._tft import VariableSelectionNetwork
from paddlets.models.forecasting.dl._tft import InputChannelEmbedding
from paddlets.models.forecasting.dl._tft import InterpretableMultiHeadAttention


class TemporalFusionTransformer(nn.Layer):
    """
    Implementation of TFT model.

    Args:
        in_chunk_len(int): The size of the loopback window, i.e., the number of time steps feed to the model.
        out_chunk_len(int): The size of the forecasting horizon, i.e., the number of time steps output by the model.
        fit_params(dict): The dimensions and dict sizes of variables.
        hidden_dim(int, Optional): The number of features in the hidden state of the TFT module.
        lstm_layers_num(int, Optional): The number of LSTM layers.
        attention_heads_num(int, Optional): The number of heads of self-attention module.
        output_quantiles(List[float], Optional): The output quantiles of the model.
        dropout(float, Optional): The fraction of neurons that are dropped in all-but-last RNN layers.
    """

    def __init__(
        self,
        in_chunk_len: int,
        out_chunk_len: int,
        fit_params: dict,
        hidden_dim: int = 64,
        lstm_layers_num: int = 1,
        attention_heads_num: int = 2,
        output_quantiles: List[float] = [0.1, 0.5, 0.9],
        dropout: float = 0.0,
    ):
        super().__init__()        
        # ============
        # data props
        # ============
        # data input & output horizon
        self._in_chunk_len = in_chunk_len
        self._out_chunk_len = out_chunk_len
        # data variables
        self._target_dim = fit_params["target_dim"]
        self._known_num_dim = fit_params["known_num_dim"]
        self._known_cat_dim = fit_params["known_cat_dim"]
        self._observed_num_dim = fit_params["observed_num_dim"]
        self._observed_cat_dim = fit_params["observed_cat_dim"]
        self._known_cat_size = fit_params["known_cat_size"]
        self._observed_cat_size = fit_params["observed_cat_size"]
        # history data
        self._num_historical_numeric = self._target_dim + self._known_num_dim + self._observed_num_dim
        self._num_historical_categorical = self._known_cat_dim + self._observed_cat_dim
        self._historical_categorical_cardinalities = self._known_cat_size + self._observed_cat_size
        # future data
        self._num_future_numeric = self._known_num_dim
        self._num_future_categorical = self._known_cat_dim
        self._future_categorical_cardinalities = self._known_cat_size
        # static data
        self._static_num_dim = fit_params["static_num_dim"]
        self._static_cat_dim = fit_params["static_cat_dim"]
        self._static_cat_size = fit_params["static_cat_size"]
        self._interpretable_output = False

        # ============
        # model props
        # ============
        self._output_quantiles = output_quantiles
        self._num_outputs = len(self._output_quantiles) * self._target_dim
        self._attention_heads_num = attention_heads_num
        self._lstm_layers = lstm_layers_num
        self._state_size = hidden_dim
        self._dropout = dropout
        
        # =====================
        # Input Transformation
        # =====================
        self._static_transform = InputChannelEmbedding(
            state_size=self._state_size,
            num_numeric=self._static_num_dim,
            num_categorical=self._static_cat_dim,
            categorical_cardinalities=self._static_cat_size,
            time_distribute=False)
        self._historical_ts_transform = InputChannelEmbedding(
            state_size=self._state_size,
            num_numeric=self._num_historical_numeric,
            num_categorical=self._num_historical_categorical,
            categorical_cardinalities=self._historical_categorical_cardinalities,
            time_distribute=True)
        self._future_ts_transform = InputChannelEmbedding(
            state_size=self._state_size,
            num_numeric=self._num_future_numeric,
            num_categorical=self._num_future_categorical,
            categorical_cardinalities=self._future_categorical_cardinalities,
            time_distribute=True)

        # =============================
        # Variable Selection Networks
        # =============================
        # selection module is None if no static covariates
        self._static_selection = VariableSelectionNetwork(
            input_dim=self._state_size,
            num_inputs=self._static_num_dim + self._static_cat_dim,
            hidden_dim=self._state_size, dropout=self._dropout) if self._static_num_dim + self._static_cat_dim else None
        
        self._historical_ts_selection = VariableSelectionNetwork(
            input_dim=self._state_size,
            num_inputs=self._num_historical_numeric + self._num_historical_categorical,
            hidden_dim=self._state_size,
            dropout=self._dropout,
            context_dim=self._state_size)
        self._future_ts_selection = VariableSelectionNetwork(
            input_dim=self._state_size,
            num_inputs=self._num_future_numeric + self._num_future_categorical,
            hidden_dim=self._state_size,
            dropout=self._dropout,
            context_dim=self._state_size)

        # =============================
        # static covariate encoders
        # =============================
        static_covariate_encoder = GatedResidualNetwork(input_dim=self._state_size,
                                                        hidden_dim=self._state_size,
                                                        output_dim=self._state_size,
                                                        dropout=self._dropout)
        self._static_encoder_selection = copy.deepcopy(static_covariate_encoder)
        self._static_encoder_enrichment = copy.deepcopy(static_covariate_encoder)
        self._static_encoder_sequential_cell_init = copy.deepcopy(static_covariate_encoder)
        self._static_encoder_sequential_state_init = copy.deepcopy(static_covariate_encoder)

        # ============================================================
        # Locality Enhancement with Sequence-to-Sequence processing
        # ============================================================
        self._past_lstm = nn.LSTM(input_size=self._state_size,
                                 hidden_size=self._state_size,
                                 num_layers=self._lstm_layers,
                                 dropout=self._dropout)
        self._future_lstm = nn.LSTM(input_size=self._state_size,
                                   hidden_size=self._state_size,
                                   num_layers=self._lstm_layers,
                                   dropout=self._dropout)
        self._post_lstm_gating = GateAddNorm(input_dim=self._state_size, dropout=self._dropout)

        # ============================================================
        # Static enrichment
        # ============================================================
        self._static_enrichment_grn = GatedResidualNetwork(input_dim=self._state_size,
                                                          hidden_dim=self._state_size,
                                                          output_dim=self._state_size,
                                                          context_dim=self._state_size,
                                                          dropout=self._dropout)

        # ============================================================
        # Temporal Self-Attention
        # ============================================================
        self._multihead_attn = InterpretableMultiHeadAttention(embed_dim=self._state_size, num_heads=self._attention_heads_num)
        self._post_attention_gating = GateAddNorm(input_dim=self._state_size, dropout=self._dropout)

        # ============================================================
        # Position-wise feed forward
        # ============================================================
        self._pos_wise_ff_grn = GatedResidualNetwork(input_dim=self._state_size,
                                                    hidden_dim=self._state_size,
                                                    output_dim=self._state_size,
                                                    dropout=self._dropout)
        self._pos_wise_ff_gating = GateAddNorm(input_dim=self._state_size, dropout=None)

        # ============================================================
        # Output layer
        # ============================================================
        self._output_layer = nn.Linear(self._state_size, self._num_outputs)

    def apply_temporal_selection(
        self, 
        temporal_representation: paddle.Tensor,
        static_selection_signal: Optional[paddle.Tensor],
        temporal_selection_module: VariableSelectionNetwork
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """
        Applying temporal variable selection network.
        
        Args:
            temporal_representation(paddle.Tensor): The temporal representation, which can be history representation or future representation.
            static_selection_signal(paddle.Tensor, Optional): The static selection signal, `None` if no static covariate. 
            temporal_selection_module(VariableSelectionNetwork): The variable selection network module.
        
        Returns:
            temporal_selection_output(paddle.Tensor):
            temporal_selection_weights(paddle.Tensor): 
        """
        num_samples, num_temporal_steps, _ = paddle.shape(temporal_representation)
        # Dimensions:
        # time_distributed_context: [num_samples x num_temporal_steps x state_size] (if any)
        # temporal_representation: [num_samples x num_temporal_steps x (total_num_temporal_inputs * state_size)]
        if static_selection_signal is not None:
            # replicate the selection signal along time
            time_distributed_context = self.replicate_along_time(static_signal=static_selection_signal,
                                                                 time_steps=num_temporal_steps)
            # for applying the same selection module on all time-steps, we stack the time dimension with the batch dimension
            time_distributed_context = self.stack_time_steps_along_batch(time_distributed_context)
        else:
            time_distributed_context = None

        # for applying the same selection module on all time-steps, we stack the time dimension with the batch dimension
        # Dimensions:
        # temporal_flattened_embedding: [(num_samples * num_temporal_steps) x (total_num_temporal_inputs * state_size)]
        temporal_flattened_embedding = self.stack_time_steps_along_batch(temporal_representation)

        # applying the selection module across time
        # Dimensions:
        # temporal_selection_output: [(num_samples * num_temporal_steps) x state_size]
        # temporal_selection_weights: [(num_samples * num_temporal_steps) x (num_temporal_inputs) x 1]
        temporal_selection_output, temporal_selection_weights = temporal_selection_module(
            flattened_embedding=temporal_flattened_embedding, context=time_distributed_context)

        # Reshape the selection outputs and selection weights - to represent the temporal dimension separately
        # Dimensions:
        # temporal_selection_output: [num_samples x num_temporal_steps x state_size)]
        # temporal_selection_weights: [num_samples x num_temporal_steps x num_temporal_inputs)]
        temporal_selection_output = temporal_selection_output.reshape([num_samples, num_temporal_steps, -1])
        temporal_selection_weights = temporal_selection_weights.squeeze(-1).reshape([num_samples, num_temporal_steps, -1])
        return temporal_selection_output, temporal_selection_weights

    @staticmethod
    def replicate_along_time(
        static_signal: paddle.Tensor, 
        time_steps: int
    ) -> paddle.Tensor:
        """
        This method gets as an input a static_signal (non-temporal tensor) [num_samples x num_features],
        and replicates it along time for 'time_steps' times, creating a tensor of [num_samples x time_steps x num_features]

        Args:
            static_signal(paddle.Tensor): the non-temporal tensor for which the replication is required.
            time_steps(int): the number of time steps according to which the replication is required.

        Returns:
            paddle.Tensor: the time-wise replicated tensor.
        """
        time_distributed_signal = paddle.tile(static_signal.unsqueeze(1), [1, time_steps, 1])
        return time_distributed_signal

    @staticmethod
    def stack_time_steps_along_batch(
        temporal_signal: paddle.Tensor
    ) -> paddle.Tensor:
        """
        This method gets as an input a temporal signal [num_samples x time_steps x num_features]
        and stacks the batch dimension and the temporal dimension on the same axis (axis=0).

        The last dimension (features dimension) is kept as is, but the rest is stacked along axis=0.
        
        Args:
            temporal_signal(paddle.Tensor): The temporal signal tensor for which stacking is required.
            
        Returns:
            paddle.Tensor: The stacked tensor.
        """
        return temporal_signal.reshape([-1, temporal_signal.shape[-1]])

    def transform_inputs(
        self, 
        batch: Dict[str, paddle.Tensor]
    ) -> Tuple[paddle.Tensor, ...]:
        """
        This method processes the batch and transform each input channel (historical_ts, future_ts, static)
        separately to eventually return the learned embedding for each of the input channels

        each feature is embedded to a vector of state_size dimension:
        - numeric features will be projected using a linear layer
        - categorical features will be embedded using an embedding layer

        eventually the embedding for all the features will be concatenated together on the last dimension of the tensor
        (i.e. axis=1 for the static features, axis=2 for the temporal data).
        
        Args:
            batch(Dict[str, paddle.Tensor]): A dict specifies all kinds of input data.
            
        Returns:
            future_ts_rep(paddle.Tensor): The representation of future input, generated by known covariates.
            historical_ts_rep: The representation of historical input, generated by known covariates, observed covariates and past targets.
            static_rep: The representation of static input, generated by static covariates.

        """
        empty_tensor = paddle.empty((0, 0, 0))
        static_rep = self._static_transform(x_numeric=batch.get('static_cov_numeric', empty_tensor).squeeze([1]),
                                           x_categorical=batch.get('static_cov_categorical', empty_tensor).squeeze([1]))
        if self._num_historical_numeric > 0:
            historical_ts_numeric = paddle.concat([batch.get("past_target", empty_tensor), \
                                                   batch.get("known_cov_numeric", empty_tensor)[:, :self._in_chunk_len, :], \
                                                   batch.get("observed_cov_numeric", empty_tensor)], \
                                                  axis=-1)
        else:
            historical_ts_numeric = empty_tensor
        if self._num_historical_categorical > 0:
            historical_ts_categorical = paddle.concat(
                [batch.get("known_cov_categorical", empty_tensor)[:, :self._in_chunk_len, :], \
                 batch.get("observed_cov_categorical", empty_tensor)], \
                axis=-1)
        else:
            historical_ts_categorical = empty_tensor
        historical_ts_rep = self._historical_ts_transform(x_numeric=historical_ts_numeric,
                                                         x_categorical=historical_ts_categorical)
        future_ts_rep = self._future_ts_transform(
            x_numeric=batch.get('known_cov_numeric', empty_tensor)[:, self._in_chunk_len:, :],
            x_categorical=batch.get('known_cov_categorical', empty_tensor)[:, self._in_chunk_len:, :])
        return future_ts_rep, historical_ts_rep, static_rep

    def get_static_encoders(
        self, 
        selected_static: paddle.Tensor
    ) -> Tuple[paddle.Tensor, ...]:
        """
        This method processes the variable selection results for the static data, yielding signals which are designed
        to allow better integration of the information from static metadata.
        Each of the resulting signals is generated using a separate GRN, and is eventually wired into various locations
        in the temporal fusion decoder, for allowing static variables to play an important role in processing.
        
        Args:
            selected_static(paddle.Tensor): The variable selection results for the static data.
        
        Returns:
            c_selection(paddle.Tensor): The tensor will be used for temporal variable selection.
            c_seq_hidden(paddle.Tensor): The hidden state will be used for local processing of temporal features.
            c_seq_cell(paddle.Tensor): The cell state will be used for local processing of temporal features.
            c_enrichment(paddle.Tensor): The tensor will be used for enriching temporal features with static information.
        """
        c_selection = self._static_encoder_selection(selected_static)
        c_enrichment = self._static_encoder_enrichment(selected_static)
        c_seq_hidden = self._static_encoder_sequential_state_init(selected_static)
        c_seq_cell = self._static_encoder_sequential_cell_init(selected_static)
        return c_enrichment, c_selection, c_seq_cell, c_seq_hidden

    def apply_sequential_processing(
        self, 
        selected_historical: paddle.Tensor, 
        selected_future: paddle.Tensor,
        c_seq_hidden: paddle.Tensor, 
        c_seq_cell: paddle.Tensor
    ) -> paddle.Tensor:
        """
        This part of the model is designated to mimic a sequence-to-sequence layer which will be used for local processing.
        On that part the historical ("observed" and "known") information will be fed into a recurrent layer called "Encoder" and
        the future information ("known") will be fed into a recurrent layer called "Decoder".
        This will generate a set of uniform temporal features which will serve as inputs into the temporal fusion
        decoder itself.
        To allow static metadata to influence local processing, we use "c_seq_hidden" and "c_seq_cell" context vectors
        from the static covariate encoders to initialize the hidden state and the cell state respectively.
        The output of the recurrent layers is gated and fused with a residual connection to the input of this block.
        
        Args:
            selected_historical(paddle.Tensor): The historical temporal signal.
            selected_future(paddle.Tensor): The futuristic temporal signal.
            c_seq_hidden(paddle.Tensor): The hidden state will be used for local processing of temporal features.
            c_seq_cell(paddle.Tensor): The cell state will be used for local processing of temporal features.
            
        Returns:
            gated_lstm_output(paddle.Tensor): The output of the gated recurrent layers.
        """

        # concatenate the historical (observed and known) temporal signal with the futuristic (known) temporal signal, along the
        # time dimension
        lstm_input = paddle.concat([selected_historical, selected_future], axis=1)

        # the historical temporal signal is fed into the first recurrent module
        # using the static metadata as initial hidden and cell state if any
        # (initial cell and hidden states are replicated for feeding to each layer in the stack)
        if c_seq_hidden is not None:
            past_lstm_output, hidden = self._past_lstm(selected_historical,
                                                      (paddle.tile(c_seq_hidden.unsqueeze(0), [self._lstm_layers, 1, 1]),
                                                       paddle.tile(c_seq_cell.unsqueeze(0), [self._lstm_layers, 1, 1])))
        else:
            past_lstm_output, hidden = self._past_lstm(selected_historical)

        # the future (known) temporal signal is fed into the second recurrent module
        # using the latest (hidden,cell) state of the first recurrent module
        # for setting the initial (hidden,cell) state.
        future_lstm_output, _ = self._future_lstm(selected_future, hidden)

        # concatenate the historical recurrent output with the futuristic recurrent output, along the time dimension
        lstm_output = paddle.concat([past_lstm_output, future_lstm_output], axis=1)

        # perform gating to the recurrent output signal, using a residual connection to input of this block
        gated_lstm_output = self._post_lstm_gating(lstm_output, residual=lstm_input)
        return gated_lstm_output

    def apply_static_enrichment(
        self, 
        gated_lstm_output: paddle.Tensor,
        static_enrichment_signal: paddle.Tensor
    ) -> paddle.Tensor:
        """
        This static enrichment stage enhances temporal features with static metadata using a GRN.
        The static enrichment signal is an output of a static covariate encoder, and the GRN is shared across time.
        
        Args:
            gated_lstm_output(paddle.Tensor): The output of the gated recurrent layers.
            static_enrichment_signal(paddle.Tensor): The tensor to enrich the temporal features.
            
        Returns:
            enriched_sequence(paddle.Tensor): The enriched sequence by static signal, if any.
        """
        num_samples, num_temporal_steps, _ = paddle.shape(gated_lstm_output)
        if static_enrichment_signal is not None:
            # replicate the selection signal along time
            time_distributed_context = self.replicate_along_time(static_signal=static_enrichment_signal,
                                                                 time_steps=num_temporal_steps)
            # for applying the same GRN module on all time-steps, we stack the time dimension with the batch dimension
            # Dimensions:
            # time_distributed_context: [num_samples x num_temporal_steps x state_size]
            time_distributed_context = self.stack_time_steps_along_batch(time_distributed_context)
        else:
            time_distributed_context = None

        # for applying the same GRN module on all time-steps, we stack the time dimension with the batch dimension
        # Dimensions:
        # flattened_gated_lstm_output: [(num_samples * num_temporal_steps) x state_size]
        flattened_gated_lstm_output = self.stack_time_steps_along_batch(gated_lstm_output)

        # applying the GRN using the static enrichment signal as context data
        # Dimensions:
        # enriched_sequence: [(num_samples * num_temporal_steps) x state_size]
        enriched_sequence = self._static_enrichment_grn(flattened_gated_lstm_output,
                                                       context=time_distributed_context)

        # reshape back to represent temporal dimension separately
        # Dimensions:
        # enriched_sequence: [num_samples x num_temporal_steps x state_size]
        enriched_sequence = enriched_sequence.reshape([num_samples, -1, self._state_size])
        return enriched_sequence

    def apply_self_attention(
        self, 
        enriched_sequence: paddle.Tensor,
        num_historical_steps: int,
        num_future_steps: int
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """
        This part performs multi-head self-attention and post-attention gating operation.
        
        Args:
            enriched_sequence(paddle.Tensor): The enriched sequence by static signal.
            num_historical_steps(int): The input chunk length of the model.
            num_future_steps(int): The output chunk length of the model.
            
        Returns:
            gated_post_attention(paddle.Tensor): The result of gated attention.
        """
        output_sequence_length = num_future_steps
        # create a mask - so that future steps will be exposed (able to attend) only to preceding steps
        # Dimensions:
        # mask: [output_sequence_length x (num_historical_steps + num_future_steps)]
        mask = paddle.concat([paddle.zeros([output_sequence_length, num_historical_steps]),
                              paddle.triu(paddle.ones([output_sequence_length, output_sequence_length]), diagonal=1)],
                             axis=1)

        # apply the interpretable multi-head attention mechanism
        # Dimensions:
        # post_attention: [num_samples x num_future_steps x state_size]
        # attention_outputs: [num_samples x num_future_steps x state_size]
        # attention_scores: [num_samples x num_future_steps x num_total_steps]
        post_attention, attention_outputs, attention_scores = self._multihead_attn(
            q=enriched_sequence[:, num_historical_steps:, :],  # query
            k=enriched_sequence,  # keys
            v=enriched_sequence,  # values
            mask=mask.astype(bool))

        # Apply gating with a residual connection to the input of this stage.
        # Because the output of the attention layer is only for the future time-steps,
        # the residual connection is only to the future time-steps of the temporal input signal
        # Dimensions:
        # gated_post_attention: [num_samples x num_future_steps x state_size]
        gated_post_attention = self._post_attention_gating(
            x=post_attention,
            residual=enriched_sequence[:, num_historical_steps:, :])
        return gated_post_attention, attention_scores

    def forward(self, batch):
        """
        forward TFT network.

        Args:
            batch(Dict[str, paddle.Tensor]): A dict specifies all kinds of input data.

        Returns:
            (case: forword computation) 
                paddle.Tensor: Forecasting Tensor of the TFT model.
            (case: interpretable result) 
                Dict[str, paddle.Tensor]: 
                    static_weights: The weights of static covariates, if any.
                    historical_selection_weights: The weights of historical variables.
                    future_selection_weights: The weights of futuristic variables.
                    attention_scores: The attention scores of self-attention layer.
            
        """        
        # =========== Transform all input channels ==============
        # Dimensions:
        # static_rep: [num_samples x (total_num_static_inputs * state_size)] (if any)
        # historical_ts_rep: [num_samples x num_historical_steps x (total_num_historical_inputs * state_size)]
        # future_ts_rep: [num_samples x num_future_steps x (total_num_future_inputs * state_size)]
        future_ts_rep, historical_ts_rep, static_rep = self.transform_inputs(batch)
        
        # =========== Static Variables Selection & Encoding==============
        # Only for the case static covariate is supplied
        if static_rep is not None:
            # Dimensions:
            # selected_static: [num_samples x state_size]
            # static_weights: [num_samples x num_static_inputs x 1]
            selected_static, static_weights = self._static_selection(static_rep)
            # each of the static encoders signals is of shape: [num_samples x state_size]
            c_enrichment, c_selection, c_seq_cell, c_seq_hidden = self.get_static_encoders(selected_static)
        else:
            selected_static, static_weights = None, None
            c_enrichment, c_selection, c_seq_cell, c_seq_hidden = None, None, None, None

        # =========== Historical variables selection ==============
        # Dimensions:
        # selected_historical: [num_samples x num_historical_steps x state_size]
        # historical_selection_weights: [num_samples x num_historical_steps x total_num_historical_inputs]
        selected_historical, historical_selection_weights = self.apply_temporal_selection(
            temporal_representation=historical_ts_rep,
            static_selection_signal=c_selection,
            temporal_selection_module=self._historical_ts_selection)

        # =========== Future variables selection ==============
        # Dimensions:
        # selected_future: [num_samples x num_future_steps x state_size]
        # future_selection_weights: [num_samples x num_future_steps x total_num_future_inputs]
        selected_future, future_selection_weights = self.apply_temporal_selection(
            temporal_representation=future_ts_rep,
            static_selection_signal=c_selection,
            temporal_selection_module=self._future_ts_selection)

        # =========== Locality Enhancement - Sequential Processing ==============
        # Dimensions:
        # gated_lstm_output : [num_samples x (num_historical_steps + num_future_steps) x state_size]
        gated_lstm_output = self.apply_sequential_processing(selected_historical=selected_historical,
                                                             selected_future=selected_future,
                                                             c_seq_hidden=c_seq_hidden,
                                                             c_seq_cell=c_seq_cell)

        # =========== Static enrichment ==============
        # Dimensions:
        # enriched_sequence: [num_samples x (num_historical_steps + num_future_steps) x state_size]
        enriched_sequence = self.apply_static_enrichment(gated_lstm_output=gated_lstm_output,
                                                         static_enrichment_signal=c_enrichment)


        # =========== self-attention ==============
        # Dimensions:
        # attention_scores: [num_samples x output_sequence_length x (num_historical_steps + num_future_steps)]
        # gated_post_attention: [num_samples x output_sequence_length x state_size]
        gated_post_attention, attention_scores = self.apply_self_attention(enriched_sequence=enriched_sequence,
                                                                           num_historical_steps=self._in_chunk_len,
                                                                           num_future_steps=self._out_chunk_len)

        # =========== position-wise feed-forward ==============
        # Applying an additional non-linear processing to the outputs of the self-attention layer using a GRN,
        # where its weights are shared across the entire layer
        post_poswise_ff_grn = self._pos_wise_ff_grn(gated_post_attention)
        # Also applying a gated residual connection skipping over the
        # attention block (using sequential processing output), providing a direct path to the sequence-to-sequence
        # layer, yielding a simpler model if additional complexity is not required
        # Dimensions:
        # gated_poswise_ff: [num_samples x output_sequence_length x state_size]
        gated_poswise_ff = self._pos_wise_ff_gating(
            post_poswise_ff_grn,
            residual=gated_lstm_output[:, self._in_chunk_len:, :])

        # =========== output projection ==============
        # Each predicted quantile has its own projection weights (all gathered in a single linear layer)
        # Dimensions:
        # predicted_quantiles: [num_samples x num_future_steps x num_quantiles]
        predicted_quantiles = self._output_layer(gated_poswise_ff).reshape([paddle.shape(gated_poswise_ff)[0], \
                                                                           self._out_chunk_len,\
                                                                           self._target_dim, -1])
        
        if not self._interpretable_output:
            return predicted_quantiles
        # below for specific interpretable results of TFT.
        else:
            return {
                # [num_samples x num_static_inputs]
                'static_weights': static_weights.squeeze(-1) if static_weights is not None else None,  
                # [num_samples x num_historical_steps x total_num_historical_inputs]
                'historical_selection_weights': historical_selection_weights,
                # [num_samples x num_future_steps x total_num_future_inputs]
                'future_selection_weights': future_selection_weights,
                # [num_samples x output_sequence_length x (num_historical_steps + num_future_steps)]
                'attention_scores': attention_scores
            }

