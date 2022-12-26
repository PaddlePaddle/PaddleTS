#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
Implements modules of TFT model.
"""

from typing import List, Optional, Tuple
import math

import numpy as np
import paddle
from paddle import nn
import paddle.nn.functional as F

from paddlets.models.forecasting.dl._tft import GatedResidualNetwork
from paddlets.models.forecasting.dl._tft import TimeDistributed
from paddlets.models.forecasting.dl._tft import NullTransform


class VariableSelectionNetwork(nn.Layer):
    """
    This module is designed to handle the fact that the relevant and specific contribution of each input variable
    to the  output is typically unknown. This module enables instance-wise variable selection, and is applied to
    both the static covariates and time-dependent covariates.

    Beyond providing insights into which variables are the most significant oones for the prediction problem,
    variable selection also allows the model to remove any unnecessary noisy inputs which could negatively impact
    performance.

    Args:
        input_dim(int): The attribute/embedding dimension of the input, associated with the ``state_size`` of th model.
        num_inputs(int): The quantity of input variables, including both numeric and categorical inputs for the relevant channel.
        hidden_dim(int): The embedding width of the output.
        dropout(float): The dropout rate associated with `GatedResidualNetwork` objects composing this object.
        context_dim(int, Optional): The embedding width of the context signal expected to be fed as an auxiliary input to this component.
        batch_first(bool, Optional): A boolean indicating whether the batch dimension is expected to be the first dimension of the input or not.
    """
    def __init__(
        self, 
        input_dim: int, 
        num_inputs: int, 
        hidden_dim: int, 
        dropout: float,
        context_dim: Optional[int] = None,
        batch_first: Optional[bool] = True
    ):
        super(VariableSelectionNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.num_inputs = num_inputs
        self.dropout = dropout
        self.context_dim = context_dim

        # A GRN to apply on the flat concatenation of the input representation (all inputs together),
        # possibly provided with context information
        self.flattened_grn = GatedResidualNetwork(input_dim=self.num_inputs * self.input_dim,
                                                  hidden_dim=self.hidden_dim,
                                                  output_dim=self.num_inputs,
                                                  dropout=self.dropout,
                                                  context_dim=self.context_dim,
                                                  batch_first=batch_first)
        # activation for transforming the GRN output to weights
        self.softmax = nn.Softmax(axis=1)

        # In addition, each input variable (after transformed to its wide representation) goes through its own GRN.
        self.single_variable_grns = nn.LayerList()
        for _ in range(self.num_inputs):
            self.single_variable_grns.append(
                GatedResidualNetwork(input_dim=self.input_dim,
                                     hidden_dim=self.hidden_dim,
                                     output_dim=self.hidden_dim,
                                     dropout=self.dropout,
                                     batch_first=batch_first))

    def forward(
        self,
        flattened_embedding: paddle.Tensor,
        context: Optional[paddle.Tensor]=None
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """
        Infer variable selection weights using the flattened representation GRN.
        
        Args:
            flattened_embedding(paddle.Tensor): The input tensor, flattened representation for variable selection.
            context(paddle.Tensor, Optional): The context tensor.
            
        Returns:
            outputs(paddle.Tensor): The output tensor of `VariableSelectionNetwork`.
            sparse_weights(paddle.Tensor): The weights tensor for interpretable use.
        """
        # the flattened embedding should be of shape [(num_samples * num_temporal_steps) x (num_inputs x input_dim)]
        # where in our case input_dim represents the model_dim or the state_size.
        # in the case of static variables selection, num_temporal_steps is disregarded and can be thought of as 1.
        sparse_weights = self.flattened_grn(flattened_embedding, context)
        sparse_weights = self.softmax(sparse_weights).unsqueeze(2)
        # After that step "sparse_weights" is of shape [(num_samples * num_temporal_steps) x num_inputs x 1]

        # Before weighting the variables - apply a GRN on each transformed input
        processed_inputs = []
        for i in range(self.num_inputs):
            # select slice of embedding belonging to a single input - and apply the variable-specific GRN
            # (the slice is taken from the flattened concatenated embedding)
            processed_inputs.append(
                self.single_variable_grns[i](flattened_embedding[..., (i * self.input_dim): (i + 1) * self.input_dim]))
        # each element in the resulting list is of size: [(num_samples * num_temporal_steps) x state_size],
        # and each element corresponds to a single input variable

        # combine the outputs of the single-var GRNs (along an additional axis)
        # Dimensions:
        # processed_inputs: [(num_samples * num_temporal_steps) x state_size x num_inputs]
        processed_inputs = paddle.stack(processed_inputs, axis=-1)

        # weigh them by multiplying with the weights tensor viewed as
        # [(num_samples * num_temporal_steps) x 1 x num_inputs]
        # so that the weight given to each input variable (for each time-step/observation) multiplies the entire state
        # vector representing the specific input variable on this specific time-step
        # Dimensions:
        # outputs: [(num_samples * num_temporal_steps) x state_size x num_inputs]
        outputs = processed_inputs * sparse_weights.transpose([0, 2, 1])

        # and finally sum up - for creating a weighted sum representation of width state_size for every time-step
        # Dimensions:
        # outputs: [(num_samples * num_temporal_steps) x state_size]
        outputs = outputs.sum(axis=-1)
        return outputs, sparse_weights


class InputChannelEmbedding(nn.Layer):
    """
    A module to handle the transformation/embedding of an input channel composed of numeric tensors and categorical
    tensors.
    It holds a NumericInputTransformation module for handling the embedding of the numeric inputs,
    and a CategoricalInputTransformation module for handling the embedding of the categorical inputs.

    Args:
        state_size(int): The state size of the model, which determines the embedding dimension/width of each input variable.
        num_numeric(int): The quantity of numeric input variables associated with the input channel.
        num_categorical(int): The quantity of categorical input variables associated with the input channel.
        categorical_cardinalities(List[int]): The quantity of categories associated with each of the categorical input variables.
        time_distribute(bool, Optional) A boolean indicating whether to wrap the composing transformations using the ``TimeDistributed`` module.
    """
    def __init__(
        self, 
        state_size: int, 
        num_numeric: int, 
        num_categorical: int, 
        categorical_cardinalities: List[int],
        time_distribute: Optional[bool] = False
    ):
        super(InputChannelEmbedding, self).__init__()
        self.state_size = state_size
        self.num_numeric = num_numeric
        self.num_categorical = num_categorical
        self.categorical_cardinalities = categorical_cardinalities
        self.time_distribute = time_distribute
        if num_numeric == 0:
            self.numeric_transform = NullTransform()
        elif self.time_distribute:
            self.numeric_transform = TimeDistributed(
                NumericInputTransformation(num_inputs=num_numeric, state_size=state_size), return_reshaped=False)
        else:
            self.numeric_transform = NumericInputTransformation(num_inputs=num_numeric, state_size=state_size)
            
        if num_categorical == 0:
            self.categorical_transform = NullTransform()
        elif self.time_distribute:
            self.categorical_transform = TimeDistributed(
                CategoricalInputTransformation(num_inputs=num_categorical, state_size=state_size,
                                               cardinalities=categorical_cardinalities), return_reshaped=False)
        else:
            self.categorical_transform = CategoricalInputTransformation(num_inputs=num_categorical,
                                                                        state_size=state_size,
                                                                        cardinalities=categorical_cardinalities)
    def forward(
        self, 
        x_numeric: paddle.Tensor, 
        x_categorical: paddle.Tensor
    ) -> paddle.Tensor:
        """
        The forward computation of the layer, for numeric input, it uses `nn.Linear` as the projected layer, 
        and for categorical input, it uses `nn.Embedding` as the embedding layer.
        
        Args:
            x_numeric(paddle.Tensor): The numeric input tensor.
            x_categorical(paddle.Tensor): The categorical input tensor.
            
        Returns:
            merged_transformations(paddle.Tensor): The result tensor which merged numeric and categorical embedding.
        """
        batch_shape = paddle.shape(x_numeric) if sum(paddle.shape(x_numeric)) > 0 else paddle.shape(x_categorical)
        processed_numeric = self.numeric_transform(x_numeric)
        processed_categorical = self.categorical_transform(x_categorical)
        # Both of the returned values, "processed_numeric" and "processed_categorical" are lists,
        # with "num_numeric" elements and "num_categorical" respectively - each element in these lists corresponds
        # to a single input variable, and is represent by its embedding, shaped as:
        # [(num_samples * num_temporal_steps) x state_size]
        # (for the static input channel, num_temporal_steps is irrelevant and can be treated as 1
        if not processed_numeric + processed_categorical:
            return None
        # the resulting embeddings for all the input varaibles are concatenated to a flattened representation
        # Dimensions:
        # merged_transformations: [(num_samples * num_temporal_steps) x (state_size * total_input_variables)]
        # total_input_variables stands for the amount of all input variables in the specific input channel, i.e
        # num_numeric + num_categorical
        merged_transformations = paddle.concat(processed_numeric + processed_categorical, axis=1)

        # for temporal data we return the resulting tensor to its 3-dimensional shape
        if self.time_distribute:
            merged_transformations = merged_transformations.reshape([batch_shape[0], batch_shape[1], -1])
            # In that case:
            # merged_transformations: [num_samples x num_temporal_steps x (state_size * total_input_variables)]
        return merged_transformations


class NumericInputTransformation(nn.Layer):
    """
    A module for transforming/embeddings the set of numeric input variables from a single input channel.
    Each input variable will be projected using a dedicated linear layer to a vector with width state_size.
    The result of applying this module is a list, with length num_inputs, that contains the embedding of each input
    variable for all the observations and time steps.

    Args:
        num_inputs(int): The quantity of numeric input variables associated with this module.
        state_size(int): The state size of the model, which determines the embedding dimension/width.
    """

    def __init__(
        self, 
        num_inputs: int, 
        state_size: int
    ):
        super(NumericInputTransformation, self).__init__()
        self.num_inputs = num_inputs
        self.state_size = state_size

        self.numeric_projection_layers = nn.LayerList()
        for _ in range(self.num_inputs):
            self.numeric_projection_layers.append(nn.Linear(1, self.state_size))

    def forward(
        self, 
        x: paddle.Tensor
    ) -> List[paddle.Tensor]:
        """
        The forward computation of `InputChannelEmbedding`, every input variable is projected using its dedicated linear layer,
        the results are stored as a list.
        
        Args:
            x(paddle.Tensor): The tensor of numeric input.
            
        Returns:
            projections(List[paddle.Tensor]): The result list composed of projected tensor of each input.
        """
        projections = []
        for i in range(self.num_inputs):
            projections.append(self.numeric_projection_layers[i](x[:, i].unsqueeze(1)))

        return projections


class CategoricalInputTransformation(nn.Layer):
    """
    A module for transforming/embeddings the set of categorical input variables from a single input channel.
    Each input variable will be projected using a dedicated embedding layer to a vector with width state_size.
    The result of applying this module is a list, with length num_inputs, that contains the embedding of each input
    variable for all the observations and time steps.

    Args:
        num_inputs(int): The quantity of categorical input variables associated with this module.
        state_size(int): The state size of the model, which determines the embedding dimension/width.
        cardinalities(List[int]): The quantity of categories associated with each of the input variables.
    """

    def __init__(
        self, 
        num_inputs: int, 
        state_size: int, 
        cardinalities: List[int]
    ):
        super(CategoricalInputTransformation, self).__init__()
        self.num_inputs = num_inputs
        self.state_size = state_size
        self.cardinalities = cardinalities

        # layers for processing the categorical inputs
        self.categorical_embedding_layers = nn.LayerList()
        for idx, cardinality in enumerate(self.cardinalities):
            self.categorical_embedding_layers.append(nn.Embedding(cardinality, self.state_size))

    def forward(
        self, 
        x: paddle.Tensor
    )-> List[paddle.Tensor]:
        """
        The forward computation of `CategoricalInputTransformation`, every input variable is projected using its embedding layer,
        the results are stored as a list.
        
        Args:
            x(paddle.Tensor): The tensor of categorical input.
            
        Returns:
            embeddings(List[paddle.Tensor]): The result list composed of embedding tensor of each input.        
        """
        embeddings = []
        for i in range(self.num_inputs):
            embeddings.append(self.categorical_embedding_layers[i](x[:, i]))

        return embeddings


class InterpretableMultiHeadAttention(nn.Layer):
    """
    The mechanism implemented in this module is used to learn long-term relationships across different time-steps.
    It is a modified version of multi-head attention, for enhancing explainability. On this modification,
    as opposed to traditional versions of multi-head attention, the "values" signal is shared for all the heads -
    and additive aggregation is employed across all the heads.
    According to the paper, each head can learn different temporal patterns, while attending to a common set of
    input features which can be interpreted as  a simple ensemble over attention weights into a combined matrix, which,
    compared to the original multi-head attention matrix, yields an increased representation capacity in an efficient
    way.

    Args:
        embed_dim(int): The dimensions associated with the `state_size` of th model, corresponding to the input as well as the output.
        num_heads(int): The number of attention heads composing the Multi-head attention component.
    """

    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int
    ):
        super(InterpretableMultiHeadAttention, self).__init__()
        self.d_model = embed_dim  # the state_size (model_size) corresponding to the input and output dimension
        self.num_heads = num_heads  # the number of attention heads
        self.all_heads_dim = embed_dim * num_heads  # the width of the projection for the keys and queries
        self.w_q = nn.Linear(embed_dim, self.all_heads_dim)  # multi-head projection for the queries
        self.w_k = nn.Linear(embed_dim, self.all_heads_dim)  # multi-head projection for the keys
        self.w_v = nn.Linear(embed_dim, embed_dim)  # a single, shared, projection for the values

        # the last layer is used for final linear mapping (corresponds to W_H in the paper)
        self.out = nn.Linear(self.d_model, self.d_model)

    def forward(
        self,
        q: paddle.Tensor, 
        k: paddle.Tensor,
        v: paddle.Tensor, 
        mask: Optional[paddle.Tensor]=None
    ) -> Tuple[paddle.Tensor, ...]:
        """
        The forward computation of `InterpretableMultiHeadAttention`.
        
        Args:
            q(paddle.Tensor): Queries tensor.
            k(paddle.Tensor): Keys tensor.
            v(paddle.Tensor):Values tensor.
            
        Returns:
            output(paddle.Tensor): The output tensor of the layer.
            attention_outputs(paddle.Tensor): The output of attention layer.
            attention_scores(paddle.Tensor): The scores of attention for interpretable use.
        """
        num_samples = paddle.shape(q)[0]
        # Dimensions:
        # queries tensor - q: [num_samples x num_future_steps x state_size]
        # keys tensor - k: [num_samples x (num_total_steps) x state_size]
        # values tensor - v: [num_samples x (num_total_steps) x state_size]

        # perform linear operation and split into h heads
        q_proj = self.w_q(q).reshape([num_samples, -1, self.num_heads, self.d_model])
        k_proj = self.w_k(k).reshape([num_samples, -1, self.num_heads, self.d_model])
        v_proj = paddle.tile(self.w_v(v), [1, 1, self.num_heads]).reshape([num_samples, -1, self.num_heads, self.d_model])

        # transpose to get the following shapes
        q_proj = q_proj.transpose([0, 2, 1, 3])  # (num_samples x num_future_steps x num_heads x state_size)
        k_proj = k_proj.transpose([0, 2, 1, 3])  # (num_samples x num_total_steps x num_heads x state_size)
        v_proj = v_proj.transpose([0, 2, 1, 3])  # (num_samples x num_total_steps x num_heads x state_size)

        # calculate attention using function we will define next
        attn_outputs_all_heads, attn_scores_all_heads = self.attention(q_proj, k_proj, v_proj, mask)
        # Dimensions:
        # attn_scores_all_heads: [num_samples x num_heads x num_future_steps x num_total_steps]
        # attn_outputs_all_heads: [num_samples x num_heads x num_future_steps x state_size]

        # take average along heads
        attention_scores = attn_scores_all_heads.mean(axis=1)
        attention_outputs = attn_outputs_all_heads.mean(axis=1)
        # Dimensions:
        # attention_scores: [num_samples x num_future_steps x num_total_steps]
        # attention_outputs: [num_samples x num_future_steps x state_size]

        # weigh attention outputs
        output = self.out(attention_outputs)
        # output: [num_samples x num_future_steps x state_size]

        return output, attention_outputs, attention_scores

    def masked_fill(
        self,
        x: paddle.Tensor, 
        mask: paddle.Tensor, 
        value: float,
    ) -> paddle.Tensor:
        """
        Mask filling operation.
        
        Args:
            x(paddle.Tensor): The input tensor to be masked.
            mask(paddle.Tensor): The mask tensor.
            value(float): A very small value.
        
        Returns:
            paddle.Tensor: The result of masked tensor.
        """
        y = paddle.full(paddle.shape(x), value, x.dtype)
        return paddle.where(mask, y, x)
        
    def attention(
        self, 
        q: paddle.Tensor, 
        k: paddle.Tensor,
        v: paddle.Tensor, 
        mask: Optional[paddle.Tensor]=None
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """
        The computation of self-attention operation.
        
        Args:
            q(paddle.Tensor): Queries tensor.
            k(paddle.Tensor): Keys tensor.
            v(paddle.Tensor):Values tensor.
            
        Returns:
            attention_outputs(paddle.Tensor): The output of attention layer.
            attention_scores(paddle.Tensor): The scores of attention for interpretable use.
        """
        # Applying the scaled dot product
        # Dimensions:
        # attention_scores: [num_samples x num_heads x num_future_steps x num_total_steps]
        attention_scores = paddle.matmul(q, k.transpose([0,1,3,2])) / math.sqrt(self.d_model)

        # Decoder masking is applied to the multi-head attention layer to ensure that each temporal dimension can only
        # attend to features preceding it
        if mask is not None:
            # the mask is broadcast along the batch(dim=0) and heads(dim=1) dimensions,
            # where the mask==True, the scores are "cancelled" by setting a very small value
            attention_scores = self.masked_fill(attention_scores, mask, -1e9)

        # still part of the scaled dot-product attention (dimensions are kept)
        attention_scores = F.softmax(attention_scores, axis=-1)
        # matrix multiplication is performed on the last two-dimensions to retrieve attention outputs
        # Dimensions:
        # attention_outputs: [num_samples x num_heads x num_future_steps x state_size]
        attention_outputs = paddle.matmul(attention_scores, v)
        return attention_outputs, attention_scores
    
