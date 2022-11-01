#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from typing import List, Dict, Any, Callable, Optional 

import numpy as np
import paddle
import tqdm

from paddle.optimizer import Optimizer
import paddle.nn.functional as F

from paddlets.models.representation.dl._ts2vec.utils import (
    create_ts2vec_inputs,
    create_contrastive_inputs,
    instance_level_encoding,
    multiscale_encoding,
    custom_collate_fn,
)
from paddlets.models.representation.dl._ts2vec.losses import hierarchical_contrastive_loss
from paddlets.models.representation.dl._ts2vec.encoder import TSEncoder
from paddlets.models.representation.dl._ts2vec.swa import AveragedModel
from paddlets.models.representation.dl.repr_base import ReprBaseModel
from paddlets.models.representation.dl.adapter import ReprDataAdapter
from paddlets.models.common.callbacks import Callback
from paddlets.datasets import TSDataset
from paddlets.logger import raise_if_not


class _TS2VecModule(paddle.nn.Layer):
    """Paddle layer implementing TS2Vec.

    Args:
        in_channels(int): The number of channels in the input series.
        out_channels(int): The number of channels in the output series.
        hidden_channels(int): The number of channels in the hidden layer.
        num_layers(int): The number of `ConvLayer` to be stacked.
    
    Attributes:
        _feat_extractor(paddle.nn.Layer): A stacked LayerList containing `DilatedConvLayer`.
        _avg_extractor(paddle.nn.Layer): An averaged model of `_feat_extractor` for Stochastic Weight Averaging (SWA).
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        num_layers: int,
    ):
        super(_TS2VecModule, self).__init__()
        self._feat_extractor = TSEncoder(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers
        )
        self._avg_extractor = AveragedModel(self._feat_extractor)
        self._avg_extractor.update_parameters(self._feat_extractor)

    def forward(
        self, 
        X: paddle.Tensor,
        mask: Optional[str],
    ) -> paddle.Tensor:
        """Forward.

        Args:
            X(paddle.Tensor): The input of TS2Vec's feature extractor.
            mask(str): The mask type, ["binomial", "all_true", "mask_last"] is optional.

        Returns:
            paddle.Tensor: Out of model.
        """
        if self.training:
            return self._feat_extractor(X, mask)
        return self._avg_extractor(X, mask)
    
    def parameters(self):
        """Returns a list of all Parameters from current layer and its sub-layers.
        """
        return self._feat_extractor.parameters()


class TS2Vec(ReprBaseModel):
    """TS2Vec\[1\] is a time series representation model introduced in 2021, 
    It is a universal framework for learning representations of time series in an arbitrary semantic level. 
    TS2Vec performs contrastive learning in a hierarchical way over augmented context views, which enables 
    a robust contextual representation for each timestamp. 

    \[1\] Yue Z, et al. "TS2Vec: Towards universal representation of time series", `<https://arxiv.org/abs/2106.10466>`_

    Args:
        segment_size(int): The size of time series segment.
        sampling_stride(int): Sampling intervals between two adjacent samples.
        optimizer_fn(Callable[..., Optimizer]): Optimizer algorithm.
        optimizer_params(Dict[str, Any]): Optimizer parameters.
        callbacks(List[Callback]): Customized callback functions.
        batch_size(int): Number of samples per batch.
        max_epochs(int): Max epochs during training.
        verbose(int): Verbosity mode.
        seed(int|None): Global random seed.

        repr_dims(int): The dimension of representation.
        hidden_dims(int): The number of channels in the hidden layer.
        num_layers(int): The number of `ConvLayer` to be stacked.
        temporal_unit(int): The minimum unit to perform temporal contrast.

    Attributes:
        _segment_size(int): The size of time series segment.
        _sampling_stride(int): Sampling intervals between two adjacent samples.
        _optimizer_fn(Callable[..., Optimizer]): Optimizer algorithm.
        _optimizer_params(Dict[str, Any]): Optimizer parameters.
        _callbacks(List[Callback]): Customized callback functions.
        _batch_size(int): Number of samples per batch.
        _max_epochs(int): Max epochs during training.
        _verbose(int): Verbosity mode.
        _seed(int|None): Global random seed.
        _repr_dims(int): The dimension of representation.
        _hidden_dims(int): The number of channels in the hidden layer.
        _num_layers(int): The number of `ConvLayer` to be stacked.
        _temporal_unit(int): The minimum unit to perform temporal contrast.
    """
    def __init__(
        self,
        segment_size: int,
        sampling_stride: int = 1,
        optimizer_fn: Callable[..., Optimizer] = paddle.optimizer.AdamW,
        optimizer_params: Dict[str, Any] = dict(learning_rate=1e-3),
        callbacks: List[Callback] = [],
        batch_size: int = 128,
        max_epochs: int = 10,
        verbose: int = 1,
        seed: Optional[int] = None,
        
        repr_dims: int = 320,
        hidden_dims: int = 64,
        num_layers: int = 10,
        temporal_unit: int = 0,
    ):
        self._repr_dims = repr_dims
        self._hidden_dims = hidden_dims
        self._num_layers = num_layers
        self._temporal_unit = temporal_unit
        super(TS2Vec, self).__init__(
            segment_size=segment_size,
            sampling_stride=sampling_stride,
            optimizer_fn=optimizer_fn,
            optimizer_params=optimizer_params,
            callbacks=callbacks,
            batch_size=batch_size,
            max_epochs=max_epochs,
            verbose=verbose,
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
                    f"TS2Vec's target dtype only supports [float16, float32, float64], " \
                    f"but received {column}: {dtype}."
                )
                continue
            raise_if_not(
                np.issubdtype(dtype, np.floating),
                f"TS2Vec's cov(observed or known) dtype currently only supports [float16, float32, float64], " \
                f"but received {column}: {dtype}."
            )
        super(TS2Vec, self)._check_tsdataset(tsdataset)

    def _update_fit_params(
        self,
        train_tsdataset: TSDataset,
    ) -> Dict[str, Any]:
        """Infer parameters by TSdataset automatically.

        Args:
            train_tsdataset(TSDataset): train dataset.

        Returns:
            Dict[str, Any]: model parameters.
        """
        input_dim = train_tsdataset.get_target().data.shape[1]
        if train_tsdataset.get_observed_cov():
            input_dim += train_tsdataset.get_observed_cov().data.shape[1]
        if train_tsdataset.get_known_cov():
            input_dim += train_tsdataset.get_known_cov().data.shape[1]
        fit_params = {
            "input_dim": input_dim
        }
        return fit_params

    def _init_fit_dataloader(
        self,
        train_tsdataset: TSDataset
    ) -> paddle.io.DataLoader:
        """Generate dataloader for train set.

        Args:
            train_tsdataset(TSDataset): Train set.

        Returns:
            paddle.io.DataLoader: Training dataloader.
        """
        self._check_tsdataset(train_tsdataset)
        data_adapter = ReprDataAdapter()
        train_dataset = data_adapter.to_paddle_dataset(
            train_tsdataset,
            segment_size=self._segment_size,
            sampling_stride=self._sampling_stride,
        )
        # In order to align with the paper of TS2Vec, a customized data organization is required.
        train_dataset._samples = custom_collate_fn(train_dataset._samples)
        return data_adapter.to_paddle_dataloader(train_dataset, self._batch_size)
    
    def _init_network(self) -> paddle.nn.Layer:
        """Setup the network.

        Returns:
            paddle.nn.Layer
        """
        return _TS2VecModule(
            in_channels=self._fit_params["input_dim"],
            out_channels=self._repr_dims,
            hidden_channels=self._hidden_dims,
            num_layers=self._num_layers,
        )

    def _train_batch(
        self,
        X: Dict[str, paddle.Tensor],
    ) -> Dict[str, Any]:
        """Trains one batch of data.

        Args:
            X(Dict[str, paddle.Tensor]): Dict of feature tensor.

        Returns:
            Dict[str, Any]: Dict of logs.
        """
        batch_logs = super(TS2Vec, self)._train_batch(X)
        self._network._avg_extractor.update_parameters(
            self._network._feat_extractor
        )
        return batch_logs

    def _compute_loss(
        self,
        X: Dict[str, paddle.Tensor]
    ) -> paddle.Tensor:
        """Compute the loss.

        Args:
            X(Dict[str, paddle.Tensor]): Dict of feature tensor.

        Returns:
            paddle.Tensor: Loss value.
        """
        feats = create_ts2vec_inputs(X)
        if feats.shape[1] > self._segment_size:
            offset = np.random.randint(feats.shape[1] - self._segment_size + 1)
            feats = feats[:, offset: offset + self._segment_size]
        aug1, aug2, overlap_len = create_contrastive_inputs(feats, self._temporal_unit)
        repr1 = self._network(aug1, mask="binomial")[:, -overlap_len:, :]
        repr2 = self._network(aug2, mask="binomial")[:, :overlap_len, :]
        loss = hierarchical_contrastive_loss(repr1, repr2)
        return loss

    def _encode(
        self,
        dataloader: paddle.io.DataLoader,
        mask: Optional[str] = "all_true",
        encoding_type: Optional[str] = None,
        sliding_len: Optional[int] = None,
        batch_size: Optional[int] = None,
        verbose: bool = True
    ) -> np.ndarray:
        """Encode function core logic.

        Args:
            dataloader(paddle.io.DataLoader): The data to be encoded.
            mask(str): The mask used by encoder can be specified with this parameter. 
                ["all_true", "mask_last"] is optional.
            encoding_type(str): When this parameter is specified, the computed representation would be 
                max pooling over all input series. ["full_series", "multiscale"] is optional.
            sliding_len(int): The contextual series length used for inference.
            batch_size(int): The batch size used for inference. If not specified, this would be the same batch size as training.
            verbose(bool): Turn on Verbose mode,set to true by default.

        Returns:
            np.ndarray: The representations for input series.
        """
        def _encode_one_batch(buffer):
            """Encode one batch of data.
            """
            padding_tensor = paddle.concat(buffer, axis=0)
            out = self._network(padding_tensor, mask)
            if encoding_type == "multiscale":
                out = multiscale_encoding(out)
            out = out[:, -1:, :].cpu()
            out = paddle.transpose(out, perm=[1, 0, 2])
            buffer.clear()
            return out

        raise_if_not(
            mask in ("all_true", "mask_last"),
            f"mask must be either `all_true` or `mask_last`"
        )
        if encoding_type is not None:
            raise_if_not(
                encoding_type in ("full_series", "multiscale"),
                f"encoding_type must be either `full_series` or `multiscale`"
            )

        batch_size = (
            self._batch_size if batch_size is None else batch_size
        )
        data = iter(dataloader).next()
        feats = create_ts2vec_inputs(data)
        self._network.eval()
        # Sliding inference(casual)
        # The timestamp t's representation is computed
        #   using the observations located in [t - sliding_len, t].
        if sliding_len is not None:
            seq_len, buffer, reprs = feats.shape[1], [], []
            for timestamp_idx in tqdm.tqdm(range(seq_len), disable=not verbose):
                start = timestamp_idx - sliding_len
                end = timestamp_idx + 1
                padding_left = (-start if start < 0 else 0)
                padding_right = (end - seq_len if end > seq_len else 0)
                padding_tensor = F.pad(
                    feats[:, max(start, 0): min(end, seq_len), :],
                    (padding_left, padding_right),
                    value=np.nan,
                    data_format="NLC"
                )
                # Accumulate a batch and encode together.
                buffer.append(padding_tensor)
                if len(buffer) < batch_size:
                    continue
                out = _encode_one_batch(buffer)
                reprs.append(out)

            if len(buffer) > 0:
                out = _encode_one_batch(buffer)
                reprs.append(out)

            out = paddle.concat(reprs, axis=1)
            if encoding_type == "full_series":
                out = instance_level_encoding(out)
                out = paddle.squeeze(out, axis=1)
            return out.numpy()

        # instance level encoding.
        if encoding_type == "full_series":
            out = self._network(feats, mask)
            out = instance_level_encoding(out)
            out = paddle.squeeze(out, axis=1)
            return out.numpy()

        # raw output.
        out = self._network(feats, mask)
        return out.numpy()
