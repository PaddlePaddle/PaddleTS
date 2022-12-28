#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from typing import List, Dict, Any, Callable, Optional 

import numpy as np
import paddle
import tqdm

from paddle.optimizer import Optimizer
import paddle.nn.functional as F

from paddlets.models.representation.dl._cost.utils import (
    create_cost_inputs,
    create_contrastive_inputs,
    custom_collate_fn,
)
from paddlets.models.representation.dl._cost.losses import (
    time_contrastive_loss, 
    frequency_contrastive_loss,
    convert_coefficient
)
from paddlets.models.representation.dl._cost.encoder import TSEncoder
from paddlets.models.representation.dl._cost.swa import AveragedModel
from paddlets.models.representation.dl.repr_base import ReprBaseModel
from paddlets.models.data_adapter import DataAdapter
from paddlets.models.common.callbacks import Callback
from paddlets.datasets import TSDataset
from paddlets.logger import raise_if_not


class _CoSTModule(paddle.nn.Layer):
    """Paddle layer implementing CoST.

    Args:
        in_channels(int): The number of channels in the input series.
        out_channels(int): The number of channels in the output series.
        hidden_channels(int): The number of channels in the hidden layer.
        num_layers(int): The number of `ConvLayer` to be stacked.
        segment_size(int): The size of time series segment.
        queue_size(int): The dynamic queue size for saving negative examples.
        temperature(float): The temperature coefficient.
        alpha(float): The weight of seasonal component in the loss.
    
    Attributes:
        _feat_extractor(paddle.nn.Layer): A stacked LayerList containing `DilatedConvLayer`.
        _avg_extractor(paddle.nn.Layer): An averaged model of `_feat_extractor` for Stochastic Weight Averaging (SWA).
        _out_proj(paddle.nn.Layer): A projection head, widely used for contrastive learning.
        _avg_head(paddle.nn.Layer): An averaged model of `_out_proj` for Stochastic Weight Averaging (SWA).
        _queue_size(int): The dynamic queue size for saving negative examples.
        _temperature(float): The temperature coefficient.
        _alpha(float): The parameter control the weightage of seasonal components in loss.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        num_layers: int,
        segment_size: int,
        queue_size: int,
        temperature: float,
        alpha: float

    ):
        super(_CoSTModule, self).__init__()
        self._feat_extractor = TSEncoder(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            seq_len=segment_size
        ) # feature extractor

        k = np.sqrt(1. / (out_channels // 2))
        weight_attr = bias_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Uniform(-k, k)
        )
        dim = (out_channels // 2)
        self._out_proj = paddle.nn.Sequential(
            paddle.nn.Linear(dim, dim, weight_attr, bias_attr),
            paddle.nn.ReLU(),
            paddle.nn.Linear(dim, dim, weight_attr, bias_attr),
        ) # projection head

        self._avg_extractor = AveragedModel(self._feat_extractor)
        self._avg_extractor.update_parameters(self._feat_extractor)
        self._avg_proj = AveragedModel(self._out_proj)
        self._avg_proj.update_parameters(self._out_proj)
         
        self._alpha = alpha
        self._queue_size = queue_size
        self._temperature = temperature
        queue = paddle.randn((out_channels // 2, queue_size))
        self.register_buffer("_queue", F.normalize(queue, axis=0))
        self.register_buffer("_ptr", paddle.zeros([1], dtype="int64"))
    
    def _dequeue_and_enqueue(
        self, 
        keys: paddle.Tensor
    ):
        """With reference to MoCO implementation, maintian a dynamic queue for saving negtive examples.

        Args:
            keys: The negative example of waiting to enter the queue.
        """
        ptr, offset = self._ptr[0], keys.shape[0]
        self._queue[:, ptr: ptr + offset] = keys.T
        self._ptr[0] = (ptr + offset) % self._queue_size

    def forward(
        self, 
        X: paddle.Tensor,
        mask: Optional[str],
    ) -> paddle.Tensor:
        """Forward.

        Args:
            X(paddle.Tensor): The input of CoST's feature extractor.
            mask(str): The mask type, ["binomial", "all_true"] is optional.

        Returns:
            paddle.Tensor: Out of model or loss value.
        """
        if not self.training:
            return self._feat_extractor(X, mask)
        
        # Generate two augmented samples.
        aug1 = create_contrastive_inputs(X)
        aug2 = create_contrastive_inputs(X)
        trend1, season1 = self._feat_extractor(aug1, mask)
        trend2, _ = self._avg_extractor(aug2, mask)
        _, season2 = self._feat_extractor(aug2, mask)

        # Trend: contrastive learning(reference MoCO) in time domain. 
        # In order to calculate efficiency, a certain time step is randomly sampled.
        batch_size, seq_len, channels = trend1.shape
        axis0 = np.arange(batch_size)[:, None, None]
        axis1 = np.random.randint(0, seq_len, (1, ))[None, :, None]
        axis2 = np.arange(channels)[None, None, :]

        trend1 = paddle.squeeze(trend1[axis0, axis1, axis2], 1)
        trend1 = self._out_proj(trend1)
        trend1 = F.normalize(trend1, axis=-1)
        season1 = F.normalize(season1, axis=-1)

        trend2 = paddle.squeeze(trend2[axis0, axis1, axis2], 1)
        trend2 = self._avg_proj(trend2)
        trend2 = F.normalize(trend2, axis=-1)
        season2 = F.normalize(season2, axis=-1)
        
        # Season: contrastive learning in frequency domain.
        season1_freq = paddle.fft.rfft(season1, axis=1)
        season2_freq = paddle.fft.rfft(season2, axis=1)
        season1_amp, season1_phase = convert_coefficient(season1_freq)
        season2_amp, season2_phase = convert_coefficient(season2_freq)

        trend_loss = time_contrastive_loss(
            trend1, trend2, paddle.assign(self._queue.detach()), self._temperature
        )
        season_loss = frequency_contrastive_loss(season1_amp, season2_amp)
        season_loss += frequency_contrastive_loss(season1_phase, season2_phase) 
        self._dequeue_and_enqueue(trend2)
        return trend_loss + (self._alpha * season_loss / 2.)


    def parameters(self):
        """Returns a list of all Parameters from current layer and its sub-layers.
        """
        return self._feat_extractor.parameters() + self._out_proj.parameters()


class CoST(ReprBaseModel):
    """CoST\[1\] is a time series representation model published in ICLR 2022, 
    It is a new time series representation learning framework for long sequence time series forecasting, 
    which applies the contrastive learning method to learn disentangled seasonal-trend representations. 
    CoST comprises both time domain and frequency domain contrastive losses to learn 
    discriminative trend and seasonal representations, respectively.

    \[1\] Woo G, et al. "CoST: Contrastive Learning of Disentangled Seasonal-Trend Representations 
    for Time Series Forecasting", `<https://arxiv.org/pdf/2202.01575.pdf>`_

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
        queue_size(int): The dynamic queue size for saving negative examples.
        temperature(float): The temperature coefficient.
        alpha(float): The weight of seasonal components in loss.

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
        _queue_size(int): The dynamic queue size for saving negative examples.
        _temperature(float): The temperature coefficient.
        _alpha(float): The parameter control the weightage of seasonal components in loss.
    """
    def __init__(
        self,
        segment_size: int,
        sampling_stride: int = 1,
        optimizer_fn: Callable[..., Optimizer] = paddle.optimizer.Momentum,
        optimizer_params: Dict[str, Any] = dict(learning_rate=1e-3),
        callbacks: List[Callback] = [],
        batch_size: int = 128,
        max_epochs: int = 10,
        verbose: int = 1,
        seed: Optional[int] = None,
        
        repr_dims: int = 320,
        hidden_dims: int = 64,
        num_layers: int = 10,
        queue_size: int = 256,
        temperature: float = 0.07,
        alpha: float = 5e-4
    ):
        raise_if_not(
            queue_size % batch_size == 0,
            f"queue_size must be divisible by batch_size." \
        ) 
        self._segment_size = segment_size
        self._repr_dims = repr_dims
        self._hidden_dims = hidden_dims
        self._num_layers = num_layers
        self._queue_size = queue_size
        self._temperature = temperature
        self._alpha = alpha
        super(CoST, self).__init__(
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
                    f"CoST's target dtype only supports [float16, float32, float64], " \
                    f"but received {column}: {dtype}."
                )
                continue
            raise_if_not(
                np.issubdtype(dtype, np.floating),
                f"CoST's cov(observed or known) dtype currently only supports [float16, float32, float64], " \
                f"but received {column}: {dtype}."
            )
        super(CoST, self)._check_tsdataset(tsdataset)

    def _update_fit_params(
        self,
        train_tsdataset: List[TSDataset],
    ) -> Dict[str, Any]:
        """Infer parameters by TSdataset automatically.

        Args:
            train_tsdataset(TSDataset): train dataset.

        Returns:
            Dict[str, Any]: model parameters.
        """
        train_tsdataset = train_tsdataset[0]
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
        train_tsdataset: List[TSDataset]
    ) -> paddle.io.DataLoader:
        """Generate dataloader for train set.

        Args:
            train_tsdataset(List[TSDataset]): Train set.

        Returns:
            paddle.io.DataLoader: Training dataloader.
        """
        data_adapter, samples = DataAdapter(), []
        for tsdataset in train_tsdataset:
            self._check_tsdataset(tsdataset)
            dataset = data_adapter.to_sample_dataset(
                rawdataset=tsdataset,
                in_chunk_len=self._segment_size,
                sampling_stride=self._sampling_stride,
                fill_last_value=np.nan
            )
            samples.extend(dataset.samples)
        # In order to align with the paper of CoST, a customized data organization is required.
        samples = custom_collate_fn(samples)
        dataset.samples = samples * (
            1 if len(samples) >= self._batch_size else int(np.ceil(self._batch_size / len(samples)))
        )
        return data_adapter.to_paddle_dataloader(dataset, self._batch_size, drop_last=True)
    
    def _init_network(self) -> paddle.nn.Layer:
        """Setup the network.

        Returns:
            paddle.nn.Layer
        """
        return _CoSTModule(
            in_channels=self._fit_params["input_dim"],
            out_channels=self._repr_dims,
            hidden_channels=self._hidden_dims,
            num_layers=self._num_layers,
            segment_size=self._segment_size,
            queue_size=self._queue_size,
            temperature=self._temperature,
            alpha=self._alpha
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
        batch_logs = super(CoST, self)._train_batch(X)
        self._network._avg_extractor.update_parameters(
            self._network._feat_extractor
        )
        self._network._avg_proj.update_parameters(
            self._network._out_proj
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
        feats = create_cost_inputs(X)
        if feats.shape[1] > self._segment_size:
            offset = np.random.randint(feats.shape[1] - self._segment_size + 1)
            feats = feats[:, offset: offset + self._segment_size]
        loss = self._network(feats, mask="binomial")
        return loss

    def _encode(
        self,
        dataloader: paddle.io.DataLoader,
        batch_size: Optional[int] = None,
        verbose: bool = True
    ) -> np.ndarray:
        """Encode function core logic.

        Args:
            dataloader(paddle.io.DataLoader): The data to be encoded.
            batch_size(int): The batch size used for inference. If not specified, 
                this would be the same batch size as training.
            verbose(bool): Turn on Verbose mode,set to true by default.

        Returns:
            np.ndarray: The representations for input series.
        """
        def _encode_one_batch(buffer):
            """Encode one batch of data.
            """
            padding_tensor = paddle.concat(buffer, axis=0)
            trend, season = self._network(padding_tensor, mask="all_true")
            out = paddle.concat([trend, season], axis=-1)
            out = out[:, -1:, :].cpu()
            out = paddle.transpose(out, perm=[1, 0, 2])
            buffer.clear()
            return out

        batch_size = (
            self._batch_size if batch_size is None else batch_size
        )
        data = iter(dataloader).next()
        feats = create_cost_inputs(data)
        self._network.eval()
        seq_len, buffer, reprs = feats.shape[1], [], []
        for timestamp_idx in tqdm.tqdm(range(seq_len), disable=not verbose):
            start = timestamp_idx - (self._segment_size - 1)
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
        return out.numpy()
