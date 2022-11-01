#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
This implementation is based on the article `DeepAR: Probabilistic forecasting with autoregressive recurrent networks <https://www.sciencedirect.com/science/article/pii/S0169207019301888>`_ .

Updated features
    Recursively decoding by mean: In prediction, we provide another decoding style besides sample paths raised in the original paper, mean of predicted distribution is used for next step's input.
"""


from typing import List, Dict, Any, Callable, Optional, Tuple, Union

import numpy as np
import paddle
from paddle import nn
from paddle.optimizer import Optimizer

from paddlets.datasets import TSDataset
from paddlets.models.forecasting.dl.paddle_base_impl import PaddleBaseModelImpl
from paddlets.models.common.callbacks import Callback
from paddlets.models.forecasting.dl.distributions import Likelihood, GaussianLikelihood
from paddlets.logger import raise_if, raise_if_not, Logger


logger = Logger(__name__)


class _DeepAR(nn.Layer):
    """
    DeepAR network implementation.
    
    Args:
        in_chunk_len(int): The size of the loopback window, i.e., the number of time steps feed to the model.
        out_chunk_len(int): The size of the forecasting horizon, i.e., the number of time steps output by the model.
        target_dim(int): The numer of targets.
        known_cov_dim(int): The number of known covariates.
        rnn_type(str): The type of the specific paddle RNN module ("GRU" or "LSTM").
        hidden_dim(int): The number of features in the hidden state `h` of the RNN module.
        num_layers_recurrent(int): The number of recurrent layers.
        dropout(float): The fraction of neurons that are dropped in all-but-last RNN layers.
        likelihood_model(Likelihood): The distribution likelihood to be used for probability forecasting.
        num_samples(int): The sampling number for validation and prediction phase, it is used for computation of quantiles loss and the point forecasting result.
        regression_mode(str): The regression mode of prediction, `mean` and `sampling` are optional.
        output_mode(str): The mode of model output, `quantiles` and `predictions`(point) are optional.
    """
    def __init__(

        self,
        in_chunk_len: int,
        out_chunk_len: int,
        target_dim: int,
        known_cov_dim: int,
        rnn_type: str,
        hidden_dim: int,
        num_layers_recurrent: int,
        drop_out: float,
        likelihood_model: Likelihood,
        num_samples: int = 10,
        regression_mode: str = "mean",
        output_mode: str = "quantiles",
    ):
        super(_DeepAR, self).__init__()
        self._in_chunk_len = in_chunk_len
        self._out_chunk_len = out_chunk_len
        self._input_size = target_dim + known_cov_dim
        self._target_dim = target_dim
        self._rnn = getattr(nn, rnn_type)(self._input_size, hidden_dim, num_layers_recurrent, dropout=drop_out)
        self._likelihood_model = likelihood_model
        self._num_samples = num_samples
        self.output_projector = nn.Linear(hidden_dim, likelihood_model.num_params * target_dim)
        self._regression_mode = regression_mode
        self._output_mode = output_mode
        self.predicting = False

    def _build_input(
        self, 
        target: paddle.Tensor,
        known_cov: Union[paddle.Tensor, None],
        first_target_replace: Union[paddle.Tensor, None] = None,
    ) -> paddle.Tensor:
        """
        Create input tensor into RNN network
        
        Args:
            target(paddle.Tensor): target tensor.
            known_cov(paddle.Tensor, None): known covariate tensor.
            first_target_replace(paddle.Tensor, None): tensor to insert into first position of target. If None (default), remove first time step.
        
        Returns:
            input_tensor(paddle.Tensor): The constructed tensor for the input of rnn.
        """
        target_roll = paddle.roll(target, shifts=1, axis=1)  # [batch_size, seq_len, target_dim]
        if first_target_replace is not None:
            target_roll[:, 0] = first_target_replace
        else:
            target_roll = target_roll[:, 1:]
            if known_cov is not None:
                known_cov = known_cov[:, 1:]

        if known_cov is not None:
            return paddle.concat([target_roll, known_cov], axis=-1)
        else:
            return target_roll
        
    def _decode_direct(
        self,
        input_tensor: paddle.Tensor,
        hidden_state: paddle.Tensor
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """
        Decode directly in training phase.
        
        Args:
            input_tensor(paddle.Tensor): The input tensor of the rnn.
            hidden_state(paddle.Tensor): The initial hidden state of the rnn.

        Returns:
            output(paddle.Tensor): The output of the model, projected to the number of distrubtion parameters.
            hidden_state(paddle.Tensor): The output of hidden state.
        """
        decoder_output, hidden_state = self._rnn(input_tensor, hidden_state)
        output = self.output_projector(decoder_output) # model output, [batch_size*num_samples, seq_len, target_dim* num_args]
        output = output.reshape([output.shape[0], output.shape[1], self._target_dim, -1])
        return output, hidden_state

    def _decode_regressive_by_mean(
        self,
        input_tensor: paddle.Tensor,
        hidden_state: paddle.Tensor
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """
        Updated implementation of DeepAR.
        Decode regressively in prediction phase, use mean of current step's output as next step's input.

        Args:
            input_tensor(paddle.Tensor): The input tensor of the rnn.
            hidden_state(paddle.Tensor): The initial hidden state of the rnn.

        Returns:
            quantiles_output(paddle.Tensor): The output of `quantiles` mode.
            prediction_output(paddle.Tensor): The output of `predictions` mode.
        """
        num_steps = input_tensor.shape[1] # out_chunk_len
        quantiles_output = []
        prediction_output = []
        # make the encoder's last target as the first target input of decoder
        last_pred_target = input_tensor[:, 0, :self._target_dim]
        # regressively generate num_steps forecasting points, each of which has num_samples sampling value
        for step in range(num_steps):
            # 0> build step input by slice input_tensor
            x = input_tensor[:, step].unsqueeze(1)
            # replace x with last step's predictions
            x[:, 0, :self._target_dim] = last_pred_target
            # 1> recall self._decode_all to generate one step's model output
            # current_target_params: [num_samples*batch_size, seq_len, target_dim, num_params]
            current_target_params, hidden_state = self._decode_direct(x, hidden_state)

            # 2> sampling num_samples each step
            # for each path, sample once, shape: [100, batch_size, 1, target_dim]
            distr_params = self._likelihood_model.output_to_params(current_target_params)
            distr = self._likelihood_model.params_to_distr(distr_params)
            current_pred_sample = distr.sample([self._num_samples])
            # remove sample dim and seq_len dim, as they are both 1, shape: [batch_size, target_dim]

            # 3> make sampling prediction as current step's output and the next step's input
            quantiles_output.append(current_pred_sample.squeeze(2))
            current_target_mean = self._likelihood_model.get_mean(distr_params)
            last_pred_target = current_target_mean.squeeze(1)
            prediction_output.append(last_pred_target)

        # quantiles mode
        # to shape: [batch_size, num_steps, target_dim, num_samples], and sort the last dim to generate quantiles
        quantiles_output = paddle.stack(quantiles_output, 2)
        quantiles_output = paddle.quantile(quantiles_output, 
                                           list(np.linspace(0, 1, 101, endpoint=True)), 
                                           axis=0).transpose([1,2,3,0])        
        # predictions mode
        prediction_output = paddle.stack(prediction_output, 1)
        return quantiles_output, prediction_output

    def _decode_regressive_by_sampling(
        self,
        input_tensor: paddle.Tensor,
        hidden_state: paddle.Tensor
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """
        The original implementation of DeepAR model.
        Decode regressively in prediction phase, construct num_samples sample path, and use sampling of current step's output as next step's input.

        Args:
            input_tensor(paddle.Tensor): The input tensor of the rnn.
            hidden_state(paddle.Tensor): The initial hidden state of the rnn.

        Returns:
            quantiles_output(paddle.Tensor): The output of `quantiles` mode.
            prediction_output(paddle.Tensor): The output of `predictions` mode.
        """
        num_steps = input_tensor.shape[1] # out_chunk_len
        quantiles_output = []
        prediction_output = []
        # initialize input tensor and hidden state by repeat num_samples times to generate num_samples prediction paths
        #input_tensor: [batch_size, seq_len, target_dim] -> [batch_size * num_samples, seq_len, target_dim]
        input_tensor = input_tensor.repeat_interleave(self._num_samples, 0)       
        if isinstance(hidden_state, tuple):  #for LSTM
            h, c = hidden_state
            hidden_state = (h.repeat_interleave(self._num_samples, 1), c.repeat_interleave(self._num_samples, 1))
        else: #for GRU
            hidden_state = hidden_state.repeat_interleave(self._num_samples, 1)
        # make the encoder's last target as the first target input of decoder
        last_pred_target = input_tensor[:, 0, :self._target_dim]
        # regressively generate num_steps forecasting points, each of which has num_samples sampling value
        for step in range(num_steps):
            # 0> build step input by slice input_tensor 
            x = input_tensor[:, step].unsqueeze(1)
            # replace x with last step's predictions
            x[:, 0, :self._target_dim] = last_pred_target
            
            # 1> recall self._decode_all to generate one step's model output
            # current_target_params: [num_samples*batch_size, seq_len, target_dim, num_params]
            current_target_params, hidden_state = self._decode_direct(x, hidden_state)

            # 2> sampling by distribution params 
            # for each path, sample once, shape: [1, batch_size* num_samples, 1, target_dim]
            current_pred_sample = self._likelihood_model.sample(current_target_params, 1) 
            # remove sample dim and seq_len dim, as they are both 1, shape: [batch_size*num_samples, target_dim]
            current_pred_sample = current_pred_sample.squeeze([0, 2])
            current_pred_mean = current_pred_sample.reshape([-1, self._num_samples, self._target_dim]).median(axis=1)
            
            # 3> make sampling prediction as current step's output and the next step's input
            prediction_output.append(current_pred_mean) # [batch_size, target_dim]
            quantiles_output.append(current_pred_sample)
            last_pred_target = current_pred_sample
            
        # quantiles mode
        quantiles_output = paddle.stack(quantiles_output).reshape([num_steps, -1, self._num_samples, self._target_dim])
        quantiles_output = quantiles_output.transpose([1, 0, 3, 2]) # to shape: [batch_size, num_steps, target_dim, num_samples]
        quantiles_output = paddle.quantile(quantiles_output, 
                                           list(np.linspace(0, 1, 101, endpoint=True)), 
                                           axis=-1).transpose([1,2,3,0])
        # predictions mode
        prediction_output = paddle.stack(prediction_output, 1)
        return quantiles_output, prediction_output

    def _encoder(
        self, 
        data: Dict[str, paddle.Tensor]
    ) -> paddle.Tensor:
        """
        Encoder for input chunk.
        
        Args:
            data(Dict[str, paddle.Tensor]): A dict specifies all kinds of input data.
            
        Returns:
            hidden_state(paddle.Tensor): The output of rnn's hidden state.
        """
        past_target = data["past_target"]
        if "known_cov_numeric" not in data:
            past_known_cov = None
        else:
            past_known_cov = data["known_cov_numeric"][:, :past_target.shape[1]]

        #1> build input tensor
        input_tensor = self._build_input(past_target, past_known_cov) # encoder do not need `first_target_replace`
        #2> encode by rnn
        _, hidden_state = self._rnn(inputs = input_tensor) # seq_len: in_chunk_len -1
        return hidden_state # for LSTM, (h, c)
    
    def _decoder(
        self, 
        data: Dict[str, paddle.Tensor], 
        hidden_state: paddle.Tensor
    ) -> paddle.Tensor:
        """
        Decoder for output chunk.
        
        Args:
            data(Dict[str, paddle.Tensor]): A dict specifies all kinds of input data.
            hidden_state(paddle.Tensor): The initial hidden state for rnn.
            
        Returns:
            output(paddle.Tensor): For training phase, output distribution parameters for computation of nllloss; for evaluation phase, output samples for quantile loss metric; for prediction phase, output quantiles(probability) or predictions(point).
        """
        future_target = data["future_target"]
        if "known_cov_numeric" not in data:
            future_known_cov = None
        else:
            future_known_cov = data["known_cov_numeric"][:, -future_target.shape[1]:, :]
        first_target_replace = data["past_target"][:, -1]
        
        #1> build input
        # in decoer make the last time step of past_target as first_target_replace
        input_tensor = self._build_input(future_target, future_known_cov, first_target_replace) 
        
        #2> judge and decode
        if not self.predicting: # for training and validation phase
            model_output, _ = self._decode_direct(input_tensor, hidden_state) #[batch_size, seq_len, target_dim, param_num]
            # for training phase, decode input tensor directly to generate all output timesteps' distribution params.
            if self.training:
                output = self._likelihood_model.output_to_params(model_output) # rescale distr parameters
            # for validation phase, sample num_samples point by distribution params for quantile_loss metric
            else: 
                output = self._likelihood_model.sample(model_output, self._num_samples)
                output = output.transpose(perm = [1,2,3,0])
            return output
        else: # for prediction phase
            # Regression mode, `mean` and `quantiles` are optional.
            if self._regression_mode == "mean":
                quantiles_output, prediction_output = self._decode_regressive_by_mean(input_tensor, hidden_state) 
            else:
                quantiles_output, prediction_output = self._decode_regressive_by_sampling(input_tensor, hidden_state) 
            # Output mode, `quantiles` and `predictions` are optional. 
            if self._output_mode == "quantiles":
                return quantiles_output
            else:
                return prediction_output
        
    def forward(
        self,
        x: Dict[str, paddle.Tensor]
    ) -> paddle.Tensor:
        """
        The main logic of DeepAR model.
        
        Args:
            x(Dict[str, paddle.Tensor]): A dict specifies all kinds of input data.
        Returns:
            output(paddle.Tensor): The output of the model.
        """
        #encode
        hidden_state = self._encoder(x)
        #decode
        if self.predicting:
            x["future_target"] = paddle.rand([x["past_target"].shape[0], self._out_chunk_len, self._target_dim])
        output = self._decoder(x, hidden_state)
        return output


class DeepARModel(PaddleBaseModelImpl):
    """
    DeepAR model.

    Args:
        in_chunk_len(int): The size of the loopback window, i.e., the number of time steps feed to the model.
        out_chunk_len(int): The size of the forecasting horizon, i.e., the number of time steps output by the model.
        rnn_type(str): The type of the specific paddle RNN module ("GRU" or "LSTM").
        hidden_size(int): The number of features in the hidden state `h` of the RNN module.
        num_layers_recurrent(int): The number of recurrent layers.
        dropout(float): The fraction of neurons that are dropped in all-but-last RNN layers.
        skip_chunk_len(int): Optional, the number of time steps between in_chunk and out_chunk for a single sample. The skip chunk is neither used as a feature (i.e. X) nor a label (i.e. Y) for a single sample. By default it will NOT skip any time steps.
        sampling_stride(int, optional): sampling intervals between two adjacent samples.
        likelihood_model(Likelihood): The distribution likelihood to be used for probability forecasting.
        num_samples(int): The sampling number for validation and prediction phase, it is used for computation of quantiles loss and the point forecasting result.
        loss_fn(Callable[..., paddle.Tensor]): The loss fucntion of probability forecasting respect to likelihood model.
        regression_mode(str): The regression mode of prediction, `mean` and `sampling` are optional.
        output_mode(str): The mode of model output, `quantiles` and `predictions` are optional.
        optimizer_fn(Callable, Optional): optimizer algorithm.
        optimizer_params(Dict, Optional): optimizer parameters.
        eval_metrics(List[str], Optional): evaluation metrics of model.
        callbacks(List[Callback], Optional): customized callback functions.
        batch_size(int, Optional): number of samples per batch.
        max_epochs(int, Optional): max epochs during training.
        verbose(int, Optional): verbosity mode.
        patience(int, Optional): number of epochs with no improvement after which learning rate wil be reduced.
        seed(int, Optional): global random seed.
    """

    def __init__(
        self,
        in_chunk_len: int,
        out_chunk_len: int,
        rnn_type_or_module: str = "LSTM",
        fcn_out_config: List[int] = None,
        hidden_size: int = 128,
        num_layers_recurrent: int = 1,
        dropout: float = 0.0,
        skip_chunk_len: int = 0,
        sampling_stride: int = 1,
        likelihood_model: Likelihood = GaussianLikelihood(),
        num_samples: int = 100,
        loss_fn: Callable[..., paddle.Tensor] = GaussianLikelihood().loss,
        regression_mode: str = "mean",
        output_mode: str = "quantiles",
        optimizer_fn: Callable[..., Optimizer] = paddle.optimizer.Adam,
        optimizer_params: Dict[str, Any] = dict(learning_rate=1e-4),
        eval_metrics: List[str] = ["quantile_loss"],
        callbacks: List[Callback] = [],
        batch_size: int = 128,
        max_epochs: int = 10,
        verbose: int = 1,
        patience: int = 4,
        seed: int = 0
    ):
        self._rnn_type_or_module = rnn_type_or_module
        self._hidden_size = hidden_size
        self._num_layers_recurrent = num_layers_recurrent
        self._dropout = dropout
        self._likelihood_model = likelihood_model
        self._num_samples = num_samples
        self._output_mode = output_mode
        self._regression_mode = regression_mode

        #check parameters validation
        raise_if_not(
                self._rnn_type_or_module in {"LSTM", "GRU"},
                "A valid RNN type should be specified, currently LSTM and GRU are supported."
                )

        super(DeepARModel, self).__init__(
            in_chunk_len=in_chunk_len,
            out_chunk_len=out_chunk_len,
            skip_chunk_len=skip_chunk_len,
            sampling_stride=sampling_stride,
            loss_fn=likelihood_model.loss,
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
    
    def _check_params(self):
        """Parameter validity verification.

        Check logic:

            batch_size: batch_size must be > 0.
            max_epochs: max_epochs must be > 0.
            verbose: verbose must be > 0.
            patience: patience must be >= 0.
        """
        raise_if(self._batch_size <= 0, f"batch_size must be > 0, got {self._batch_size}.")
        raise_if(self._max_epochs <= 0, f"max_epochs must be > 0, got {self._max_epochs}.")
        raise_if(self._verbose <= 0, f"verbose must be > 0, got {self._verbose}.")
        raise_if(self._patience < 0, f"patience must be >= 0, got {self._patience}.")
        raise_if(self._output_mode not in {"quantiles", "predictions"}, \
                 f"output mode must be one of {{`quantiles`, `predictions`}}, got `{self._output_mode}`.")
        raise_if(self._regression_mode not in {"mean", "sampling"}, \
                 f"regression mode must be one of {{`mean`, `sampling`}}, got `{self._regression_mode}`.")        
        
        # If user does not specify an evaluation metric, a metric is provided by default.
        # Currently, only support quantile_loss. TODO: construct more metrics: NLLLoss
        if self._eval_metrics != ["quantile_loss"]:
            logger.warning(f"Evaluation metric is transformed to ['quantile_loss'], got {self._eval_metrics}.")
            self._eval_metrics = ["quantile_loss"]
        
    def _check_tsdataset(
        self,
        tsdataset: TSDataset
    ):
        """
        Rewrite _check_tsdataset to fit the specific model.
        For DeepAR, all data variables are expected to be float32.
        """
        for column, dtype in tsdataset.dtypes.items():
            raise_if_not(
                np.issubdtype(dtype, np.floating),
                f"deepar variables' dtype only supports [float16, float32, float64], " \
                f"but received {column}: {dtype}."
            )
        super(DeepARModel, self)._check_tsdataset(tsdataset)

    def _update_fit_params(
        self,
        train_tsdataset: List[TSDataset],
        valid_tsdataset: Optional[List[TSDataset]] = None
    ) -> Dict[str, Any]:
        """
        Infer parameters by TSdataset automatically.

        Args:
            train_tsdataset(List[TSDataset]): list of train dataset
            valid_tsdataset(List[TSDataset], optional): list of validation dataset

        Returns:
            Dict[str, Any]: model parameters
        """
        fit_params = {
                "target_dim": train_tsdataset[0].get_target().data.shape[1],
                "known_cov_dim": 0,
                "observed_cov_dim": 0
                }
        if train_tsdataset[0].get_known_cov() is not None:
            fit_params["known_cov_dim"] = train_tsdataset[0].get_known_cov().data.shape[1]
        if train_tsdataset[0].get_observed_cov() is not None:
            fit_params["observed_cov_dim"] = train_tsdataset[0].get_observed_cov().data.shape[1]
        return fit_params

    def _init_network(self) -> paddle.nn.Layer:
        """
        Init network.

        Returns:
            paddle.nn.Layer
        """
        return _DeepAR(
            self._in_chunk_len,
            self._out_chunk_len,
            self._fit_params["target_dim"],
            self._fit_params["known_cov_dim"],
            self._rnn_type_or_module,
            self._hidden_size,
            self._num_layers_recurrent,
            self._dropout,
            self._likelihood_model,
            self._num_samples,
            self._regression_mode,
            self._output_mode
        )
    
    def _prepare_X_y(self, 
        X: Dict[str, paddle.Tensor]
    ) -> Tuple[Dict[str, paddle.Tensor], paddle.Tensor]:
        """Split the packet into X, y.

        Args:
            X(Dict[str, paddle.Tensor]): Dict of feature/target tensor.

        Returns:
            X(Dict[str, paddle.Tensor]): Dict of feature tensor.
            y(paddle.Tensor): Target tensor.
        """
        y = X["future_target"]
        return X, y
    
    def _predict(
        self, 
        dataloader: paddle.io.DataLoader,
    ) -> np.ndarray:
        """
        Predict function core logic.

        Args:
            dataloader(paddle.io.DataLoader): Data to be predicted.

        Returns:
            np.ndarray.
        """
        self._network.predicting = True
        return super(DeepARModel, self)._predict(dataloader)

