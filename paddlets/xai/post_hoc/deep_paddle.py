# !/usr/bin/env python3
# -*- coding:utf-8 -*-

from typing import List, Optional, Union, Dict
from packaging import version
from collections import defaultdict

import numpy as np
import paddle
from paddle import Tensor
from shap import Explainer

from paddlets.models.forecasting.dl.paddle_base import PaddleBaseModel
from paddlets.logger import Logger, raise_if_not, raise_if, raise_log

logger = Logger(__name__)


class PaddleDeep(Explainer):
    """
    Using paddlepaddle framework to implement the deep shap. 
    Currently we directly calculate the gradient but not using the deep lift method between the output and input because some operations are not supported.
    
    Args:
        model(PaddleBaseModel): A model object that supports `forward` function.
        data(Dict[str, Tensor]):  A dict tensor for training the deep explainer
    """
    def __init__(
                self, 
                model: Optional[Union[PaddleBaseModel,]],
                data: Dict[str, Tensor],
    ) -> None:
        self.data = [data]
        # To keep the DeepExplainer base value
        self.expected_value = None 
        # Get module(nn.Layer)
        if not issubclass(type(model), PaddleBaseModel):
            raise_log(f"The model type ({type(model)}) is not supported by deep explainer.")
        self.model = model._network
        if not issubclass(type(self.model), paddle.nn.Layer):
            raise_log("Only the type(paddle.nn.Layer) is supported. Please check the type of model._network!")
            
        self.multi_output = False
        self.num_outputs = 1
        
        with paddle.no_grad():
            outputs = self.model(*self.data)
            
            if outputs.shape[1] > 1:
                self.multi_output = True
                self.num_outputs = outputs.shape[1]
            self.expected_value = outputs.mean(0).cpu().numpy()

    def gradient(self, idx: int, inputs: Dict[str, Tensor]) -> List[Dict[str, np.ndarray]]:
        """
        Calculate the gradient of input and output on each output step.

        Args:
            idx(int): output step.
            inputs(Dict[str, Tensor]): input data.

        Returns:
            List[Dict[str, np.ndarray]]: gradient values.
        """
        self.model.clear_gradients()
        
        X = []
        for input_ in inputs:
            for key in input_.keys():
                input_[key].stop_gradient = False
            X.append(input_)
            
        # Convert model mode to compute grad
        self.model.train()
        
        outputs = self.model(*X)

        selected = [val for val in outputs[:, idx]]
        grads = []
        for idx, x in enumerate(X):
            grad_dict = {}
            retain_graph = True
            for i, key in enumerate(x.keys()):
                if key == 'future_target':
                    continue
                if i == len(x.keys()) - 1 and idx == len(X) - 1:
                    retain_graph = None
                grad = paddle.autograd.grad(selected, x[key],
                                           retain_graph=retain_graph,
                                           allow_unused=True)[0]

                if grad is not None:
                    grad = grad.cpu().numpy()
                else:
                    grad = paddle.zeros_like(X[idx][key]).cpu().numpy()
                grad_dict[key] = grad
            grads.append(grad_dict)
        # Convert model mode
        self.model.eval()
        return grads

    def shap_values(self, X: Dict[str, Tensor]) -> List[Dict[str, np.ndarray]]:
        """
        Calculate the shap value of X.

        Args:
            X(Dict[str, Tensor]): input data.

        Returns:
            List[Dict[str, np.ndarray]]: shap values.
        """
        raise_if_not(isinstance(X, Dict) and 'past_target' in X, 'Input(X) must be dict[str, tensor] and past_target in X!')
        # Now just one input
        X = [X]
        model_output_ranks = (paddle.ones((X[0]['past_target'].shape[0], self.num_outputs)).astype('int') * \
                              paddle.arange(0, self.num_outputs).astype('int'))
        # Compute the attribution
        output_phis = []
        for i in range(model_output_ranks.shape[1]):
            phis = []
            # Record the attribution
            for k in range(len(X)):
                dicts = {}
                for key in X[k].keys():
                    dicts[key] = np.zeros(X[k][key].shape)
                phis.append(dicts)
            # Compute the attribution
            for j in range(X[0]['past_target'].shape[0]):
                # Tile the inputs to line up with X
                tiled_X = [{key: paddle.tile(X[l][key][j:j + 1], (self.data[l][key].shape[0],) \
                                             + tuple([1 for k in range(len(X[l][key].shape) - 1)])) \
                            for key in X[l].keys()} for l in range(len(X))]
                joint_x = [{key: paddle.concat((tiled_X[l][key], self.data[l][key]), axis=0) \
                            for key in X[l].keys()} for l in range(len(X))]
                # Run attribution with the computation graph
                feature_ind = model_output_ranks[j, i]
                sample_phis = self.gradient(feature_ind, joint_x)
                # Assign the attributions to the right part of the output arrays
                for l in range(len(X)):
                    for key in X[l].keys():
                        phis[l][key][j] = (paddle.to_tensor(sample_phis[l][key][self.data[l][key].shape[0]:]) \
                                           * (X[l][key][j: j + 1] - self.data[l][key])).cpu().detach().numpy().mean(0)
            output_phis.append(phis[0])
        return output_phis

