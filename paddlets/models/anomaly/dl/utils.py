#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from typing import List, Dict, Any, Callable, Optional
import functools

from paddle.optimizer import Optimizer
import numpy as np
import pandas as pd
import paddle
import paddle.nn.functional as F
import more_itertools as mit

from paddlets.logger import raise_if_not, raise_if, raise_log, Logger
from paddlets.datasets import TSDataset
from paddlets.models import BaseModel

logger = Logger(__name__)


def percentile(anomaly_score: np.ndarray, q: float=100):
    """
    The percentile function to get anomaly threshold.

    Args:
        anomaly_score(np.ndarray): Anomaly score.
        q(float): The parameter used to calculate the quantile, range is [0, 100].

    Return:
        threshold(float): Anomaly threshold.
    """
    return np.percentile(anomaly_score, q)


def my_kl_loss(p: paddle.Tensor, q: paddle.Tensor):
    """
    The Kullback–Leibler divergence.
    
    Args:
        p(paddle.Tensor): Tensor of arbitrary shape in log-probabilities.
        q(paddle.Tensor): Tensor of the same shape as input.

    Return:
        loss(paddle.Tensor): Got loss.
    """
    res = p * (paddle.log(p + 0.0001) - paddle.log(q + 0.0001))
    return paddle.mean(paddle.sum(res, axis=-1), axis=1)
    
    
def adjust_learning_rate(optimizer: Callable[..., Optimizer], epoch:int, lr_:float):
    """
    Dynamic Learning Rate Adjustment.
    
    Args:
        optimizer(Callable[..., Optimizer]): Optimizer algorithm.
        epoch(int): Max epochs during training.
        lr_(float): Learning rate.    
    """
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        optimizer.set_lr(lr)
        logger.info('epoch: {}, Updating learning rate to {}'.format(epoch, optimizer.get_lr()))


def series_prior_loss(output_list: List[paddle.Tensor], 
                      input: paddle.Tensor, 
                      criterion: Callable[..., paddle.Tensor]=paddle.nn.MSELoss(), 
                      win_size: int=100, 
                      k: int=3):
    """
    Calculate Association discrepancy in train.
    
    Args:
        output_list(List[paddle.Tensor]): Model ouput tensor list.
        input(paddle.Tensor): Target tensor.
        criterion(Callable[..., paddle.Tensor]): Loss function.
        win_size(int): The size of the loopback window, i.e. the number of time steps feed to the model.
        k(int): The optimization is to enlarge the association discrepancy.

    Return:
        loss1(paddle.Tensor): Series_loss and rec_loss.
        loss2(paddle.Tensor): Prior_loss and rec_loss.
        for_loss_one(paddle.Tensor): Rec_loss and Association discrepancy.
    """
    output, series, prior, _  = output_list
    series_loss = 0.0
    prior_loss = 0.0
    for u in range(len(prior)):
        series_kl = prior[u] / paddle.tile(paddle.unsqueeze(paddle.sum(prior[u], axis=-1), axis=-1), 
                                                                   repeat_times=[1, 1, 1, win_size])
        series_loss += (paddle.mean(my_kl_loss(series[u], series_kl.detach())) 
                      + paddle.mean(my_kl_loss(series_kl.detach(), series[u]))
                       )
        prior_loss += (paddle.mean(my_kl_loss(series_kl, series[u].detach()))
                      + paddle.mean(my_kl_loss(series[u].detach(), series_kl))
                      )
    series_loss = series_loss / len(prior)
    prior_loss = prior_loss / len(prior)
    rec_loss = criterion(output, input)
    loss1 = rec_loss - k * series_loss
    loss2 = rec_loss + k * prior_loss
    for_loss_one = (rec_loss - k * series_loss).item()
    return loss1, loss2, for_loss_one


def series_prios_energy(output_list, loss, temperature=50, win_size=100):
    """
    calculate Association discrepancy in test.
    
    Args:
        output_list(List[paddle.Tensor]): Model ouput tensor list.
        loss(paddle.Tensor): Got loss.
        temperature(int|float): A parameter to adjust series loss and prior loss. 
        win_size(int): The size of the loopback window, i.e. the number of time steps feed to the model.
       
    return:
        cri(np.ndarray): Anomaly score for predict.
    """
    output, series, prior, _ = output_list
    series_loss = 0.0
    prior_loss = 0.0
    for u in range(len(prior)):
        series_kl = prior[u] / paddle.tile(paddle.unsqueeze(paddle.sum(prior[u], axis=-1), axis=-1), 
                                                             repeat_times=[1, 1, 1, win_size])
        if u == 0:
            series_loss = my_kl_loss(series[u], series_kl.detach()) * temperature
            prior_loss = my_kl_loss(series_kl, series[u].detach()) * temperature
        else:
            series_loss += my_kl_loss(series[u], series_kl.detach()) * temperature
            prior_loss += my_kl_loss(series_kl, series[u].detach()) * temperature
    # Metric
    metric = paddle.nn.functional.softmax((-series_loss - prior_loss), axis=-1)
    cri = metric * loss
    cri = cri.detach().cpu().numpy()
    return cri


def anomaly_get_threshold(model: Callable[..., paddle.Tensor], 
                         train_dataloader: paddle.io.DataLoader, 
                         thre_dataloader: paddle.io.DataLoader, 
                         temperature: float= 50,  
                         anormly_ratio: float=4,
                         criterion: Callable[..., paddle.Tensor] = paddle.nn.MSELoss(), 
                         my_kl_loss: Callable[..., paddle.Tensor] = my_kl_loss,
                         win_size: int=100, 
                         ):
    """
    Threshold is calculated based on Association-based Anomaly Criterion.
    
    Args:
        model(Callable[..., paddle.Tensor]): Anomaly transformer model.
        train_dataloader(paddle.io.DataLoader): Train set. 
        thre_dataloader(List[paddle.io.DataLoader]|None): Test set.
        temperature(int|float): A parameter to adjust series loss and prior loss.  
        anormly_ratio(int|float): The Proportion of Anomaly data in train set and test set.
        criterion(Callable[..., paddle.Tensor]|None): Loss function for the reconstruction loss.
        my_kl_loss(Callable[..., paddle.Tensor]|None): Loss function for association discrepancy.
        win_size(int): The size of the loopback window, i.e. the number of time steps feed to the model.

    Return:
        threshold(float|None): The threshold to judge anomaly.
    
    """
    model.eval()
    # (1) stastic on the train set
    attens_energy = []
    for i , (input_data) in  enumerate(train_dataloader):
        output, series, prior, _ = model(input_data)
        if "observed_cov_numeric" in input_data.keys():    #  Not support categorical temporarily 
            input_data = input_data['observed_cov_numeric']
        else:
            raise_log(ValueError(f"observed_cov_numeric doesn't exist!"))
        loss = paddle.mean(criterion(input_data, output), axis=-1)
        series_loss = 0.0
        prior_loss = 0.0
        for u in range(len(prior)):
            series_kl = prior[u] / paddle.tile(paddle.unsqueeze(paddle.sum(prior[u], axis=-1), axis=-1), 
                                                                       repeat_times=[1, 1, 1, win_size])
            if u == 0:            
                series_loss = my_kl_loss(series[u], series_kl.detach()) * temperature
                prior_loss = my_kl_loss(series_kl, series[u].detach()) * temperature
            else:
                series_loss += my_kl_loss(series[u], series_kl.detach()) * temperature
                prior_loss += my_kl_loss(series_kl, series[u].detach()) * temperature
        metric = paddle.nn.functional.softmax((-series_loss - prior_loss), axis=-1)    # AnomalyScore
        cri = metric * loss
        cri = cri.detach().cpu().numpy()
        attens_energy.append(cri)
    attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
    train_energy = np.array(attens_energy)
    # (2) find the threshold
    attens_energy = []
    for i, (input_data) in enumerate(thre_dataloader):
        output, series, prior, _ = model(input_data)
        if "observed_cov_numeric" in input_data.keys():    #  Not support categorical temporarily 
            input_data = input_data['observed_cov_numeric']
        else:
            raise_log(ValueError(f"observed_cov_numeric doesn't exist!"))
        loss = paddle.mean(criterion(input_data, output), axis=-1)
        series_loss = 0.0
        prior_loss = 0.0
        for u in range(len(prior)):
            series_kl = prior[u] / paddle.tile(paddle.unsqueeze(paddle.sum(prior[u], axis=-1), axis=-1), 
                                                                   repeat_times=[1, 1, 1, win_size])
            if u == 0:
                series_loss = my_kl_loss(series[u], series_kl.detach()) * temperature
                prior_loss = my_kl_loss(series_kl, series[u].detach()) * temperature
            else:
                series_loss += my_kl_loss(series[u], series_kl.detach()) * temperature
                prior_loss += my_kl_loss(series_kl, series[u].detach()) * temperature
        # Metric
        metric = paddle.nn.functional.softmax((-series_loss - prior_loss), axis=-1)  # AnomalyScore
        cri = metric * loss
        cri = cri.detach().cpu().numpy()
        attens_energy.append(cri)
        
    attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
    test_energy = np.array(attens_energy)
    # comb energy
    combined_energy = np.concatenate([train_energy, test_energy], axis=0)
    threshold = np.percentile(combined_energy, 100 - anormly_ratio)
    return threshold


def result_adjust(pred: np.ndarray, real: np.ndarray):
    """
    The adjustment is a widely-used convention in time series anomaly detection.
    
    Args:
        pred(List[float]|np.ndarray): The model prediction results.
        real(List[float]|np.ndarray): Ground truth target values.

    Return:
        pred(np.ndarray): Adjusted prediction results.
    """
    anomaly_state = False
    for i in range(len(real)):
        if real[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if real[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(real)):
                if real[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif real[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return np.array(pred)


def smooth_l1_loss_vae(output_tensor_list: List[paddle.Tensor], 
                       kld_beta: float=0.2):
    """ 
    smooth l1 loss.
    
    Args:
        output_tensor_list(list[paddle.Tensor]): Model ouput.
        kld_beta(float): Kld beta in 0~1.

    Return:
        loss(float): Got loss.
    """
    [recon, mu, logvar, obs] = output_tensor_list
    recon_loss = F.smooth_l1_loss(recon, obs, reduction='sum')
    kld = -0.5 * paddle.sum(1 + logvar - mu ** 2 - logvar.exp())
    loss = recon_loss + kld_beta * kld
    return loss


def to_tsdataset(
    scenario: str = "anomaly_label"
    ) -> Callable[..., Callable[..., TSDataset]]:
    """A decorator, used for converting ndarray to tsdataset in anomaly dl models. 

    Args:
        scenario(str): The task type. ["anomaly_label", "anomaly_score"] is optional.
    Returns:
        Callable[..., Callable[..., TSDataset]]: Wrapped core function.
    """
    def decorate(func) -> Callable[..., TSDataset]:
        @functools.wraps(func)
        def wrapper(
            obj: BaseModel,
            test_data: TSDataset,
            train_data: TSDataset,
        ) -> TSDataset:
            """Core processing logic.

            Args:
                obj(BaseModel): BaseModel instance.
                test_data(TSDataset): Test tsdataset.
                train_data(TSDataset): Train tsdataset.
                
            Returns:
                TSDataset: Predict results.
            """
            raise_if_not(
                scenario in ("anomaly_label", "anomaly_score"),
                f"{scenario} not supported, ['anomaly_label', 'anomaly_score'] is optional."
            )
            
            results = func(obj, test_data, train_data)
            # Generate target cols
            target_cols = test_data.get_target()
            if target_cols is None:
                target_cols = [scenario]
            else:
                target_cols = target_cols.data.columns
                if scenario == "anomaly_score":
                    target_cols = target_cols + '_score'
            # Generate target index freq
            target_index = test_data.get_observed_cov().data.index
            if isinstance(target_index, pd.RangeIndex):
                freq = target_index.step
            else:
                freq = target_index.freqstr
            results_size = results.size
            raise_if(
                results_size == 0,
                f"There is something wrong, anomaly predict size is 0, you'd better check the tsdataset or the predict logic."
            )
            target_index = target_index[:results_size]
            anomaly_target = pd.DataFrame(results, index=target_index, columns=target_cols)
            return TSDataset.load_from_dataframe(anomaly_target, freq=freq)
        return wrapper
    return decorate

    
def epsilon_th(anomaly_score: np.ndarray, reg_level: int = 1):
    """
    Threshold method proposed by Hundman et. al. (https://arxiv.org/abs/1802.04431)
    Code from TelemAnom (https://github.com/khundman/telemanom)
    
    Args:
        anomaly_score(np.ndarray): Anomaly score.
        reg_level(int): The parameter used to calculate the threshold.

    Return:
        threshold(float): Anomaly threshold.
    
    """
    a_s = anomaly_score
    best_epsilon = None
    max_score = -10000000
    mean_a_s = np.mean(a_s)
    sd_a_s = np.std(a_s)

    for z in np.arange(2.5, 12, 0.5):
        epsilon = mean_a_s + sd_a_s * z
        pruned_a_s = a_s[a_s < epsilon]

        i_anom = np.argwhere(a_s >= epsilon).reshape(-1,)
        buffer = np.arange(1, 50)
        i_anom = np.sort(
            np.concatenate(
                (
                    i_anom,
                    np.array([i + buffer for i in i_anom]).flatten(),
                    np.array([i - buffer for i in i_anom]).flatten(),
                )
            )
        )
        i_anom = i_anom[(i_anom < len(a_s)) & (i_anom >= 0)]
        i_anom = np.sort(np.unique(i_anom))

        if len(i_anom) > 0:
            groups = [list(group) for group in mit.consecutive_groups(i_anom)]
            mean_perc_decrease = (mean_a_s - np.mean(pruned_a_s)) / mean_a_s
            sd_perc_decrease = (sd_a_s - np.std(pruned_a_s)) / sd_a_s
            if reg_level == 0:
                denom = 1
            elif reg_level == 1:
                denom = len(i_anom)
            elif reg_level == 2:
                denom = len(i_anom) ** 2
            score = (mean_perc_decrease + sd_perc_decrease) / denom
            if score >= max_score and len(i_anom) < (len(a_s) * 0.5):
                max_score = score
                best_epsilon = epsilon

    if best_epsilon is None:
        best_epsilon = np.max(a_s)
            
    return best_epsilon
