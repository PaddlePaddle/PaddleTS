#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from numbers import Integral
import uuid
import hashlib

from inspect import isclass
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from paddlets.models.base import Trainable
from paddlets.logger import raise_if_not, raise_if, raise_log, Logger
from paddlets.datasets.tsdataset import TSDataset

logger = Logger(__name__)

def check_model_fitted(model: Trainable, msg: str = None):
    """
    check if model has fitted, Raise Exception if not fitted

    Args:
        model(Trainable): model instance.
        msg(str): str, default=None
                  The default error message is, "This %(name)s instance is not fitted
                  yet. Call 'fit' with appropriate arguments before using this
                  estimator."
                  For custom messages if "%(name)s" is present in the message string,
                  it is substituted for the estimator name.
                  Eg. : "Estimator, %(name)s, must be fitted before sparsifying".
    Returns:
        None

    Raise:
        ValueError
    """
    from paddlets.pipeline import Pipeline
    from paddlets.models.forecasting.ml.ml_base import MLBaseModel
    from paddlets.models.forecasting.dl.paddle_base import PaddleBaseModel
    from paddlets.ensemble.ensemble_forecaster_base import EnsembleForecasterBase
    try:
        from paddlets.automl import AutoTS
    except Exception as e:
        logger.warning(f"error occurred while import autots, err: {str(e)}")
        AutoTS = None
    #不需要fit的模型列表  
    MODEL_NEED_NO_FIT = ["ArimaModel"]    
    if model.__class__.__name__ in MODEL_NEED_NO_FIT:
        return
    if isclass(model):
        raise_log(ValueError(f"{type(model).__name__}is a class, not an instance."))
    if msg is None:
        msg = (
            "This %(name)s instance is not fitted yet. Call 'fit' with "
            "appropriate arguments before using this estimator."
        )
    if not isinstance(model, Trainable):
        raise_log(ValueError(f"{type(model).__name__} is not a Trainable Object."))

    fitted = False
    # PipeLine
    if isinstance(model, Pipeline):
        fitted = model._fitted
    # Paddle 模型
    elif isinstance(model, PaddleBaseModel):
        fitted = True if model._network else False
    # ML 模型
    elif isinstance(model, MLBaseModel):
        #TODO:后续如果将 self._models 提到 MLBaseModel后，这里需要同步修改为判断 self._models ，而不是 "_models" 字符串。
        fitted = True if "model" in vars(model) or "_model" in vars(model) else False
    elif isinstance(model, EnsembleForecasterBase):
        fitted = True if model.fitted else False
    elif AutoTS is not None and isinstance(model, AutoTS):
        fitted = model.is_refitted()

    raise_if_not(fitted, msg % {"name": type(model).__name__})

def get_uuid(prefix: str = "", suffix: str = ""):
    """
    Get a random string of 16 characters.

    Args:
        prefix(str, optional): The prefix of the returned string.
        suffix(str, optional): The suffix of the returned string.

    Returns:
        str: String of 16 characters.
    """
    digits = "01234abcdefghijklmnopqrstuvwxyz56789"
    new_uuid = uuid.uuid1()
    md = hashlib.md5()
    md.update(str(new_uuid).encode())
    for i in md.digest():
        x = (i + 128) % 34
        prefix = prefix + digits[x]
    res = prefix + suffix if suffix is not None else prefix
    return res


def check_train_valid_continuity(train_data: TSDataset, valid_data: TSDataset)-> bool:
    """
    Check if train and test TSDataset are continous

    Args:
        train_data(TSDataset): Train dataset.
        test_data(TSDataset): Test dataset.

    Return:
        bool: if train and test TSDataset are continous

    """
    train_index = train_data.target.data.index
    valid_index = valid_data.target.data.index

    continuious = False
    if isinstance(train_index, pd.DatetimeIndex):
        if isinstance(valid_index, pd.DatetimeIndex):
            continuious = (valid_index[0] - train_index[-1] == pd.to_timedelta(train_index.freq))
    elif isinstance(train_index, pd.RangeIndex):
        if isinstance(valid_index, pd.RangeIndex):
            continuious = (valid_index[0] - train_index[-1] == train_index.step)
    else:
        raise_log("Unsupport data index format")

    return continuious


def split_dataset(dataset: TSDataset, split_point: int) ->  TSDataset:
    """
    Split dataset (accroding to the max length)

    Args:
        dataset(TSDataset): dataset to be splited.
        split_point(int): split point.

    Return:
        TSDataset

    """
    target_index = None
    observed_index = None
    known_index = None
    index_list = []

    if dataset.target:
        target_index = dataset.target.data.index
        index_list.append(target_index)
    if dataset.known_cov:
        known_index = dataset.known_cov.data.index
        index_list.append(known_index)
    if dataset.observed_cov:
        observed_index = dataset.observed_cov.data.index
        index_list.append(observed_index)
    
    #sort to avoid  wrong positions index
    index_list.sort(key=lambda x: x[0])

    all_index = pd.concat([x.to_series() for x in index_list]).index.drop_duplicates()
    max_len = len(all_index)
    split_index = all_index[split_point-1]

    raise_if(split_point >= max_len, "split point should smaller than dataset length")
    raise_if(split_point <= 0, "split point should > 0")
    raise_if_not(isinstance(split_point, Integral),
                 f"split point should be  Integral type, instead of {type(split_point)}")

    target_pre = None
    target_after = None
    if dataset.target:

        if split_index < target_index[0]:
            target_after = dataset.target
        elif split_index >= target_index[-1]:
            target_pre = dataset.target
        elif split_index in target_index:
            if isinstance(dataset.target.data.index, pd.RangeIndex):
                target_pre, target_after = dataset.target.split(int((split_index - dataset.target.data.index[0]) / dataset.target.data.index.step +1))
            else:
                target_pre, target_after = dataset.target.split(split_index)

    known_pre = None
    known_after = None
    if dataset.known_cov:
        if split_index < known_index[0]:
            known_after = dataset.known_cov
        elif split_index >= known_index[-1]:
            known_pre = dataset.known_cov
        elif split_index in known_index:
            if isinstance(dataset.known_cov.data.index, pd.RangeIndex):
                known_pre, known_after = dataset.known_cov.split(int((split_index - dataset.known_cov.data.index[0]) / dataset.known_cov.data.index.step + 1))
            else:
                known_pre, known_after = dataset.known_cov.split(split_index)

    observed_pre = None
    observed_after = None
    if dataset.observed_cov:
        if split_index < observed_index[0]:
            observed_after = dataset.observed_cov
        elif split_index >= observed_index[-1]:
            observed_pre = dataset.observed_cov
        elif split_index in observed_index:
            if isinstance(dataset.observed_cov.data.index, pd.RangeIndex):
                observed_pre, observed_after = dataset.observed_cov.split(int(((split_index - dataset.observed_cov.data.index[0])) / dataset.observed_cov.data.index.step + 1))
            else:
                observed_pre, observed_after = dataset.observed_cov.split(split_index)
    return (TSDataset(target_pre, observed_pre, known_pre, dataset.static_cov),
            TSDataset(target_after, observed_after, known_after, dataset.static_cov))

def get_tsdataset_max_len(dataset:TSDataset) -> int:
    """
    Get dataset max length

    Args:
        dataset(TSDataset): dataset use to get length.

    Return:
        int

    """
    target_index = None
    observed_index = None
    known_index = None
    index_list = []

    if dataset.target:
        target_index = dataset.target.data.index
        index_list.append(target_index)
    if dataset.known_cov:
        known_index = dataset.known_cov.data.index
        index_list.append(known_index)
    if dataset.observed_cov:
        observed_index = dataset.observed_cov.data.index
        index_list.append(observed_index)
    
    #sort to avoid  wrong positions index
    index_list.sort(key=lambda x: x[0])

    all_index = pd.concat([x.to_series() for x in index_list]).index.drop_duplicates()

    return len(all_index)

def repr_results_to_tsdataset(reprs: np.array, dataset: TSDataset)-> TSDataset:
    """
    Convert representation model output to a TSDataset 

    Args:
        reprs(np.array): output results of representation model 
        dataset(TSDataset): dataset use to get target 

    Return:
        TSDataset
    """
    labels_df = dataset.get_target().data
    reprs = reprs[0]
    if reprs.shape[0] != labels_df.shape[0]:
        raise_log("The length of labels_df and reprs should be equal")
    reprs_df = pd.DataFrame(reprs, columns=[f"repr_{i}" for i in range(reprs.shape[-1])], index=dataset.get_target().data.index)
    new_dataset = TSDataset.load_from_dataframe(
                    df=reprs_df.join(labels_df),
                    observed_cov_cols=reprs_df.columns.tolist(),
                    target_cols=labels_df.columns.tolist())
    return new_dataset

def plot_anoms(predict_data:TSDataset = None, 
                origin_data:TSDataset = None , 
                feature_name:str = None):
    """
    Plots anomalies

    Args:
        predict_data(TSDataset): Data used to print predict anom labels.
        origin_data(TSDataset|None): Data used to print features or origin anom labels. only print predict anom labels if set to None.
        feature_name(str|None): feature name in origin data to print
    """
    def plot_anoms_point(ax: plt.Axes, anomaly_data: TSDataset):
        """
        plot anoms point
        """
        ax2 = ax.twinx()
        anom_vals = anomaly_data.target.data.to_numpy()
        anom_label = anomaly_data.target.data.columns[0]
        ax2.plot(anomaly_data.target.data.index, anom_vals, color="r",label=anom_label)
        ax2.set_ylabel(anom_label)
        minval, maxval = min(anom_vals), max(anom_vals)
        delta = maxval - minval
        if delta > 0:
            ax2.set_ylim(minval - delta / 8, maxval + 2 * delta)
        else:
            ax2.set_ylim(minval - 1 / 30, maxval + 1)


    def plot_anoms_window(ax: plt.Axes, anomaly_data: TSDataset):
        """
        Plots anomalies as windows 
        """
        anomaly_labels = anomaly_data.get_target().data
        t, y = anomaly_labels.index, anomaly_labels.values
        splits = np.where(y[1:] != y[:-1])[0] + 1
        splits = np.concatenate(([0], splits, [len(y) - 1]))
        for k in range(len(splits) - 1):
            if y[splits[k]]:  # If splits[k] is anomalous
                ax.axvspan(t[splits[k]], t[splits[k + 1]], color="purple", alpha=0.5)

    ax = plt

    if predict_data is None:
        pass
    else:       
        raise_if_not(isinstance(predict_data, TSDataset), f"origin data type ({type(origin_data)}) must be TSDataset")
        predict_data = predict_data.copy()

    if origin_data is None:
        plot_anoms_point(ax, predict_data)
        return
    else:
        raise_if_not(isinstance(origin_data, TSDataset), f"origin data type ({type(origin_data)}) must be TSDataset")  
        origin_data = origin_data.copy()

    if feature_name is not None:
        ax = origin_data.plot(columns=feature_name, x_compat=True)

    if origin_data.target is not None:
        plot_anoms_window(ax,origin_data)
    
    if predict_data is None:
        pass
    else:
        plot_anoms_point(ax, predict_data)
