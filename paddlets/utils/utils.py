#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import uuid
import hashlib

from inspect import isclass

from paddlets.logger.logger import raise_log
from paddlets.models.base import Trainable
from paddlets.pipeline import Pipeline
from paddlets.models.dl.paddlepaddle.paddle_base import PaddleBaseModel
from paddlets.logger import Logger, raise_if, raise_if_not
from paddlets.models.base import BaseModel

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
    if isinstance(model, PaddleBaseModel):
        fitted = True if model._network else False

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

