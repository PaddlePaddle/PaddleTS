# !/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import json

from paddlets.models.base import BaseModel
from paddlets.logger import raise_if, raise_if_not, raise_log


def load(path: str) -> BaseModel:
    """
    Loads a saved model from a file path.

    Args:
        path(str): A path string containing a model file name.

    Returns:
        BaseModel: Loaded model.
    """
    abs_model_path = os.path.abspath(path)

    raise_if_not(os.path.exists(abs_model_path), "path not exists: %s" % abs_model_path)
    raise_if(os.path.isdir(abs_model_path), "path must be a file path, not a directory: %s" % abs_model_path)

    abs_root_path = os.path.dirname(abs_model_path)
    abs_model_path = os.path.join(abs_root_path, os.path.basename(abs_model_path))
    modelname = os.path.basename(abs_model_path)
    abs_modelmeta_path = os.path.join(abs_root_path, "%s_%s" % (modelname, "model_meta"))

    try:
        with open(abs_modelmeta_path, "r") as f:
            model_meta_map = json.load(f)
    except Exception as e:
        raise_log(ValueError("failed to open file: %s, err: %s" % (abs_modelmeta_path, str(e))))

    modelmeta_key_ancestor_classname_set = "ancestor_classname_set"
    modelmeta_key_modulename = "modulename"
    missed_keys = {modelmeta_key_ancestor_classname_set, modelmeta_key_modulename} - model_meta_map.keys()
    raise_if(
        len(missed_keys) > 0,
        "unable to get meta info %s, file path: %s, content: %s" % (missed_keys, abs_modelmeta_path, model_meta_map)
    )

    # (currently deprecated) if PaddleBaseModel.__name__ in model_meta_map["ancestor_classname_set"]
    if "PaddleBaseModel" in model_meta_map[modelmeta_key_ancestor_classname_set]:
        # lazy import
        from paddlets.models.dl.paddlepaddle.paddle_base import PaddleBaseModel
        return PaddleBaseModel.load(abs_model_path)
    raise_log(ValueError(
        "The given model class is not supported: %s.%s" %
        (
            model_meta_map[modelmeta_key_modulename],
            # model_meta_map["ancestor_classname_set"] = [child, parent, grandparent, ancestor]
            model_meta_map[modelmeta_key_ancestor_classname_set][0]
        )
    ))
