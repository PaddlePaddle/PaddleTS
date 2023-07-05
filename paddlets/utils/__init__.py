# !/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
    utils
"""
from paddlets.utils.utils import get_uuid
from paddlets.utils.utils import check_model_fitted, check_train_valid_continuity, split_dataset
from paddlets.utils.backtest import backtest
from paddlets.utils.validation import cross_validate, fit_and_score
from paddlets.utils.utils import plot_anoms