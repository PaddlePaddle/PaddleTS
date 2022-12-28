# !/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
paddlets models forecasting.
"""

from paddlets.models.forecasting.dl.mlp import MLPRegressor
from paddlets.models.forecasting.dl.rnn import RNNBlockRegressor
from paddlets.models.forecasting.dl.lstnet import LSTNetRegressor
from paddlets.models.forecasting.dl.nbeats import NBEATSModel
from paddlets.models.forecasting.dl.tcn import TCNRegressor
from paddlets.models.forecasting.dl.nhits import NHiTSModel
from paddlets.models.forecasting.dl.transformer import TransformerModel
from paddlets.models.forecasting.dl.informer import InformerModel
from paddlets.models.forecasting.dl.deepar import DeepARModel
from paddlets.models.forecasting.dl.tft import TFTModel
from paddlets.models.forecasting.dl.scinet import SCINetModel

