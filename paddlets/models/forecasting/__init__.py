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
from paddlets.models.forecasting.dl.NLinear import NLinearModel
from paddlets.models.forecasting.dl.DLinear import DLinearModel
from paddlets.models.forecasting.dl.PatchTST import PatchTSTModel
from paddlets.models.forecasting.dl.RLinear import RLinearModel
from paddlets.models.forecasting.dl.Nonstationary import Nonstationary_Transformer
from paddlets.models.forecasting.dl.Crossformer import Crossformer
from paddlets.models.forecasting.dl.TiDE import TiDE
