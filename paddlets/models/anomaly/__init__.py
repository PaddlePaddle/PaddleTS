# !/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
paddlets anomaly.
"""

from paddlets.models.anomaly.dl.autoencoder import AutoEncoder
from paddlets.models.anomaly.dl.anomaly_transformer import AnomalyTransformer
from paddlets.models.anomaly.dl.vae import VAE
from paddlets.models.anomaly.dl.usad import USAD
from paddlets.models.anomaly.dl.mtad_gat import MTADGAT
from paddlets.models.anomaly.dl.timesnet_ad import TimesNet_AD
from paddlets.models.anomaly.dl.dlinear_ad import DLinear_AD
from paddlets.models.anomaly.dl.rlinear_ad import RLinear_AD
from paddlets.models.anomaly.dl.nlinear_ad import NLinear_AD
from paddlets.models.anomaly.dl.patchtst_ad import PatchTST_AD
from paddlets.models.anomaly.dl.nstransformer_ad import NonStationary_AD