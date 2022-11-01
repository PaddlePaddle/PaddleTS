# !/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
analysis
"""
from paddlets.analysis.base_analyzers import Summary, Max
from paddlets.analysis.base_analyzers import summary, max
from paddlets.analysis.time_domain import Seasonality
from paddlets.analysis.time_domain import Acf
from paddlets.analysis.time_domain import Correlation
from paddlets.analysis.frequency_domain import FFT
from paddlets.analysis.frequency_domain import STFT
from paddlets.analysis.frequency_domain import CWT
from paddlets.analysis.analysis_report import AnalysisReport


#TSDataset内置的分析算子实例
#不是所有的analysis算子都可以集成到TSDataset中，要考虑函数的复杂程度以及第三方库引用
TSDataset_Inner_Analyzer = {
    "summary": summary,
    "max": max,
}
