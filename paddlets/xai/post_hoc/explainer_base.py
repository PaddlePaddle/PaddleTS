# !/usr/bin/env python3
# -*- coding:utf-8 -*-
from abc import ABC, abstractmethod
from typing import List

import shap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class BaseExplainer(ABC):
    @abstractmethod
    def __init__(self) -> None:
        """
        Initialization
        """
        pass
    
    @abstractmethod
    def explain(self) -> np.ndarray:
        """
        Calculate the explanatory value of the test sample.
        """
        pass
        
    @abstractmethod
    def plot(self) -> 'plt':
        """
        Display the explanatory value.
        """
        pass
    
    def _out_vs_in_plot(self, 
                        attribution: np.ndarray,
                        out_cols: List[int], 
                        in_cols: List[int],
                        unique_cols: List[str],
                        **kwargs) -> None:
        """
        Display the attribution of each input time step for each output time step. Such as: 
        
            out_cols[0] |       0.1      |         0.4      |      0.2      
            ---------------------------------------------------------------
            out_cols[1] |       0.2      |         0.2      |      0.1     
            ---------------------------------------------------------------
            out_cols[2] |       0.6      |         0.3      |      0.9     
            ---------------------------------------------------------------
                        |   in_cols[0]   |    in_cols[1]    |    in_cols[2] 
        
        Args:
            attribution(np.ndarray): explanation value(samples * time steps * feature).
            out_cols(List[int]): the time points of output.
            in_cols(List[int]): the time points of input.
            unique_cols(List[str]): feature names.
            kwargs: Optionally, additional keyword arguments passed to `sns.heatmap`.
            
        Returns:
            None
        """
        values_np = attribution.sum(axis=2)
        values_df = pd.DataFrame(values_np, index=out_cols, columns=in_cols)

        figsize = kwargs.get('figsize', (40, 10))
        if 'figsize' in kwargs:
            del kwargs['figsize']
        plt.figure(figsize=figsize)
        g = sns.heatmap(values_df, square=True, annot=True, fmt='0.2f', linewidths=1, **kwargs)
        plt.xlabel('in_chunk_len')
        plt.ylabel('out_chunk_len')
        plt.title('OI')
        plt.show()
        
    def _out_vs_feature_plot(self, 
                              attribution: np.ndarray,
                              out_cols: List[int], 
                              in_cols: List[int],
                              unique_cols: List[str],
                              **kwargs) -> None:
        """
        Display the attribution of each feature for each output time step. Such as:
        
            out_cols[0]   |      0.1       |       0.4         |         0.2     
            ------------------------------------------------------------------------
            out_cols[1]   |      0.2       |       0.2         |         0.1    
            ------------------------------------------------------------------------
            out_cols[2]   |      0.6       |       0.3         |         0.9    
            ------------------------------------------------------------------------
                          | unique_cols[0] |   unique_cols[1]  |     unique_cols[2]  
        
        Args:
            attribution(np.ndarray): explanation value(samples * time steps * feature).
            out_cols(List[int]): the time points of output.
            in_cols(List[int]): the time points of input.
            unique_cols(List[str]): feature names.
            kwargs: Optionally, additional keyword arguments passed to `sns.heatmap`.
            
        Returns:
            None
        """
        values_np = attribution.sum(axis=1)
        values_df = pd.DataFrame(values_np, index=out_cols, columns=unique_cols).transpose()
        
        figsize = kwargs.get('figsize', (30, 10))
        if 'figsize' in kwargs:
            del kwargs['figsize']
        plt.figure(figsize=figsize)
        g = sns.heatmap(values_df, square=True, annot=True, fmt='0.2f', linewidths=1, **kwargs)
        plt.xlabel('out_chunk_len')
        plt.ylabel('feature')
        plt.title('OV')
        plt.show()
        
    def _in_vs_feature_plot(self, 
                             attribution: np.ndarray,
                             out_cols: List[int], 
                             in_cols: List[int],
                             unique_cols: List[str],
                             **kwargs) -> None:
        """
        Display the attribution of each input time step for each feature. Such as:
        
          unique_cols[0] |   0.1      |      0.4      |       0.2    
          ------------------------------------------------------------
          unique_cols[1] |   0.2      |      0.2      |       0.1      
          ------------------------------------------------------------
          unique_cols[2] |   0.6      |      0.3      |       0.9     
          ------------------------------------------------------------
                         | in_cols[0] |  in_cols[1]   |    in_cols[2] 
        
        Args:
            attribution(np.ndarray): explanation value(samples * time steps * feature).
            out_cols(List[int]): the time points of output.
            in_cols(List[int]): the time points of input.
            unique_cols(List[str]): feature names.
            kwargs: Optionally, additional keyword arguments passed to `sns.heatmap`.
            
        Returns:
            None
        """
        values_np = attribution.sum(axis=0)
        values_df = pd.DataFrame(values_np, index=in_cols, columns=unique_cols).transpose()
        
        figsize = kwargs.get('figsize', (30, 10))
        if 'figsize' in kwargs:
            del kwargs['figsize']
        plt.figure(figsize=figsize)
        g = sns.heatmap(values_df, square=True, annot=True, fmt='0.2f', linewidths=1, **kwargs)
        plt.xlabel('in_chunk_len')
        plt.ylabel('feature')
        plt.title('IV')
        plt.show()
        
    def _in_plot(self, 
                 attribution: np.ndarray,
                 out_cols: List[int], 
                 in_cols: List[int],
                 unique_cols: List[str],
                 **kwargs) -> None:
        """
        Display the attribution of each input time step. Such as:
        
                   |     0.6       |      0.3       |     0.9   
            ------------------------------------------------------
                   | in_cols[0]    |   in_cols[1]   |   in_cols[2]     
        
        Args:
            attribution(np.ndarray): explanation value(samples * time steps * feature).
            out_cols(List[int]): the time points of output.
            in_cols(List[int]): the time points of input.
            unique_cols(List[str]): feature names.
            kwargs: Optionally, additional keyword arguments passed to `shap.summary_plot`.
            
        Returns:
            None
        """
        values_np = abs(attribution.sum(axis=0).sum(axis=1))
        
        kwargs['show'] = False
        kwargs['plot_type'] = 'bar' if not 'plot_type' in kwargs else kwargs['plot_type']
        shap.summary_plot(values_np.reshape(1, -1), in_cols, **kwargs)
        plt.ylabel('in_chunk_len')
        plt.title('I')
        plt.show()
        
    def _feature_plot(self, 
                       attribution: np.ndarray,
                       out_cols: List[int], 
                       in_cols: List[int],
                       unique_cols: List[str],
                       **kwargs) -> None:
        """
        Display the attribution of each feature. Such as:
        
                   |     0.6         |       0.3         |       0.9   
            ------------------------------------------------------------------
                   | unique_cols[0]  |  unique_cols[1]   |  unique_cols[2]   
        
        Args:
            attribution(np.ndarray): explanation value(samples * time steps * feature).
            out_cols(List[int]): the time points of output.
            in_cols(List[int]): the time points of input.
            unique_cols(List[str]): feature names.
            kwargs: Optionally, additional keyword arguments passed to `shap.summary_plot`.
            
        Returns:
            None
        """
        values_np = attribution.sum(axis=0).sum(axis=0)

        kwargs['show'] = False
        kwargs['plot_type'] = 'bar' if not 'plot_type' in kwargs else kwargs['plot_type']
        shap.summary_plot(values_np.reshape(1, -1), unique_cols, **kwargs)
        plt.title('V')
        plt.show()