# !/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
Implementation of different frequency domain and time-frequency domain analysis operators, including FFT, STFT, CWT.
"""

from typing import Any, Callable, List, Optional, Sequence, Tuple, Union, Dict

import numpy as np
import pandas as pd
from scipy.fftpack import fft, fftfreq
from scipy.signal import stft
from pywt import cwt
import matplotlib.pyplot as plt

from paddlets import TimeSeries, TSDataset
from paddlets.logger import Logger, raise_if_not, raise_if, raise_log
from paddlets.analysis.base import Analyzer

logger = Logger(__name__)


class FFT(Analyzer):
    """
    Fast Fourier transform (FFT) analysis operator performs FFT on a signal(1-D) to 
    obtain the amplitude spectrum and phase spectrum of the signal at different frequencies.

    This operator returns both X and Y coordinates for visual display. 
    Where X represents frequency and Y represents amplitude spectrum or phase spectrum.

    Args:
        fs(float): The sampling frequency of the signal(default=0), when the fs is not specified, 
            the range of x coordinate defaults to the length of the data.
            If the data to be analyzed is multiple columns, the default frequency of all columns is the same.
            If the fs of different columns is different, it is recommended to call this operator separately for each column.
        norm(bool): Whether to normalize the amplitude or phase after FFT transformation, default=True.
        half(bool): Whether to take half of the amplitude or phase after FFT transform(when the signal is a real signal,
            its frequency domain signal after Fourier transform is symmetrical about the 0-frequency axis), default=True.
        kwargs: Other parameters.

    Returns:
        None
    """

    def __init__(self, fs: float = 0, norm: bool = True, half: bool = True, **kwargs):
        super(FFT, self).__init__(**kwargs)
        self._fs = fs
        self._norm = norm
        self._half = half
        if self._fs == 0:
            logger.warning("It's suggested to assign a positive number to the fs parameter.")
        self._columns = []


    def analyze(
        self,
        X: Union[pd.Series, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Implementation logic of fast Fourier transform analysis operator

        Args:
            X(pd.Series|pd.DataFrame): X to be analyzed

        Returns:
            fft_result(DataFrame): The result of fft, each column to be analyzed returns 3 keys 

        Examples:
            .. code-block:: python

                #Given:
                X = pd.DataFrame(np.array(range(100)), columns=['a'])
                #Returns:
                1. 'a_x': np.ndarray(1-D), Representative frequency),
                2. 'a_amplitude': np.ndarray(1-D), Representative cwt of amplitude spectrum
                3. 'a_phase': np.ndarray(1-D), Representative phase spectrum
        
        Raise:
            ValueError
        """

        def _compute_fft(col: pd.Series):
            """
            Calculation process of fast Fourier transform analysis operator

            Args:
                col(pd.Series): X to be calculation

            Return:
                x(np.ndarray): Abscissa (indicating frequency range)
                col_amplitude(np.ndarray): amplitude
                col_phase(np.ndarray): phase
            """
            col_len = len(col)
            #Do fast Fourier transform
            col_fft = fft(col.values)
            #Take the absolute value of the complex number to get the amplitude spectrum
            col_amplitude = np.abs(col_fft)
            #Take the angle of the complex number to get the phase spectrum
            col_phase = np.angle(col_fft)
            #Get abscissa
            x = np.arange(col_len)
            if self._fs != 0:
                x = fftfreq(col_len, 1.0 / self._fs)
            #Normalized or not
            if self._norm:
                col_amplitude = col_amplitude / col_len
                col_phase = col_phase / col_len
            #Whether to take half of the result
            if self._half:
                data_range = range(int(col_len / 2))
                col_amplitude = col_amplitude[data_range]
                col_phase = col_phase[data_range]
                x = x[data_range]

            return x, col_amplitude, col_phase

        if isinstance(X, pd.Series):
            if X.name is None:
                logger.warning("The column name will be set to '0'.")
            X = X.to_frame()
        if isinstance(X, pd.DataFrame):
            fft_dict = {}
            for col_name in X:
                self._columns.append(col_name)
                col = X[col_name]
                #Skip columns that are not numerical
                if not (np.issubdtype(col.dtype, np.integer) or np.issubdtype(col.dtype, np.floating)):
                    logger.warning("The values in the column %s should be numerical." % (col_name))
                    continue
                x, col_amplitude, col_phase = _compute_fft(col)
                col_name = str(col_name)
                fft_dict[col_name + '_x'] = x
                fft_dict[col_name + '_amplitude'] = col_amplitude
                fft_dict[col_name + '_phase'] = col_phase

            if len(fft_dict) == 0:
                raise_log(ValueError("All the values in the columns are invalid, please check the data."))
            
            return pd.DataFrame(fft_dict)
        else:
            raise_log(ValueError("The data format must be pd.Series or pd.DataFrame."))

    @classmethod
    def get_properties(cls) -> Dict:
        """
        Get the properties of the analyzer.
        """
        return {
            "name": "fft",
            "report_heading": "FFT",
            "report_description": "Frequency domain analysis of signal based on fast Fourier transform."
        }

    def plot(self) -> "pyplot":
        """
        display fft result.

        Args:
            None

        Returns:
            plt(matplotlib.pyplot object): The fft figure

        Raise:
            None
        """
        columns_num = len(self._columns)
        fig, ax = plt.subplots(columns_num, 1, squeeze=False)
        for i in range(0, columns_num):
            col_name = self._columns[i]
            x = self._res[col_name + '_x'].tolist()
            y = self._res[col_name + '_amplitude'].tolist()
            ax[i, 0].plot(x, y)
            ax[i, 0].set_title(col_name + ' FFT Magnitude')
            ax[i, 0].set_xlabel('Frequency')
            ax[i, 0].set_ylabel('Amplitude')
        
        plt.tight_layout()
        
        return plt


class STFT(Analyzer):
    """
    Short time Fourier transform (STFT) is used to analyze non-stationary signals because the waveform 
    of non-stationary signal changes irregularly and there is no concept of instantaneous frequency. 
    In this case, the effect of using fast Fourier transform analysis is poor.

    In STFT, the windowing mechanism is used to stabilize the signal(truncated in time, so that the 
    waveform does not change significantly in a short time), and then FFT can be used for windowed signal segmentation. 
    STFT obtains the spectrum of n-segment signals arranged in time sequence.

    The length of the window determines the time resolution and frequency resolution of the spectrum. 
    The longer the window is, the higher the frequency resolution is, the lower the time resolution is. 
    Conversely, the lower the frequency resolution is, the higher the time resolution is. 
    For time-varying unsteady signals, high frequency is suitable for small windows, and low frequency is suitable for large windows.

    In order to improve the time-domain characteristics on the basis of ensuring the frequency-domain characteristics, 
    Often choose to overlap between segments to improve the time-domain analysis ability. However, 
    the more overlapping points will greatly increase the amount of calculation, resulting in low efficiency.

    Args:
        fs(float): The sampling frequency of the signal(default=1.0).
            If the data to be analyzed is multiple columns, the default frequency of all columns is the same. 
            If the fs of different columns is different, it is recommended to call this operator separately for each column.
        window(str|tuple|array_like): Desired window to use, default="hann".
        nperseg(int): Length of each segment, default=256.
        noverlap(None|int): Number of points to overlap between segments. If None, noverlap = nperseg // 2, default=None
        nfft(None|int): Length of the FFT used, if a zero padded FFT is desired. If None, the FFT length is nperseg, default=None.
        detrend(str|function|False): Specifies how to detrend each segment, default=False.
        return_onesided(bool): If True, return a one-sided spectrum for real data. If False return a two-sided spectrum. default=True.
        boundary(None|str): Specifies whether the input signal is extended at both ends, and how to generate the new values, default='zeros'.
        padded(bool): Specifies whether the input signal is zero-padded at the end to make the signal 
            fit exactly into an integer number of window segments, default=True.
        axis(int): Axis along which the STFT is computed, default=-1.
        kwargs: Other parameters.
    For more details about parameters, please refer to: 
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.stft.html?highlight=stft#scipy.signal.stft

    Returns:
        None
    """

    def __init__(
        self,
        fs: float = 1.0,
        window: Union[str, Tuple[str], List[str]] = 'hann',
        nperseg: int = 256,
        noverlap: Union[None, int] = None,
        nfft: Union[None, int] = None,
        detrend: Union[str, bool] = False,
        return_onesided: bool = True,
        boundary: Union[str, None] = 'zeros',
        padded: bool = True,
        axis: int = -1,
        **kwargs
        ):
        super(STFT, self).__init__(**kwargs)
        self._fs = fs
        self._window = window
        self._nperseg = nperseg
        self._noverlap = noverlap
        self._nfft = nfft
        self._detrend = detrend
        self._return_onesided = return_onesided
        self._boundary = boundary
        self._padded = padded
        self._axis = axis
        if self._nperseg < 0:
            raise_log(ValueError("nperseg must be a positive integer."))
        if self._fs == 0:
            raise_log(ValueError("fs parameter can't be 0."))
        self._columns = []

    def analyze(
        self,
        X: Union[pd.Series, pd.DataFrame]
    ) -> Dict:
        """
        Implementation logic of short-time Fourier transform analysis operator

        Args:
            X(pd.Series|pd.DataFrame): X to be analyzed

        Returns:
            stft_result(Dict): The result of stft, each column to be analyzed returns 3 keys 
        
        Examples:
            .. code-block:: python

                #Given:
                X = pd.DataFrame(np.array(range(100)), columns=['a'])
                #Returns:
                1. 'a_f': np.ndarray(1-D), Representative frequency
                2. 'a_t': np.ndarray(1-D), Representative time
                3. 'a_Zxx': np.ndarray(2-D), Representative stft of x

        Raise:
            ValueError
        """

        def _compute_stft(col: pd.Series):
            """
            Calculation process of short-time Fourier transform analysis

            Args:
                col[pd.Series]: X to be calculation

            Return:
                f(np.ndarray): Array of sample frequencies.
                t[np.ndarray]: Array of segment times.
                Zxx[np.ndarray]: STFT of x. By default, the last axis of Zxx corresponds to the segment times.
            """
            col_len = len(col)
            #It is best to ensure that the set nperseg is less than the length of the data
            if col_len < self._nperseg:
                logger.warning("nperseg = %s is greater than input length = %s, please using nperseg < %s." % (self._nperseg, col_len, col_len))
            f, t, Zxx = stft(col.values, fs=self._fs, window=self._window, nperseg=self._nperseg, noverlap=self._noverlap,\
                    nfft=self._nfft, detrend=self._detrend, return_onesided=self._return_onesided, boundary=self._boundary,\
                    padded=self._padded, axis=self._axis)

            return f, t, Zxx

        if isinstance(X, pd.Series):
            if X.name is None:
                logger.warning("The column name will be set to '0'.")
            X = X.to_frame()
        if isinstance(X, pd.DataFrame):
            stft_dict = {}
            for col_name in X:
                col = X[col_name]
                self._columns.append(col_name)
                #Skip columns that are not numerical
                if not (np.issubdtype(col.dtype, np.integer) or np.issubdtype(col.dtype, np.floating)):
                    logger.warning("The values in the column %s should be numerical." % (col_name))
                    continue
                f, t, Zxx = _compute_stft(col)
                col_name = str(col_name)
                stft_dict[col_name + '_f'] = f
                stft_dict[col_name + '_t'] = t
                stft_dict[col_name + '_Zxx'] = Zxx

            if len(stft_dict) == 0:
                raise_log(ValueError("All the values in the columns are invalid, please check the data."))

            return stft_dict
        else:
            raise_log(ValueError("The data format must be pd.Series or pd.DataFrame."))

    @classmethod
    def get_properties(cls) -> Dict:
        """
        Get the properties of the analyzer.
        """
        return {
            "name": "stft",
            "report_heading": "STFT",
            "report_description": "Time-frequency analysis of signal based on short-time Fourier transform."
        }
    
    def plot(self) -> "pyplot":
        """
        display stft result.

        Args:
            None

        Returns:
            plt(matplotlib.pyplot object): The stft figure

        Raise:
            None
        """
        columns_num = len(self._columns)
        fig, ax = plt.subplots(columns_num, 1, squeeze=False)
        for i in range(0, columns_num):
            col_name = self._columns[i]
            f = self._res[col_name + '_f'].tolist()
            t = self._res[col_name + '_t'].tolist()
            Zxx = self._res[col_name + '_Zxx'].tolist()
            ax[i, 0].pcolormesh(t, f, np.abs(Zxx), shading='auto')
            ax[i, 0].set_title(col_name + ' STFT Magnitude')
            ax[i, 0].set_xlabel('Time')
            ax[i, 0].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        return plt


class CWT(Analyzer):
    """
    Fast Fourier transform (FFT) can only get the frequency domain information of a signal, 
    and it is impossible to know the frequency information of the signal at different times. 
    Short time Fourier transform (STFT) solves this problem to a certain extent by introducing the windowing mechanism, 
    but the effect of STFT is affected by the size of the window. The window is too small, the frequency resolution is low, 
    the window is too large, and the time resolution is low.

    Continuous wavelet transform(CWT) can solve the above problems. 
    It inherits and develops the idea of localization of short-time Fourier transform, 
    and overcomes the shortcomings that the window size does not change with frequency. 
    It can provide a "time-frequency" window that changes with frequency. It is an ideal tool for signal time-frequency analysis and processing.

    Args:
        scales(int): The wavelet scales to use, It can be set to half the length of the data.
            It should be noted that when the half of the data is relatively large (such as greater than 1000),
            the larger wavelet scale will cause the calculation to be more time-consuming and you can change 
            the wavelet scale into a value of 100, 200 or other small number.
        wavelet(str): Wavelet to use, options include ['cgau1', 'cgau2', 'cgau3', 'cgau4', 'cgau5', 'cgau6', 'cgau7', 'cgau8', 'cmor', 'fbsp', 'gaus1',
            'gaus2', 'gaus3', 'gaus4', 'gaus5', 'gaus6', 'gaus7', 'gaus8', 'mexh', 'morl', 'shan'], default='cgau8'.
        fs(float): The sampling frequency of the signal(default=1.0).
            If the data to be analyzed is multiple columns, the default frequency of all columns are considered  the same.
            If the fs of different columns is different, it is recommended to call this operator separately for each column.
        method(str): The method used to compute the CWT, options include ['conv', 'fft', 'auto'], default='conv'.
        axis(int): Axis along which the STFT is computed, default=-1.
        kwargs: Other parameters.
    For more details about parameters, please refer to: https://pywavelets.readthedocs.io/en/latest/ref/cwt.html?highlight=cwt

    Returns:
        None
    """

    def __init__(
        self,
        scales: int = 64,
        wavelet: str = 'cgau8',
        fs: float = 1.0,
        method: str = 'conv',
        axis: int = -1,
        **kwargs
        ):
        super(CWT, self).__init__(**kwargs)
        self._scales = scales
        self._wavelet = wavelet
        self._fs = fs
        self._method = method
        self._axis = axis
        #Scales cannot be less than 2
        if self._scales < 2:
            raise_log(ValueError("scales must be greater than 1."))
        if self._fs == 0:
            raise_log(ValueError("fs parameter can't be 0."))
        self._columns = []

    def analyze(
        self,
        X: Union[pd.Series, pd.DataFrame]
    ) -> Dict:
        """
        Implementation logic of continuous wavelet transform

        Args:
            X(pd.Series|pd.DataFrame): X to be analyzed

        Returns:
            cwt_result(Dict): The result of cwt, each column to be analyzed returns 3 keys
        
        Examples:
            .. code-block:: python

                #Given:
                X = pd.DataFrame(np.array(range(100)), columns=['a'])
                #Returns:
                1. 'a_t': np.ndarray(1-D), Representative time
                2. 'a_coefs': np.ndarray(2-D), Representative cwt of x
                3. 'a_frequencies': np.ndarray(1-D), Representative frequency

        Raise:
            ValueError
        """

        def _compute_cwt(col: pd.Series):
            """
            Calculation process of continuous wavelet transform

            Args:
                col[pd.Series]: X to be calculation

            Return:
                t[np.ndarray]: time
                coefs[np.ndarray]: the result of Continuous wavelet transform
                frequencies[np.ndarray]: frequency
            """
            col_len = len(col)
            scales_list = np.arange(1, self._scales)
            coefs, frequencies = cwt(col.values, scales=scales_list, wavelet=self._wavelet, sampling_period=1.0/self._fs, method=self._method, axis=self._axis)
            #Time coordinate
            t = np.linspace(0, 1, col_len, endpoint=False)

            return t, coefs, frequencies

        if isinstance(X, pd.Series):
            if X.name is None:
                logger.warning("The column name will be set to '0'.")
            X = X.to_frame()
        if isinstance(X, pd.DataFrame):
            cwt_dict = {}
            for col_name in X:
                self._columns.append(col_name)
                col = X[col_name]
                #Skip columns that are not numerical
                if not (np.issubdtype(col.dtype, np.integer) or np.issubdtype(col.dtype, np.floating)):
                    logger.warning("The values in the column %s should be numerical." % (col_name))
                    continue
                t, coefs, frequencies = _compute_cwt(col)
                col_name = str(col_name)
                cwt_dict[col_name + '_t'] = t
                cwt_dict[col_name + '_coefs'] = coefs
                cwt_dict[col_name + '_frequencies'] = frequencies

            if len(cwt_dict) == 0:
                raise_log(ValueError("All the values in the columns are invalid, please check the data."))

            return cwt_dict
        else:
            raise_log(ValueError("The data format must be pd.Series or pd.DataFrame."))

    @classmethod
    def get_properties(cls) -> Dict:
        """
        Get the properties of the analyzer.
        """
        return {
            "name": "cwt",
            "report_heading": "CWT",
            "report_description": "Time-frequency analysis of signal based on continuous wavelet transform."
        }
    
    def plot(self) -> "pyplot":
        """
        display cwt result.

        Args:
            None

        Returns:
            plt(matplotlib.pyplot object): The cwt figure

        Raise:
            None
        """
        columns_num = len(self._columns)
        fig, ax = plt.subplots(columns_num, 1, squeeze=False)
        for i in range(0, columns_num):
            col_name = self._columns[i]
            f = self._res[col_name + '_frequencies'].tolist()
            t = self._res[col_name + '_t'].tolist()
            coefs = self._res[col_name + '_coefs'].tolist()
            ax[i, 0].contourf(t, f, np.abs(coefs))
            ax[i, 0].set_title(col_name + ' CWT Magnitude')
            ax[i, 0].set_xlabel('Time')
            ax[i, 0].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        return plt
