# !/usr/bin/env python3
# -*- coding:utf-8 -*-

from typing import List, Dict, Optional, Tuple

import pandas as pd
import numpy as np
from paddle.io import Dataset as PaddleDataset

from paddlets.logger import Logger
from paddlets.datasets.tsdataset import TSDataset
from paddlets.logger.logger import raise_if, raise_log

logger = Logger(__name__)


class PaddleDsFromDf(PaddleDataset):
    """
    Convert dataframe to Paddle loader.
        
    Args:
        df(pd.DataFrame): Data to be converted.
        in_chunk_len(int): The size of the loopback window, i.e. the number of time steps feed to the model.
        out_chunk_len(int): The size of the forecasting horizon, i.e. the number of time steps output by the model.
        skip_chunk_len(int): Optional, the number of time steps between in_chunk and out_chunk for a single sample. 
        freq(str): A string representing the Pandas DateTimeIndex's frequency.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        in_chunk_len: int = 1,
        out_chunk_len: int = 1,
        skip_chunk_len: int = 0,
        freq: str = '1H') -> None:

        super(PaddleDsFromDf).__init__()

        self._df = df
        self._freq = freq
        self._in_chunk_len = in_chunk_len
        self._out_chunk_len = out_chunk_len
        self._skip_chunk_len = skip_chunk_len
        self._samples = self._build_samples()

    def __len__(self) -> int:
        """Length of TimeSeries"""
        return len(self._samples)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Indexing operation.

        Args:
            idx(int): The data indice.

        Returns:
            Dict[str, np.ndarray]
        """
        return self._samples[idx]

    def _build_samples(self) -> List[Dict[str, np.ndarray]]:
        """Construction sample"""
        _samples = []
        
        target_set = set()
        observed_set = set()
        known_set = set()
        static_set = set()
        
        target_list = []
        observed_list = []
        known_list = []
        static_list = []
        # Make sure that the feature sequence is consistent with the tsdataset
        for col in self._df.columns:
            var_name = col.rsplit(':', 1)[0]
            var_type_prefix = col.rsplit(':', 1)[1]
            var_type = var_type_prefix.split('_', 1)[0]

            if (var_type == 'target'):
                if var_name not in target_set:
                    target_list.append(var_name)
                target_set.add(var_name) 
            elif (var_type == 'observed'):
                if var_name not in observed_set:
                    observed_list.append(var_name)
                observed_set.add(var_name)
            elif (var_type == 'known'):
                if var_name not in known_set:
                    known_list.append(var_name)
                known_set.add(var_name)
            elif (var_type == 'static'):
                if var_name not in static_set:
                    static_list.append(var_name)
                static_set.add(var_name)
                
        for i in range(self._df.shape[0]):
            df_test = self._df[i: i+1]

            # Target numpy
            np_target, _ = self._build_feature(-self._in_chunk_len + 1, 1, 'target', target_list, df_test)

            raise_if(
                len(np_target) == 0,
                "past target length should be > 0"
            )

            sample = {'past_target': np_target}

            # Future target
            np_future_target = np.zeros((1, self._out_chunk_len), np.float32).T

            if (len(np_future_target) > 0):
                sample['future_target'] = np_future_target

            # Observed numpy
            if observed_list:
                np_ob_num_sum, np_ob_cat_sum = self._build_feature(-self._in_chunk_len + 1, 1, 'observed', 
                                                                   observed_list, df_test)

                if (len(np_ob_cat_sum)):
                    sample['observed_cov_categorical'] = np_ob_cat_sum

                if (len(np_ob_num_sum)):
                    sample['observed_cov_numeric'] = np_ob_num_sum

            # Known numpy
            if known_list:
                np_known_num_sum, np_known_cat_sum = self._build_feature(-self._in_chunk_len + 1,
                    1, 'known', known_list, df_test)
                
                np_known_num_sum_future, np_known_cat_sum_future = self._build_feature(self._skip_chunk_len + 1,
                    self._out_chunk_len + 1 + self._skip_chunk_len, 'known', known_list, df_test)

                if (len(np_known_num_sum) > 0):
                    sample['known_cov_numeric'] = np_known_num_sum
                if (len(np_known_num_sum_future) > 0):
                    if'known_cov_numeric' in sample:
                        sample['known_cov_numeric'] = np.concatenate([sample['known_cov_numeric'], 
                                                                      np_known_num_sum_future], axis=0)
                    else:
                        sample['known_cov_numeric'] = np_known_num_sum_future

                if (len(np_known_cat_sum) > 0):
                    sample['known_cov_categorical'] = np_known_cat_sum
                if (len(np_known_cat_sum_future) > 0):
                    if 'known_cov_categorical' in sample:
                        sample['known_cov_categorical'] = np.concatenate([sample['known_cov_categorical'], 
                                                                          np_known_cat_sum_future], axis=0)
                    else:
                        sample['known_cov_categorical'] = np_known_cat_sum_future
                
            # Static numpy
            if static_list:
                np_static_num_sum, np_static_cat_sum = self._build_feature(-self._in_chunk_len + 1,
                    1, 'static', static_list, df_test)
                np_static_num_sum_future, np_static_cat_sum_future = self._build_feature(1 + self._skip_chunk_len,
                    self._out_chunk_len + 1 + self._skip_chunk_len, 'static', static_list, df_test)
                if (len(np_static_num_sum) > 0):
                    sample['static_cov_numeric'] = np_static_num_sum[0: 1, :]
                if (len(np_static_num_sum_future) > 0):
                    if 'static_cov_numeric' in sample:
                        sample['static_cov_numeric'] = np.concatenate([sample['static_cov_numeric'], 
                                                                       np_static_num_sum_future], axis=0)[0: 1, :]
                    else:
                        sample['static_cov_numeric'] = np_static_num_sum_future[0: 1, :]

                if (len(np_static_cat_sum) > 0):
                    sample['static_cov_categorical'] = np_static_cat_sum[0: 1, :]
                if (len(np_static_cat_sum_future) > 0):
                    if 'static_cov_categorical' in sample:
                        sample['static_cov_categorical'] = np.concatenate([sample['static_cov_categorical'], 
                                                                           np_static_cat_sum_future], axis=0)[0: 1, :]
                    else:
                        sample['static_cov_categorical'] = np_static_cat_sum_future[0: 1, :]

            _samples.append(sample)
        
        return _samples

    def _build_feature(self, 
                       index_from: int, 
                       index_to: int, 
                       prefix: str, 
                       cov_list: List[str], 
                       df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build features numpy from dataframe

        Args:
            index_from(int): start index of lag features
            index_to(int): end index of lag features
            prefix(str): prefix of feature
            cov_list(List[str]): Needed columns
            df(pd.DataFrame): original data.

        Returns:
            Tuple[np.ndarray, np.ndarray]
        """
        cat_np_list = []
        num_np_list = []
        np_cat_sum = np.array([])
        np_num_sum = np.array([])

        for cov_col_prefix in cov_list:
            cat_name_list = []
            num_name_list = []

            for i in range(index_from, index_to, 1):
                _cov_col_name = cov_col_prefix + ":{}_lag_{}".format(prefix, i)
                if np.issubdtype(df.dtypes[_cov_col_name], np.integer):
                    cat_name_list.append(_cov_col_name)
                elif np.issubdtype(df.dtypes[_cov_col_name], np.floating):
                    num_name_list.append(_cov_col_name)
                else:
                    raise_if(True, "can't support data type: {} of col:{}".format(df.dtypes[_cov_col_name], 
                                                                                  _cov_col_name))
            if cat_name_list:
                cat_np_list.append(df[cat_name_list].values)
            if num_name_list:
                num_np_list.append(df[num_name_list].values)
        if cat_np_list:
            np_cat_sum = np.concatenate(cat_np_list, axis=0).astype(np.int32).T
        if num_np_list:
            np_num_sum = np.concatenate(num_np_list, axis=0).astype(np.float32).T
        return np_num_sum, np_cat_sum


class DatasetWrapper(object):
    """
    Format conversion between pd.DataFrame and TSDataset.
        
    Args:
        df(pd.DataFrame): Data to be converted
        in_chunk_len(int): The size of the loopback window, i.e. the number of time steps feed to the model.
        out_chunk_len(int): The size of the forecasting horizon, i.e. the number of time steps output by the model.
        skip_chunk_len(int): Optional, the number of time steps between in_chunk and out_chunk for a single sample. 
        sampling_stride(int): Sampling intervals between two adjacent samples.
        freq(str): A string representing the Pandas DateTimeIndex's frequency.
    """
    def __init__(
        self,
        in_chunk_len: int = 1,
        out_chunk_len: int = 1,
        skip_chunk_len: int = 0,
        sampling_stride: int = 1,
        freq: str = '1H') -> None:

        self._in_chunk_len = in_chunk_len
        self._out_chunk_len = out_chunk_len
        self._skip_chunk_len = skip_chunk_len
        self._sampling_stride = sampling_stride
        self._freq = freq
        self.freq_time = pd.tseries.frequencies.to_offset(freq)
        self.target_cols = None
        self.observed_cols = None
        self.known_cols = None
        self.record_type = {}
        self.record_col = {}

        self._check_params()

    def _check_params(self) -> None:
        """"Parameter validity verification"""
        raise_if(
            self._in_chunk_len <= 0,
            "in_chunk_len should > 0"
        )

        raise_if(
            self._out_chunk_len <= 0,
            "out_chunk_len should > 0"
        )

        raise_if(
            self._skip_chunk_len < 0,
            "skip_chunk_len should >= 0"
        )

        raise_if(
            self._sampling_stride < 0,
            "sampling_stride should > 0"
        )

    def dataset_to_dataframe(self, ts: TSDataset) -> pd.DataFrame:
        """
        Convert TSDataset to pd.DataFrame
        
        Args:
            ts(TSDataset): original data.

        Returns:
            pd.DataFrame
        """
        df_list = []
        target = ts.get_target().to_dataframe()
        known = ts.get_known_cov().to_dataframe() if ts.get_known_cov() else pd.DataFrame()
        observed = ts.get_observed_cov().to_dataframe() if ts.get_observed_cov() else pd.DataFrame()
        static = ts.get_static_cov()
        self.record_type = ts.dtypes.to_dict()
        
        for i in range(0, self._in_chunk_len):
            df_list.append(target.shift(periods=i, axis=0).rename(
                columns=lambda x: f"{x}:target_lag_{-i}"
            ))
            if not observed.empty:
                df_list.append(observed.shift(periods=i, axis=0).rename(
                    columns=lambda x: f"{x}:observed_lag_{-i}"
                ))
            if not known.empty:
                df_list.append(known.shift(periods=i, axis=0).rename(
                    columns=lambda x: f"{x}:known_lag_{-i}"
                ))
        if not known.empty:
            for i in range(1, self._out_chunk_len + 1 + self._skip_chunk_len):
                df_list.append(known.shift(periods=-i, axis=0).rename(
                    columns=lambda x: f"{x}:known_lag_{i}"
                ))

        _df = pd.concat(df_list, axis=1, join='inner')
        
        if static:
            for i in range(-self._in_chunk_len + 1, self._out_chunk_len + self._skip_chunk_len + 1):
                for col in static:
                    _df["%s:static_lag_%d" % (col, i)] = int(static[col]) \
                                                         if isinstance(static[col], int) else static[col]
                    self.record_type[col] = type(static[col])
        keep_list = [ind for x, ind in enumerate(target.index[:: -1]) if (x % self._sampling_stride == 0) \
                     and x <= target.shape[0] - self._in_chunk_len]
        keep_list = keep_list[:: -1]
        
        _df = _df.loc[keep_list, :]
        return _df

    def dataframe_to_paddledsfromdf(self, df: pd.DataFrame) -> PaddleDsFromDf:
        """
        Convert pd.DataFrame to Paddle loader
        
        Args:
            df(pd.DataFrame): data to be converted.

        Returns:
            PaddleDsFromDf object.
        """
        # Unified data type
        if self.record_type:
            for col in df.columns:
                if col.rsplit(':')[0] in self.record_type:
                    df[col] = df[col].astype(self.record_type[col.rsplit(':')[0]])
        return PaddleDsFromDf(df, self._in_chunk_len, self._out_chunk_len, self._skip_chunk_len, self._freq)
    
    def dataframe_to_ts(self, df: pd.DataFrame) -> List[TSDataset]:
        """
        Convert pd.DataFrame to TSDataset
        
        Args:
            df(pd.DataFrame): data to be converted.

        Returns:
            List[TSDataset].
        """
        target_set = set()
        observed_set = set()
        known_set = set()
        static_set = set()
        
        target_list = []
        observed_list = []
        known_list = []
        static_list = []
        # Make sure that the feature sequence is consistent with the tsdataset
        for col in df.columns:
            var_name = col.rsplit(':', 1)[0]
            var_type_prefix = col.rsplit(':', 1)[1]
            var_type = var_type_prefix.split('_', 1)[0]

            if (var_type == 'target'):
                if var_name not in target_set:
                    target_list.append(var_name)
                target_set.add(var_name) 
            elif (var_type == 'observed'):
                if var_name not in observed_set:
                    observed_list.append(var_name)
                observed_set.add(var_name)
            elif (var_type == 'known'):
                if var_name not in known_set:
                    known_list.append(var_name)
                known_set.add(var_name)
            elif (var_type == 'static'):
                if var_name not in static_set:
                    static_list.append(var_name)
                static_set.add(var_name)

        sample_num = df.shape[0]
        tss = []
        for sample_index in range(sample_num):
            pred_time = df.index[sample_index]
            dicts = {'target': {}, 'observed': {}, 'known': {}, 'static': {}}
            for col_index, col in enumerate(df.columns):
                name, col = col.rsplit(':', 1)
                format_, col = col.split('_', 1)
                index = int(col.rsplit('_', 1)[1])
                if format_ not in dicts:
                    continue
                if name not in dicts[format_]:
                    if not dicts[format_]:
                        dicts[format_] = {}
                    dicts[format_][name] = pd.Series([df.iloc[sample_index, col_index]], 
                                                     index=[pred_time + self.freq_time * index])
                else:
                    dicts[format_][name][pred_time + self.freq_time * index] = df.iloc[sample_index, col_index]

            target_dataframe = pd.DataFrame(dicts['target'])
            observed_dataframe = pd.DataFrame(dicts['observed'])
            known_dataframe = pd.DataFrame(dicts['known'])
            static_dataframe = pd.DataFrame(dicts['static'])

            dataframe = target_dataframe
            if not observed_dataframe.empty:
                dataframe = pd.concat([dataframe, observed_dataframe], axis=1)
            if not known_dataframe.empty:
                dataframe = pd.concat([dataframe, known_dataframe], axis=1)
            if not static_dataframe.empty:
                dataframe = pd.concat([dataframe, static_dataframe], axis=1)
            ts = TSDataset.load_from_dataframe(dataframe, target_cols=target_list, observed_cov_cols=observed_list, 
                                               known_cov_cols=known_list, static_cov_cols=static_list, 
                                               drop_tail_nan=True)
            # Unified data type
            ts.astype(self.record_type)
            tss.append(ts)
        return tss
