# !/usr/bin/env python3
# -*- coding:utf-8 -*-

import unittest
import pandas as pd
import numpy as np
import random

from paddlets import TSDataset, TimeSeries
from paddlets.models.utils import get_target_from_tsdataset, check_tsdataset, to_tsdataset
from paddlets.models.forecasting import RNNBlockRegressor


class TestUtils(unittest.TestCase):
    """
    Utils unittest
    """
    def __init__(self):
        """
        unittest setup
        """
        super(TestUtils, self).__init__()
        np.random.seed(2022)
        target = pd.Series(
                np.random.randn(200).astype(np.float16),
                index=pd.date_range("2022-01-01", periods=200, freq="15T", name='timestamp'),
                name="value")
        label = pd.Series(
                np.random.randint(0, 2, 200),
                index=pd.date_range("2022-01-01", periods=200, freq="15T", name='timestamp'),
                name="label")
        feature = pd.DataFrame(
                np.random.randn(200, 2).astype(np.float32),
                index=pd.date_range("2022-01-01", periods=200, freq="15T", name='timestamp'),
                columns=["a", "b"])
        
        # index is DatetimeIndex
        #for anomaly
        self.tsdataset1 = TSDataset.load_from_dataframe(pd.concat([label,feature],axis=1), 
                target_cols='label', feature_cols='a')
        self.tsdataset2 = TSDataset.load_from_dataframe(pd.concat([label,feature],axis=1), 
                feature_cols='a')
        # for forecasting
        self.tsdataset3 = TSDataset.load_from_dataframe(pd.concat([target,feature],axis=1), 
                target_cols='value', observed_cov_cols='a')
        
        # index is RangeIndex
        # for anomaly
        index = pd.RangeIndex(0, 200, 1)
        label = label.reset_index(drop=True).reindex(index)
        feature = feature.reset_index(drop=True).reindex(index)
        target = target.reset_index(drop=True).reindex(index)
        self.tsdataset4 = TSDataset.load_from_dataframe(pd.concat([label,feature],axis=1), 
                label_col='label', feature_cols='a')
        self.tsdataset5 = TSDataset.load_from_dataframe(pd.concat([label,feature],axis=1), 
                feature_cols='a')
         # for forecasting
        self.tsdataset6 = TSDataset.load_from_dataframe(pd.concat([target,feature],axis=1), 
                target_cols='value', observed_cov_cols='a')
        
        model = RNNBlockRegressor(
            in_chunk_len=16,
            out_chunk_len=2,
        )
        self.model = model
               
def utils_test_get_target_from_tsdataset(test_utils):
    """unittest function for get_target_from_tsdataset
    """
    with test_utils.assertLogs("paddlets", level="WARNING") as captured:
        tsdataset = get_target_from_tsdataset(test_utils.tsdataset1)
        test_utils.assertEqual(
            captured.records[0].getMessage(), 
            "covariant exists and will be filtered."
        )
        test_utils.assertEqual(tsdataset.get_observed_cov(), None)
        test_utils.assertNotEqual(tsdataset.get_target(), None)

def utils_test_check_tsdataset(test_utils):
    """unittest function for check_tsdataset
    """
    # case 1: test float
    tsdataset = test_utils.tsdataset3.copy()
    check_tsdataset(tsdataset)
    test_utils.assertEqual(tsdataset.get_target().data['value'].dtype, np.dtype('float32'))
    # case 2: test int
    tsdataset = test_utils.tsdataset1.copy()
    check_tsdataset(tsdataset)
    test_utils.assertEqual(tsdataset.get_target().data['label'].dtype, np.dtype('int64'))
    # case 3: test nan
    tsdataset = test_utils.tsdataset3.copy()
    tsdataset["value"][0] = np.NaN
    with test_utils.assertLogs("paddlets", level="WARNING") as captured:
            check_tsdataset(tsdataset)
            test_utils.assertEqual(
                captured.records[0].getMessage(), 
                "Input `value` contains np.inf or np.NaN, which may lead to unexpected results from the model."
            )
    
def utils_test_to_tsdataset(test_utils):
    """unittest function for to_tsdataset
    """
    # case 1 invalid scenario
    succeed = True
    message = ''
    try:
        res = utils_test_to_tsdataset1(test_utils.model, test_utils.tsdataset1)
    except Exception as e:
        succeed = False
        message = e
    test_utils.assertFalse(succeed)
    test_utils.assertEqual(
        str(message), 
        "anomaly_scor not supported, ['forecasting', 'anomaly_label', 'anomaly_score'] is optional."
    )
    # case2 scenario = anomaly_label, index is DatetimeIndex, target is not none
    res = utils_test_to_tsdataset2(test_utils.model, test_utils.tsdataset1)
    test_utils.assertIsInstance(res, TSDataset)
    test_utils.assertIsInstance(res.get_target().data.index, pd.DatetimeIndex)
    test_utils.assertEqual(res.get_target().data.shape, (5, 1))
    test_utils.assertEqual(res.get_target().data.columns[0], 'label')
    
    # case3 scenario = anomaly_label, index is DatetimeIndex, target is none
    res = utils_test_to_tsdataset2(test_utils.model, test_utils.tsdataset2)
    test_utils.assertIsInstance(res, TSDataset)
    test_utils.assertIsInstance(res.get_target().data.index, pd.DatetimeIndex)
    test_utils.assertEqual(res.get_target().data.shape, (5, 1))
    test_utils.assertEqual(res.get_target().data.columns[0], 'anomaly_label')
    
    # case4 scenario = anomaly_label, index is RangeIndex, target is not none
    res = utils_test_to_tsdataset2(test_utils.model, test_utils.tsdataset4)
    test_utils.assertIsInstance(res, TSDataset)
    test_utils.assertIsInstance(res.get_target().data.index, pd.RangeIndex)
    test_utils.assertEqual(res.get_target().data.shape, (5, 1))
    test_utils.assertEqual(res.get_target().data.columns[0], 'label')
    
    # case5 scenario = anomaly_label, index is RangeIndex, target is none
    res = utils_test_to_tsdataset2(test_utils.model, test_utils.tsdataset5)
    test_utils.assertIsInstance(res, TSDataset)
    test_utils.assertIsInstance(res.get_target().data.index, pd.RangeIndex)
    test_utils.assertEqual(res.get_target().data.shape, (5, 1))
    test_utils.assertEqual(res.get_target().data.columns[0], 'anomaly_label')
    
    # case6 scenario = anomaly_score, index is DatetimeIndex, target is not none
    res = utils_test_to_tsdataset3(test_utils.model, test_utils.tsdataset1)
    test_utils.assertIsInstance(res, TSDataset)
    test_utils.assertIsInstance(res.get_target().data.index, pd.DatetimeIndex)
    test_utils.assertEqual(res.get_target().data.shape, (2, 1))
    test_utils.assertEqual(res.get_target().data.columns[0], 'label_score')
    
    # case7 scenario = anomaly_score, index is DatetimeIndex, target is none
    res = utils_test_to_tsdataset3(test_utils.model, test_utils.tsdataset2)
    test_utils.assertIsInstance(res, TSDataset)
    test_utils.assertIsInstance(res.get_target().data.index, pd.DatetimeIndex)
    test_utils.assertEqual(res.get_target().data.shape, (2, 1))
    test_utils.assertEqual(res.get_target().data.columns[0], 'anomaly_score')
    
    # case8 scenario = anomaly_score, index is RangeIndex, target is not none
    res = utils_test_to_tsdataset3(test_utils.model, test_utils.tsdataset4)
    test_utils.assertIsInstance(res, TSDataset)
    test_utils.assertIsInstance(res.get_target().data.index, pd.RangeIndex)
    test_utils.assertEqual(res.get_target().data.shape, (2, 1))
    test_utils.assertEqual(res.get_target().data.columns[0], 'label_score')
    
    # case9 scenario = anomaly_score, index is RangeIndex, target is none
    res = utils_test_to_tsdataset3(test_utils.model, test_utils.tsdataset5)
    test_utils.assertIsInstance(res, TSDataset)
    test_utils.assertIsInstance(res.get_target().data.index, pd.RangeIndex)
    test_utils.assertEqual(res.get_target().data.shape, (2, 1))
    test_utils.assertEqual(res.get_target().data.columns[0], 'anomaly_score')
    
    # case10 scenario = forecasting, index is DatetimeIndex
    res = utils_test_to_tsdataset4(test_utils.model, test_utils.tsdataset3)
    test_utils.assertIsInstance(res, TSDataset)
    test_utils.assertIsInstance(res.get_target().data.index, pd.DatetimeIndex)
    test_utils.assertEqual(res.get_target().data.shape, (2, 1))
    test_utils.assertEqual(res.get_target().data.columns[0], 'value')
    
    # case11 scenario = forecasting, index is RangeIndex
    res = utils_test_to_tsdataset4(test_utils.model, test_utils.tsdataset6)
    test_utils.assertIsInstance(res, TSDataset)
    test_utils.assertIsInstance(res.get_target().data.index, pd.RangeIndex)
    test_utils.assertEqual(res.get_target().data.shape, (2, 1))
    test_utils.assertEqual(res.get_target().data.columns[0], 'value')
        
@to_tsdataset(scenario="anomaly_scor")
def utils_test_to_tsdataset1(model, tsdataset):
    return np.array([1, 0, 1, 0, 1])
                 
@to_tsdataset(scenario="anomaly_label")
def utils_test_to_tsdataset2(model, tsdataset):
    return np.array([1, 0, 1, 0, 1])
  
@to_tsdataset(scenario="anomaly_score")
def utils_test_to_tsdataset3(model, tsdataset):
    return np.array([1.1, 2.2]) 

@to_tsdataset(scenario="forecasting")
def utils_test_to_tsdataset4(model, tsdataset):
    return np.array([[[ 0.04294178], [-0.51277405]]])
        
            
if __name__ == "__main__":
    test_utils = TestUtils()
    # Test get_target_from_tsdataset
    utils_test_get_target_from_tsdataset(test_utils)
    # Test check_tsdataset
    utils_test_check_tsdataset(test_utils)
    # Test to_tsdataset
    utils_test_to_tsdataset(test_utils)
    
