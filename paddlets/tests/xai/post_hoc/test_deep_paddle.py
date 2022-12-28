# !/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import sys
import time
import unittest
from unittest import TestCase

import paddle

from paddlets import TSDataset
from paddlets.models.forecasting import NBEATSModel
from paddlets.transform import StandardScaler
from paddlets.pipeline.pipeline import Pipeline
from paddlets.datasets.repository import dataset_list, get_dataset, DATASETS
from paddlets.xai.post_hoc.deep_paddle import PaddleDeep

class TestPaddleDeep(TestCase):
    def setUp(self):
        """
        unittest function
        """
        super().setUp()
        data = get_dataset('ECL')
        keep_cols = ['MT_320', 'MT_000', 'MT_001', ]
        ts_cols = data.columns
        remove_cols = []
        for col, types in ts_cols.items():
            if (col not in keep_cols):
                remove_cols.append(col)

        data.drop(remove_cols)
        self.data = data
        
    def test_gradient(self):
        """
        unittest function
        """
        df = self.data.to_dataframe()
        data = TSDataset.load_from_dataframe(df, target_cols=['MT_320'],)
        # Sample
        data, _ = data.split('2014-06-30')
        train_data, test_data = data.split('2014-06-15')
        train_data, val_data = train_data.split('2014-06-01')
        # Parameters
        in_chunk_len = 24
        out_chunk_len = 24
        skip_chunk_len = 0
        sampling_stride = 24
        max_epochs = 1
        patience = 5
        # Pipeline
        pipeline_list = [(StandardScaler, {}), 
                 (NBEATSModel, {'in_chunk_len': in_chunk_len, 
                                'out_chunk_len': out_chunk_len, 
                                'skip_chunk_len': skip_chunk_len, 
                                'max_epochs': max_epochs, 
                                'patience': patience})
                ]
        # Fit
        pipe = Pipeline(pipeline_list)
        pipe.fit(train_data, val_data)
        #shap value
        background = {'past_target': paddle.rand((1, 24, 1))}
        foreground = [{'past_target': paddle.rand((1, 24, 1))}]
        pd = PaddleDeep(pipe._model, background)
        grad = pd.gradient(0, foreground)

        self.assertEqual(len(grad), 1)
        self.assertEqual(grad[0]['past_target'].shape, (1,24,1))
        
    def test_shap_values(self):
        """
        unittest function
        """
        df = self.data.to_dataframe()
        data = TSDataset.load_from_dataframe(df, target_cols=['MT_320'],)
        # Sample
        data, _ = data.split('2014-06-30')
        train_data, test_data = data.split('2014-06-15')
        train_data, val_data = train_data.split('2014-06-01')
        # Parameters
        in_chunk_len = 24
        out_chunk_len = 24
        skip_chunk_len = 0
        sampling_stride = 24
        max_epochs = 1
        patience = 5
        # Pipeline
        pipeline_list = [(StandardScaler, {}), 
                 (NBEATSModel, {'in_chunk_len': in_chunk_len, 
                                'out_chunk_len': out_chunk_len, 
                                'skip_chunk_len': skip_chunk_len, 
                                'max_epochs': max_epochs, 
                                'patience': patience})
                ]
        # Fit
        pipe = Pipeline(pipeline_list)
        pipe.fit(train_data, val_data)
        #shap value
        background = {'past_target': paddle.rand((1, 24, 1))}
        foreground = {'past_target': paddle.rand((1, 24, 1))}
        pd = PaddleDeep(pipe._model, background)
        sv = pd.shap_values(foreground)

        self.assertEqual(len(sv), 24)
        self.assertEqual(sv[0]['past_target'].shape, (1,24,1))

        
        

if __name__ == "__main__":
    unittest.main()
