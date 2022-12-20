# !/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import sys
import time
import unittest
from unittest import TestCase

from paddlets import TSDataset
from paddlets.models.forecasting import NBEATSModel, RNNBlockRegressor, MLPRegressor, DeepARModel
from paddlets.transform import StandardScaler
from paddlets.pipeline.pipeline import Pipeline
from paddlets.datasets.repository import dataset_list, get_dataset, DATASETS
from paddlets.xai.post_hoc import ShapExplainer
from paddlets.datasets.tsdataset import TimeSeries, TSDataset

class TestShapExplainer(TestCase):
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
        

    def test_explain(self):
        """
        unittest function
        """
        # Known/observed exists. use_paddleloader=False
        df = self.data.to_dataframe()
        data = TSDataset.load_from_dataframe(df, target_cols=['MT_320'], observed_cov_cols=['MT_000'], 
                                            known_cov_cols=['MT_001',])
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
        # Explainer
        se = ShapExplainer(pipe, train_data, background_sample_number=1, keep_index=True, use_paddleloader=False)
        shap_value = se.explain(test_data, nsamples=100) #default the latest sample
        self.assertEqual(shap_value.shape, (out_chunk_len, 1, 96))
        ###########################################################################################################
        # Known exists, observed=None, use_paddleloader=False
        df = self.data.to_dataframe()
        data = TSDataset.load_from_dataframe(df, target_cols=['MT_320'], observed_cov_cols=[], 
                                            known_cov_cols=['MT_000',])
        # Sample
        data, _ = data.split('2014-06-30')
        train_data, test_data = data.split('2014-06-15')
        train_data, val_data = train_data.split('2014-06-01')
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
        # Explainer
        se = ShapExplainer(pipe, train_data, background_sample_number=1, keep_index=True, use_paddleloader=False)
        shap_value = se.explain(test_data, nsamples=80) #default the latest sample
        self.assertEqual(shap_value.shape, (out_chunk_len, 1, 72))
        ###########################################################################################################
        # Known=None, observed exists, use_paddleloader=False
        df = self.data.to_dataframe()
        data = TSDataset.load_from_dataframe(df, target_cols=['MT_320'], observed_cov_cols=['MT_000', 'MT_001'], 
                                            known_cov_cols=[])
        # Sample
        data, _ = data.split('2014-06-30')
        train_data, test_data = data.split('2014-06-15')
        train_data, val_data = train_data.split('2014-06-01')
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
        # Explainer
        se = ShapExplainer(pipe, train_data, background_sample_number=1, keep_index=True, use_paddleloader=False)
        shap_value = se.explain(test_data, nsamples=80) #default the latest sample
        self.assertEqual(shap_value.shape, (out_chunk_len, 1, 72))
        ###########################################################################################################
        # Known/observed = None, use_paddleloader=False
        df = self.data.to_dataframe()
        data = TSDataset.load_from_dataframe(df, target_cols=['MT_320'], observed_cov_cols=[], 
                                            known_cov_cols=[])
        # Sample
        data, _ = data.split('2014-06-30')
        train_data, test_data = data.split('2014-06-15')
        train_data, val_data = train_data.split('2014-06-01')
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
        # Explainer
        se = ShapExplainer(pipe, train_data, background_sample_number=1, keep_index=True, use_paddleloader=False)
        shap_value = se.explain(test_data, nsamples=30) #default the latest sample
        self.assertEqual(shap_value.shape, (out_chunk_len, 1, 24))
        ###########################################################################################################
        # Known/observed exists. use_paddleloader=True
        df = self.data.to_dataframe()
        data = TSDataset.load_from_dataframe(df, target_cols=['MT_320'], observed_cov_cols=['MT_000'], 
                                            known_cov_cols=['MT_001',])
        # Sample
        data, _ = data.split('2014-06-30')
        train_data, test_data = data.split('2014-06-15')
        train_data, val_data = train_data.split('2014-06-01')
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
        # Explainer
        se = ShapExplainer(pipe._model, train_data, background_sample_number=1, keep_index=True, use_paddleloader=True)
        shap_value = se.explain(test_data, nsamples=100) #default the latest sample
        self.assertEqual(shap_value.shape, (out_chunk_len, 1, 96))
        ###########################################################################################################
        # Known exists, observed=None, use_paddleloader=True
        df = self.data.to_dataframe()
        data = TSDataset.load_from_dataframe(df, target_cols=['MT_320'], observed_cov_cols=[], 
                                            known_cov_cols=['MT_000', 'MT_001',])
        # Sample
        data, _ = data.split('2014-06-30')
        train_data, test_data = data.split('2014-06-15')
        train_data, val_data = train_data.split('2014-06-01')
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
        # Explainer
        se = ShapExplainer(pipe._model, train_data, background_sample_number=1, keep_index=True, use_paddleloader=True)
        shap_value = se.explain(test_data, nsamples=130) #default the latest sample
        self.assertEqual(shap_value.shape, (out_chunk_len, 1, 120))
        ###########################################################################################################
        # Known=None, observed exists, use_paddleloader=True
        df = self.data.to_dataframe()
        data = TSDataset.load_from_dataframe(df, target_cols=['MT_320'], observed_cov_cols=['MT_000', 'MT_001', ], 
                                            known_cov_cols=[])
        # Sample
        data, _ = data.split('2014-06-30')
        train_data, test_data = data.split('2014-06-15')
        train_data, val_data = train_data.split('2014-06-01')
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
        # Explainer
        se = ShapExplainer(pipe._model, train_data, background_sample_number=1, keep_index=True, use_paddleloader=True)
        shap_value = se.explain(test_data, nsamples=80) #default the latest sample
        self.assertEqual(shap_value.shape, (out_chunk_len, 1, 72))
        ###########################################################################################################
        # Known/observed = None, use_paddleloader=True
        df = self.data.to_dataframe()
        data = TSDataset.load_from_dataframe(df, target_cols=['MT_320'], observed_cov_cols=[], 
                                            known_cov_cols=[])
        # Sample
        data, _ = data.split('2014-06-30')
        train_data, test_data = data.split('2014-06-15')
        train_data, val_data = train_data.split('2014-06-01')
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
        # Explainer
        se = ShapExplainer(pipe._model, train_data, background_sample_number=1, keep_index=True, use_paddleloader=True)
        shap_value = se.explain(test_data, nsamples=30) #default the latest sample
        self.assertEqual(shap_value.shape, (out_chunk_len, 1, 24))
        ###########################################################################################################
        # Static exists
        df = self.data.to_dataframe()
        df['s'] = 0
        data = TSDataset.load_from_dataframe(df, target_cols=['MT_320'], observed_cov_cols=['MT_000'], 
                                            known_cov_cols=['MT_001',], static_cov_cols=['s'])
        # Sample
        data, _ = data.split('2014-06-30')
        train_data, test_data = data.split('2014-06-15')
        train_data, val_data = train_data.split('2014-06-01')
        # Pipeline
        pipeline_list = [(StandardScaler, {}), 
                 (RNNBlockRegressor, {'in_chunk_len': in_chunk_len, 
                                'out_chunk_len': out_chunk_len, 
                                'skip_chunk_len': skip_chunk_len, 
                                'max_epochs': max_epochs, 
                                'patience': patience})
                ]
        # Fit
        pipe = Pipeline(pipeline_list)
        pipe.fit(train_data, val_data)
        # Explainer
        se = ShapExplainer(pipe, train_data, background_sample_number=1, keep_index=True, use_paddleloader=False)
        shap_value = se.explain(test_data, nsamples=150) #default the latest sample
        self.assertEqual(shap_value.shape, (out_chunk_len, 1, 144))
        ###########################################################################################################
        # Static exists, use_paddleloader=True, skip_chunk_len > 0
        skip_chunk_len = 4
        df = self.data.to_dataframe()
        df['s'] = 0
        data = TSDataset.load_from_dataframe(df, target_cols=['MT_320'], observed_cov_cols=['MT_000'], 
                                            known_cov_cols=['MT_001', ], static_cov_cols=['s'])
        # Sample
        data, _ = data.split('2014-06-30')
        train_data, test_data = data.split('2014-06-15')
        train_data, val_data = train_data.split('2014-06-01')
        # Pipeline
        pipeline_list = [(StandardScaler, {}), 
                 (RNNBlockRegressor, {'in_chunk_len': in_chunk_len, 
                                'out_chunk_len': out_chunk_len, 
                                'skip_chunk_len': skip_chunk_len, 
                                'max_epochs': max_epochs, 
                                'patience': patience})
                ]
        # Fit
        pipe = Pipeline(pipeline_list)
        pipe.fit(train_data, val_data)
        # Explainer
        se = ShapExplainer(pipe._model, train_data, background_sample_number=1, keep_index=True, use_paddleloader=True)
        shap_value = se.explain(test_data, nsamples=160) #default the latest sample
        self.assertEqual(shap_value.shape, (out_chunk_len, 1, 152))
        ###########################################################################################################
        # Skip_chunk_len > 0
        df = self.data.to_dataframe()
        df['s'] = 0
        data = TSDataset.load_from_dataframe(df, target_cols=['MT_320'], observed_cov_cols=['MT_000'], 
                                            known_cov_cols=['MT_001', ], static_cov_cols=['s'])
        # Sample
        data, _ = data.split('2014-06-30')
        train_data, test_data = data.split('2014-06-15')
        train_data, val_data = train_data.split('2014-06-01')
        # Pipeline
        pipeline_list = [(StandardScaler, {}), 
                 (RNNBlockRegressor, {'in_chunk_len': in_chunk_len, 
                                'out_chunk_len': out_chunk_len, 
                                'skip_chunk_len': skip_chunk_len, 
                                'max_epochs': max_epochs, 
                                'patience': patience})
                ]
        # Fit
        pipe = Pipeline(pipeline_list)
        pipe.fit(train_data, val_data)
        # Explainer
        se = ShapExplainer(pipe, train_data, background_sample_number=1, keep_index=True, use_paddleloader=False)
        shap_value = se.explain(test_data, nsamples=160) #default the latest sample
        self.assertEqual(shap_value.shape, (out_chunk_len, 1, 152))
        
    def test_get_explanation(self):
        """
        unittest function
        """
        df = self.data.to_dataframe()
        data = TSDataset.load_from_dataframe(df, target_cols=['MT_320'], observed_cov_cols=['MT_000'], 
                                            known_cov_cols=['MT_001', ])
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
        # Explainer
        se = ShapExplainer(pipe, train_data, background_sample_number=1, keep_index=True, use_paddleloader=False)
        shap_value = se.explain(test_data, nsamples=100) #default the latest sample
        value = se.get_explanation(1, 0)
        self.assertEqual(value.shape, (96, ))
        
    def test_plot(self):
        """
        unittest function
        """
        df = self.data.to_dataframe()
        data = TSDataset.load_from_dataframe(df, target_cols=['MT_320'], observed_cov_cols=['MT_000'], 
                                            known_cov_cols=['MT_001',])
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
        # Explainer
        se = ShapExplainer(pipe, train_data, background_sample_number=1, keep_index=True, use_paddleloader=False)
        shap_value = se.explain(test_data, nsamples=100) #default the latest sample
        
        se.plot(method='OI')
        se.plot(method='OV')
        se.plot(method='IV')
        se.force_plot(out_chunk_indice=[1, 3], sample_index=0, contribution_threshold=0.01)
        se.summary_plot(out_chunk_indice=[1, 3], sample_index=0)
        self.assertEqual(True, True)
        
    def test_different_model(self):
        """
        unittest function
        """
        # Mlp
        df = self.data.to_dataframe()
        data = TSDataset.load_from_dataframe(df, target_cols=['MT_320'], observed_cov_cols=['MT_000'], 
                                            known_cov_cols=['MT_001', ])
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
                 (MLPRegressor, {'in_chunk_len': in_chunk_len, 
                                'out_chunk_len': out_chunk_len, 
                                'skip_chunk_len': skip_chunk_len, 
                                'max_epochs': max_epochs, 
                                'patience': patience})
                ]
        # Fit
        pipe = Pipeline(pipeline_list)
        pipe.fit(train_data, val_data)
        # Explainer
        se = ShapExplainer(pipe, train_data, background_sample_number=1, keep_index=True, use_paddleloader=False)
        shap_value = se.explain(test_data, nsamples=100) #default the latest sample
        value = se.get_explanation(1, 0)
        self.assertEqual(value.shape, (96, ))
        ###########################################################################################################
        # Deepar-point predictions
        df = self.data.to_dataframe()
        data = TSDataset.load_from_dataframe(df, target_cols=['MT_320'], observed_cov_cols=['MT_000'], 
                                            known_cov_cols=['MT_001', ])
        # Sample
        data, _ = data.split('2014-06-30')
        train_data, test_data = data.split('2014-06-15')
        train_data, val_data = train_data.split('2014-06-01')
        # Pipeline
        pipeline_list = [(StandardScaler, {}), 
                 (DeepARModel, {'in_chunk_len': in_chunk_len, 
                                'out_chunk_len': out_chunk_len, 
                                'skip_chunk_len': skip_chunk_len, 
                                'max_epochs': max_epochs, 
                                'patience': patience,
                                'output_mode': 'predictions'})
                ]
        # Fit
        pipe = Pipeline(pipeline_list)
        pipe.fit(train_data, val_data)
        # Explainer
        se = ShapExplainer(pipe, train_data, background_sample_number=1, keep_index=True, use_paddleloader=False)
        shap_value = se.explain(test_data, nsamples=100) #default the latest sample
        value = se.get_explanation(1, 0)
        self.assertEqual(value.shape, (96, ))
        ###########################################################################################################
        # Deepar-quantile predictions(raise error)
        df = self.data.to_dataframe()
        data = TSDataset.load_from_dataframe(df, target_cols=['MT_320'], observed_cov_cols=['MT_000'], 
                                            known_cov_cols=['MT_001', ])
        # Sample
        data, _ = data.split('2014-06-30')
        train_data, test_data = data.split('2014-06-15')
        train_data, val_data = train_data.split('2014-06-01')
        # Pipeline
        pipeline_list = [(StandardScaler, {}), 
                 (DeepARModel, {'in_chunk_len': in_chunk_len, 
                                'out_chunk_len': out_chunk_len, 
                                'skip_chunk_len': skip_chunk_len, 
                                'max_epochs': max_epochs, 
                                'patience': patience,})
                ]
        # Fit
        pipe = Pipeline(pipeline_list)
        pipe.fit(train_data, val_data)
        # Explainer
        flag = False
        try:
            se = ShapExplainer(pipe, train_data, background_sample_number=1, keep_index=True, use_paddleloader=False)
        except:
            flag = True
        self.assertEqual(flag, True)
        ###########################################################################################################
        # Deepar-point predictions, use_paddleloader = True
        df = self.data.to_dataframe()
        data = TSDataset.load_from_dataframe(df, target_cols=['MT_320'], observed_cov_cols=['MT_000'], 
                                            known_cov_cols=['MT_001', ])
        # Sample
        data, _ = data.split('2014-06-30')
        train_data, test_data = data.split('2014-06-15')
        train_data, val_data = train_data.split('2014-06-01')
        # Pipeline
        pipeline_list = [(StandardScaler, {}), 
                 (DeepARModel, {'in_chunk_len': in_chunk_len, 
                                'out_chunk_len': out_chunk_len, 
                                'skip_chunk_len': skip_chunk_len, 
                                'max_epochs': max_epochs, 
                                'patience': patience,
                                'output_mode': 'predictions'})
                ]
        # Fit
        pipe = Pipeline(pipeline_list)
        pipe.fit(train_data, val_data)
        # Explainer
        se = ShapExplainer(pipe._model, train_data, background_sample_number=5, keep_index=True, use_paddleloader=True)
        shap_value = se.explain(test_data, nsamples=100) #default the latest sample
        value = se.get_explanation(1, 0)
        self.assertEqual(value.shape, (96, ))
        
        

if __name__ == "__main__":
    unittest.main()
