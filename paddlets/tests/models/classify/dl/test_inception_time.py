# !/usr/bin/env python3
# -*- coding:utf-8 -*-

from unittest import TestCase, mock
import unittest
import random

import pandas as pd
import numpy as np
import paddle

from paddlets.models.common.callbacks import Callback
from paddlets.models.classify.dl.inception_time import InceptionTimeClassifier
from paddlets.datasets import TimeSeries, TSDataset


class TestInceptionTimeClassifier(TestCase):
    def setUp(self):
        """unittest function
        """
        np.random.seed(2022)
        paddlets_ds, labels = self._build_mock_data_and_label(range_index=True,instance_cnt=10,target_dims=1)
        paddlets_ds2, labels2 = self._build_mock_data_and_label(range_index=False,instance_cnt=10,target_dims=2)
        self._paddlets_ds = paddlets_ds
        self._labels = labels
        self._paddlets_ds2 = paddlets_ds2
        self._labels2 = labels2
        self.test_init()
        super().setUp()

    @staticmethod
    def _build_mock_data_and_label(
            target_periods: int = 200,
            target_dims: int = 5,
            n_classes: int = 4,
            instance_cnt: int = 100,
            random_data: bool = True,
            range_index: bool = False,
            seed: bool = False
    ):
        """
        build train datasets and labels.
        todo:not equal target_periods?
        """
        if seed:
            np.random.seed(2022)

        target_cols = [f"dim_{k}" for k in range(target_dims)]
        labels = [f"class" + str(item) for item in np.random.randint(0, n_classes, instance_cnt)]

        ts_datasets = []
        for i in range(instance_cnt):
            if random_data:
                target_data = np.random.randint(0, 10, (target_periods, target_dims))
            else:
                target_data = target_periods * [target_dims * [0]]
            if range_index:
                target_df = pd.DataFrame(
                    target_data,
                    index=pd.RangeIndex(0, target_periods, 1),
                    columns=target_cols
                )
            else:
                target_df = pd.DataFrame(
                    target_data,
                    index=pd.date_range("2022-01-01", periods=target_periods, freq="1D"),
                    columns=target_cols
                )
            ts_datasets.append(
                TSDataset(target=TimeSeries.load_from_dataframe(data=target_df).astype(np.float32))
            )

        return ts_datasets, labels

    def test_init(self):
        """test init
        """
        # case1 
        self.model1 = InceptionTimeClassifier(

        )

        #case2
        self.model2 = InceptionTimeClassifier(
            max_epochs=10,
            batch_size= 16
        )

        #case3
        self.model3 = InceptionTimeClassifier(
            max_epochs=10,
            batch_size= 16,
            use_bottleneck=False,
            use_residual= False
        )

        #case4
        self.model4 = InceptionTimeClassifier(
            max_epochs=10,
            batch_size= 16,
            block_depth=8,
            kernel_size=80
        )

        #case5
        self.model5 = InceptionTimeClassifier(
            max_epochs=10,
            batch_size= 16,
            activation="Sigmoid"
        )
    
    def test_fit(self):
        """test fit
        """
        self.model1.fit(self._paddlets_ds, self._labels)
        self.model2.fit(self._paddlets_ds, self._labels)
        self.model3.fit(self._paddlets_ds, self._labels)
        self.model4.fit(self._paddlets_ds, self._labels)
        self.model5.fit(self._paddlets_ds, self._labels)
    
    def test_predict(self):
        """test predict
        """
        self.model1.fit(self._paddlets_ds, self._labels)
        p1 = self.model1.predict(self._paddlets_ds)
        p2 = self.model1.predict_proba(self._paddlets_ds)
        assert len(p1) == 10
        assert len(p2) == 10

        self.model2.fit(self._paddlets_ds, self._labels)
        p1 = self.model2.predict(self._paddlets_ds)
        p2 = self.model2.predict_proba(self._paddlets_ds)
        assert len(p1) == 10
        assert len(p2) == 10

        self.model3.fit(self._paddlets_ds, self._labels)
        p1 = self.model3.predict(self._paddlets_ds)
        p2 = self.model3.predict_proba(self._paddlets_ds)
        assert len(p1) == 10
        assert len(p2) == 10

        self.model4.fit(self._paddlets_ds, self._labels)
        p1 = self.model4.predict(self._paddlets_ds)
        p2 = self.model4.predict_proba(self._paddlets_ds)
        assert len(p1) == 10
        assert len(p2) == 10

        self.model5.fit(self._paddlets_ds, self._labels)
        p1 = self.model5.predict(self._paddlets_ds)
        p2 = self.model5.predict_proba(self._paddlets_ds)
        assert len(p1) == 10
        assert len(p2) == 10
    
    def test_save_and_load(self):
        """test save and load
        """
        self.model1.fit(self._paddlets_ds, self._labels)
        #model1
        self.model1.save(path="model1")
        model1 = self.model1.load(path="model1")
        import os
        os.remove("model1")
        predictions = model1.predict(self._paddlets_ds)
        assert (len(predictions) == 10)

if __name__ == "__main__":
    unittest.main()
