# !/usr/bin/env python3
# -*- coding:utf-8 -*-

from unittest import TestCase, mock
import unittest
import random

import pandas as pd
import numpy as np
import paddle

from paddlets.models.common.callbacks import Callback
from paddlets.models.classify.dl.cnn import CNNClassifier
from paddlets.datasets import TimeSeries, TSDataset


class TestCNNClassifier(TestCase):
    def setUp(self):
        """unittest function
        """
        np.random.seed(2022)
        paddlets_ds, labels = self._build_mock_data_and_label(range_index=True)
        paddlets_ds2, labels2 = self._build_mock_data_and_label(range_index=False)
        self._paddlets_ds = paddlets_ds
        self._labels = labels
        self._paddlets_ds2 = paddlets_ds2
        self._labels2 = labels2
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
        """unittest function
        """
        # case1 (参数全部合法)
        param1 = {
            "batch_size": 1,
            "max_epochs": 1,
            "verbose": 1,
            "patience": 1,
        }
        cnn = CNNClassifier(
            **param1
        )

        # case2 (batch_size 不合法)
        param2 = {
            "batch_size": 0,
            "max_epochs": 1,
            "verbose": 1,
            "patience": 1,
        }
        with self.assertRaises(ValueError):
            cnn = CNNClassifier(
                **param2
            )

        # case3 (max_epochs 不合法)
        param3 = {
            "batch_size": 1,
            "max_epochs": 0,
            "verbose": 1,
            "patience": 1,
        }
        with self.assertRaises(ValueError):
            cnn = CNNClassifier(
                **param3
            )

        # case4 (verbose 不合法)
        param4 = {
            "batch_size": 1,
            "max_epochs": 1,
            "verbose": 0,
            "patience": 1,
        }
        with self.assertRaises(ValueError):
            cnn = CNNClassifier(
                **param4
            )

        # case5 (patience 不合法)
        param5 = {
            "batch_size": 1,
            "max_epochs": 1,
            "verbose": 1,
            "patience": -1,
        }
        with self.assertRaises(ValueError):
            cnn = CNNClassifier(
                **param5
            )

        # case6 hidden_config 不合法
        param6 = {
            "batch_size": 128,
            "max_epochs": 1,
            "verbose": 1,
            "patience": 1,
            "hidden_config": [100, -100]
        }
        with self.assertRaises(AssertionError):
            cnn = CNNClassifier(
                **param6
            )
            cnn.fit(self._paddlets_ds, self._labels)

        # case7 kernel_size过大或hidden layer过多 导致output_chunk_len小于1
        param7 = {
            "batch_size": 128,
            "max_epochs": 1,
            "verbose": 1,
            "patience": 1,
            "kernel_size": 7,
            "avg_pool_size": 9,
            "hidden_config": [12, 6, 12, 6]
        }

        with self.assertRaises(ValueError):
            cnn = CNNClassifier(
                **param7
            )
            cnn.fit(self._paddlets_ds, self._labels)

        # case8 自定义kernel_size/avg_pool_size/hidden_config
        param8 = {
            "batch_size": 128,
            "max_epochs": 1,
            "verbose": 1,
            "patience": 1,
            "kernel_size": 5,
            "avg_pool_size": 2,
            "hidden_config": [12, 6, 3]
        }
        cnn = CNNClassifier(
            **param8
        )
        cnn.fit(self._paddlets_ds, self._labels)

        # case9 (use_bn = True)
        param9 = {
            "batch_size": 128,
            "max_epochs": 1,
            "verbose": 1,
            "patience": 1,
            "use_bn": True
        }
        cnn = CNNClassifier(
            **param9
        )
        cnn.fit(self._paddlets_ds, self._labels)
        self.assertIsInstance(cnn._network._nn[1], paddle.nn.BatchNorm1D)

    def test_init_dataloader(self):
        """unittest function
        """
        cnn = CNNClassifier(
            max_epochs=1
        )
        # case1 (评估集未传入函数)
        _, valid_dataloader = cnn._init_fit_dataloaders(self._paddlets_ds, self._labels)
        self.assertIsNone(valid_dataloader)

        # calse2 (评估集传入函数)
        _, valid_dataloaders = cnn._init_fit_dataloaders(self._paddlets_ds, self._labels, self._paddlets_ds, self._labels)
        self.assertNotEqual(len(valid_dataloaders), 0)

        # case3 (训练集包含非float32)
        paddlets_ds = self._paddlets_ds.copy()
        paddlets_ds[0].astype({"dim_1": "float64"})
        cnn.fit(paddlets_ds, self._labels)
        dtypes = paddlets_ds[0].dtypes.to_dict()
        self.assertEqual(dtypes["dim_1"], np.float32)

        paddlets_ds = self._paddlets_ds.copy()
        paddlets_ds[0].astype({"dim_2": "int32"})
        with self.assertRaises(TypeError):
            cnn.fit(paddlets_ds, self._labels)

        # case4 (训练集包含非法数据类型如object字符串类型)
        paddlets_ds = self._paddlets_ds.copy()
        paddlets_ds[0].astype({"dim_1": "O"})
        with self.assertRaises(TypeError):
            cnn.fit(paddlets_ds, self._labels)

        # case5 (训练集包含NaN)
        paddlets_ds2 = self._paddlets_ds2.copy()
        paddlets_ds2[0]["dim_1"][0] = np.NaN
        with self.assertLogs("paddlets", level="WARNING") as captured:
            cnn.fit(paddlets_ds2, self._labels)
            self.assertEqual(
                captured.records[0].getMessage(),
                "Input `dim_1` contains np.inf or np.NaN, which may lead to unexpected results from the model."
            )

    def test_init_metrics(self):
        """unittest function
        """
        # case1 (以用户传入的metric为第一优先)
        cnn = CNNClassifier(
            eval_metrics=["mae"]
        )
        _, metrics_names, _ = cnn._init_metrics(["val"])
        self.assertEqual(metrics_names[-1], "val_mae")

        # case2 (用户未传入的metric, 取默认metric)
        cnn = CNNClassifier(
            patience=1
        )
        _, metrics_names, _ = cnn._init_metrics(["val"])
        self.assertEqual(metrics_names[-1], "val_mse")

    def test_init_callbacks(self):
        """unittest function
        """
        # case1 (patience = 0)
        cnn = CNNClassifier(
            patience=0
        )
        cnn._metrics, cnn._metrics_names, _ = cnn._init_metrics(["val"])
        with self.assertLogs("paddlets", level="WARNING") as captured:
            cnn._init_callbacks()
            self.assertEqual(len(captured.records), 1)  # check that there is only one log message
            self.assertEqual(
                captured.records[0].getMessage(),
                "No early stopping will be performed, last training weights will be used."
            )

        # case2 (patience > 0)
        cnn = CNNClassifier(
            patience=1
        )
        cnn._metrics, cnn._metrics_names, _ = cnn._init_metrics(["val"])
        _, callback_container = cnn._init_callbacks()

        # case3 (用户未传入callbacks)
        self.assertEqual(len(callback_container._callbacks), 2)

        # case4 (用户传入callbacks)
        callback = Callback()
        cnn = CNNClassifier(
            callbacks=[callback]
        )
        cnn._metrics, cnn._metrics_names, _ = cnn._init_metrics(["val"])
        _, callback_container = cnn._init_callbacks()
        self.assertEqual(len(callback_container._callbacks), 3)

    def test_fit(self):
        """unittest function
        """
        # case1 (用户只传入训练集, log不显示评估指标, 达到最大epochs训练结束)
        cnn = CNNClassifier(
            optimizer_params=dict(learning_rate=1e-1),
            eval_metrics=["mse", "mae"],
            batch_size=512,
            max_epochs=10,
            patience=1
        )
        cnn.fit(self._paddlets_ds, self._labels)

        # case2 (用户同时传入训练/评估集, log显示评估指标, 同时early_stopping生效)
        cnn = CNNClassifier(
            optimizer_params=dict(learning_rate=1e-1),
            eval_metrics=["mse", "mae"],
            batch_size=32,
            max_epochs=10,
            patience=1
        )
        cnn.fit(self._paddlets_ds, self._labels, self._paddlets_ds, self._labels)
        self.assertEqual(cnn._stop_training, True)

    def test_predict(self):
        """unittest function
        """
        # case1 (index为DatetimeIndex)
        cnn = CNNClassifier(
            eval_metrics=["mse", "mae"],
            max_epochs=1,
            patience=1
        )
        cnn.fit(self._paddlets_ds, self._labels, self._paddlets_ds, self._labels)
        res = cnn.predict(self._paddlets_ds)
        self.assertIsInstance(res, np.ndarray)
        self.assertEqual(len(res), len(self._labels))

        # case2 (index为DatetimeIndex)
        cnn.fit(self._paddlets_ds2, self._labels2, self._paddlets_ds2, self._labels2)
        res2 = cnn.predict(self._paddlets_ds2)
        self.assertIsInstance(res, np.ndarray)
        self.assertEqual(len(res2), len(self._labels2))


if __name__ == "__main__":
    unittest.main()