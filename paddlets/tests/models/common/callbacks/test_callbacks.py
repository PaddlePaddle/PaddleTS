# !/usr/bin/env python3
# -*- coding:utf-8 -*-

from unittest import TestCase, mock
import unittest
import random
import time

import numpy as np

from paddlets.models.common.callbacks import (
    CallbackContainer,
    EarlyStopping,
    Callback,
    History
)


class CallbackHelper(Callback):
    """辅助测试
    """
    def __init__(self, name):
        """初始化函数
        """
        self._name = name

    def on_epoch_begin(self, epoch, logs):
        """在每个epoch开始调用
        """
        msg = f"{self._name}, on_epoch_{epoch}_begin, logs = {logs}"
        print(msg)

    def on_epoch_end(self, epoch, logs):
        """在每个epoch开始调用
        """
        msg = f"{self._name}, on_epoch_{epoch}_end, logs = {logs}"
        print(msg)

    def on_batch_begin(self, batch, logs):
        """在每个epoch开始调用
        """
        msg = f"{self._name}, on_batch_{batch}_begin, logs = {logs}"
        print(msg)

    def on_batch_end(self, batch, logs):
        """在每个epoch开始调用
        """
        msg = f"{self._name}, on_batch_{batch}_end, logs = {logs}"
        print(msg)

    def on_train_begin(self, logs):
        """在每个epoch开始调用
        """
        msg = f"{self._name}, on_train_begin, logs = {logs}"
        print(msg)

    def on_train_end(self, logs):
        """在每个epoch开始调用
        """
        msg = f"{self._name}, on_train_end, logs = {logs}"
        print(msg)


class TestCallbak(TestCase):
    def setUp(self):
        """unittest function
        """
        super().setUp()

    def test_callback(self):
        """unittest function
        """
        # case1
        cbks = CallbackHelper("test")
        model = mock.Mock(name="model")
        cbks.set_trainer(model)
        self.assertEqual(cbks._trainer, model)

        # case2
        epochs, steps = 2, 5
        logs = {"loss": 50.341673, "acc": 0.00256}
        cbks.on_train_begin(logs)
        for epoch in range(epochs):
            cbks.on_epoch_begin(epoch, logs)
            for batch in range(steps):
                cbks.on_batch_begin(batch, logs)
                logs["loss"] -= random.random() * 0.1
                logs["acc"] += random.random() * 0.1
                cbks.on_batch_end(batch, logs)
            cbks.on_epoch_end(epoch, logs)
        cbks.on_train_end(logs)


class TestCallbackContainer(TestCase):
    def setUp(self):
        """unittest function
        """
        super().setUp()

    def test_callback_contrainer(self):
        """unittest function
        """
        # case1
        cbk1 = CallbackHelper("cbk1")
        cbk2 = CallbackHelper("cbk2")
        cbks = CallbackContainer([cbk1])
        self.assertEqual(cbk1, cbks._callbacks[0]) 
        
        # case2
        cbks.append(cbk2)
        self.assertEqual(cbk2, cbks._callbacks[1])
        
        # case3
        model = mock.Mock(name="model")
        cbks.set_trainer(model)
        for ckb in cbks._callbacks:
            self.assertEqual(ckb._trainer, model)

        # case4
        epochs, steps = 2, 5
        logs = {"loss": 50.341673, "acc": 0.00256}
        cbks.on_train_begin(logs)
        for epoch in range(epochs):
            cbks.on_epoch_begin(epoch, logs)
            for batch in range(steps):
                cbks.on_batch_begin(batch, logs)
                logs["loss"] -= random.random() * 0.1
                logs["acc"] += random.random() * 0.1
                cbks.on_batch_end(batch, logs)
            cbks.on_epoch_end(epoch, logs)
        cbks.on_train_end(logs)


class TestEarlyStopping(TestCase):
    def setUp(self):
        """unittest function
        """
        super().setUp()

    def test_early_stopping(self):
        """unittest function
        """
        # case1
        for tag, ret in enumerate([np.inf, -np.inf]):
            earlystopping = EarlyStopping(
                early_stopping_metric="acc",
                is_maximize=tag,
                tol=0.,
                patience=1
            )
            self.assertEqual(earlystopping._best_loss, ret)

        # case2
        model = mock.Mock(name="model")
        model._stop_training = False
        model._max_epoch = 3
        model.network = mock.Mock(name="network")
        model.network.state_dict = mock.Mock(return_value={})
        model.network.set_state_dict = mock.Mock(return_value=None)
        earlystopping.set_trainer(model)

        epochs, steps = 3, 5
        logs = {"loss": 50.341673, "acc": 0.00256}
        delta = [0.01] * 1 + [-0.01] * 2
        for epoch in range(epochs):
            for batch in range(steps):
                logs["loss"] -= random.random() * 0.1
            logs["acc"] += delta[epoch]
            earlystopping.on_epoch_end(epoch, logs)
            if model._stop_training:
                break
        earlystopping.on_train_end(logs)
        self.assertEqual(model._stop_training, True)
        self.assertEqual(earlystopping._best_epoch, 0)
        self.assertAlmostEqual(earlystopping._best_loss, 0.01256, delta=1e-5)

        # case3
        earlystopping = EarlyStopping(
            early_stopping_metric="acc", 
            is_maximize=True,
            patience=4
        )
        model = mock.Mock(name="model")
        model._stop_training = False
        model._max_epoch = 3
        model.network = mock.Mock(name="network")
        model.network.state_dict = mock.Mock(return_value={})
        model.network.set_state_dict = mock.Mock(return_value=None)
        earlystopping.set_trainer(model)

        epochs, steps = 3, 5
        logs = {"loss": 50.341673, "acc": 0.00256}
        delta = [-0.001] * 3
        for epoch in range(epochs):
            for batch in range(steps):
                logs["loss"] -= random.random() * 0.1
            logs["acc"] += delta[epoch]
            earlystopping.on_epoch_end(epoch, logs)
            if model._stop_training:
                break
        earlystopping.on_train_end(logs)
        self.assertEqual(model._stop_training, False)
        self.assertEqual(earlystopping._best_epoch, 0)
        self.assertAlmostEqual(earlystopping._best_loss, 0.00156, delta=1e-5)


class TestHistory(TestCase):
    def setUp(self):
        """unittest function
        """
        super().setUp()

    def test_history(self):
        """unittest function
        """
        # case1, case2, case3
        for verbose in range(3):
            history = History(verbose)
            model = mock.Mock(name="model")
            model._metrics_names = ["mse", "mae"]
            history.set_trainer(model)

            logs = {"start_time": time.time()}
            history.on_train_begin(logs)

            epochs, steps = 3, 5
            logs = {"loss": 50.341673, "batch_size": 64}
            epoch_logs = {"lr": 1e-1, "mse": 25.3618, "mae": 5.6325}
            for epoch in range(epochs):
                history.on_epoch_begin(epoch)
                for batch in range(steps):
                    logs["loss"] -= random.random() * 0.1
                    history.on_batch_end(batch, logs)
                epoch_logs["lr"] -= random.random() * 0.1
                epoch_logs["mse"] -= random.random() * 0.1
                epoch_logs["mae"] -= random.random() * 0.1
                history._epoch_metrics.update({
                    "lr": epoch_logs["lr"],
                    "mse": epoch_logs["mse"],
                    "mae": epoch_logs["mae"],
                })
                history.on_epoch_end(epoch, history._epoch_metrics)


if __name__ == "__main__":
    unittest.main()

