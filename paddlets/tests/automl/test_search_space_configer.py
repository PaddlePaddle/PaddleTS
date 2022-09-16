# !/usr/bin/env python3
# -*- coding:utf-8 -*-
import unittest

from unittest import TestCase
from ray.tune import quniform, randint

from paddlets.automl.autots import SearchSpaceConfiger
from paddlets.models.forecasting import MLPRegressor
from paddlets.transform import Fill


class TestSearchSpaceConfiger(TestCase):
    def setUp(self):
        """
        unittest function
        """
        super().setUp()

    def test_init(self):
        """
        unittest function
        """
        configer = SearchSpaceConfiger()
        # DL
        sp = configer.get_default_search_space(MLPRegressor)
        sp = configer.get_default_search_space("MLPRegressor")
        sp_str = configer.search_space_to_str(sp)
        sp = configer.recommend(MLPRegressor)

        # Pipeline
        sp = configer.get_default_search_space([Fill, MLPRegressor])
        sp = configer.get_default_search_space(["Fill", "MLPRegressor"])
        sp_str = configer.search_space_to_str(sp)
        sp = configer.recommend([Fill, MLPRegressor])

        # param to str
        randint_sp = randint(1, 2)
        randint_sp_str = configer._param_search_space_to_str(randint_sp)
        self.assertEqual(randint_sp_str, "randint(1, 2)")

        quniform_sp = quniform(2, 5, q=1)
        quniform_sp_str = configer._param_search_space_to_str(quniform_sp)
        self.assertEqual(quniform_sp_str, "quniform(2, 5, q=1)")

        # Unknown estimator
        with self.assertRaises(NotImplementedError):
            sp = configer.get_default_search_space("fake")
        # empty pipeline
        with self.assertRaises(NotImplementedError):
            sp = configer.get_default_search_space(["fake-1", "fake-2", "fake-2-1"])
        # partial empty
        sp = configer.get_default_search_space(["fake-0", "MLPRegressor"])
        sp_str = configer.search_space_to_str(sp)
        sp = configer.get_default_search_space(["fake-0", MLPRegressor])
        sp_str = configer.search_space_to_str(sp)
        sp = configer.get_default_search_space([SearchSpaceConfiger, MLPRegressor])
        sp_str = configer.search_space_to_str(sp)


if __name__ == "__main__":
    unittest.main()
