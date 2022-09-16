# !/usr/bin/env python3
# -*- coding:utf-8 -*-
import unittest

from unittest import TestCase

from paddlets.automl.searcher import Searcher


class TestSearcher(TestCase):
    def setUp(self):
        """
        unittest function
        """
        super().setUp()

    def test_get_searcher(self):
        """
        unittest function
        """
        searcher_list = Searcher.get_supported_algs()
        for e in searcher_list:
            search_alg = Searcher.get_searcher(e)
        # Unknown searcher
        with self.assertRaises(NotImplementedError):
            Searcher.get_searcher("Unknown")


if __name__ == "__main__":
    unittest.main()