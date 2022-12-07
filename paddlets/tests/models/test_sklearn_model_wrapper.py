# !/usr/bin/env python3
# -*- coding:utf-8 -*-

import unittest
import os
import random
import json
import shutil
import datetime
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from itertools import product

import sklearn
from sklearn.linear_model import (
    ARDRegression,
    BayesianRidge,
    ElasticNet,
    ElasticNetCV,
    HuberRegressor,
    Lars,
    LarsCV,
    Lasso,
    LassoCV,
    LassoLars,
    LassoLarsCV,
    LassoLarsIC,
    LinearRegression,
    LogisticRegression,
    LogisticRegressionCV,
    MultiTaskElasticNet,
    MultiTaskElasticNetCV,
    MultiTaskLasso,
    MultiTaskLassoCV,
    OrthogonalMatchingPursuit,
    OrthogonalMatchingPursuitCV,
    PassiveAggressiveClassifier,
    PassiveAggressiveRegressor,
    Perceptron,
    # QuantileRegressor,
    Ridge,
    RidgeCV,
    RidgeClassifier,
    RidgeClassifierCV,
    SGDClassifier,
    SGDRegressor,
    # SGDOneClassSVM,
    TheilSenRegressor,
    RANSACRegressor,
    PoissonRegressor,
    GammaRegressor,
    TweedieRegressor
)

from sklearn.svm import (
    LinearSVC,
    LinearSVR,
    NuSVC,
    NuSVR,
    OneClassSVM,
    SVC,
    SVR
)

from sklearn.cross_decomposition import (
    PLSCanonical,
    PLSRegression,
    PLSSVD,
    CCA
)

from sklearn.decomposition import (
    DictionaryLearning,
    FastICA,
    IncrementalPCA,
    KernelPCA,
    MiniBatchDictionaryLearning,
    MiniBatchSparsePCA,
    NMF,
    PCA,
    SparseCoder,
    SparsePCA,
    FactorAnalysis,
    TruncatedSVD,
    LatentDirichletAllocation
)

from sklearn.gaussian_process import (
    GaussianProcessRegressor,
    GaussianProcessClassifier
)

from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis
)

from sklearn.naive_bayes import (
    BernoulliNB,
    GaussianNB,
    MultinomialNB,
    ComplementNB,
    CategoricalNB
)

from sklearn.kernel_ridge import KernelRidge

from sklearn.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor,
    KNeighborsTransformer,
    NearestCentroid,
    NearestNeighbors,
    RadiusNeighborsClassifier,
    RadiusNeighborsRegressor,
    RadiusNeighborsTransformer,
    KernelDensity,
    LocalOutlierFactor,
    NeighborhoodComponentsAnalysis
)

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier, OutputCodeClassifier
from sklearn.semi_supervised import SelfTrainingClassifier, LabelPropagation, LabelSpreading
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.cluster import (
    AffinityPropagation,
    AgglomerativeClustering,
    Birch,
    DBSCAN,
    OPTICS,
    KMeans,
    FeatureAgglomeration,
    MeanShift,
    MiniBatchKMeans,
    SpectralClustering,
    SpectralBiclustering,
    SpectralCoclustering
)

from paddlets.models.forecasting.ml.ml_base import MLBaseModel
from paddlets.models.ml_model_wrapper import SklearnModelWrapper, make_ml_model
from paddlets.datasets import TSDataset, TimeSeries
from paddlets.models.data_adapter import MLDataLoader


class MockNotSklearnModel(object):
    """Mock class that is NOT from sklearn."""
    pass


class TestSklearnModelWrapper(unittest.TestCase):
    """
    test SklearnModelWrapper.

    Note:
        The following models cannot be imported from sklearn 0.24.2, but can be imported in sklearn 1.0.2:
            sklearn.linear_model.QuantileRegressor
            sklearn.linear_model.SGDOneClassSVM
    """
    def setUp(self):
        """
        unittest setup.

        self._good_to_init_sklearn_model_list: can be init, but some of them may NOT be fitted / predicted.
        self._not_implement_predict_method_sklearn_model_class_set: models that do NOT implement predict method.
        self.__cannot_fit_or_predict_sklearn_model_class_set: models that implement predict method, but have other 
            errors while fitting / predicting.
        """
        self._default_modelname = "modelname"
        self._default_model_class = KNeighborsClassifier
        self._default_in_chunk_len = 3
        self._default_out_chunk_len = 1

        self._good_to_init_sklearn_model_list = [
            # linear_model
            {"clazz": ARDRegression, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            {"clazz": BayesianRidge, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            {"clazz": ElasticNet, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            {"clazz": ElasticNetCV, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            {"clazz": HuberRegressor, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            {"clazz": Lars, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            {"clazz": LarsCV, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            {"clazz": Lasso, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            {"clazz": LassoCV, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            {"clazz": LassoLars, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            {"clazz": LassoLarsCV, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            {"clazz": LassoLarsIC, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            {
                "clazz": LinearRegression,
                "init_params": dict(),
                "fit_params": {"sample_weight": None},
                "predict_params": dict()
            },
            {"clazz": LogisticRegression, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            {"clazz": LogisticRegressionCV, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            {"clazz": MultiTaskElasticNet, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            {"clazz": MultiTaskElasticNetCV, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            {"clazz": MultiTaskLasso, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            {"clazz": MultiTaskLassoCV, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            {"clazz": OrthogonalMatchingPursuit, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            {
                "clazz": OrthogonalMatchingPursuitCV,
                "init_params": dict(),
                "fit_params": dict(),
                "predict_params": dict()
            },
            {
                "clazz": PassiveAggressiveClassifier,
                "init_params": dict(),
                "fit_params": dict(),
                "predict_params": dict()
            },
            {
                "clazz": PassiveAggressiveRegressor,
                "init_params": dict(),
                "fit_params": dict(),
                "predict_params": dict()
            },
            {"clazz": Perceptron, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            # {"clazz": QuantileRegressor, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            {"clazz": Ridge, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            {"clazz": RidgeCV, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            {"clazz": RidgeClassifier, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            {"clazz": RidgeClassifierCV, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            {"clazz": SGDClassifier, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            {"clazz": SGDRegressor, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            # {"clazz": SGDOneClassSVM, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            {"clazz": TheilSenRegressor, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            {"clazz": RANSACRegressor, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            {"clazz": PoissonRegressor, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            {"clazz": GammaRegressor, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            {"clazz": TweedieRegressor, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            
            # svm
            {"clazz": LinearSVC, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            {"clazz": LinearSVR, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            {"clazz": NuSVC, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            {"clazz": NuSVR, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            {"clazz": OneClassSVM, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            {"clazz": SVC, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            {"clazz": SVR, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            
            # cross_decomposition
            # {"clazz": PLSCanonical, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            {"clazz": PLSRegression, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            # {"clazz": PLSSVD, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            # {"clazz": CCA, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            
            # decomposition
            # {"clazz": DictionaryLearning, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            # {"clazz": FastICA, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            # {"clazz": IncrementalPCA, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            # {"clazz": KernelPCA, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            # {
            #     "clazz": MiniBatchDictionaryLearning,
            #     "init_params": dict(),
            #     "fit_params": dict(),
            #     "predict_params": dict()
            # },
            # {"clazz": MiniBatchSparsePCA, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            # {"clazz": NMF, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            # {"clazz": PCA, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            # {
            #     "clazz": SparseCoder,
            #     "init_params": {"dictionary": np.array([[0, 1, 0], [-1, -1, 2]])},
            #     "fit_params": dict(),
            #     "predict_params": dict()
            # },
            # {"clazz": SparsePCA, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            # {"clazz": FactorAnalysis, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            # {"clazz": TruncatedSVD, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            # {
            #     "clazz": LatentDirichletAllocation,
            #     "init_params": dict(),
            #     "fit_params": dict(),
            #     "predict_params": dict()
            # },
            
            # gaussian_process
            {"clazz": GaussianProcessRegressor, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            {"clazz": GaussianProcessClassifier, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            
            # discriminant_analysis
            {
                "clazz": LinearDiscriminantAnalysis,
                "init_params": dict(),
                "fit_params": dict(),
                "predict_params": dict()
            },
            {
                "clazz": QuadraticDiscriminantAnalysis,
                "init_params": dict(),
                "fit_params": dict(),
                "predict_params": dict()
            },
            
            # naive_bayes
            {"clazz": BernoulliNB, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            {"clazz": GaussianNB, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            {"clazz": MultinomialNB, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            {"clazz": ComplementNB, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            {"clazz": CategoricalNB, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            
            # kernel_ridge
            {"clazz": KernelRidge, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            
            # neighbors
            {"clazz": KNeighborsClassifier, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            {"clazz": KNeighborsRegressor, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            # {"clazz": KNeighborsTransformer, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            {"clazz": NearestCentroid, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            # {"clazz": NearestNeighbors, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            {"clazz": RadiusNeighborsClassifier, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            {"clazz": RadiusNeighborsRegressor, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            # {
            #     "clazz": RadiusNeighborsTransformer,
            #     "init_params": dict(),
            #     "fit_params": dict(),
            #     "predict_params": dict()
            # },
            # {"clazz": KernelDensity, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            {
                "clazz": LocalOutlierFactor,
                # This model is special: getattr(LocalOutlierFactor(), "predict") is True if novelty = True,
                # Otherwise, getattr(LocalOutlierFactor(), "predict") is False if novelty = False.
                "init_params": {"novelty": True},
                "fit_params": dict(),
                "predict_params": dict()
            },
            # {
            #     "clazz": NeighborhoodComponentsAnalysis,
            #     "init_params": dict(),
            #     "fit_params": dict(),
            #     "predict_params": dict()
            # },

            # tree
            {"clazz": DecisionTreeClassifier, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            {"clazz": DecisionTreeRegressor, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            
            # multiclass
            {
                "clazz": OneVsRestClassifier, 
                "init_params": {"estimator": SVC()}, 
                "fit_params": dict(),
                "predict_params": dict()
            },
            {
                "clazz": OneVsOneClassifier, 
                "init_params": {"estimator": SVC()}, 
                "fit_params": dict(),
                "predict_params": dict()
            },
            {
                "clazz": OutputCodeClassifier, 
                "init_params": {"estimator": SVC()}, 
                "fit_params": dict(),
                "predict_params": dict()
            },
            
            # semi_supervised
            {
                "clazz": SelfTrainingClassifier, 
                "init_params": {"base_estimator": SVC(probability=True, gamma="auto")},
                "fit_params": dict(),
                "predict_params": dict()
            },
            {"clazz": LabelPropagation, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            {"clazz": LabelSpreading, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            
            # isotonic
            {"clazz": IsotonicRegression, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            
            # calibration
            {"clazz": CalibratedClassifierCV, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            
            # mixture
            {"clazz": GaussianMixture, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            {"clazz": BayesianGaussianMixture, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            
            # cluster
            {"clazz": AffinityPropagation, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            # {"clazz": AgglomerativeClustering, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            {"clazz": Birch, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            # {"clazz": DBSCAN, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            # {"clazz": OPTICS, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            {"clazz": KMeans, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            # {"clazz": FeatureAgglomeration, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            {"clazz": MeanShift, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            {"clazz": MiniBatchKMeans, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            # {"clazz": SpectralClustering, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            # {"clazz": SpectralBiclustering, "init_params": dict(), "fit_params": dict(), "predict_params": dict()},
            # {"clazz": SpectralCoclustering, "init_params": dict(), "fit_params": dict(), "predict_params": dict()}
        ]
        self._not_implement_predict_method_sklearn_model_class_set = {
            # sklearn.cluster bad models start
            AgglomerativeClustering,
            DBSCAN,
            OPTICS,
            FeatureAgglomeration,

            # sklearn.decomposition bad models start
            DictionaryLearning,
            FastICA,
            IncrementalPCA,
            KernelPCA,
            MiniBatchDictionaryLearning,
            MiniBatchSparsePCA,
            NMF,
            PCA,
            SparseCoder,
            SparsePCA,
            FactorAnalysis,
            TruncatedSVD,
            LatentDirichletAllocation,
            SpectralClustering,
            SpectralBiclustering,
            SpectralCoclustering,

            # sklearn.cross_decomposition bad models start
            PLSSVD,

            # sklearn.neighbors
            KNeighborsTransformer,
            NearestNeighbors,
            RadiusNeighborsTransformer,
            KernelDensity,
            NeighborhoodComponentsAnalysis
        }
        self._cannot_fit_or_predict_sklearn_model_class_set = {
            # These models have implemented predict method, but has other different errors.
            # You are using LassoLarsIC in the case where the number of samples is smaller than the number
            # of features. In this setting, getting a good estimate for the variance of the noise is not possible.
            # Provide an estimate of the noise variance in the constructor.
            LassoLarsIC,

            # n_samples=6 should be >= n_clusters=8.
            KMeans,

            # The number of samples must be more than the number of classes
            LinearDiscriminantAnalysis,

            # n_splits=5 cannot be greater than the number of members in each class.
            LogisticRegressionCV,

            # For mono-task outputs, use ElasticNetCVCV
            MultiTaskElasticNetCV,

            # For mono-task outputs, use ElasticNet
            MultiTaskLasso,

            # Negative values in data passed to ComplementNB (input X)
            ComplementNB,

            # Negative values in data passed to CategoricalNB (input X)
            CategoricalNB,

            # For mono-task outputs, use LassoCVCV
            MultiTaskLassoCV,

            # n_samples=6 should be >= n_clusters=8.
            MiniBatchKMeans,

            # No neighbors found for test samples array([0]), you can try using larger radius, giving a label for
            # outliers, or considering removing them from your dataset.
            RadiusNeighborsClassifier,

            # For mono-task outputs, use ElasticNet
            MultiTaskElasticNet,

            # `min_samples` may not be larger than number of samples: n_samples = 6.
            RANSACRegressor,

            # y has only 1 sample in class 4.0, covariance is ill defined.
            QuadraticDiscriminantAnalysis,

            # Negative values in data passed to MultinomialNB (input X)
            MultinomialNB,

            # n_splits=5 cannot be greater than the number of members in each class.
            CalibratedClassifierCV,

            # Isotonic regression input X should be a 1d array or 2d array with 1 feature
            IsotonicRegression,

            # This model only failed for scikit-learn 1.1.2, it works for scikit-learn 1.0.2
            # n_components == 2, must be <= 1.
            PLSCanonical,

            # This model only failed for scikit-learn 1.1.2, it works for scikit-learn 1.0.2
            # n_components == 2, must be <= 1.
            CCA
        }
        self._good_to_fit_and_predict_sklearn_model_list = list()
        for model in self._good_to_init_sklearn_model_list:
            if model["clazz"] not in self._cannot_fit_or_predict_sklearn_model_class_set and \
                    model["clazz"] not in self._not_implement_predict_method_sklearn_model_class_set:
                self._good_to_fit_and_predict_sklearn_model_list.append(model)

        super().setUp()

    def test_init_model(self):
        """test SklearnModelWrapper::__init__"""
        ###############################################################
        # case 0 (good case)                                          #
        # 1) models are inherited from sklearn.base.BaseEstimator.    #
        # 1) models has implemented fit and predict callable methods. #
        ###############################################################
        for model in self._good_to_init_sklearn_model_list:
            succeed = True
            try:
                _ = SklearnModelWrapper(
                    in_chunk_len=self._default_in_chunk_len,
                    out_chunk_len=self._default_out_chunk_len,
                    model_class=model["clazz"],
                    model_init_params=model["init_params"]
                )
            except ValueError:
                succeed = False
            self.assertTrue(succeed)

        ###########################
        # case 1 (bad case)       #
        # 1) model_class is None. #
        ###########################
        bad_model_class = None
        succeed = True
        try:
            _ = SklearnModelWrapper(
                in_chunk_len=self._default_in_chunk_len,
                out_chunk_len=self._default_out_chunk_len,
                model_class=bad_model_class
            )
        except ValueError:
            succeed = False
        self.assertFalse(succeed)

        ##############################################
        # case 2 (bad case)                          #
        # 1) isinstance(model_class, type) is False. #
        ##############################################
        bad_model_class = MockNotSklearnModel()
        succeed = True
        try:
            _ = SklearnModelWrapper(
                in_chunk_len=self._default_in_chunk_len,
                out_chunk_len=self._default_out_chunk_len,
                model_class=bad_model_class
            )
        except ValueError:
            succeed = False
        self.assertFalse(succeed)

        #################################################
        # case 3 (bad case)                             #
        # 1) model is NOT inherited from BaseEstimator. #
        #################################################
        bad_model_class = MockNotSklearnModel
        succeed = True
        try:
            _ = SklearnModelWrapper(
                in_chunk_len=self._default_in_chunk_len,
                out_chunk_len=self._default_out_chunk_len,
                model_class=bad_model_class
            )
        except ValueError:
            succeed = False
        self.assertFalse(succeed)

        ####################################################
        # case 4 (bad case)                                #
        # 1) model does NOT implement callable fit method. #
        ####################################################
        # Currently all sklearn models has implemented fit method, thus need to use BaseEstimator for testing.
        bad_model_class = sklearn.base.BaseEstimator
        succeed = True
        try:
            _ = SklearnModelWrapper(
                in_chunk_len=self._default_in_chunk_len,
                out_chunk_len=self._default_out_chunk_len,
                model_class=bad_model_class
            )
        except ValueError:
            succeed = False
        self.assertFalse(succeed)

        ########################################################
        # case 5 (bad case)                                    #
        # 1) model does NOT implement callable predict method. #
        ########################################################
        # lots of models in sklearn.cluster and sklearn.decomposition does not implement predict method, thus not
        # supported by SklearnModelWrapper.
        for bad_model_class in self._not_implement_predict_method_sklearn_model_class_set:
            succeed = True
            try:
                _ = SklearnModelWrapper(
                    in_chunk_len=self._default_in_chunk_len,
                    out_chunk_len=self._default_out_chunk_len,
                    model_class=bad_model_class
                )
            except ValueError:
                succeed = False
            self.assertFalse(succeed)

        #######################
        # case 6 (bad case)   #
        # 1) in_chunk_len < 0 #
        #######################
        bad_in_chunk_len = -1

        succeed = True
        try:
            _ = SklearnModelWrapper(
                in_chunk_len=bad_in_chunk_len,
                out_chunk_len=self._default_out_chunk_len,
                model_class=self._default_model_class
            )
        except ValueError:
            succeed = False
        self.assertFalse(succeed)

        #########################
        # case 7 (bad case)     #
        # 1) out_chunk_len == 0 #
        #########################
        bad_out_chunk_len = 0
        succeed = True
        try:
            _ = SklearnModelWrapper(
                in_chunk_len=self._default_in_chunk_len,
                out_chunk_len=bad_out_chunk_len,
                model_class=self._default_model_class
            )
        except ValueError:
            succeed = False
        self.assertFalse(succeed)

        ########################
        # case 8 (bad case)    #
        # 1) out_chunk_len > 0 #
        ########################
        # Wrapper only supports single-step time series forecasting. The caller can use BaseModel.recursive_predict as
        # an alternative approach if time series forecasting is needed.
        bad_out_chunk_len = 2
        succeed = True
        try:
            _ = SklearnModelWrapper(
                in_chunk_len=self._default_in_chunk_len,
                out_chunk_len=bad_out_chunk_len,
                model_class=self._default_model_class,
            )
        except ValueError:
            succeed = False
        self.assertFalse(succeed)

        #########################
        # case 9 (bad case)     #
        # 1) skip_chunk_len < 0 #
        #########################
        bad_skip_chunk_len = -1
        succeed = True
        try:
            _ = SklearnModelWrapper(
                in_chunk_len=self._default_in_chunk_len,
                skip_chunk_len=bad_skip_chunk_len,
                out_chunk_len=self._default_out_chunk_len,
                model_class=self._default_model_class
            )
        except ValueError:
            succeed = False
        self.assertFalse(succeed)

        ###########################
        # case 10 (bad case)       #
        # 1) sampling_stride == 0 #
        ###########################
        bad_sampling_stride = 0
        succeed = True
        try:
            _ = SklearnModelWrapper(
                in_chunk_len=self._default_in_chunk_len,
                out_chunk_len=self._default_out_chunk_len,
                sampling_stride=bad_sampling_stride,
                model_class=self._default_model_class
            )
        except ValueError:
            succeed = False
        self.assertFalse(succeed)

        ###########################
        # case 11 (bad case)       #
        # 1) sampling_stride < 0  #
        ###########################
        bad_sampling_stride = -1
        succeed = True
        try:
            _ = SklearnModelWrapper(
                in_chunk_len=self._default_in_chunk_len,
                out_chunk_len=self._default_out_chunk_len,
                sampling_stride=bad_sampling_stride,
                model_class=self._default_model_class
            )
        except ValueError:
            succeed = False
        self.assertFalse(succeed)

        ################################
        # case 12 (bad case)            #
        # 1) invalid_model_init_params #
        ################################
        bad_model_init_params = {"mock_bad_key": "mock_bad_value"}
        succeed = True
        try:
            _ = SklearnModelWrapper(
                in_chunk_len=self._default_in_chunk_len,
                out_chunk_len=self._default_out_chunk_len,
                model_class=self._default_model_class,
                model_init_params=bad_model_init_params
            )
        except ValueError:
            succeed = False
        self.assertFalse(succeed)

    def test_fit_and_predict(self):
        """
        test SklearnModelWrapper::fit and SklearnModelWrapper::predict.
        """
        #######################################
        # case 0 (good case)                  #
        # 1) all optional params are default. #
        # 2) target.columns == 1.             #
        # 3) known cov not None.              #
        # 4) observed cov not None.           #
        #######################################
        target_col_num = 1
        target_periods = 10
        known_periods = target_periods + 10
        observed_periods = target_periods

        for model in self._good_to_fit_and_predict_sklearn_model_list:
            if model["fit_params"] != dict() or model["predict_params"] != dict():
                continue
            paddlets_ds = self._build_mock_ts_dataset(
                target_col_num=target_col_num,
                target_periods=target_periods,
                known_periods=known_periods,
                observed_periods=observed_periods
            )
            model_wrapper = SklearnModelWrapper(
                in_chunk_len=self._default_in_chunk_len,
                out_chunk_len=self._default_out_chunk_len,
                model_class=model["clazz"],
                model_init_params=model["init_params"]
            )
            model_wrapper.fit(train_data=paddlets_ds)
            predicted_ds = model_wrapper.predict(paddlets_ds)
            self.assertIsNotNone(predicted_ds.get_target())

        #######################################
        # case 1 (good case)                  #
        # 1) all optional params are default. #
        # 2) target.columns == 1.             #
        # 3) known cov is None.               #
        # 4) observed cov is NOT None.        #
        #######################################
        target_col_num = 1
        target_periods = 10
        known_periods = target_periods + 10
        observed_periods = target_periods

        for model in self._good_to_fit_and_predict_sklearn_model_list:
            if model["fit_params"] != dict() or model["predict_params"] != dict():
                continue
            paddlets_ds = self._build_mock_ts_dataset(
                target_col_num=target_col_num,
                target_periods=target_periods,
                known_periods=known_periods,
                observed_periods=observed_periods
            )
            paddlets_ds.known_cov = None

            model_wrapper = SklearnModelWrapper(
                in_chunk_len=self._default_in_chunk_len,
                out_chunk_len=self._default_out_chunk_len,
                model_class=model["clazz"],
                model_init_params=model["init_params"]
            )
            model_wrapper.fit(train_data=paddlets_ds)
            predicted_ds = model_wrapper.predict(paddlets_ds)
            self.assertIsNotNone(predicted_ds.get_target())

        #######################################
        # case 2 (good case)                  #
        # 1) all optional params are default. #
        # 2) target.columns == 1.             #
        # 3) known cov is NOT None.           #
        # 4) observed cov is None.            #
        #######################################
        target_col_num = 1
        target_periods = 10
        known_periods = target_periods + 10
        observed_periods = target_periods

        for model in self._good_to_fit_and_predict_sklearn_model_list:
            if model["fit_params"] != dict() or model["predict_params"] != dict():
                continue
            paddlets_ds = self._build_mock_ts_dataset(
                target_col_num=target_col_num,
                target_periods=target_periods,
                known_periods=known_periods,
                observed_periods=observed_periods
            )
            paddlets_ds.observed_cov = None

            model_wrapper = SklearnModelWrapper(
                in_chunk_len=self._default_in_chunk_len,
                out_chunk_len=self._default_out_chunk_len,
                model_class=model["clazz"],
                model_init_params=model["init_params"]
            )
            model_wrapper.fit(train_data=paddlets_ds)
            predicted_ds = model_wrapper.predict(paddlets_ds)
            self.assertIsNotNone(predicted_ds.get_target())

        #######################################
        # case 3 (good case)                  #
        # 1) all optional params are default. #
        # 2) target.columns == 1.             #
        # 3) known cov is None.               #
        # 4) observed cov is None.            #
        #######################################
        target_col_num = 1
        target_periods = 10
        known_periods = target_periods + 10
        observed_periods = target_periods

        for model in self._good_to_fit_and_predict_sklearn_model_list:
            if model["fit_params"] != dict() or model["predict_params"] != dict():
                continue
            paddlets_ds = self._build_mock_ts_dataset(
                target_col_num=target_col_num,
                target_periods=target_periods,
                known_periods=known_periods,
                observed_periods=observed_periods
            )
            paddlets_ds.observed_cov = None
            paddlets_ds.known_cov = None

            model_wrapper = SklearnModelWrapper(
                in_chunk_len=self._default_in_chunk_len,
                out_chunk_len=self._default_out_chunk_len,
                model_class=model["clazz"],
                model_init_params=model["init_params"]
            )
            model_wrapper.fit(train_data=paddlets_ds)
            predicted_ds = model_wrapper.predict(paddlets_ds)
            self.assertIsNotNone(predicted_ds.get_target())

        ##########################################
        # case 4 (good case)                      #
        # 1) in_chunk_len = 0.                   #
        # 2) known cov is None.                  #
        # 3) observed cov is None.               #
        # 4) others optional params are default. #
        ##########################################
        target_col_num = 1
        target_periods = 10
        known_periods = target_periods + 10
        observed_periods = target_periods

        for model in self._good_to_fit_and_predict_sklearn_model_list:
            if model["fit_params"] != dict() or model["predict_params"] != dict():
                continue
            paddlets_ds = self._build_mock_ts_dataset(
                target_col_num=target_col_num,
                target_periods=target_periods,
                known_periods=known_periods,
                observed_periods=observed_periods
            )
            paddlets_ds.observed_cov = None
            paddlets_ds.known_cov = None

            model_wrapper = SklearnModelWrapper(
                in_chunk_len=self._default_in_chunk_len,
                out_chunk_len=self._default_out_chunk_len,
                model_class=model["clazz"],
                model_init_params=model["init_params"]
            )

            model_wrapper.fit(train_data=paddlets_ds)
            predicted_ds = model_wrapper.predict(paddlets_ds)
            self.assertIsNotNone(predicted_ds.get_target())

        #################################################################
        # case 5 (good case)                                            #
        # 1) Non default ml_dataloader_to_fit_ndarray function.         #
        # 2) Non default udf_ml_dataloader_to_predict_ndarray function. #
        # 3) others optional params are default.                        #
        # 4) target.columns == 1.                                       #
        #################################################################
        target_col_num = 1
        target_periods = 10
        known_periods = target_periods + 10
        observed_periods = target_periods

        for model in self._good_to_fit_and_predict_sklearn_model_list:
            if model["fit_params"] != dict() or model["predict_params"] != dict():
                continue
            paddlets_ds = self._build_mock_ts_dataset(
                target_col_num=target_col_num,
                target_periods=target_periods,
                known_periods=known_periods,
                observed_periods=observed_periods
            )

            model_wrapper = SklearnModelWrapper(
                in_chunk_len=self._default_in_chunk_len,
                out_chunk_len=self._default_out_chunk_len,
                model_class=model["clazz"],
                model_init_params=model["init_params"],
                udf_ml_dataloader_to_fit_ndarray=self.udf_ml_dataloader_to_fit_ndarray,
                udf_ml_dataloader_to_predict_ndarray=self.udf_ml_dataloader_to_predict_ndarray
            )
            model_wrapper.fit(train_data=paddlets_ds)
            predicted_ds = model_wrapper.predict(paddlets_ds)
            self.assertIsNotNone(predicted_ds.get_target())

        ##################################################################
        # case 6 (good case)                                             #
        # 1) has extra fit params.                                       #
        # 2) has extra predict params.                                   #
        # 3) other optional params are default.                          #
        # 4) target.columns == 1.                                        #
        ##################################################################
        target_col_num = 1
        target_periods = 10
        known_periods = target_periods + 10
        observed_periods = target_periods

        for model in self._good_to_fit_and_predict_sklearn_model_list:
            if model["fit_params"] == dict() or model["predict_params"] == dict():
                continue
            paddlets_ds = self._build_mock_ts_dataset(
                target_col_num=target_col_num,
                target_periods=target_periods,
                known_periods=known_periods,
                observed_periods=observed_periods
            )
            model_wrapper = SklearnModelWrapper(
                in_chunk_len=self._default_in_chunk_len,
                out_chunk_len=self._default_out_chunk_len,
                model_class=model["clazz"],
                model_init_params=model["init_params"],
                fit_params=model["fit_params"],
                predict_params=model["predict_params"]
            )
            model_wrapper.fit(train_data=paddlets_ds)
            predicted_ds = model_wrapper.predict(paddlets_ds)
            self.assertIsNotNone(predicted_ds.get_target())

        ######################
        # case 7 (bad case)  #
        # 1) target is None. #
        ######################
        paddlets_ds = self._build_mock_ts_dataset()
        paddlets_ds.target = None

        model_wrapper = SklearnModelWrapper(
            in_chunk_len=self._default_in_chunk_len,
            out_chunk_len=self._default_out_chunk_len,
            model_class=self._default_model_class
        )

        succeed = True
        try:
            model_wrapper.fit(train_data=paddlets_ds)
        except ValueError:
            succeed = False
        self.assertFalse(succeed)

        ##########################
        # case 8 (bad case)      #
        # 1) target.columns > 1. #
        ##########################
        target_col_num = 2
        target_periods = 10
        known_periods = target_periods + 10
        observed_periods = target_periods
        paddlets_ds = self._build_mock_ts_dataset(
            target_col_num=target_col_num,
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods
        )

        model_wrapper = SklearnModelWrapper(
            in_chunk_len=self._default_in_chunk_len,
            out_chunk_len=self._default_out_chunk_len,
            model_class=self._default_model_class
        )

        succeed = True
        try:
            model_wrapper.fit(train_data=paddlets_ds)
        except ValueError:
            succeed = False
        self.assertFalse(succeed)

        ################################################
        # case 9 (bad case)                            #
        # 1) target.dtype != numeric (i.e. np.float32) #
        ################################################
        target_col_num = 1
        target_periods = 10
        known_periods = target_periods + 10
        observed_periods = target_periods
        paddlets_ds = self._build_mock_ts_dataset(
            target_col_num=target_col_num,
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods,
            # Explicitly set target to (invalid) np.int64 to repro this bad case.
            target_dtype=np.int64
        )
        model_wrapper = SklearnModelWrapper(
            in_chunk_len=self._default_in_chunk_len,
            out_chunk_len=self._default_out_chunk_len,
            model_class=self._default_model_class
        )

        succeed = True
        try:
            # target.dtype is NOT numeric (np.float32), bad.
            model_wrapper.fit(train_data=paddlets_ds)
        except ValueError:
            succeed = False
        self.assertFalse(succeed)
        
    def test_save(self):
        """
        test SklearnModelWrapper::save (inherited from MLBaseModel::save).

        Here will only test good case to ensure that it works for child class SklearnModelWrapper. To know more about
        bad cases, see unittest for MLBaseModel (tests.models.ml.test_ml_base.py::test_save).
        """
        ########################################
        # case 0 (good case)                   #
        # 1) path exists.                      #
        # 2) No file conflicts.                #
        ########################################
        model_wrapper = SklearnModelWrapper(
            in_chunk_len=self._default_in_chunk_len,
            out_chunk_len=self._default_out_chunk_len,
            model_class=self._default_model_class
        )

        target_col_num = 1
        target_periods = 10
        known_periods = target_periods + 10
        observed_periods = target_periods
        paddlets_ds = self._build_mock_ts_dataset(
            target_col_num=target_col_num,
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods
        )

        model_wrapper.fit(train_data=paddlets_ds)
        predicted_ds = model_wrapper.predict(paddlets_ds)
        self.assertIsNotNone(predicted_ds.get_target())

        path = os.path.join(os.getcwd(), str(random.randint(1, 10000000)))
        os.mkdir(path)
        model_wrapper.save(os.path.join(path, self._default_modelname))

        files = set(os.listdir(path))
        internal_filename_map = {
            "model_meta": "%s_%s" % (self._default_modelname, "model_meta")
        }
        self.assertEqual(files, {self._default_modelname, *internal_filename_map.values()})

        # model type
        with open(os.path.join(path, internal_filename_map["model_meta"]), "r") as f:
            model_meta = json.load(f)
        self.assertTrue(SklearnModelWrapper.__name__ in model_meta["ancestor_classname_set"])
        self.assertTrue(MLBaseModel.__name__ in model_meta["ancestor_classname_set"])
        self.assertEqual(SklearnModelWrapper.__module__, model_meta["modulename"])
        shutil.rmtree(path)

        #########################################################
        # case 1 (good case)                                    #
        # 1) path exists.                                       #
        # 2) No file conflicts.                                 #
        # 3) the same model can be saved twice at the same dir. #
        #########################################################
        # save the same model instance twice with different name at same path.
        path = os.path.join(os.getcwd(), str(random.randint(1, 10000000)))
        os.mkdir(path)

        # build and fit the only model instance.
        model_wrapper = SklearnModelWrapper(
            in_chunk_len=self._default_in_chunk_len,
            out_chunk_len=self._default_out_chunk_len,
            model_class=self._default_model_class
        )

        target_col_num = 1
        target_periods = 10
        known_periods = target_periods + 10
        observed_periods = target_periods
        paddlets_ds = self._build_mock_ts_dataset(
            target_col_num=target_col_num,
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods
        )
        model_wrapper.fit(train_data=paddlets_ds)

        # save the first one.
        model_1_name = "a"
        model_1_internal_filename_map = {"model_meta": "%s_%s" % (model_1_name, "model_meta")}
        model_wrapper.save(os.path.join(path, model_1_name))

        # save the second one.
        model_2_name = "b"
        model_2_internal_filename_map = {"model_meta": "%s_%s" % (model_2_name, "model_meta")}
        model_wrapper.save(os.path.join(path, model_2_name))

        files = set(os.listdir(path))
        self.assertEqual(
            files,
            {
                model_1_name,
                *model_1_internal_filename_map.values(),
                model_2_name,
                *model_2_internal_filename_map.values()
            }
        )

        shutil.rmtree(path)

    def test_load(self):
        """
        test SklearnModelWrapper::load (inherited from MLBaseModel::load).

        Here will only test good case to ensure that it works for child class SklearnModelWrapper. To know more about
        bad cases, see unittest for MLBaseModel (tests.models.ml.test_ml_base.py::test_load).
        """
        ####################################################################################
        # case 0 (good case)                                                               #
        # 1) model exists in the given path.                                               #
        # 2) the saved model is fitted before loading.                                     #
        # 3) the predicted result for loaded model is identical to the one before loading. #
        ####################################################################################
        # build + fit + save
        model_wrapper = SklearnModelWrapper(
            in_chunk_len=self._default_in_chunk_len,
            out_chunk_len=self._default_out_chunk_len,
            model_class=self._default_model_class
        )

        target_col_num = 1
        target_periods = 10
        known_periods = target_periods + 10
        observed_periods = target_periods
        paddlets_ds = self._build_mock_ts_dataset(
            target_col_num=target_col_num,
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods
        )
        model_wrapper.fit(train_data=paddlets_ds)

        # store predicted dataset before load
        pred_ds_before_load = model_wrapper.predict(paddlets_ds)
        self.assertEqual(
            (self._default_out_chunk_len, len(paddlets_ds.get_target().data.columns)),
            pred_ds_before_load.get_target().data.shape
        )

        path = os.path.join(os.getcwd(), str(random.randint(1, 10000000)))
        os.mkdir(path)

        abs_model_path = os.path.join(path, self._default_modelname)
        model_wrapper.save(abs_model_path)

        # load model
        loaded_model_wrapper = MLBaseModel.load(abs_model_path)

        # model type expected
        self.assertTrue(isinstance(loaded_model_wrapper, SklearnModelWrapper))

        # predict using loaded model
        pred_ds_after_load = loaded_model_wrapper.predict(paddlets_ds)

        # compare predicted dataset
        self.assertTrue(np.alltrue(
            pred_ds_before_load.get_target().to_numpy(False) == pred_ds_after_load.get_target().to_numpy(False)
        ))
        shutil.rmtree(path)

        #############################################################################################################
        # case 1 (good case)                                                                                        #
        # 1) model exists in the given path.                                                                        #
        # 2) the saved model is fitted before loading.                                                              #
        # 3) will load 2 model instances from the same saved model file, namely, loaded_model_1 and loaded_model_2. #
        # 4) guarantee that loaded_model_1.predict(data) == loaded_model_2.predict(data)                            #
        #############################################################################################################
        # build + fit + save
        model_wrapper = SklearnModelWrapper(
            in_chunk_len=self._default_in_chunk_len,
            out_chunk_len=self._default_out_chunk_len,
            model_class=self._default_model_class
        )

        target_col_num = 1
        target_periods = 10
        known_periods = target_periods + 10
        observed_periods = target_periods
        paddlets_ds = self._build_mock_ts_dataset(
            target_col_num=target_col_num,
            target_periods=target_periods,
            known_periods=known_periods,
            observed_periods=observed_periods
        )
        model_wrapper.fit(train_data=paddlets_ds)

        # store predicted dataset before load
        pred_ds_before_load = model_wrapper.predict(paddlets_ds)
        self.assertEqual(
            (self._default_out_chunk_len, len(paddlets_ds.get_target().data.columns)),
            pred_ds_before_load.get_target().data.shape
        )

        path = os.path.join(os.getcwd(), str(random.randint(1, 10000000)))
        os.mkdir(path)

        # use same mode instance to save the first model file.
        model_1_name = "a"
        abs_model_1_path = os.path.join(path, model_1_name)
        model_wrapper.save(abs_model_1_path)

        # use same mode instance to save the second model file.
        model_2_name = "b"
        abs_model_2_path = os.path.join(path, model_2_name)
        model_wrapper.save(abs_model_2_path)

        # load 2 models
        loaded_model_1 = MLBaseModel.load(abs_model_1_path)
        loaded_model_2 = MLBaseModel.load(abs_model_2_path)

        # model type expected
        self.assertEqual(model_wrapper.__class__, loaded_model_1.__class__)
        self.assertEqual(model_wrapper.__class__, loaded_model_2.__class__)

        # predicted results expected.
        loaded_model_1_pred_ds = loaded_model_1.predict(paddlets_ds)
        loaded_model_2_pred_ds = loaded_model_2.predict(paddlets_ds)
        self.assertTrue(np.alltrue(
            pred_ds_before_load.get_target().to_numpy(False) == loaded_model_1_pred_ds.get_target().to_numpy(False)
        ))
        self.assertTrue(np.alltrue(
            pred_ds_before_load.get_target().to_numpy(False) == loaded_model_2_pred_ds.get_target().to_numpy(False)
        ))
        shutil.rmtree(path)

    def test_make_ml_model(self):
        """
        test ml_model_wrapper::make_ml_model.
        """
        ######################
        # case 0 (good case) #
        # 1) sklearn model.  #
        ######################
        target_col_num = 1
        target_periods = 10
        known_periods = target_periods + 10
        observed_periods = target_periods
        for model in self._good_to_fit_and_predict_sklearn_model_list:
            paddlets_ds = self._build_mock_ts_dataset(
                target_col_num=target_col_num,
                target_periods=target_periods,
                known_periods=known_periods,
                observed_periods=observed_periods
            )

            model_wrapper = make_ml_model(
                in_chunk_len=self._default_in_chunk_len,
                out_chunk_len=self._default_out_chunk_len,
                model_class=model["clazz"],
                model_init_params=model["init_params"]
            )

            model_wrapper.fit(train_data=paddlets_ds)

            predicted_ds = model_wrapper.predict(paddlets_ds)
            self.assertIsNotNone(predicted_ds.get_target())

        ###########################
        # case 1 (bad case)       #
        # 1) model_class is None. #
        ###########################
        bad_model_class = None
        succeed = True
        try:
            _ = make_ml_model(
                in_chunk_len=self._default_in_chunk_len,
                out_chunk_len=self._default_out_chunk_len,
                model_class=bad_model_class
            )
        except ValueError:
            succeed = False
        self.assertFalse(succeed)

        ##############################################
        # case 2 (bad case)                          #
        # 1) isinstance(model_class, type) is False. #
        ##############################################
        bad_model_class = MockNotSklearnModel()
        succeed = True
        try:
            _ = make_ml_model(
                in_chunk_len=self._default_in_chunk_len,
                out_chunk_len=self._default_out_chunk_len,
                model_class=bad_model_class
            )
        except ValueError:
            succeed = False
        self.assertFalse(succeed)

    @staticmethod
    def _build_mock_ts_dataset(
        target_col_num: int = 1,
        known_col_num: int = 2,
        observed_col_num: int = 2,
        target_periods: int = 10,
        known_periods: int = 10,
        observed_periods: int = 10,
        target_start_timestamp: pd.Timestamp = pd.Timestamp(datetime.datetime.now().date()),
        known_start_timestamp: pd.Timestamp = pd.Timestamp(datetime.datetime.now().date()),
        observed_start_timestamp: pd.Timestamp = pd.Timestamp(datetime.datetime.now().date()),
        freq: str = "1D",
        cov_dtypes_contain_numeric: bool = True,
        cov_dtypes_contain_categorical: bool = True,
        target_dtype: type = np.float32
    ):
        """
        Build mock paddlets dataset.

        all timeseries must have same freq.
        """
        numeric_dtype = np.float32
        categorical_dtype = np.int64

        # target (ml model requires target col num MUST == 1, thus cannot both contain numeric + categorical).
        target_df = pd.DataFrame(
            np.array([[i for n in range(target_col_num)] for i in range(target_periods)], dtype=target_dtype),
            index=pd.date_range(start=target_start_timestamp, periods=target_periods, freq=freq),
            columns=[f"target{n}" for n in range(target_col_num)],
        )

        # known
        known_raw_data = [[i * (10 ** (n + 1)) for n in range(known_col_num)] for i in range(known_periods)]
        # known_raw_data = [(i * 10, i * 100) for i in range(known_periods)]
        known_numeric_df = None
        if cov_dtypes_contain_numeric:
            # numeric
            known_numeric_data = np.array(known_raw_data, dtype=numeric_dtype)
            known_numeric_df = pd.DataFrame(
                data=known_numeric_data,
                index=pd.date_range(start=known_start_timestamp, periods=known_periods, freq=freq),
                columns=["known_numeric_0", "known_numeric_1"]
            )

        known_categorical_df = None
        if cov_dtypes_contain_categorical:
            # categorical
            known_categorical_data = np.array(known_raw_data, dtype=categorical_dtype)
            known_categorical_df = pd.DataFrame(
                data=known_categorical_data,
                index=pd.date_range(start=known_start_timestamp, periods=known_periods, freq=freq),
                columns=["known_categorical_0", "known_categorical_1"]
            )
        if (known_numeric_df is None) and (known_categorical_df is None):
            raise Exception(f"failed to build known cov data, both numeric df and categorical df are all None.")
        if (known_numeric_df is not None) and (known_categorical_df is not None):
            # both are NOT None.
            known_cov_df = pd.concat([known_numeric_df, known_categorical_df], axis=1)
        else:
            known_cov_df = [known_numeric_df, known_categorical_df][1 if known_numeric_df is None else 0]

        # observed
        observed_raw_data = [[i * (-10 ** (n + 1)) for n in range(observed_col_num)] for i in range(observed_periods)]
        # observed_raw_data = [(i * -1, i * -10) for i in range(observed_periods)]
        observed_numeric_df = None
        if cov_dtypes_contain_numeric:
            # numeric
            observed_numeric_data = np.array(observed_raw_data, dtype=numeric_dtype)
            observed_numeric_df = pd.DataFrame(
                data=observed_numeric_data,
                index=pd.date_range(start=observed_start_timestamp, periods=observed_periods, freq=freq),
                columns=["observed_numeric_0", "observed_numeric_1"]
            )

        observed_categorical_df = None
        if cov_dtypes_contain_categorical:
            # categorical
            observed_categorical_data = np.array(observed_raw_data, dtype=categorical_dtype)
            observed_categorical_df = pd.DataFrame(
                data=observed_categorical_data,
                index=pd.date_range(start=observed_start_timestamp, periods=observed_periods, freq=freq),
                columns=["observed_categorical_0", "observed_categorical_1"]
            )

        if (observed_numeric_df is None) and (observed_categorical_df is None):
            raise Exception(f"failed to build observed cov data, both numeric df and categorical df are all None.")
        if (observed_numeric_df is not None) and (observed_categorical_df is not None):
            # both are NOT None.
            observed_cov_df = pd.concat([observed_numeric_df, observed_categorical_df], axis=1)
        else:
            observed_cov_df = [observed_numeric_df, observed_categorical_df][
                1 if observed_numeric_df is None else 0]

        # static
        static = dict()
        if cov_dtypes_contain_numeric:
            # numeric
            static["static_numeric"] = np.float32(1)
        if cov_dtypes_contain_categorical:
            # categorical
            static["static_categorical"] = np.int64(2)

        return TSDataset(
            target=TimeSeries.load_from_dataframe(data=target_df),
            known_cov=TimeSeries.load_from_dataframe(data=known_cov_df),
            observed_cov=TimeSeries.load_from_dataframe(data=observed_cov_df),
            static_cov=static
        )

    @staticmethod
    def udf_ml_dataloader_to_fit_ndarray(
        ml_dataloader: MLDataLoader,
        model_init_params: Dict[str, Any],
        in_chunk_len: int,
        skip_chunk_len: int,
        out_chunk_len: int
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        data = next(ml_dataloader)

        sample_x_keys = data.keys() - {"future_target"}
        if in_chunk_len < 1:
            # lag scenario cannot use past_target as features.
            sample_x_keys -= {"past_target"}
        # concatenated ndarray will follow the below ordered list rule:
        # [rule 1] past_target features will ALWAYS be on the left side of known_cov features.
        # [rule 2] numeric features will ALWAYS be on the left side of categorical features.
        full_ordered_x_key_list = ["past_target"]
        full_ordered_x_key_list.extend(
            [f"{t[1]}_{t[0]}" for t in product(["numeric", "categorical"], ["known_cov", "observed_cov", "static_cov"])]
        )

        # For example, given:
        # sample_keys (un-ordered) = {"static_cov_categorical", "known_cov_numeric", "observed_cov_categorical"}
        # full_ordered_x_key_list = [
        #   "past_target",
        #   "known_cov_numeric",
        #   "observed_cov_numeric",
        #   "static_cov_numeric",
        #   "known_cov_categorical",
        #   "observed_cov_categorical",
        #   "static_cov_categorical"
        # ]
        # Thus, actual_ordered_x_key_list = [
        #   "known_cov_numeric",
        #   "observed_cov_categorical",
        #   "static_cov_categorical"
        # ]
        actual_ordered_x_key_list = []
        for k in full_ordered_x_key_list:
            if k in sample_x_keys:
                actual_ordered_x_key_list.append(k)

        reshaped_x_ndarray_list = []
        for k in actual_ordered_x_key_list:
            ndarray = data[k]
            # 3-dim -> 2-dim
            reshaped_ndarray = ndarray.reshape(ndarray.shape[0], ndarray.shape[1] * ndarray.shape[2])
            reshaped_x_ndarray_list.append(reshaped_ndarray)
        x = np.hstack(tup=reshaped_x_ndarray_list)

        # sklearn requires that y.shape must be (n_samples, ), so the invocation of np.squeeze() is required.
        # As we already make pre-assertions in _validate_train_data(), thus we can ensure the following:
        # 1. target.dtype must be np.float32 (i.e., numeric);
        # 2. len(target.columns) must == 1;
        y = np.squeeze(data["future_target"])
        return x, y

    @staticmethod
    def udf_ml_dataloader_to_predict_ndarray(
        ml_dataloader: MLDataLoader,
        model_init_params: Dict[str, Any],
        in_chunk_len: int,
        skip_chunk_len: int,
        out_chunk_len: int
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        data = next(ml_dataloader)

        sample_x_keys = data.keys() - {"future_target"}
        if in_chunk_len < 1:
            # lag scenario cannot use past_target as features.
            sample_x_keys -= {"past_target"}
        # concatenated ndarray will follow the below ordered list rule:
        # [rule 1] past_target features will ALWAYS be on the left side of known_cov features.
        # [rule 2] numeric features will ALWAYS be on the left side of categorical features.
        product_keys = product(["numeric", "categorical"], ["known_cov", "observed_cov", "static_cov"])
        full_ordered_x_key_list = ["past_target"] + [f"{t[1]}_{t[0]}" for t in product_keys]

        # For example, given:
        # sample_keys (un-ordered) = {"static_cov_categorical", "known_cov_numeric", "observed_cov_categorical"}
        # full_ordered_x_key_list = [
        #   "past_target",
        #   "known_cov_numeric",
        #   "observed_cov_numeric",
        #   "static_cov_numeric",
        #   "known_cov_categorical",
        #   "observed_cov_categorical",
        #   "static_cov_categorical"
        # ]
        # Thus, actual_ordered_x_key_list = [
        #   "known_cov_numeric",
        #   "observed_cov_categorical",
        #   "static_cov_categorical"
        # ]
        actual_ordered_x_key_list = []
        for k in full_ordered_x_key_list:
            if k in sample_x_keys:
                actual_ordered_x_key_list.append(k)

        reshaped_x_ndarray_list = []
        for k in actual_ordered_x_key_list:
            ndarray = data[k]
            # 3-dim -> 2-dim
            reshaped_ndarray = ndarray.reshape(ndarray.shape[0], ndarray.shape[1] * ndarray.shape[2])
            reshaped_x_ndarray_list.append(reshaped_ndarray)
        x = np.hstack(tup=reshaped_x_ndarray_list)
        return x, None
