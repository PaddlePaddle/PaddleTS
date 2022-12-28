# !/usr/bin/env python3
# -*- coding:utf-8 -*-
from typing import Callable, Tuple, List

import numpy as np
from sklearn.base import clone
from sklearn.cluster import KMeans

from paddlets.datasets.tsdataset import TSDataset
from paddlets.models.representation.dl.repr_base import ReprBaseModel
from paddlets.models.representation import TS2Vec
from paddlets.ensemble.base import EnsembleBase
from paddlets.ensemble.stacking_ensemble import StackingEnsembleBase
from paddlets.logger.logger import raise_if, Logger, raise_log, raise_if_not

logger = Logger(__name__)


class ReprCluster(StackingEnsembleBase):
    """
    The ReprCluster Class.

    Args:
        repr_model(ReprBasemodel): Representation model to use for cluster.
        repr_model_params(dict):params for reprmodel init.
        encode_params(dict):params for reprmodel encode.
        downstream_learner(Callable): The downstream learner, should be a sklearn-like cluster, set to KMeans() by default.
        verbose(bool): Turn on Verbose mode,set to true by default.

    """

    def __init__(self,
                 repr_model: ReprBaseModel,
                 repr_model_params: dict = None,
                 encode_params: dict = None,
                 downstream_learner: Callable = None,
                 verbose: bool = False
                 ) -> None:

        if repr_model_params is None:
            repr_model_params = {}
        if encode_params is None:
            encode_params = {}
        raise_if(not isinstance(repr_model_params, dict), "model_params should be a params dict")
        raise_if(not isinstance(encode_params, dict), "encode_params should be a params dict")
        self._repr_model = repr_model
        self._repr_model_params = repr_model_params
        self._encode_params = encode_params

        super().__init__([(repr_model, repr_model_params)], downstream_learner, verbose)

    def _set_params(self, estimators) -> None:
        """
        Set estimators params

        Set params and initial estimators.

        Args:

            estimators(List[Tuple[object, dict]] ): A list of tuple (class,params) consisting of several paddlets models.

        """
        if "verbose" not in self._encode_params:
            self._encode_params["verbose"] = False
        #initalize in fit()
        self._estimators = estimators

    def _check_estimators(self, estimators) -> None:
        """
        Check estimators

        Check and valid estimators

        Args:

            estimators(List[Tuple[object, dict]] ): A list of tuple (class,params) consisting of several paddlets models.

        """
        super()._check_estimators(estimators)

        if all([issubclass(e[0], ReprBaseModel) for e in estimators]):
            pass
        else:
            raise ValueError("Estimators have unsupported or uncompatible models")  

    def _check_tsdataset_list(self,
            tsdataset_list: List[TSDataset]) -> None:

        raise_if(not isinstance(tsdataset_list,list) or 
                not all([isinstance(data,TSDataset) for data in tsdataset_list]),
                "Fit dataset should be type of List[TSDataset] ")
        raise_if(len(tsdataset_list) == 0, "Fit data list length == 0")
        raise_if(tsdataset_list[0].target is None, "Target is None")
        len_each_data = len(tsdataset_list[0].target)
        raise_if(not all([len(data.target) == len_each_data for data in tsdataset_list]),
                "Only support equal length tsdataset_lists as predict data")

    def fit(self,
            train_datasets: List[TSDataset]) -> None:
        """
        fit 

        Args:
            tsdataset_list(TSDataset): train data.

        """
        self._check_tsdataset_list(train_datasets)
        X_meta = self._generate_fit_meta_data(train_datasets)

        self._final_learner.fit(X_meta)

        len_each_data = len(train_datasets[0].target)
        self._fit_data_length = len_each_data 
        self._fitted = True

    def _generate_fit_meta_data(self, tsdataset_list: List[TSDataset]) -> np.ndarray:
        """
        Generate fit meta data

        Args:
            tsdataset_list(TSDataset): train data.
            labels: labels, length equal to length of tsdataset_list
        """
        # fit repr
        len_each_data = len(tsdataset_list[0].target)
        repr_model_params = self._repr_model_params.copy()
        if "segment_size" in repr_model_params:
            raise_if(repr_model_params["segment_size"]!= len_each_data, 
            f"Repr model param segment_size {repr_model_params['segment_size']}, should equal to each TSdata size {len_each_data}") 
        else:
            repr_model_params["segment_size"] = len_each_data

        #
        self._estimators = [self._repr_model(**repr_model_params)]
        self._estimators[0].fit(tsdataset_list)

        # encode
        encode_params = self._encode_params.copy()
        if self._repr_model is TS2Vec:
            if "sliding_len" in encode_params:
                raise_if(encode_params["sliding_len"] != len_each_data -1,
                         f" TS2Vec encode param sliding_len ({encode_params['sliding_len']})should \
                 equal to len_each_data{len_each_data} when use ts2vec")
            else:
                encode_params["sliding_len"] = len_each_data - 1

        encode_res = [self._estimators[0].encode(data, **encode_params) for data in tsdataset_list]

        X_meta = np.stack([e[0][-1] for e in encode_res])


        return X_meta

        
    def predict(self,
                tsdatasets: List[TSDataset]) -> np.ndarray:
        """
        Predict

        Args:
            tsdataset_list(TSDataset): predict data.
        """
        raise_if(not self._fitted,"Please fit model first")

        self._check_tsdataset_list(tsdatasets)

        len_each_data = len(tsdatasets[0].target)
        raise_if_not(len_each_data==self._fit_data_length,"predict data should have equal length with fit data")

        X_meta = self._generate_predict_meta_data(tsdatasets)
        y_pred = self._final_learner.predict(X_meta)

        return y_pred

    def _generate_predict_meta_data(self, tsdataset_list: List[TSDataset]) -> np.ndarray:
        """
        Generate predict meta data

        Args:
            tsdataset_list(TSDataset): predict data.
        """
        len_each_data = len(tsdataset_list[0].target)
        encode_params = self._encode_params.copy()
        if self._repr_model is TS2Vec:
            if "sliding_len" in encode_params:
                raise_if(encode_params["sliding_len"] != len_each_data -1,
                         f" TS2Vec encode param sliding_len ({encode_params['sliding_len']})should \
                 equal to len_each_data{len_each_data} when use ts2vec")
            else:
                encode_params["sliding_len"] = len_each_data - 1
            
        encode_res = [self._estimators[0].encode(data, **encode_params) for data in tsdataset_list]

        X_meta = np.stack([e[0][-1] for e in encode_res])
        return X_meta

    def _check_final_learner(self, final_learner) -> None:
        """
        Check if a final learner is given and if it is valid, otherwise set default cluster.

        Args:
            final_learner(Callable):A sklearn-like cluster
        Returns:

            cluster
        Raises:

            ValueError
                Raise error if given cluster is not a valid sklearn-like cluster.

        """

        if final_learner is None:
            final_learner = KMeans()
        else:
            final_learner = clone(final_learner)
        return final_learner

    def save(self, path: str, repr_cluster_file_name: str = "repr-cluster-partial.pkl") -> None:
        """
        Save the repr-cluster model to a directory.

        Args:
            path(str): Output directory path.
            ensemble_file_name(str): Name of repr-cluster model object. This file contains meta information of repr-cluster model.
        """
        return super().save(path, repr_cluster_file_name)

    @staticmethod
    def load(path: str, repr_cluster_file_name: str = "repr-cluster-partial.pkl") -> "ReprCluster":
        """
        Load the repr-cluster model from a directory.

        Args:
            path(str): Input directory path.
            ensemble_file_name(str): Name of repr-cluster model object. This file contains meta information of repr-cluster model.

        Returns:
            The loaded ensemble model.
        """
        return EnsembleBase.load(path, repr_cluster_file_name)