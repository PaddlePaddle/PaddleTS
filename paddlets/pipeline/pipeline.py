# !/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import math
import numpy as np
import pandas as pd
import pickle

from typing import List, Optional, Tuple, Union
from paddlets.models.base import Trainable
from paddlets.datasets.tsdataset import TSDataset, TimeSeries
from paddlets.logger import Logger, raise_if_not, raise_if, raise_log
from paddlets.logger.logger import log_decorator
from paddlets.models.model_loader import load as paddlets_model_load
from paddlets.utils.utils import get_tsdataset_max_len, split_dataset

logger = Logger(__name__)


class Pipeline(Trainable):
    """
    The pipeline is designed to build a workflow for time series modeling which may be comprised of a set of
    transformers and an model.

    **Note**: The model is optional.

    Args:
        steps(List[Tuple[object, str]]): A list of transformers and a final model.

    Examples:
        >>> ...
        >>> ksigma_params = {"cols":['example_columns'], "k": 0.5}
        >>> mlp_params = {'in_chunk_len': 7, 'out_chunk_len': 3, 'skip_chunk_len': 0, 'eval_metrics': ["mse", "mae"]}
        >>> pipeline = Pipeline([(KSigma, ksigma_params), (TimeFeatureGenerator, {}), (MLPRegressor, mlp_params)])
    """

    def __init__(self, steps: List[Tuple[object, str]]):
        raise_if(steps is None, ValueError("steps must not be None"))
        for e in steps:
            if 2 != len(e):
                raise_log(ValueError("The expected length of the tuple is 2, but actual element len: %s" % len(e)))

        self._steps = steps
        self._fitted = False
        self._model = None
        self._model_exist = False
        self._transform_list = []
        # Init transformers
        for index in range(len(self._steps) - 1):
            e = self._steps[index]
            transform_params = e[-1]
            try:
                transform = e[0](**transform_params)
            except Exception as e:
                raise_log(ValueError("init error: %s" % (str(e))))
            self._transform_list.append(transform)
        # Init final model
        try:
            last_object = self._steps[-1][0](**self._steps[-1][-1])
        except Exception as e:
            raise_log(ValueError("init error: %s" % (str(e))))
        if hasattr(last_object, "fit_transform"):
            self._transform_list.append(last_object)
        else:
            self._model_exist = True
            self._model = last_object

    @log_decorator
    def fit(
            self,
            train_tsdataset: Union[TSDataset, List[TSDataset]],
            valid_tsdataset: Optional[Union[TSDataset, List[TSDataset]]] = None):
        """
        Fit transformers and transform the data then fit the model.

        Args:
            train_tsdataset(Union[TSDataset, List[TSDataset]]): Train dataset.
            valid_tsdataset(Union[TSDataset, List[TSDataset]], optional): Valid dataset.

        Returns:
            Pipeline: Pipeline with fitted transformers and fitted model.
        """
        if isinstance(train_tsdataset, list):
            train_tsdataset_copy = [data.copy() for data in train_tsdataset]
        else:
            train_tsdataset_copy = train_tsdataset.copy()
        if valid_tsdataset:
            if isinstance(valid_tsdataset, list):
                valid_tsdataset_copy = [data.copy() for data in valid_tsdataset]
            else:
                valid_tsdataset_copy = valid_tsdataset.copy()
        # Transform
        for transform in self._transform_list:
            train_tsdataset_copy = transform.fit_transform(train_tsdataset_copy)
            if valid_tsdataset:
                valid_tsdataset_copy = transform.fit_transform(valid_tsdataset_copy)
        # Final model
        if self._model:
            if valid_tsdataset:
                self._model.fit(train_tsdataset_copy, valid_tsdataset_copy)
            else:
                self._model.fit(train_tsdataset_copy)
        self._fitted = True
        return self

    def transform(self,
                  tsdataset: Union[TSDataset, List[TSDataset]],
                  inplace: bool = False,
                  cache_transform_steps: bool = False,
                  previous_caches: List[TSDataset] = None) -> Union[TSDataset, Tuple[TSDataset, List[TSDataset]]]:
        """
        Transform the `TSDataset` using the fitted transformers in the pipeline.

        Args:
            tsdataset(Union[TSDataset, List[TSDataset]]): Data to be transformed.
            inplace(bool): Set to True to perform inplace transform and avoid a data copy. Default is False.
            cache_transform_steps: Cache each transform step's transorm result into a list.
            previous_caches : previous transform results cache

        Returns:
            Tuple[TSDataset,Tuple[List[TSDataset],TSDataset]]: Return transformed results by default. Return Both
                transformed results and each transform step's caches if set cache_transform_steps = True.
        """
        self._check_fitted()
        raise_if(cache_transform_steps is True and isinstance(tsdataset, list),
                 "Not implement error. Not support cache when input tsdataset is a list.")
        tsdataset_transformed = tsdataset
        if not inplace:
            if isinstance(tsdataset, list):
                tsdataset_transformed = [data.copy() for data in tsdataset]
            else:
                tsdataset_transformed = tsdataset.copy()
        if cache_transform_steps is False:
            # normal transform
            for transform in self._transform_list:
                tsdataset_transformed = transform.transform(tsdataset_transformed)
            return tsdataset_transformed

        # Recursive predict Transform
        tansform_list_length = len(self._transform_list)
        # Init transform copys with same length as tansform list, fill with None
        transform_caches = [None] * tansform_list_length
        # the first transformer's cache is the origin data
        if cache_transform_steps and self._transform_list[0].need_previous_data:
            transform_caches[0] = tsdataset_transformed

        for i in range(tansform_list_length):
            transformed_data_len = get_tsdataset_max_len(tsdataset_transformed)
            data_pre = previous_caches[i] if previous_caches else None
            transformer = self._transform_list[i]

            if data_pre:
                tsdataset_transformed = TSDataset.concat([data_pre, tsdataset_transformed])
            tsdataset_transformed = transformer.transform_n_rows(tsdataset_transformed, transformed_data_len)

            # caches
            if cache_transform_steps:
                next_transformer_index = i + 1
                last_transformer_index = tansform_list_length - 1
                # final transfomer do not has next transformer, break
                if i == last_transformer_index:
                    break
                # next transformer's cache is this transformer's results
                if self._transform_list[next_transformer_index].need_previous_data:
                    transform_caches[next_transformer_index] = tsdataset_transformed

        res = tsdataset_transformed
        return (res, transform_caches) if cache_transform_steps else res

    def inverse_transform(self,
                          tsdataset: Union[TSDataset, List[TSDataset]],
                          inplace: bool = False) -> TSDataset:
        """
        The inverse transformation of `self.transform`.
        Apply `inverse_transform` using the fitted transformers in the pipeline.
        Note that not all transformers implement `inverse_transform` method.
        If a transformer do not implement `inverse_transform`, it would not inversely transform the input data.

        Args:
            tsdataset(Union[TSDataset, List[TSDataset]]): Data to apply `inverse_transform`.
            inplace(bool): Set to True to perform inplace transform and avoid a data copy. Default is False.

        Returns:
            TSDataset: Inversely transformed TSDataset.
        """
        self._check_fitted()
        tsdataset_transformed = tsdataset
        if not inplace:
            if isinstance(tsdataset, list):
                tsdataset_transformed = [data.copy() for data in tsdataset]
            else:
                tsdataset_transformed = tsdataset.copy()
        # Transform
        for transform in reversed(self._transform_list):
            try:
                tmp_ts = transform.inverse_transform(tsdataset_transformed)
                tsdataset_transformed = tmp_ts
            except NotImplementedError:
                logger.debug("%s not implement inverse_transform, continue" % (transform.__class__.__name__))
                continue
            except Exception as e:
                raise_log(RuntimeError("error occurred while inverse_transform, error: %s" % (str(e))))

        return tsdataset_transformed

    def predict(self, tsdataset: TSDataset) -> TSDataset:
        """
        Transform the `TSDataset` using the fitted transformers and perform prediction with the fitted model in the
        pipeline, only effective when the model exists in the pipeline.

        Args:
            tsdataset(TSDataset): Data to be predicted.

        Returns:
            TSDataset: Predicted results of calling `self.predict` on the final model.
        """
        self._check_model_exist()
        self._check_fitted()
        tsdataset_transformed = self.transform(tsdataset)
        predictions = self._model.predict(tsdataset_transformed)
        if "anomaly" not in str(self._model):
            predictions = self.inverse_transform(predictions)
        return predictions

    def predict_proba(self, tsdataset: TSDataset) -> TSDataset:
        """
        Transform the `TSDataset` using the fitted transformers and perform probability prediction with the fitted
        model in the pipeline, only effective when the model exists in the pipeline.

        Args:
            tsdataset(TSDataset): Data to be predicted.

        Returns:
            TSDataset: Predicted results of calling `self.predict_proba` on the final model.
        """
        self._check_model_exist()
        self._check_fitted()
        tsdataset_transformed = self.transform(tsdataset)
        # Only valid if the final model implements predict_proba.
        raise_if_not(hasattr(self._model, "predict_proba"), \
                     "predict_proba is only valid if the final model implements predict_proba")
        return self._model.predict_proba(tsdataset_transformed)

    def predict_score(self, tsdataset: TSDataset) -> TSDataset:
        """
        Transform the `TSDataset` using the fitted transformers and perform anomaly detection score prediction with the fitted
        model in the pipeline, only effective when the model exists in the pipeline.

        Args:
            tsdataset(TSDataset): Data to be predicted.

        Returns:
            TSDataset: Predicted results of calling `self.predict_score` on the final model.
        """
        self._check_model_exist()
        self._check_fitted()
        tsdataset_transformed = self.transform(tsdataset)
        # Only valid if the final model implements predict_score.
        raise_if_not(hasattr(self._model, "predict_score"), \
                     "predict_score is only valid if the final model implements predict_score")
        return self._model.predict_score(tsdataset_transformed)

    def recursive_predict(
            self,
            tsdataset: TSDataset,
            predict_length: int
    ) -> TSDataset:
        """
        Apply `self.predict` method iteratively for multi-step time series forecasting, the predicted results from the
        current call will be appended to the `TSDataset` object and will appear in the loopback window for next call.
        Note that each call of `self.predict` will return a result of length `out_chunk_len`, so it will be called
        ceiling(`predict_length`/`out_chunk_len`) times to meet the required length.

        Args:
            tsdataset(TSDataset): Data to be predicted.
            predict_length(int): Length of predicted results.

        Returns:
            TSDataset: Predicted results.
        """
        return self._recursive_predict(tsdataset, predict_length)

    def recursive_predict_proba(
            self,
            tsdataset: TSDataset,
            predict_length: int,
    ) -> TSDataset:
        """
        Apply `self.predict_proba` method iteratively for multi-step time series forecasting, the predicted results
        from the current call will be appended to the `TSDataset` object and will appear in the loopback window for
        next call. Note that each call of `self.predict_proba` will return a result of length `out_chunk_len`,
        so it will be called ceiling(`predict_length`/`out_chunk_len`) times to meet the required length.

        Args:
            tsdataset(TSDataset): Data to be predicted.
            predict_length(int): Length of predicted results.

        Returns:
            TSDataset: Predicted results.
        """
        return self._recursive_predict(tsdataset, predict_length, need_proba=True)

    def _recursive_predict(
            self,
            tsdataset: TSDataset,
            predict_length: int,
            need_proba: bool = False
    ) -> TSDataset:
        """
        _recursive_predict

        Args:
            tsdataset(TSDataset): Data to be predicted.
            predict_length(int): Length of predicted results.
            need_proba(bool): Whether to use predict_proba to infer the class probabilities.

        Returns:
            TSDataset: Predicted results.
        """
        self._check_model_exist()
        self._check_fitted()
        raise_if(
            "anomaly" in str(self._model),
            "The anomaly detection model does not support recursive_predict."
        )
        self._check_recursive_predict_valid(predict_length, need_proba=need_proba)
        recursive_rounds = math.ceil(predict_length / self._model._out_chunk_len)
        """
        Use recursive_transform , which means:
        Use the predicted value of the current time 
        step to determine its feature transform data in the next time step.
        """
        tsdataset_copy = tsdataset.copy()
        # check tsdataset
        out_chunk_time_freq = None
        if isinstance(tsdataset.get_target().data.index, pd.RangeIndex):
            out_chunk_time_freq = self._model._out_chunk_len * \
                                  (tsdataset_copy.get_target().time_index.step)
        elif isinstance(tsdataset.get_target().data.index, pd.DatetimeIndex):
            out_chunk_time_freq = self._model._out_chunk_len * \
                                  (tsdataset_copy.get_target().time_index.freq)
        else:
            raise_log(ValueError(f"time col type not support, \
                                                 index type:{type(tsdataset.get_target().data.index)}"))

        target_res_end_time = tsdataset_copy.get_target().end_time + \
                              recursive_rounds * out_chunk_time_freq
        if tsdataset_copy.get_known_cov() is not None \
                and target_res_end_time > tsdataset_copy.get_known_cov().end_time:
            raise_log(RuntimeError(
                "recursive_rounds is %s, "
                "recursive predict output end time : %s, while no enough known_cov can be used as in_chunk, "
                "known_cov's end_time must >= %s'" % (str(recursive_rounds),
                                                      str(target_res_end_time),
                                                      str(target_res_end_time))))
        if tsdataset_copy.get_observed_cov() is not None \
                and target_res_end_time > tsdataset_copy.get_observed_cov().end_time + out_chunk_time_freq:
            raise_log(RuntimeError(
                "recursive_rounds is %s, "
                "recursive predict output end time : %s, while no enough observed_cov can be used as in_chunk, "
                "observed_cov's end_time must >= %s'" % (str(recursive_rounds),
                                                         str(target_res_end_time),
                                                         str(target_res_end_time - out_chunk_time_freq))))

        # Reindex data and the default fill value is np.nan
        # fill_value = np.nan
        # if tsdataset_copy.get_known_cov() is not None:
        #     if isinstance(tsdataset_copy.get_known_cov().data.index, pd.RangeIndex):
        #         tsdataset_copy.get_known_cov().reindex(
        #             pd.RangeIndex(start=tsdataset_copy.get_known_cov().start_time,
        #                           stop=dataset_end_time + 1,
        #                           step=tsdataset_copy.get_known_cov().time_index.step),
        #             fill_value=fill_value
        #         )
        #     else:
        #         tsdataset_copy.get_known_cov().reindex(
        #             pd.date_range(start=tsdataset_copy.get_known_cov().start_time,
        #                           end=dataset_end_time,
        #                           freq=tsdataset_copy.get_known_cov().time_index.freq),
        #             fill_value=fill_value
        #         )
        # if tsdataset_copy.get_observed_cov() is not None:
        #     if isinstance(tsdataset_copy.get_observed_cov().data.index, pd.RangeIndex):
        #         tsdataset_copy.get_observed_cov().reindex(
        #             pd.RangeIndex(start=tsdataset_copy.get_observed_cov().start_time,
        #                           stop=dataset_end_time + 1,
        #                           step=tsdataset_copy.get_observed_cov().time_index.step),
        #             fill_value=fill_value
        #         )
        #     else:
        #         tsdataset_copy.get_observed_cov().reindex(
        #             pd.date_range(start=tsdataset_copy.get_observed_cov().start_time,
        #                           end=dataset_end_time,
        #                           freq=tsdataset_copy.get_observed_cov().time_index.freq),
        #             fill_value=fill_value
        #         )
        target_length = len(tsdataset_copy.target)

        # feature process on pre data
        if tsdataset_copy.known_cov:
            pre_data, _ = split_dataset(tsdataset_copy, target_length + self._model._out_chunk_len)
        else:
            pre_data = tsdataset_copy
        data_pre_transformed, data_pre_transformed_caches = self.transform(tsdataset=pre_data,
                                                                           cache_transform_steps=True)

        results = []

        # recursive predict start
        for i in range(recursive_rounds):

            # predict
            if need_proba == True:
                predictions = self._model.predict_proba(data_pre_transformed)
            else:
                predictions = self._model.predict(data_pre_transformed)
            predictions = self.inverse_transform(predictions)
            results.append(predictions)
            # break in last round
            if i == recursive_rounds - 1:
                break

            # predict concat to origindata
            tsdataset_copy = TSDataset.concat([tsdataset_copy, predictions], keep="last")
            target_length = target_length + self._model._out_chunk_len

            # split new predict chunk
            _, new_chunk = tsdataset_copy.split(target_length - self._model._out_chunk_len)
            if tsdataset_copy.known_cov:
                new_chunk, _ = split_dataset(new_chunk, 2 * self._model._out_chunk_len)

            # transform one chunk
            chunk_transformed, chunk_transformed_caches = self.transform(new_chunk,
                                                                         cache_transform_steps=True,
                                                                         previous_caches=data_pre_transformed_caches,
                                                                         inplace=False)

            # concate transform results
            data_pre_transformed = TSDataset.concat([data_pre_transformed, chunk_transformed], keep="last")

            # concat transform caches
            for i in range(len(data_pre_transformed_caches)):
                if data_pre_transformed_caches[i]:
                    data_pre_transformed_caches[i] = TSDataset.concat(
                        [data_pre_transformed_caches[i], chunk_transformed_caches[i]])

        # Concat results
        result = TSDataset.concat(results)
        # Resize result
        result.set_target(
            TimeSeries(result.get_target().data[0: predict_length], result.freq)
        )
        return result

    def save(self, path: str, pipeline_file_name: str = "pipeline-partial.pkl", model_file_name: str = "paddlets_model"):
        """
        Save the pipeline to a directory.

        Args:
            path(str): Output directory path.
            pipeline_file_name(str): Name of pipeline object. This file contains transformers and
                meta information of pipeline.
            model_file_name(str): Name of model object. See `BaseModel.save` for more information.
        """
        if not os.path.exists(path):
            # Check path
            os.makedirs(path)
        elif not os.path.isdir(path):
            raise_log(ValueError(f"path is not a directory, path : {path}"))
        # Check file not exist
        pipeline_file_path = os.path.join(path, pipeline_file_name)
        if os.path.exists(pipeline_file_path):
            raise_log(FileExistsError(f"pipeline-partial file already exist, path : {pipeline_file_path}"))
        # 1.Save model
        if self._model is not None:
            self._model.save(os.path.join(path, model_file_name))
        # 2.Save pipeline(without final model)
        model_tmp = self._model
        self._model = None
        try:
            with open(pipeline_file_path, "wb") as f:
                pickle.dump(self, f)
        except Exception as e:
            raise_log(ValueError("error occurred while saving pipeline, file path: %s, err: %s" \
                                 % (pipeline_file_path, str(e))))
        # Reset model
        self._model = model_tmp

    @classmethod
    def load(cls, path: str, pipeline_file_name: str = "pipeline-partial.pkl", model_file_name: str = "paddlets_model"):
        """
        Load the pipeline from a directory.

        Args:
            path(str): Input directory path.
            pipeline_file_name(str): Name of pipeline object. This file contains transformers and
                meta information of pipeline.
            model_file_name(str): Name of model object. See `BaseModel.save` for more information.

        Returns:
            Pipeline: The loaded pipeline.
        """
        if not os.path.exists(path):
            raise_log(FileNotFoundError(f"path not exist, path : {path}"))
        if not os.path.isdir(path):
            raise_log(ValueError(f"path is not a directory, path : {path}"))
        # 1.Load pipeline
        # Check file exist
        pipeline_file_path = os.path.join(path, pipeline_file_name)
        if not os.path.exists(pipeline_file_path):
            raise_log(FileExistsError(f"pipeline-partial file not exist, path : {pipeline_file_path}"))
        try:
            with open(pipeline_file_path, "rb") as f:
                pipeline = pickle.load(f)
        except Exception as e:
            raise_log(RuntimeError(
                "error occurred while loading pipeline, path: %s, error: %s" % (pipeline_file_path, str(e))))
        # 2.Load model
        if pipeline._model_exist is True:
            model = paddlets_model_load(os.path.join(path, model_file_name))
            # Add model to pipeline
            pipeline._model = model
        return pipeline

    def _check_fitted(self):
        """
        Check that pipeline is fitted.
        Raise error if pipeline not fitted.
        """
        if not self._fitted:
            raise_log(RuntimeError("please do fit first!"))

    def _check_model_exist(self):
        """
        Check that self._model exists.
        Raise error if self._model does not exist.
        """
        if self._model is None:
            raise_log(RuntimeError("model not exist"))

    def _check_recursive_predict_valid(self, predict_length: int, need_proba: bool = False):
        """
        Check that `recursive_predict` is valid.
        Raise error if `recursive_predict` is invalid.
        """
        if need_proba == True:
            raise_if_not(hasattr(self._model, "recursive_predict_proba"), \
                         "predict_proba is only valid if the final model implements predict_proba")
        # Not supported when _skip_chunk !=0
        raise_if(self._model._skip_chunk_len != 0, f"recursive_predict not supported when \
            _skip_chunk_len!=0, got {self._model._skip_chunk_len}.")
        raise_if(predict_length <= 0, f"predict_length must be > \
            0, got {predict_length}.")

    @property
    def steps(self):
        return self._steps
