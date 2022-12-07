# !/usr/bin/env python3
# -*- coding:utf-8 -*-

import abc
import numpy as np
import sklearn
import pyod
from pyod.models.base import BaseDetector as PyodBaseDetector
from typing import Type, Dict, Any, Optional, Callable, Tuple
from itertools import product

from paddlets.models.base import BaseModel
from paddlets.models.forecasting.ml.ml_base import MLBaseModel
from paddlets.models.data_adapter import SampleDataset, MLDataLoader, DataAdapter
from paddlets.models.utils import to_tsdataset, check_tsdataset
from paddlets.datasets import TSDataset
from paddlets.logger import Logger, raise_log, raise_if, raise_if_not

logger = Logger(__name__)


class MLModelBaseWrapper(MLBaseModel, metaclass=abc.ABCMeta):
    """
    Time series model base wrapper for third party models.

    Args:
        model_class(Type): Class type of the third party model.
        in_chunk_len(int): The size of the loopback window, i.e., the number of time steps feed to the model.
        out_chunk_len(int): The size of the forecasting horizon, i.e., the number of time steps output by the model.
        skip_chunk_len(int, optional): The number of time steps between in_chunk and out_chunk for a single sample.
            The skip chunk is neither used as a feature (i.e. X) nor a label (i.e. Y) for a single sample. By
            default, it will NOT skip any time steps.
        sampling_stride(int, optional): Time steps to stride over the i-th sample and (i+1)-th sample.
            More precisely, let `t` be the time index of target time series,
            `t[i]` be the start time of the i-th sample, `t[i+1]` be the start time of the (i+1)-th sample, then
            `sampling_stride` represents the result of `t[i+1] - t[i]`.
        model_init_params(Dict[str, Any]): All params for initializing the third party model.
        fit_params(Dict[str, Any], optional): All params for fitting third party model except x_train / y_train.
        predict_params(Dict[str, Any], optional): All params for forecasting third party model except x_test / y_test.
    """
    def __init__(
        self,
        model_class: Type,
        in_chunk_len: int,
        out_chunk_len: int = 1,
        skip_chunk_len: int = 0,
        sampling_stride: int = 1,
        model_init_params: Dict[str, Any] = None,
        fit_params: Dict[str, Any] = None,
        predict_params: Dict[str, Any] = None
    ):
        super(MLModelBaseWrapper, self).__init__(
            in_chunk_len=in_chunk_len,
            skip_chunk_len=skip_chunk_len,
            out_chunk_len=out_chunk_len
        )
        self._sampling_stride = sampling_stride

        self._model_class = model_class
        self._model_init_params = model_init_params if model_init_params is not None else dict()
        self._fit_params = fit_params if fit_params is not None else dict()
        self._predict_params = predict_params if predict_params is not None else dict()


class SklearnModelWrapper(MLModelBaseWrapper):
    """
    Time series model wrapper for sklearn third party models.

    Args:
        model_class(Type): Class type of the third party model.
        in_chunk_len(int): The size of the loopback window, i.e., the number of time steps feed to the model.
        out_chunk_len(int): The size of the forecasting horizon, i.e., the number of time steps output by the model.
        skip_chunk_len(int, optional): The number of time steps between in_chunk and out_chunk for a single sample.
            The skip chunk is neither used as a feature (i.e. X) nor a label (i.e. Y) for a single sample. By
            default, it will NOT skip any time steps.
        sampling_stride(int, optional): Time steps to stride over the i-th sample and (i+1)-th sample.
            More precisely, let `t` be the time index of target time series,
            `t[i]` be the start time of the i-th sample, `t[i+1]` be the start time of the (i+1)-th sample, then
            `sampling_stride` represents the result of `t[i+1] - t[i]`.
        model_init_params(Dict[str, Any]): All params for initializing the third party model.
        fit_params(Dict[str, Any], optional): All params for fitting third party model except x_train / y_train.
        predict_params(Dict[str, Any], optional): All params for forecasting third party model except x_test / y_test.
        udf_ml_dataloader_to_fit_ndarray(Callable, optional): User defined function for converting MLDataLoader object
            to a numpy.ndarray object that can be processed by `fit` method of the third party model.
        udf_ml_dataloader_to_predict_ndarray(Callable, optional): User defined function for converting MLDataLoader
            object to a numpy.ndarray object that can be processed by `predict` method of the third party model.
    """
    def __init__(
        self,
        model_class: Type,
        in_chunk_len: int,
        out_chunk_len: int,
        skip_chunk_len: int = 0,
        sampling_stride: int = 1,
        model_init_params: Dict[str, Any] = None,
        fit_params: Dict[str, Any] = None,
        predict_params: Dict[str, Any] = None,
        udf_ml_dataloader_to_fit_ndarray: Optional[Callable] = None,
        udf_ml_dataloader_to_predict_ndarray: Optional[Callable] = None
    ):
        raise_if(in_chunk_len < 0, f"in_chunk_len ({in_chunk_len}) must >= 0.")
        raise_if(skip_chunk_len < 0, f"skip_chunk_len ({skip_chunk_len}) must >= 0.")
        raise_if(
            out_chunk_len != 1,
            f"""out_chunk_len ({out_chunk_len}) must == 1. 
            Please refer to {BaseModel}.recursive_predict for multistep time series forecasting."""
        )
        raise_if(sampling_stride < 1, f"sampling_stride ({sampling_stride}) must >= 1.")

        super(SklearnModelWrapper, self).__init__(
            in_chunk_len=in_chunk_len,
            skip_chunk_len=skip_chunk_len,
            out_chunk_len=out_chunk_len,
            model_class=model_class,
            sampling_stride=sampling_stride,
            model_init_params=model_init_params,
            fit_params=fit_params,
            predict_params=predict_params
        )
        raise_if(self._model_class is None, "model_class must not be None.")
        raise_if_not(isinstance(self._model_class, type), "isinstance(model_class, type) must be True.")

        self._model = self._init_model()
        self._data_adapter = self._init_data_adapter()

        self._udf_ml_dataloader_to_fit_ndarray = default_sklearn_ml_dataloader_to_fit_ndarray
        if udf_ml_dataloader_to_fit_ndarray is not None:
            self._udf_ml_dataloader_to_fit_ndarray = udf_ml_dataloader_to_fit_ndarray

        self._udf_ml_dataloader_to_predict_ndarray = default_sklearn_ml_dataloader_to_predict_ndarray
        if udf_ml_dataloader_to_predict_ndarray is not None:
            self._udf_ml_dataloader_to_predict_ndarray = udf_ml_dataloader_to_predict_ndarray

    def fit(self, train_data: TSDataset, valid_data: Optional[TSDataset] = None) -> None:
        """
        Fit a machine learning model.

        Args:
            train_data(TSDataset): training dataset.
            valid_data(TSDataset, optional): validation dataset.
        """
        self._validate_train_data(train_data)

        train_ml_dataset = self._tsdataset_to_ml_dataset(train_data)
        train_ml_dataloader = self._ml_dataset_to_ml_dataloader(train_ml_dataset)

        train_x, train_y = None, None
        try:
            train_x, train_y = self._udf_ml_dataloader_to_fit_ndarray(
                ml_dataloader=train_ml_dataloader,
                model_init_params=self._model_init_params,
                in_chunk_len=self._in_chunk_len,
                out_chunk_len=self._out_chunk_len,
                skip_chunk_len=self._skip_chunk_len
            )
        except Exception as e:
            raise_log(
                ValueError(
                    f"""failed to convert train_data to sklearn model trainable numpy array. 
                    Please check udf_ml_dataloader_to_fit_ndarray function. Error: {str(e)}"""
                )
            )

        if hasattr(self._model, "fit") and callable(getattr(self._model, "fit")):
            self._model.fit(train_x, train_y, **self._fit_params)
            return
        raise_log(ValueError(f"{self._model_class} must implement callable fit method."))

    @to_tsdataset(scenario="forecasting")
    def predict(self, tsdataset: TSDataset) -> TSDataset:
        """
        Make prediction.

        Args:
            tsdataset(TSDataset): TSDataset to predict.

        Returns:
            TSDataset: TSDataset with predictions.
        """
        self._validate_predict_data(tsdataset)

        w0 = len(tsdataset.get_target().time_index) - 1 + self._skip_chunk_len + self._out_chunk_len
        w1 = w0
        time_window = (w0, w1)

        test_ml_dataset = self._tsdataset_to_ml_dataset(tsdataset, time_window=time_window)
        test_ml_dataloader = self._data_adapter.to_ml_dataloader(test_ml_dataset, batch_size=len(test_ml_dataset))

        test_x, test_y = None, None
        try:
            test_x, test_y = self._udf_ml_dataloader_to_predict_ndarray(
                ml_dataloader=test_ml_dataloader,
                model_init_params=self._model_init_params,
                in_chunk_len=self._in_chunk_len,
                out_chunk_len=self._out_chunk_len,
                skip_chunk_len=self._skip_chunk_len
            )
        except Exception as e:
            raise_log(
                ValueError(
                    f"""failed to convert train_data to sklearn model predictable numpy array. 
                    Please check udf_ml_dataloader_to_predict_ndarray function. Error: {str(e)}"""
                )
            )

        if hasattr(self._model, "predict") and callable(getattr(self._model, "predict")):
            return self._model.predict(test_x, **self._predict_params)
        raise_log(ValueError(f"original model {self._model_class} must have callable predict method."))

    def _init_model(self) -> sklearn.base.BaseEstimator:
        """
        Internal method, init sklearn model.

        1) The model class must be inherited from sklearn.base.BaseEstimator.
        2) The initialized model object (or its ancestor) must implement fit and predict callable method.
        """
        # 1 model must be inherited from sklearn.base.BaseEstimator.
        expected_parent_class = sklearn.base.BaseEstimator
        parent_classes = self._model_class.mro()

        raise_if_not(
            expected_parent_class in parent_classes,
            f"{self._model_class} must inherit from {expected_parent_class}, but actual inherit chain: {parent_classes}"
        )

        model = None
        try:
            model = self._model_class(**self._model_init_params)
        except Exception as e:
            # all other possible errors are captured here:
            # (1) TypeError: __init__() got an unexpected keyword argument "xxx"
            raise_log(
                ValueError(
                    f"init model failed: {self._model_class}, model_init_params: {self._model_init_params}, error: {e}."
                )
            )

        # It is possible that the class has "predict" method, but the initialized model object does NOT have one.
        # Given this scenario, we need to check the initialized object rather than the class type.
        # For example:
        # >>> from sklearn.neighbors import LocalOutlierFactor
        # >>> model = LocalOutlierFactor()
        # >>> hasattr(model, "predict")
        # False
        # >>> hasattr(model.__class__, "predict")
        # True
        not_implemented_method_set = set()
        for method in {"fit", "predict"}:
            if (hasattr(model, method) and callable(getattr(model, method))) is not True:
                not_implemented_method_set.add(method)
        raise_if(
            len(not_implemented_method_set) > 0,
            f"The initialized {self._model_class} object must implement {not_implemented_method_set} methods."
        )
        return model

    def _init_data_adapter(self) -> DataAdapter:
        """
        Internal method, initialize data adapter.

        Returns:
            DataAdapter: Initialized data adapter object.
        """
        return DataAdapter()

    def _tsdataset_to_ml_dataset(
        self,
        tsdataset: TSDataset,
        time_window: Optional[Tuple[int, int]] = None
    ) -> SampleDataset:
        """
        Internal method, convert TSDataset to MLDataset.

        Returns:
            MLDataset: Converted MLDataset object.
        """
        return self._data_adapter.to_sample_dataset(
            rawdataset=tsdataset,
            in_chunk_len=self._in_chunk_len,
            out_chunk_len=self._out_chunk_len,
            skip_chunk_len=self._skip_chunk_len,
            sampling_stride=self._sampling_stride,
            time_window=time_window
        )

    def _ml_dataset_to_ml_dataloader(self, ml_dataset: SampleDataset) -> MLDataLoader:
        """
        Internal method, convert MLDataset to MLDataLoader.

        Returns:
            MLDataLoader: Converted MLDataLoader object.
        """
        return self._data_adapter.to_ml_dataloader(ml_dataset, batch_size=len(ml_dataset))

    def _validate_train_data(self, train_data: TSDataset) -> None:
        """
        Internal method, validate training data. Raises if invalid.

        Args:
            train_data(TSDataset): Training data to be validated.
        """
        raise_if(train_data is None, "training dataset must not be None.")
        raise_if(train_data.get_target() is None, "target timeseries must not be None.")

        check_tsdataset(train_data)
        target_ts = train_data.get_target()
        # multi target is NOT supported.
        raise_if(
            len(target_ts.columns) != 1,
            f"training dataset target timeseries columns number ({len(target_ts.columns)}) must be 1."
        )

        # target dtype must be numeric (np.float32), not categorical (np.int64).
        target_dtype = target_ts.data.iloc[:, 0].dtype
        raise_if(
            np.float32 != target_dtype,
            f"The dtype of TSDataset.target ({target_dtype}) must be sub-type of numpy.floating."
        )

    def _validate_predict_data(self, tsdataset: TSDataset) -> None:
        """
        Internal method, validate data for prediction. Raises if invalid.

        Args:
            tsdataset(TSDataset): Predict data to be validated.
        """
        raise_if(tsdataset is None, "The dataset to be predicted must not be None.")
        raise_if(tsdataset.get_target() is None, "target timeseries to be predicted must not be None.")

        check_tsdataset(tsdataset)
        target_ts = tsdataset.get_target()
        # multi target is NOT supported.
        raise_if(
            len(target_ts.columns) != 1,
            f"columns number of target timeseries to be predicted ({len(target_ts.columns)}) must be 1."
        )

        # target dtype must be numeric (np.float32), not categorical (np.int64).
        target_dtype = target_ts.data.iloc[:, 0].dtype
        raise_if(
            np.float32 != target_dtype,
            f"The dtype of TSDataset.target ({target_dtype}) must be sub-type of numpy.floating."
        )


def default_sklearn_ml_dataloader_to_fit_ndarray(
    ml_dataloader: MLDataLoader,
    model_init_params: Dict[str, Any],
    in_chunk_len: int,
    skip_chunk_len: int,
    out_chunk_len: int
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Default function for converting MLDataLoader to a numpy array that can be used for fitting the sklearn model.

    Args:
        ml_dataloader(MLDataLoader): MLDataLoader object to be converted.
        model_init_params(Dict): parameters when initializing sklearn models, possibly be used while converting.
        in_chunk_len(int): The size of the loopback window, i.e., the number of time steps feed to the model. Possibly
            be used while converting.
        skip_chunk_len(int, optional): The number of time steps between in_chunk and out_chunk for a single sample.
            The skip chunk is neither used as a feature (i.e. X) nor a label (i.e. Y) for a single sample. By
            default, it will NOT skip any time steps. Possibly be used while converting.
        out_chunk_len(int): The size of the forecasting horizon, i.e., the number of time steps output by the model.
            Possibly be used while converting.

    Returns:
        Tuple[np.ndarray, Optional[np.ndarray]]: Converted numpy array. The first and second element in the tuple
        represent x_train and y_train, respectively.
    """
    data = next(ml_dataloader)

    sample_x_keys = data.keys() - {"future_target"}
    if in_chunk_len < 1:
        # lag scenario cannot use past_target as features.
        sample_x_keys -= {"past_target"}
    # concatenated ndarray will follow the below ordered list rule:
    # [rule 1] left -> right = target features, ..., known features, ..., observed features, ..., static_cov_features.
    # [rule 2] left -> right = numeric features, ..., categorical features.
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
    #   "past_target_categorical",
    #   "known_cov_categorical",
    #   "observed_cov_categorical",
    #   "static_cov_categorical"
    # ]
    # Thus, actual_ordered_x_key_list = [
    #   "known_cov_numeric",
    #   "observed_cov_categorical",
    #   "static_cov_categorical"
    # ]
    # The built sample ndarray will be like below:
    # [
    #   [
    #       known_cov_numeric_feature, observed_cov_categorical_feature, static_cov_categorical_feature
    #   ],
    # [
    #       known_cov_numeric_feature, observed_cov_categorical_feature, static_cov_categorical_feature
    #   ],
    #   ...
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
    # Note: if a_ndarray.dtype = np.int64, b_ndarray.dtype = np.float32, then
    # np.hstack(tup=(a_ndarray, b_ndarray)).dtype will ALWAYS BE np.float32
    x = np.hstack(tup=reshaped_x_ndarray_list)

    # Why needs to call np.squeeze? See below:
    # Because sklearn requires that y.shape must be (n_samples, ), so it is required to call np.squeeze().
    # Meanwhile, as we already make pre-assertions in _validate_train_data(), thus we can ensure the following:
    # 1. data["future_target"].shape[1] (i.e., out_chunk_len) must == 1.
    # 2. data["future_target"].shape[2] (i.e., len(target.columns)) must == 1;
    # Thus, np.squeeze() call can successfully remove these single-dimensional entries (shape[1] and shape[2]) and
    # only make shape[0] (i.e., batch_size dim) reserved.
    # As a result, after the below call, y.shape == (batch_size, ), which fits sklearn requirement.
    y = np.squeeze(data["future_target"])
    return x, y


def default_sklearn_ml_dataloader_to_predict_ndarray(
    ml_dataloader: MLDataLoader,
    model_init_params: Dict[str, Any],
    in_chunk_len: int,
    skip_chunk_len: int,
    out_chunk_len: int
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Default function for converting MLDataLoader to a numpy array that can be predicted by the sklearn model.

    Args:
        ml_dataloader(MLDataLoader): MLDataLoader object to be converted.
        model_init_params(Dict): parameters when initializing sklearn models, possibly be used while converting.
        in_chunk_len(int): The size of the loopback window, i.e., the number of time steps feed to the model. Possibly
            be used while converting.
        skip_chunk_len(int, optional): The number of time steps between in_chunk and out_chunk for a single sample.
            The skip chunk is neither used as a feature (i.e. X) nor a label (i.e. Y) for a single sample. By
            default, it will NOT skip any time steps. Possibly be used while converting.
        out_chunk_len(int): The size of the forecasting horizon, i.e., the number of time steps output by the model.
            Possibly be used while converting.

    Returns:
        Tuple[np.ndarray, Optional[np.ndarray]]: Converted numpy array. The first and second element in the tuple
        represent x and y, respectively, where y is optional.
        """
    data = next(ml_dataloader)

    sample_x_keys = data.keys() - {"future_target"}
    if in_chunk_len < 1:
        # lag scenario cannot use past_target as features.
        sample_x_keys -= {"past_target"}
    # concatenated ndarray will follow the below ordered list rule:
    # [rule 1] left -> right = target features, ..., known features, ..., observed features, ..., static_cov_features.
    # [rule 2] left -> right = numeric features, ..., categorical features.
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
    #   "past_target_categorical",
    #   "known_cov_categorical",
    #   "observed_cov_categorical",
    #   "static_cov_categorical"
    # ]
    # Thus, actual_ordered_x_key_list = [
    #   "known_cov_numeric",
    #   "observed_cov_categorical",
    #   "static_cov_categorical"
    # ]
    # The built sample ndarray will be like below:
    # [
    #   [
    #       known_cov_numeric_feature, observed_cov_categorical_feature, static_cov_categorical_feature
    #   ],
    #   [
    #       known_cov_numeric_feature, observed_cov_categorical_feature, static_cov_categorical_feature
    #   ],
    #   ...
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
    # Note: if a_ndarray.dtype = np.int64, b_ndarray.dtype = np.float32, then
    # np.hstack(tup=(a_ndarray, b_ndarray)).dtype will ALWAYS BE np.float32
    x = np.hstack(tup=reshaped_x_ndarray_list)
    return x, None


class PyodModelWrapper(MLModelBaseWrapper):
    """
    Time series model wrapper for pyod third party models.

    Args:
        model_class(Type): Class type of the third party model.
        in_chunk_len(int): The size of the loopback window, i.e., the number of time steps feed to the model.
        sampling_stride(int, optional): Time steps to stride over the i-th sample and (i+1)-th sample.
            More precisely, let `t` be the time index of target time series,
            `t[i]` be the start time of the i-th sample, `t[i+1]` be the start time of the (i+1)-th sample, then
            `sampling_stride` represents the result of `t[i+1] - t[i]`.
        model_init_params(Dict[str, Any]): All params for initializing the third party model.
        predict_params(Dict[str, Any], optional): All params for forecasting third party model except x_test / y_test.
        udf_ml_dataloader_to_fit_ndarray(Callable, optional): User defined function for converting MLDataLoader object
            to a numpy.ndarray object that can be processed by `fit` method of the third party model.
        udf_ml_dataloader_to_predict_ndarray(Callable, optional): User defined function for converting MLDataLoader
            object to a numpy.ndarray object that can be processed by `predict` method of the third party model.
    """
    def __init__(
        self,
        model_class: Type,
        in_chunk_len: int,
        sampling_stride: int = 1,
        model_init_params: Dict[str, Any] = None,
        predict_params: Dict[str, Any] = None,
        udf_ml_dataloader_to_fit_ndarray: Optional[Callable] = None,
        udf_ml_dataloader_to_predict_ndarray: Optional[Callable] = None
    ):
        raise_if(in_chunk_len <= 0, f"in_chunk_len ({in_chunk_len}) must > 0.")
        raise_if(sampling_stride < 1, f"sampling_stride ({sampling_stride}) must >= 1.")

        super(PyodModelWrapper, self).__init__(
            in_chunk_len=in_chunk_len,
            model_class=model_class,
            sampling_stride=sampling_stride,
            model_init_params=model_init_params,
            predict_params=predict_params
        )

        raise_if(self._model_class is None, "model_class must not be None.")
        raise_if_not(isinstance(self._model_class, type), "isinstance(model_class, type) must be True.")
        raise_if(
            self._predict_params.get("return_confidence", False) is True,
            "predict method does NOT support return_confidence=True, please set to False or do not pass it."
        )

        self._model = self._init_model()
        self._data_adapter = self._init_data_adapter()

        self._udf_ml_dataloader_to_fit_ndarray = default_pyod_ml_dataloader_to_fit_ndarray
        if udf_ml_dataloader_to_fit_ndarray is not None:
            self._udf_ml_dataloader_to_fit_ndarray = udf_ml_dataloader_to_fit_ndarray

        self._udf_ml_dataloader_to_predict_ndarray = default_pyod_ml_dataloader_to_predict_ndarray
        if udf_ml_dataloader_to_predict_ndarray is not None:
            self._udf_ml_dataloader_to_predict_ndarray = udf_ml_dataloader_to_predict_ndarray

    @to_tsdataset(scenario="anomaly_score")
    def predict_score(self, tsdataset: TSDataset) -> np.ndarray:
        """
        Predict raw anomaly scores of tsdataset using the fitted model, outliers are assigned with higher scores.

        Args:
            tsdataset(TSDataset): The input samples for which will be computed.

        Returns:
            np.ndarray: numpy array of shape (n_samples,), the anomaly score of the input samples.
        """
        # as this call will use fitted model, thus here is validating predictable data (rather than train data).
        self._validate_predict_data(tsdataset)

        ml_dataset = self._tsdataset_to_ml_dataset(tsdataset)
        ml_dataloader = self._ml_dataset_to_ml_dataloader(ml_dataset)

        x, y = None, None
        try:
            x, y = self._udf_ml_dataloader_to_predict_ndarray(
                ml_dataloader=ml_dataloader,
                model_init_params=self._model_init_params,
                in_chunk_len=self._in_chunk_len
            )
        except Exception as e:
            raise_log(
                ValueError(
                    f"""failed to convert train_data to numpy array as pyod model.decision_function's input. 
                    Please check _udf_ml_dataloader_to_predict_ndarray function. Error: {str(e)}"""
                )
            )

        # As pyod.BaseDetector uses @abc.abstractmethod to decorate decision_function method, thus we can ensure that
        # any classes inherited from BaseDetector must implement it, thus below call is safe, no need to try-except.
        return self._model.decision_function(x)

    def fit(self, train_data: TSDataset, valid_data: Optional[TSDataset] = None) -> None:
        """
        Fit a machine learning model.

        Args:
            train_data(TSDataset): training dataset.
            valid_data(TSDataset, optional): validation dataset. Not used, present for API consistency by convention.
        """
        if valid_data is not None:
            # currently validation dataset is not supported.
            logger.logger.warning("valid_data is CURRENTLY NOT used in fit method, pass this parameter has no effect.")
        self._validate_train_data(train_data)

        train_ml_dataset = self._tsdataset_to_ml_dataset(train_data)
        train_ml_dataloader = self._ml_dataset_to_ml_dataloader(train_ml_dataset)

        train_x, train_y = None, None
        try:
            train_x, train_y = self._udf_ml_dataloader_to_fit_ndarray(
                ml_dataloader=train_ml_dataloader,
                model_init_params=self._model_init_params,
                in_chunk_len=self._in_chunk_len
            )
        except Exception as e:
            raise_log(
                ValueError(
                    f"""failed to convert train_data to pyod model trainable numpy array. 
                    Please check udf_ml_dataloader_to_fit_ndarray function. Error: {str(e)}"""
                )
            )

        # As pyod.BaseDetector uses @abc.abstractmethod to decorate fit method, thus we can ensure that
        # any classes inherited from BaseDetector must implement it, thus below call is safe, no need to try-except.
        self._model.fit(train_x, train_y)

    @to_tsdataset(scenario="anomaly_label")
    def predict(self, tsdataset: TSDataset) -> TSDataset:
        """
        Make prediction.

        Args:
            tsdataset(TSDataset): TSDataset to predict.

        Returns:
            TSDataset: TSDataset with predictions.
        """
        self._validate_predict_data(tsdataset)

        test_ml_dataset = self._tsdataset_to_ml_dataset(tsdataset)
        test_ml_dataloader = self._data_adapter.to_ml_dataloader(test_ml_dataset, batch_size=len(test_ml_dataset))

        test_x, test_y = None, None
        try:
            test_x, test_y = self._udf_ml_dataloader_to_predict_ndarray(
                ml_dataloader=test_ml_dataloader,
                model_init_params=self._model_init_params,
                in_chunk_len=self._in_chunk_len
            )
        except Exception as e:
            raise_log(
                ValueError(
                    f"""failed to convert train_data to pyod model predictable numpy array. 
                    Please check udf_ml_dataloader_to_predict_ndarray function. Error: {str(e)}"""
                )
            )

        # As pyod.BaseDetector uses @abc.abstractmethod to decorate predict method, thus we can ensure that
        # any classes inherited from BaseDetector must implement it, thus below call is safe, no need to try-except.
        return self._model.predict(test_x, **self._predict_params)

    def _init_model(self) -> PyodBaseDetector:
        """
        Internal method, init pyod model.

        1) The model class must be inherited from pyod.model.base.BaseDetector.
        2) The initialized model object (or its ancestor) must implement fit, predict and decision_function methods.
        """
        # 1 model must be inherited from pyod.model.base.BaseDetector.
        expected_parent_class = PyodBaseDetector
        parent_classes = self._model_class.mro()

        raise_if_not(
            expected_parent_class in parent_classes,
            f"{self._model_class} must inherit from {expected_parent_class}, but actual inherit chain: {parent_classes}"
        )

        try:
            return self._model_class(**self._model_init_params)
        except Exception as e:
            # all other possible errors are captured here:
            # (1) TypeError: __init__() got an unexpected keyword argument "xxx"
            raise_log(
                ValueError(
                    f"init model failed: {self._model_class}, model_init_params: {self._model_init_params}, error: {e}."
                )
            )

    def _init_data_adapter(self) -> DataAdapter:
        """
        Internal method, initialize data adapter for pyod anomaly models.

        Returns:
            DataAdapter: Initialized data adapter object.
        """
        return DataAdapter()

    def _tsdataset_to_ml_dataset(self, tsdataset: TSDataset) -> SampleDataset:
        """
        Internal method, convert TSDataset to MLDataset.

        Returns:
            MLDataset: Converted MLDataset object.
        """
        return self._data_adapter.to_sample_dataset(
            rawdataset=tsdataset,
            in_chunk_len=self._in_chunk_len,
            sampling_stride=self._sampling_stride
        )

    def _ml_dataset_to_ml_dataloader(self, ml_dataset: SampleDataset) -> MLDataLoader:
        """
        Internal method, convert MLDataset to MLDataLoader.

        Returns:
            MLDataLoader: Converted MLDataLoader object.
        """
        return self._data_adapter.to_ml_dataloader(sample_dataset=ml_dataset, batch_size=len(ml_dataset))

    def _validate_train_data(self, train_data: TSDataset) -> None:
        """
        Internal method, validate training data. Raises if invalid.

        Args:
            train_data(TSDataset): Training data to be validated.
        """
        raise_if(train_data is None, "training dataset must not be None.")
        raise_if(train_data.get_observed_cov() is None, "observed_cov timeseries must not be None.")

        check_tsdataset(train_data)

    def _validate_predict_data(self, tsdataset: TSDataset) -> None:
        """
        Internal method, validate data for prediction. Raises if invalid.

        Args:
            tsdataset(TSDataset): Predictable data to be validated.
        """
        raise_if(tsdataset is None, "predict dataset must not be None.")
        raise_if(tsdataset.get_observed_cov() is None, "observed_cov timeseries must not be None.")

        check_tsdataset(tsdataset)


def default_pyod_ml_dataloader_to_fit_ndarray(
    ml_dataloader: MLDataLoader,
    model_init_params: Dict[str, Any],
    in_chunk_len: int
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Default function for converting MLDataLoader to a numpy array that can be used for fitting the pyod model.

    In this method will remove in_chunk_len dimension for the passed data. The reason is that all models in pyod
    requires X.ndim must == (n_samples, n_features), where n_samples is identical to batch_size, n_features is
    identical to observed_cov_col_num (In paddlets context, we define n_samples as batch_size, define n_features as
    observed_cov_col_num for anomaly detection models). However, the samples built by data adapter are 3-dim ndarray
    with shape of (batch_size, in_chunk_len, observed_cov_col_num), thus needs to flatten (i.e. remove) the first
    dimension (i.e., batch_size) and make it a 2-dim array.

    Args:
        ml_dataloader(MLDataLoader): MLDataLoader object to be converted.
        model_init_params(Dict): parameters when initializing sklearn models, possibly be used while converting.
        in_chunk_len(int): The size of the loopback window, i.e., the number of time steps feed to the model. Possibly
            be used while converting.

    Returns:
        Tuple[np.ndarray, Optional[np.ndarray]]: Converted numpy array. The first and second element in the tuple
        represent x_train and y_train, respectively.
    """
    data = next(ml_dataloader)

    # Please note that anomaly samples will NEVER contain the following keys:
    # "past_target_*"
    # "future_target_*"
    # "known_cov_*"
    # Refers to models.anomaly.ml.adapter.ml_dataset.MLDataset::_build_samples() to get more details.
    sample_x_keys = data.keys()
    # concatenated ndarray will follow the below ordered list rule:
    # [rule 1] left -> right = observed_cov_features, ..., static_cov_features.
    # [rule 2] left -> right = numeric features, ..., categorical features.
    product_keys = product(["numeric", "categorical"], ["observed_cov", "static_cov"])
    full_ordered_x_key_list = [f"{t[1]}_{t[0]}" for t in product_keys]

    # For example, given:
    # sample_keys (un-ordered) = {"observed_cov_categorical", "static_cov_numeric", "observed_cov_numeric"}
    # full_ordered_x_key_list = [
    #   "observed_cov_numeric",
    #   "static_cov_numeric",
    #   "observed_cov_categorical",
    #   "static_cov_categorical"
    # ]
    # Thus, actual_ordered_x_key_list = [
    #   "observed_cov_numeric",
    #   "static_cov_numeric",
    #   "observed_cov_categorical"
    # ]
    # The built sample ndarray will be like below:
    # [
    #   [
    #       observed_cov_numeric_feature, static_cov_numeric_feature, observed_cov_categorical_feature
    #   ],
    #   [
    #       observed_cov_numeric_feature, static_cov_numeric_feature, observed_cov_categorical_feature
    #   ],
    #   ...
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
    # Note: if a_ndarray.dtype = np.int64, b_ndarray.dtype = np.float32, then
    # np.hstack(tup=(a_ndarray, b_ndarray)).dtype will ALWAYS BE np.float32
    x = np.hstack(tup=reshaped_x_ndarray_list)
    return x, None


def default_pyod_ml_dataloader_to_predict_ndarray(
    ml_dataloader: MLDataLoader,
    model_init_params: Dict[str, Any],
    in_chunk_len: int
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Default function for converting MLDataLoader to a numpy array that can be predicted by the pyod model.

    Args:
        ml_dataloader(MLDataLoader): MLDataLoader object to be converted.
        model_init_params(Dict): parameters when initializing sklearn models, possibly be used while converting.
        in_chunk_len(int): The size of the loopback window, i.e., the number of time steps feed to the model. Possibly
            be used while converting.

    Returns:
        Tuple[np.ndarray, Optional[np.ndarray]]: Converted numpy array. The first and second element in the tuple
        represent x and y, respectively, where y is optional.
        """
    # Currently, the way convert from dataloader to fit/predict ndarray are identical.
    return default_pyod_ml_dataloader_to_fit_ndarray(
        ml_dataloader=ml_dataloader,
        model_init_params=model_init_params,
        in_chunk_len=in_chunk_len
    )


def make_ml_model(
    model_class: Type,
    in_chunk_len: int,
    out_chunk_len: int = 1,
    skip_chunk_len: int = 0,
    sampling_stride: int = 1,
    model_init_params: Dict[str, Any] = None,
    fit_params: Dict[str, Any] = None,
    predict_params: Dict[str, Any] = None,
    udf_ml_dataloader_to_fit_ndarray: Optional[Callable] = None,
    udf_ml_dataloader_to_predict_ndarray: Optional[Callable] = None
) -> MLModelBaseWrapper:
    """
    Make Wrapped time series model based on the third-party model.

    Args:
        model_class(Type): Class type of the third party model.
        in_chunk_len(int): The size of the loopback window, i.e., the number of time steps feed to the model.
        out_chunk_len(int): The size of the forecasting horizon, i.e., the number of time steps output by the model.
        skip_chunk_len(int, optional): The number of time steps between in_chunk and out_chunk for a single sample.
            The skip chunk is neither used as a feature (i.e. X) nor a label (i.e. Y) for a single sample. By
            default, it will NOT skip any time steps.
        sampling_stride(int, optional): Time steps to stride over the i-th sample and (i+1)-th sample.
            More precisely, let `t` be the time index of target time series,
            `t[i]` be the start time of the i-th sample, `t[i+1]` be the start time of the (i+1)-th sample, then
            `sampling_stride` represents the result of `t[i+1] - t[i]`.
        model_init_params(Dict[str, Any]): All params for initializing the third party model.
        fit_params(Dict[str, Any], optional): All params for fitting third party model except x_train / y_train.
        predict_params(Dict[str, Any], optional): All params for forecasting third party model except x_test / y_test.
        udf_ml_dataloader_to_fit_ndarray(Callable, optional): User defined function for converting MLDataLoader object
            to a numpy.ndarray object that can be processed by `fit` method of the third party model. Any third party
            models that accept numpy array as fit inputs can use this function to build the data for training.
        udf_ml_dataloader_to_predict_ndarray(Callable, optional): User defined function for converting MLDataLoader
            object to a numpy.ndarray object that can be processed by `predict` method of the third party model. Any
            third-party models that accept numpy array as predict inputs can use this function to build the data for
            prediction.

    Returns:
        MLModelBaseWrapper: Wrapped time series model wrapper object, currently support SklearnModelWrapper and
        PyodModelWrapper.
    """
    raise_if(model_class is None, "model_class must not be None.")
    raise_if_not(isinstance(model_class, type), "isinstance(model_class, type) must be True.")

    module = model_class.__module__.split(".")[0]

    if module == sklearn.__name__:
        return SklearnModelWrapper(
            model_class=model_class,
            in_chunk_len=in_chunk_len,
            out_chunk_len=out_chunk_len,
            skip_chunk_len=skip_chunk_len,
            sampling_stride=sampling_stride,
            model_init_params=model_init_params,
            fit_params=fit_params,
            predict_params=predict_params,
            udf_ml_dataloader_to_fit_ndarray=udf_ml_dataloader_to_fit_ndarray,
            udf_ml_dataloader_to_predict_ndarray=udf_ml_dataloader_to_predict_ndarray
        )

    if module == pyod.__name__:
        return PyodModelWrapper(
            model_class=model_class,
            in_chunk_len=in_chunk_len,
            sampling_stride=sampling_stride,
            model_init_params=model_init_params,
            predict_params=predict_params,
            udf_ml_dataloader_to_fit_ndarray=udf_ml_dataloader_to_fit_ndarray,
            udf_ml_dataloader_to_predict_ndarray=udf_ml_dataloader_to_predict_ndarray
        )
    raise_log(ValueError(f"Unable to make ml model for {model_class}."))
