# !/usr/bin/env python3
# -*- coding:utf-8 -*-

import abc
import numpy as np
import sklearn
from typing import Type, Dict, Any, Optional, Callable, Tuple

from paddlets.models.base import BaseModel
from paddlets.models.forecasting.ml.ml_base import MLBaseModel
from paddlets.models.forecasting.ml.adapter.data_adapter import DataAdapter
from paddlets.models.forecasting.ml.adapter.ml_dataset import MLDataset
from paddlets.models.forecasting.ml.adapter.ml_dataloader import MLDataLoader
from paddlets.models.utils import to_tsdataset
from paddlets.datasets import TSDataset
from paddlets.logger import raise_log, raise_if, raise_if_not


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
        out_chunk_len: int,
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
        raise_if_not(isinstance(self._model_class, type), "isinstance(model_class, type) must be True.)")

        self._model = self._init_model()
        self._data_adapter = self._init_data_adapter()

        self._udf_ml_dataloader_to_fit_ndarray = default_ml_dataloader_to_fit_ndarray
        if udf_ml_dataloader_to_fit_ndarray is not None:
            self._udf_ml_dataloader_to_fit_ndarray = udf_ml_dataloader_to_fit_ndarray

        self._udf_ml_dataloader_to_predict_ndarray = default_ml_dataloader_to_predict_ndarray
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

    @to_tsdataset
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
    ) -> MLDataset:
        """
        Internal method, convert TSDataset to MLDataset.

        Returns:
            MLDataset: Converted MLDataset object.
        """
        return self._data_adapter.to_ml_dataset(
            rawdataset=tsdataset,
            in_chunk_len=self._in_chunk_len,
            out_chunk_len=self._out_chunk_len,
            skip_chunk_len=self._skip_chunk_len,
            sampling_stride=self._sampling_stride,
            time_window=time_window
        )

    def _ml_dataset_to_ml_dataloader(self, ml_dataset: MLDataset) -> MLDataLoader:
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

        target_ts = train_data.get_target()
        # multi target is NOT supported.
        raise_if(
            len(target_ts.columns) > 1,
            f"training dataset target timeseries columns number ({len(target_ts.columns)}) must be 1."
        )

    def _validate_predict_data(self, test_data: TSDataset) -> None:
        """
        Internal method, validate data for prediction. Raises if invalid.

        Args:
            test_data(TSDataset): Predict data to be validated.
        """
        raise_if(test_data is None, "The dataset to be predicted must not be None.")
        raise_if(test_data.get_target() is None, "target timeseries to be predicted must not be None.")

        target_ts = test_data.get_target()
        # multi target is NOT supported.
        raise_if(
            len(target_ts.columns) > 1,
            f"columns number of target timeseries to be predicted ({len(target_ts.columns)}) must be 1."
        )


def default_ml_dataloader_to_fit_ndarray(
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

    observed_cov = data['observed_cov']
    observed_cov = observed_cov.reshape(observed_cov.shape[0], observed_cov.shape[1] * observed_cov.shape[2])
    known_cov = data['known_cov']
    known_cov = known_cov.reshape(known_cov.shape[0], known_cov.shape[1] * known_cov.shape[2])
    if in_chunk_len < 1:
        x_train = np.hstack((known_cov, observed_cov))
    else:
        past_target = data['past_target']
        past_target = past_target.reshape(past_target.shape[0], past_target.shape[1] * past_target.shape[2])
        x_train = np.hstack((known_cov, observed_cov, past_target))

    # data["future_target"].shape = (n_samples, out_chunk_len, target_col_num)
    # y_train.shape must be (n_samples, )
    # the pre-check in the self.__init__ already guarantee that out_chunk_len must be equal to 1.
    # the pre-check in the self.predict already guarantee that target_col_num must be equal to 1.
    y_train = np.squeeze(data["future_target"])
    return x_train, y_train


def default_ml_dataloader_to_predict_ndarray(
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

    observed_cov = data['observed_cov']
    observed_cov = observed_cov.reshape(observed_cov.shape[0], observed_cov.shape[1] * observed_cov.shape[2])
    known_cov = data['known_cov']
    known_cov = known_cov.reshape(known_cov.shape[0], known_cov.shape[1] * known_cov.shape[2])
    if in_chunk_len < 1:
        x_test = np.hstack((known_cov, observed_cov))
    else:
        past_target = data['past_target']
        past_target = past_target.reshape(past_target.shape[0], past_target.shape[1] * past_target.shape[2])
        x_test = np.hstack((known_cov, observed_cov, past_target))
    return x_test, None


def make_ml_model(
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
        SklearnModelWrapper: Wrapped time series model wrapper object.
    """
    raise_if(model_class is None, "model_class must not be None.")
    raise_if_not(isinstance(model_class, type), "isinstance(model_class, type) must be True.")

    if model_class.__module__.split(".")[0] == sklearn.__name__:
        return SklearnModelWrapper(
            in_chunk_len=in_chunk_len,
            out_chunk_len=out_chunk_len,
            model_class=model_class,
            skip_chunk_len=skip_chunk_len,
            sampling_stride=sampling_stride,
            model_init_params=model_init_params,
            fit_params=fit_params,
            predict_params=predict_params,
            udf_ml_dataloader_to_fit_ndarray=udf_ml_dataloader_to_fit_ndarray,
            udf_ml_dataloader_to_predict_ndarray=udf_ml_dataloader_to_predict_ndarray
        )
    raise_log(ValueError(f"Unable to make ml model for {model_class}."))
