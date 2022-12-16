# !/usr/bin/env python3
# -*- coding:utf-8 -*-

import abc
import copy
from typing import Callable, List, Optional, Union
from paddlets.utils.utils import get_tsdataset_max_len, split_dataset

import numpy as np
import pandas as pd

from paddlets import TSDataset, TimeSeries
from paddlets.logger import raise_if_not, raise_if, raise_log
from paddlets.logger.logger import log_decorator

TSDATASET_COL_TYPES = ['target', 'observed_cov', 'known_cov']

class BaseTransform(object, metaclass=abc.ABCMeta):
    """
    Base class for all data transformation classes (named `transformers` in this module)

    Any subclass or transformer needs to inherit from this base class and
    implement :func:`fit`, :func:`transform` and :func:`fit_transform` methods.
    """
    def __init__(self):
        #if transformer need previous data to generate features
        self.need_previous_data = False
        self.n_rows_pre_data_need = 0

    def _check_multi_tsdataset(self, datasets: List[TSDataset]):
        """
        Check the validity of multi time series combination transform

        Args:
            datasets(List[TSDataset]): datasets from which to fit or transform the transformer.
        """
        raise_if(
            len(datasets) == 0,
            "The Length of datasets cannot be 0!"
        )
        columns_set = set(tuple(sorted(dataset.columns.items())) for dataset in datasets)
        raise_if_not(
            len(columns_set) == 1,
            f"Invalid tsdatasets. The given tsdataset column schema ({[ts.columns for ts in datasets]}) must be same."
        )

    def fit(self, dataset: Union[TSDataset, List[TSDataset]]):
        """
        Learn the parameters from the dataset needed by the transformer.

        Any non-abstract class inherited from this class should implement this method.

        The parameters fitted by this method is transformer-specific. For example, the `MinMaxScaler` needs to 
        compute the MIN and MAX, and the `StandardScaler` needs to compute the MEAN and STD (standard deviation)
        from the dataset. 

        Args:
            dataset(Union[TSDataset, List[TSDataset]]): dataset from which to fit the transformer.
        """
        if isinstance(dataset, list):
            self._check_multi_tsdataset(dataset)
            attr_list = ['target', 'observed_cov', 'known_cov']
            ts_build_param = {}
            for attr in attr_list:
                if getattr(dataset[0], attr) is not None:
                    ts_build_param[attr] = TimeSeries(
                        pd.concat([getattr(data, attr).data for data in dataset]).reset_index(drop=True),
                        1
                    )
                else:
                    ts_build_param[attr] = None
            #new_dataset is not a standard TSDataset, only use fit
            new_dataset = TSDataset(**ts_build_param)
            return self.fit_one(new_dataset) 
        else:
            return self.fit_one(dataset)

    @abc.abstractmethod
    def fit_one(self, dataset: TSDataset):
        """
        Learn the parameters from the dataset needed by the transformer.

        Any non-abstract class inherited from this class should implement this method.

        The parameters fitted by this method is transformer-specific. For example, the `MinMaxScaler` needs to 
        compute the MIN and MAX, and the `StandardScaler` needs to compute the MEAN and STD (standard deviation)
        from the dataset. 

        Args:
            dataset(TSDataset): dataset from which to fit the transformer.
        """
        pass

    def transform(
        self,
        dataset: Union[TSDataset, List[TSDataset]],
        inplace: bool = False
    ) -> Union[TSDataset, List[TSDataset]]:
        """
        Apply the fitted transformer on the dataset

        Any non-abstract class inherited from this class should implement this method.

        Args:
            dataset(Union[TSDataset, List[TSDataset]): dataset to be transformed.
            inplace(bool, optional): Set to True to perform inplace transformation. Default is False.
            
        Returns:
            Union[TSDataset, List[TSDataset]]: transformed dataset.
        """
        if isinstance(dataset, list):
            self._check_multi_tsdataset(dataset)
            return [self.transform_one(data, inplace) for data in dataset]
        else:
            return self.transform_one(dataset, inplace)

    @abc.abstractmethod
    def transform_one(
        self,
        dataset: TSDataset,
        inplace: bool = False
    ) -> TSDataset:
        """
        Apply the fitted transformer on the dataset

        Any non-abstract class inherited from this class should implement this method.

        Args:
            dataset(TSDataset): dataset to be transformed.
            inplace(bool, optional): Set to True to perform inplace transformation. Default is False.
            
        Returns:
            TSDataset: transformed dataset.
        """
        pass

    def transform_n_rows(
            self,
            dataset: TSDataset,
            n_rows:int,
            inplace: bool = False,
    ) -> TSDataset:
        """
        Apply the fitted transformer on the part of the dataset


        Args:
            dataset(TSDataset): dataset to be transformed.
            n_rows(int):n_rows to be transformed.
            inplace(bool, optional): Set to True to perform inplace transformation. Default is False.
            
        Returns:
            TSDataset: transformed dataset.
        """
        data_len = get_tsdataset_max_len(dataset)
        if not self.need_previous_data or data_len == n_rows:
            return self.transform_one(dataset, inplace)
        if self.n_rows_pre_data_need == -1:
            transformed_dataset = self.transform_one(dataset, inplace)
            _, res = split_dataset(transformed_dataset , data_len - n_rows)
        else:
            _, dataset = split_dataset(dataset, data_len - n_rows - self.n_rows_pre_data_need)
            transformed_dataset = self.transform_one(dataset, inplace)
            _, res = split_dataset(transformed_dataset, self.n_rows_pre_data_need)
        return res

    def fit_transform(
        self,
        dataset: Union[TSDataset, List[TSDataset]],
        inplace: bool = False
    ) -> Union[TSDataset, List[TSDataset]]:
        """
        Combine the above fit and transform into one method, firstly fitting the transformer from the dataset 
        and then applying the fitted transformer on the dataset.

        Any non-abstract class inherited from this class should implement this method.

        Args:
            dataset(Union[TSDataset, List[TSDataset]]): dataset to process.
            inplace(bool, optional): Set to True to perform inplace transformation. Default is False.

        Returns:
            Union[TSDataset, List[TSDataset]]: transformed data.
        """
        return self.fit(dataset).transform(dataset, inplace)

    def inverse_transform(
        self,
        dataset: Union[TSDataset, List[TSDataset]],
        inplace: bool = False
    ) -> Union[TSDataset, List[TSDataset]]:
        """
        Inversely transform the dataset output by the `transform` method.

        Differ from other abstract methods, this method is not decorated by abc.abstractmethod. The reason is that not
        all the transformations can be transformed back inversely, thus, it is neither possible nor mandatory
        for all sub classes inherited from this base class to implement this method.

        In general, other modules such as Pipeline will possibly call this method WITHOUT knowing if the called
        transform instance has implemented this method. To work around this, instead of simply using `pass`
        expression as the default placeholder, this method raises a NotImplementedError to enable the callers
        (e.g. Pipeline) to use try-except mechanism to identify those data transformation operators that do NOT 
        implement this method.

        Args:
            dataset(Union[TSDataset, List[TSDataset]]): dataset to be inversely transformed.
            inplace(bool, optional): Set to True to perform inplace transformation. Default is False.

        Returns:
            TSDataset: inverserly transformed dataset.

        Raises:
            NotImplementedError
        """
        if isinstance(dataset, list):
            self._check_multi_tsdataset(dataset)
            return [self.inverse_transform_one(data, inplace) for data in dataset]
        else:
            return self.inverse_transform_one(dataset, inplace)

    def inverse_transform_one(
        self,
        dataset: TSDataset,
        inplace: bool = False
    ) -> TSDataset:
        """
        Inversely transform the dataset output by the `transform` method.

        Differ from other abstract methods, this method is not decorated by abc.abstractmethod. The reason is that not
        all the transformations can be transformed back inversely, thus, it is neither possible nor mandatory
        for all sub classes inherited from this base class to implement this method.

        In general, other modules such as Pipeline will possibly call this method WITHOUT knowing if the called
        transform instance has implemented this method. To work around this, instead of simply using `pass`
        expression as the default placeholder, this method raises a NotImplementedError to enable the callers
        (e.g. Pipeline) to use try-except mechanism to identify those data transformation operators that do NOT 
        implement this method.

        Args:
            dataset(TSDataset): dataset to be inversely transformed.
            inplace(bool, optional): Set to True to perform inplace transformation. Default is False.

        Returns:
            TSDataset: inverserly transformed dataset.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError

class UdBaseTransform(BaseTransform):
    """
    User define base transform.

    Args:
        ud_transformer(object): User define or third-party transformer object.
        in_col_names(Optional[Union[str, List[str]]]): Column name or names to be transformed.
        per_col_transform(bool): Whether each column of data is transformed independently, default False.
        drop_origin_columns(bool): Whether to delete the original column, default=False.
        out_col_types(Optional[Union[str, List[str]]]): The type of output columns, None values represent automatic inference based on input.
        out_col_names(Optional[List[str]]): The name of output columns, None values represent automatic inference based on input.
        
    """

    def __init__(
        self,
        ud_transformer: object,
        in_col_names: Optional[Union[str, List[str]]]=None,
        per_col_transform: bool=False,
        drop_origin_columns: bool=False,
        out_col_types: Optional[Union[str, List[str]]]=None,
        out_col_names: Optional[List[str]]=None,
    ):
        super().__init__()
        self._ud_transformer = ud_transformer
        self._drop_origin_columns = drop_origin_columns
        self._out_col_types = out_col_types
        self._out_col_names = out_col_names
        self._cols = [in_col_names] if isinstance(in_col_names, str) else in_col_names
        self._fitted = False
        self._per_col_transform = per_col_transform
        if self._per_col_transform:
            self._ud_transformer_col_list = {}

    def _check_output(
        self,
        raw_dataset: TSDataset,
        input: pd.DataFrame,
        output: np.ndarray
    ):
        """
        Check the legitimacy of the output.

        Args:
            raw_dataset(TSDataset): dataset to be transformed.
            input(pd.DataFrame): The input of ud transformer base on the raw_dataset.
            output(np.ndarray): The output of ud transformer.

        Returns:
            None

        Raises:
            ValueError
        """
        raise_if_not(
            input.shape[0] == output.shape[0],
            "The row of input is not equal to the row of output!"
        )

        output_col_num = output.shape[1] if len(output.shape) >= 2 else output.shape[0]

        if self._out_col_names:
            if isinstance(self._out_col_names, list):
                raise_if_not(
                    len(self._out_col_names) == output_col_num,
                    "The out_col_names does not match the actual output!"
                )                
        
        def check_start_time():
            start_time_set = set(raw_dataset.get_item_from_column(column).start_time for column in input.columns)
            return len(start_time_set) == 1

        def get_input_col_type():
            return set(raw_dataset.columns[column] for column in input.columns)

        if self._out_col_types:
            raise_if_not(
                check_start_time(),
                "The start time point of input cols is different!"
            )
            if isinstance(self._out_col_types, list):
                raise_if_not(
                    len(self._out_col_types) == output_col_num,
                    "The out_col_types does not match the actual output"
                )
                for type in self._out_col_types:
                    raise_if(
                        type not in TSDATASET_COL_TYPES,
                        f"Invalid col type: {type}"
                    )
        else:
            if len(get_input_col_type()) != 1:
                raise_if_not(
                    input.shape == output.shape and \
                    check_start_time() and \
                    self._drop_origin_columns,
                    "The type of input column must be the same in this case!"
                )

    def _infer_output_column_types(
        self,
        raw_dataset: TSDataset,
        input: pd.DataFrame,
    )-> Union[str, List[str]]:
        """
        Infer output column's types.

        Args:
            raw_dataset(TSDataset): dataset to be transformed.
            input(pd.DataFrame): The input of ud transformer base on the raw_dataset.
            output(np.ndarray): The output of ud transformer.

        Returns:
            out_col_types(Union[str, List[str]])
        """
        if self._out_col_types:
            return self._out_col_types
        else:
            columns = list(raw_dataset.columns[column] for column in input.columns)
            if len(set(columns)) == 1:
                return columns[0]
            else:
                return columns

    def _get_output_column_names(
        self,
        input: pd.DataFrame,
        output: np.ndarray
    )-> List[str]:
        """
        Get output column's names.

        Args:
            input(pd.DataFrame): The input of ud transformer base on the raw_dataset.
            output(np.ndarray): The output of ud transformer.

        Returns:
            out_col_names(List[str])
        """
        if self._out_col_names:
            return self._out_col_names
        else:
            if input.shape == output.shape and self._drop_origin_columns:
                return list(input.columns)
            else:
                name_prefix = "_".join(
                    [
                        self._ud_transformer.__class__.__name__,
                        "-".join(
                            [column for column in input.columns]
                        )
                    ]
                )
                output_col_num = output.shape[1] if len(output.shape) >= 2 else output.shape[0]
                return [f"{name_prefix}_{i}" for i in range(output_col_num)]

    def _gen_input(
        self,
        raw_dataset: TSDataset,
        cols: Union[str, List[str]],
        strict: bool=True
    )->pd.DataFrame:
        """
        Generate the input to ud transformer base on raw_dataset.

        Args:
            raw_dataset(TSDataset): dataset to be transformed.
            cols(Union[str, List[str]]): The input col names.
            strict(bool): Strict matching or not.

        Returns:
            input(pd.DataFrame)
        """
        cols = cols if isinstance(cols, list) else [cols]
        if strict:
            input = raw_dataset[cols]
        else:
            cols = [col for col in cols if col in raw_dataset.columns]
            raise_if(
                len(cols) == 0,
                "No cols was matched!"
            )
            input = raw_dataset[cols]
        if isinstance(input, pd.Series):
            input = input.to_frame()
        return input

    def _gen_output(
        self,
        raw_output
    )->np.ndarray:
        """
        Generate the np.ndarray output base on the raw_output from ud transform.

        Args:
            raw_output(TSDataset): raw_output from ud transform.

        Returns:
            output(np.ndarray)
        """
        if isinstance(raw_output, np.ndarray):
            return raw_output
        else:
            raise_log(
                TypeError(f"Invalid output type: {type(raw_output)}")
            )

    @log_decorator
    def fit_one(self, dataset: TSDataset):
        """
        Learn the parameters from the dataset needed by the transformer.
        
        Args:
            dataset(TSDataset): dataset from which to fit the transformer
        
        Returns:
            self
        """
        if self._cols is None:
            self._cols = list(dataset.columns.keys())
        if self._per_col_transform:
            tmp_tansformer = self._ud_transformer
            for col in self._cols:
                self._ud_transformer = copy.deepcopy(tmp_tansformer)
                input = self._gen_input(dataset, col)
                self._fit(input)
                self._ud_transformer_col_list[col] = self._ud_transformer
        else:
            input = self._gen_input(dataset, self._cols)
            self._fit(input)
        self._fitted = True
        return self

    def _transform_logic(
        self, 
        dataset: TSDataset, 
        cols: Union[str, List[str]],
        transform_func: Callable
    ) -> TSDataset:
        """
        Transform or inverse_transform the dataset with the fitted transformer.
        
        Args:
            dataset(TSDataset): dataset to be transformed.
            cols(Union[str, List[str]]): The input col names.
            transform_func(Callable): The transform function.
        
        Returns:
            TSDataset
        """
        cols = cols if isinstance(cols, list) else [cols]

        input = self._gen_input(dataset, cols)

        raw_output = transform_func(input)

        output = self._gen_output(raw_output)
        
        self._check_output(dataset, input, output)

        col_names = self._get_output_column_names(input, output)
        col_types = self._infer_output_column_types(dataset, input)
    
        insert_col_name = []
        def set_columns(output, col_types):
            for col in output.columns:
                dataset.set_column(col, output[col], col_types)
                insert_col_name.append(col)
        
        def gen_index():
            start_time = dataset.get_item_from_column(cols[0]).start_time
            if isinstance(dataset.freq, str):
                return pd.date_range(
                    start=start_time, 
                    periods=output.shape[0], 
                    freq=dataset.freq
                )
            else:
                return pd.RangeIndex(
                    start=start_time, 
                    stop=start_time + output.shape[0] * dataset.freq,
                    step=dataset.freq
                )
        time_index = gen_index()

        if isinstance(col_types, str):
            tmp_output = pd.DataFrame(output, index=time_index, columns=col_names)
            set_columns(tmp_output, col_types)
        else:
            for i in range(output.shape[1]):
                tmp_output = pd.DataFrame(output[:, i], index=time_index, columns=[col_names[i]])
                set_columns(tmp_output, col_types[i])
        if self._drop_origin_columns:
            for col in input.columns:
                if col not in insert_col_name:
                    dataset.drop(col)        
        return dataset

    @log_decorator
    def transform_one(
        self, 
        dataset: TSDataset, 
        inplace: bool = False
    ) -> TSDataset:
        """
        Transform or inverse_transform the dataset with the fitted transformer.
        
        Args:
            dataset(TSDataset): dataset to be transformed.
            inplace(bool): whether to replace the original data. default=False
        
        Returns:
            TSDataset
        """
        new_ts = dataset if inplace else dataset.copy()
        if self._per_col_transform:
            for col in self._cols:
                if col not in dataset.columns:
                    continue
                self._ud_transformer = self._ud_transformer_col_list[col]
                self._transform_logic(new_ts, col, self._transform)
            return new_ts
        else:
            return self._transform_logic(new_ts, self._cols, self._transform)

    @log_decorator
    def inverse_transform_one(
        self, 
        dataset: TSDataset,
        inplace: bool=False
    ) -> TSDataset:
        """
        Inversely transform the dataset output by the `transform` method.

        Args:
            dataset(TSDataset): dataset to be inversely transformed.
            inplace(bool): Set to True to perform inplace operation and avoid data copy.

        Returns:
            TSDataset: Inversely transformed TSDataset.
        """
        new_ts = dataset if inplace else dataset.copy()
        if self._per_col_transform:
            for col in self._cols:
                if col not in dataset.columns:
                    continue
                self._ud_transformer = self._ud_transformer_col_list[col]
                self._transform_logic(new_ts, col, self._inverse_transform)
            return new_ts
        else:
            return self._transform_logic(dataset, self._cols, self._inverse_transform)

    @abc.abstractmethod
    def _fit(self, input: pd.DataFrame):
        """
        Learn the parameters from the dataset needed by the transformer.
        
        Args:
            input(pd.DataFrame): The input to transformer.
        
        Returns:
            None
        """
        pass

    @abc.abstractmethod
    def _transform(
        self, 
        input: pd.DataFrame
    ):
        """
        Transform the dataset with the fitted transformer.
        
        Args:
            input(pd.DataFrame): The input to transformer.
         
        """
        pass

    def _inverse_transform(
            self, 
            input: pd.DataFrame
        ):
        """
        Inversely transform the dataset output by the `transform` method.

        Args:
            input(pd.DataFrame): The input to transformer.
        
        """
        raise NotImplementedError
