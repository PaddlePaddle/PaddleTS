#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from typing import Dict, List, Union, Callable, Optional, Any
from IPython.display import display
from tqdm import tqdm
import math

import numpy as np
import pandas as pd
from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from paddlets.utils.utils import check_model_fitted
from paddlets.models.forecasting import TFTModel
from paddlets.logger import Logger, raise_if
from paddlets.datasets import TSDataset, TimeSeries

logger = Logger(__name__)

rcParams.update({'figure.autolayout': True,
                 'figure.figsize': [10, 5],
                 'font.size': 10})


class TFTExplainer(TFTModel):
    """
    Inherit TFT, and implement an explainer, which provides display of the explanation result.
    """
    def __init__(self, *args, **kwargs):
        """
        Init base TFT.
        """
        super().__init__(*args, **kwargs)

    def _update_fit_params(
        self,
        train_tsdataset: List[TSDataset],
        valid_tsdataset: Optional[List[TSDataset]] = None
    ) -> Dict[str, Any]:
        """
        Rewrite base `_update_fit_params` function to add more data members for explanation function.
        
        Args:
            train_tsdataset(List[TSDataset]): list of train dataset
            valid_tsdataset(List[TSDataset], optional): list of validation dataset

        Returns:
            Dict[str, Any]: model parameters
        """
        fit_params = super()._update_fit_params(train_tsdataset, valid_tsdataset)
        self.static_cols = self.static_num_cols + self.static_cat_cols
        self.history_cols = self.target_cols + self.known_num_cols + \
        self.observed_num_cols + self.known_cat_cols + self.observed_cat_cols
        self.future_cols = self.known_num_cols + self.known_cat_cols
        self.mapping = {
            'Static Weights': {'arr_key': 'static_weights', 'feat_names': self.static_cols},
            'Historical Weights': {'arr_key': 'historical_selection_weights', 'feat_names': self.history_cols},
            'Future Weights': {'arr_key': 'future_selection_weights', 'feat_names': self.future_cols},
        }
        return fit_params
           
    def _display_explanations(
        self,
        explain_data: Dict[str, np.ndarray],
        weights_prctile: Optional[List[float]]=[10, 50, 90],
        observation_index: int=0,
        horizons: Union[List[int], int]=[1,5,10],
        top_feature_num: Optional[int]=20,
        unit: Optional[str] = 'Units'
    ):
        """
        Main visualization logic, which contains selection weights and attention scores. 
        
        Args:
            explain_data(Dict[str, np.ndarray]): A dictionary of numpy arrays containing the explanation outputs of the model for a set of observations.
            weights_prctile(List[float]): A list of percentile to compute as a distribution describer for the scores. 
            observation_index(int): The index with the dataset, corresponding to the observation for which the visualization will be generated.
            horizons(List[int], Optional): A list horizon, specified in time-steps units, for which the statistics will be computed. 
            top_feature_num(int, Optional): An integer specifying the quantity of the top weighted features to display.
            unit(str, Optional): The units associated with the time-steps. This variable is used for labeling the corresponding axes.
            
        """
        if not isinstance(weights_prctile, list):
            weights_prctile = [weights_prctile]
        # ========================
        # Selection Weights
        # ========================
        self._display_selection_weights_stats(outputs_dict=explain_data,
                                             prctiles=weights_prctile,
                                             mapping=self.mapping
                                            )
        self._display_sample_wise_selection_stats(weights_arr=explain_data['static_weights'],
                                               observation_index=observation_index,
                                               feature_names=self.static_cols,
                                               top_n=top_feature_num,
                                               title='Static Features')
        self._display_sample_wise_selection_stats(weights_arr=explain_data['historical_selection_weights'],
                                           observation_index=observation_index,
                                           feature_names=self.history_cols,
                                           top_n=top_feature_num,
                                           title='Historical Features',
                                           rank_stepwise=True)
        self._display_sample_wise_selection_stats(weights_arr=explain_data['future_selection_weights'],
                                           observation_index=observation_index,
                                           feature_names=self.future_cols,
                                           top_n=top_feature_num,
                                           title='Future Features',
                                           historical=False,
                                           rank_stepwise=False)
        # ========================
        # Attention Scores
        # ========================
        if not isinstance(horizons, list):
            horizons = [horizons]
            
        # One step ahead
        if len(weights_prctile) > 1: # only for backtest scenario 
            for horizon in horizons:
                self._display_attention_scores(attention_scores=explain_data['attention_scores'],
                                          horizons=horizon,
                                          prctiles=weights_prctile,
                                          unit=unit)
        # Multihorizon Attention
        # for prediction scenario and backtest scenario
        for prctile in weights_prctile:
            self._display_attention_scores(attention_scores=explain_data['attention_scores'],
                                          horizons=horizons,
                                          prctiles=prctile,
                                          unit=unit)
        # Single specific sample
        if explain_data['attention_scores'].shape[0] > 1:
            self._display_sample_wise_attention_scores(attention_scores=explain_data['attention_scores'],
                                                      observation_index=observation_index,
                                                      horizons=horizons,
                                                      unit=unit)
         
    def _check_horizons(
        self,
        horizons: Union[List[int], int],
        out_chunk_len: int,
    ):
        """
        check validation of horizons for backtest and prediction.
        
        Args:
            horizons(Union[List[int], int]): List of horizon to be explained.
            out_chunk_len(int): The size of the model's forecasting time steps.
        """
        if isinstance(horizons, int):
            horizons = [horizons]
        for horizon in horizons:
            raise_if(horizon > out_chunk_len, 
                     f"all horizons should be no bigger than model's `out_chunk_len`: {out_chunk_len}, got {horizons}.")
     
    def _check_backtest_params(
        self,
        target_length: int,
        in_chunk_len: int,
        skip_chunk_len: int,
        start: Union[pd.Timestamp, int, str ,float] = None,
    ):
        """
        For backtest's explanation, check validation of parameters.
        
        Args:
            target_length(int): The length of target.
            in_chunk_len(int): The size of the loopback window, i.e., the number of time steps feed to the model.
            skip_chunk_len(int): Optional, the number of time steps between in_chunk and out_chunk for a single sample.
            start(Union[pd.Timestamp, int, str ,float]): The first prediction time, at which a prediction is computed for a future time.
        """
        # check whether model fitted or not.
        check_model_fitted(self)
        # start time should no less than in_chunk_len + skip_chunk_len
        raise_if(start < in_chunk_len + skip_chunk_len, 
            f"Parameter 'start' value should >= in_chunk_len {in_chunk_len} + skip_chunk_len {skip_chunk_len}")
        # start time should no bigger than target length
        raise_if(start > target_length, 
            f"Parameter 'start' value should not exceed data target_len {target_length}")
        # if skip_chunk_len !=0, prediction will start from start + skip_chunk_len.
        if skip_chunk_len != 0:
            logger.info(f"model.skip_chunk_len is {skip_chunk_len}, \
            backtest will start at index {start + skip_chunk_len} (start + skip_chunk_len)")
    
    def explain_backtest(
        self,
        data: TSDataset,
        start: Union[pd.Timestamp, int, str ,float] = None,
        observation_index: Optional[int]=0,
        horizons: Union[List[int], int]=[1],
        unit: Optional[str] = 'Units',
        display: Optional[bool] = True
    ):
        """
        Explain backtest data, the backtest logic is a simplied version of `utils.backtest` by setting `predict_window` and `stride` as `out_chunk_len`.
        
        Args:
            data(TSDataset): The TSdataset used for successively generating explanation result and visualizing.
            start(Union[pd.Timestamp, int, str ,float]): The first prediction time, at which a prediction is computed for a future time.
            observation_index(int, Optional): The index with the dataset, corresponding to the observation for which the visualization will be generated.
            horizons(Union[List[int], int]): A list horizon, specified in time-steps units, for which the statistics will be computed. 
            unit(str, Optional): The units associated with the time-steps. This variable is used for labeling the corresponding axes.
            display(bool, Optional): Whether to display the explanation results.
            
        Returns:
            Dict[str, np.ndarray]: Aggregated explanation data predicted by the model.
        """
        predicts_agg = {}
        data = data.copy()
        predict_window = self._out_chunk_len
        all_target = data.get_target()
        all_observe = data.get_observed_cov() if data.get_observed_cov() else None
        if start is None:
            start = self._in_chunk_len + self._skip_chunk_len
        start = all_target.get_index_at_point(start)
        # check horizons
        self._check_horizons(horizons, self._out_chunk_len)
        # check backtest parameters
        self._check_backtest_params(len(all_target), self._in_chunk_len, self._skip_chunk_len, start)
        predict_rounds = math.ceil((len(all_target) - start) / self._sampling_stride)
        index = start - self._skip_chunk_len
        # iterative prediction
        for _ in tqdm(range(predict_rounds), desc="Backtest Progress"):
            data._target, rest = all_target.split(index)
            data._observed_cov, _ = all_observe.split(index) if all_observe else (None, None)
            rest_len = len(rest)
            if rest_len < predict_window + self._skip_chunk_len:
                if data.known_cov is not None:
                    target_end_time = data._target.end_time
                    known_index = data.known_cov.get_index_at_point(target_end_time)
                    if len(data.known_cov) - known_index - 1 < predict_window + self._skip_chunk_len:
                        break
                predict_window = rest_len - self._skip_chunk_len
            output = self.predict_interpretable(data)
            for key, array in output.items():
                predicts_agg.setdefault(key, []).append(array)
            # step to next sample
            index = index + self._sampling_stride
            
        # aggregates all sample's explanation results
        outputs = dict()
        for k in list(predicts_agg.keys()):
            outputs[k] = np.concatenate(predicts_agg[k], axis=0)
        # visualization
        if display:
            self._display_explanations(explain_data=outputs, 
                                       weights_prctile=self._q_points.tolist(),
                                       observation_index=observation_index, 
                                       horizons=horizons,
                                       unit=unit)
        return outputs

    def explain_prediction(
        self,
        data: TSDataset,
        horizons: Union[List[int], int]=[1],
        unit: Optional[str] = 'Units',
        display: Optional[bool] = True
    ):
        """
        Explain prediction data, in cases of single sample prediction.
        
        Args:
            data(TSDataset): The TSdataset used for predicting explanation result and visualizing.
            horizons(Union[List[int], int]): A list or a single horizon, specified in time-steps units, for which the statistics will be computed. 
            unit(str, Optional): The units associated with the time-steps. This variable is used for labeling the corresponding axes.
            display(bool, Optional): Whether to display the explanation results.
            
        Returns:
            Dict[str, np.ndarray]: Explanation data predicted by the model.
        """
        # check horizons
        self._check_horizons(horizons, self._out_chunk_len)
        # explained result for single prediction
        explain_data = self.predict_interpretable(data)
        if display:
            # for prediction scenario, only one sample generated, set percentile as 50.
            self._display_explanations(explain_data=explain_data, 
                                       weights_prctile=50, 
                                       observation_index=0, 
                                       horizons=horizons, 
                                       unit=unit)
        return explain_data
              
    def _aggregate_weights(
        self,
        output_arr: np.ndarray,
        prctiles: List[float],
        feat_names: List[str]) -> pd.DataFrame:
        """
        Implements a utility function for aggregating selection weights for a set (array) of observations,
        whether these selection weights are associated with the static input attributes, or with a set of temporal selection
        weights.
        The aggregation of the weights is performed through the computation of several percentiles (provided by the caller)
        for describing the distribution of the weights, for each attribute.

        Args:
            output_arr(np.ndarray): A 2D or 3D array containing the selection weights output by the model. A 3D tensor will imply selection weights associated with temporal inputs.
            prctiles(List[float]): A list of percentiles according to which the distribution of selection weights will be described.
            feat_names(List[str]):A list of strings associated with the relevant attributes (according to the their order).

        Returns:
            agg_df(pd.DataFrame): A pandas dataframe, indexed with the relevant feature names, containing the aggregation of selection weights.
        """

        prctiles_agg = []  # a list to contain the computation for each percentile
        for q in prctiles:  # for each of the provided percentile
            # infer whether the provided weights are associated with a temporal input channel
            if len(output_arr.shape) > 2:
                # lose the temporal dimension and then describe the distribution of weights
                flatten_time = output_arr.reshape(-1, output_arr.shape[-1])
            else:  # if static - take as is
                flatten_time = output_arr
            # accumulate
            prctiles_agg.append(np.percentile(flatten_time, q=q, axis=0))

        # combine the computations and index according to feature names
        agg_df = pd.DataFrame({prctile: aggs for prctile, aggs in zip(prctiles, prctiles_agg)})
        agg_df.index = feat_names
        return agg_df

    def _display_selection_weights_stats(
        self,
        outputs_dict: Dict[str, np.ndarray],
        prctiles: List[float],
        mapping: Dict,
    ):
        """
        Implements a utility function for displaying the selection weights statistics of multiple input channels according
        to the outputs provided by the model for a set of input observations.
        It requires a mapping which specifies which output key corresponds to each input channel, and the associated list
        of attributes.

        Args:
            outputs_dict(Dict[str,np.ndarray]): A dictionary of numpy arrays containing the outputs of the model for a set of observations.
            prctiles(List[float]): A list of percentiles according to which the distribution of selection weights will be described.
            mapping(Dict): A dictionary specifying the output key corresponding to which input channel and the associated feature names.
            sort_by(Optional[float]): The percentile according to which the weights statistics will be sorted before displaying (Must be included as
            part of ``prctiles``).
        """
        
        sort_by = 50.0 if 50.0 in prctiles else prctiles[0]
        # for each input channel included in the mapping
        for name, config in mapping.items():
            if not config["feat_names"]:
                continue
            # perform weight aggregation according to the provided configuration
            weights_agg = self._aggregate_weights(output_arr=outputs_dict[config['arr_key']],
                                            prctiles=prctiles,
                                            feat_names=config['feat_names'])
            print(name)
            print('=========')
            # display the computed statistics, sorted, and color highlighted according to the value.
            display(weights_agg.sort_values([sort_by], ascending=False).style.background_gradient(cmap='viridis'))

    def _display_attention_scores(
        self,
        attention_scores: np.ndarray,
        horizons: Union[int, List[int]],
        prctiles: Union[float, List[float]],
        unit: Optional[str] = 'Units'
    ):
        """
        Implements a utility function for displaying the statistics of attention scores according
        to the outputs provided by the model for a set of input observations.
        The statistics of the scores will be described using specified percentiles, and for specified horizons.

        Args:
            attention_scores(np.ndarray): A numpy array containing the attention scores for the relevant dataset.
            horizons(Union[int, List[int]]): A list or a single horizon, specified in time-steps units, for which the statistics will be computed. If more than one horizon was configured, then only a single percentile computation will be allowed.
            prctiles(Union[int, List[int]]): A list or a single percentile to compute as a distribution describer for the scores. If more than percentile was configured, then only a single horizon will be allowed.
            unit(Optional[str]): The units associated with the time-steps. This variable is used for labeling the corresponding axes.
        """

        # if any of ``horizons`` or ``prctiles`` is provided as int, transform into a list.
        if not isinstance(horizons, list):
            horizons = [horizons]
        if not isinstance(prctiles, list):
            prctiles = [prctiles]

        # make sure only maximum one of ``horizons`` and ``prctiles`` has more than one element.
        assert len(prctiles) == 1 or len(horizons) == 1

        # compute the configured percentiles of the attention scores, for each percentile separately
        attn_stats = {}
        for prctile in prctiles:
            attn_stats[prctile] = np.percentile(attention_scores, q=prctile, axis=0)

        #fig, ax = plt.subplots(figsize=(10, 5))
        fig, ax = plt.subplots()
        if len(prctiles) == 1:  # in case only a single percentile was configured
            relevant_prctile = prctiles[0]
            title = f"Multi-Step - Attention ({relevant_prctile}% Percentile)"
            scores_percentile = attn_stats[relevant_prctile]
            for horizon in horizons:  # a single line for each horizon
                # infer the corresponding x_axis according to the shape of the scores array
                siz = scores_percentile.shape
                x_axis = np.arange(siz[0] - siz[1], siz[0])
                ax.plot(x_axis, scores_percentile[horizon - 1], lw=1, label=f"t + {horizon} scores", marker='o')
        else:
            title = f"{horizons[0]} Steps Ahead - Attention Scores"
            for prctile, scores_percentile in attn_stats.items():  # for each percentile
                # infer the corresponding x_axis according to the shape of the scores array
                siz = scores_percentile.shape
                x_axis = np.arange(siz[0] - siz[1], siz[0])
                ax.plot(x_axis, scores_percentile[horizons[0] - 1], lw=1, label=f"{prctile}%", marker='o')

        ax.axvline(x=0, lw=1, color='r', linestyle='--')
        ax.grid(True)
        ax.set_xlabel(f"Relative Time-step [{unit}]")
        ax.set_ylabel('Attention Scores')
        ax.set_title(title)
        ax.legend()
        plt.show(block=False)

    def _display_sample_wise_attention_scores(
        self,
        attention_scores: np.ndarray,
        observation_index: int,
        horizons: Union[int, List[int]],
        unit: Optional[str] = None):
        """
        Implements a utility function for displaying, on a single observation level,
        the attention scores output by the model, for, possibly, a multitude of horizons.

        Args:
            attention_scores(np.ndarray): A numpy array containing the attention scores for the relevant dataset.
            observation_index(int): The index with the dataset, corresponding to the observation for which the visualization will be generated.
            horizons(Union[int, List[int]]): A list or a single horizon, specified in time-steps units, for which the scores will be displayed.
            unit(Optional[str]): The units associated with the time-steps. This variable is used for labeling the corresponding axes.
        """
        # if ``horizons`` is provided as int, transform into a list.
        if isinstance(horizons, int):
            horizons = [horizons]

        # take the relevant record from  the provided array, using the specified index
        sample_attn_scores = attention_scores[observation_index, ...]

        fig, ax = plt.subplots()
        # infer the corresponding x_axis according to the shape of the scores array
        attn_shape = sample_attn_scores.shape
        x_axis = np.arange(attn_shape[0] - attn_shape[1], attn_shape[0])

        # for each horizon, plot the associated attention score signal for all the steps
        for step in horizons:
            ax.plot(x_axis, sample_attn_scores[step - 1], marker='o', lw=3, label=f"t+{step}")

        ax.axvline(x=-0.5, lw=1, color='k', linestyle='--')
        ax.grid(True)
        ax.legend()

        ax.set_xlabel('Relative Time-Step ' + (f"[{unit}]" if unit else ""))
        ax.set_ylabel('Attention Score')
        ax.set_title('Attention Mechanism Scores - Per Horizon')
        plt.show(block=False)

    def _display_sample_wise_selection_stats(
        self,
        weights_arr: np.ndarray,
        observation_index: int,
        feature_names: List[str],
        top_n: Optional[int] = None,
        title: Optional[str] = '',
        historical: Optional[bool] = True,
        rank_stepwise: Optional[bool] = False
    ):
        """
        Implements a utility function for displaying, on a single observation level, the selection weights output by the
        model. This function can handle selection weights of both temporal input channels and static input channels.

        Args:
            weights_arr(np.ndarray): A 2D or 3D array containing the selection weights output by the model. A 3D tensor will implies selection weights associated with temporal inputs.
            observation_index(int): The index with the dataset, corresponding to the observation for which the visualization will be generated.
            feature_names(List[str]): A list of strings associated with the relevant attributes (according to the their order).
            top_n(Optional[int]): An integer specifying the quantity of the top weighted features to display.
            title(Optional[str]): A string which will be used when creating the title for the visualization.
            historical(Optional[bool]): Specifies whether the corresponding input channel contains historical data or future data. Relevant only for temporal input channels, and used for display purposes.
            rank_stepwise(Optional[bool]): Specifies whether to rank the features according to their weights, on each time-step separately, or simply display the raw selection weights output by the model. Relevant only for temporal input channels, and used for display purposes.
        """

        # a-priori assume non-temporal input channel
        num_temporal_steps = None

        # infer number of attributes according to the shape of the weights array
        weights_shape = weights_arr.shape
        num_features = weights_shape[-1]
        if num_features <= 1:
            # no feature, return
            return
        # infer whether the input channel is temporal or not
        is_temporal: bool = len(weights_shape) > 2

        # bound maximal number of features to display by the total amount of features available (in case provided)
        top_n = min(num_features, top_n) if top_n else num_features

        # take the relevant record from  the provided array, using the specified index
        sample_weights = weights_arr[observation_index, ...]

        if is_temporal:
            # infer number of temporal steps
            num_temporal_steps = weights_shape[1]
            # aggregate the weights (by averaging) across all the time-steps
            sample_weights_trans = sample_weights.T
            weights_df = pd.DataFrame({'weight': sample_weights_trans.mean(axis=1)}, index=feature_names)
        else:
            # in case the input channel is not temporal, just use the weights as is
            weights_df = pd.DataFrame({'weight': sample_weights}, index=feature_names)

        # ========================
        # Aggregative Barplot
        # ========================
        #fig, ax = plt.subplots(figsize=(10, 5))
        fig, ax = plt.subplots()
        weights_df.sort_values('weight', ascending=False).iloc[:top_n].plot.bar(ax=ax)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(11)
            tick.label.set_rotation(45)

        ax.grid(True)
        ax.set_xlabel('Feature Name')
        ax.set_ylabel('Selection Weight')
        ax.set_title(title + (" - " if title != "" else "") + \
                     f"Selection Weights " + ("Aggregation " if is_temporal else "") + \
                     (f"- Top {top_n}" if top_n < num_features else ""))
        plt.show(block=False)

        if is_temporal:
            # ========================
            # Temporal Display
            # ========================
            # infer the order of the features, according to the average selection weight across time
            order = sample_weights_trans.mean(axis=1).argsort()[::-1]

            # order the weights sequences as well as their names accordingly
            ordered_weights = sample_weights_trans[order]
            ordered_names = [feature_names[i] for i in order.tolist()]

            if rank_stepwise:
                # the weights are now considered to be the ranking after ordering the features in each time-step separately
                ordered_weights = np.argsort(ordered_weights, axis=0)

            #fig, ax = plt.subplots(figsize=(9, 6))
            fig, ax = plt.subplots()
            # create a corresponding x-axis, going forward/backwards, depending on the configuration
            if historical:
                map_x = {idx: val for idx, val in enumerate(np.arange(0 - num_temporal_steps, 1))}
            else:
                map_x = {idx: val for idx, val in enumerate(np.arange(1, num_temporal_steps + 1))}

            def format_fn(tick_val, tick_pos):
                if int(tick_val) in map_x:
                    return map_x[int(tick_val)]
                else:
                    return ''

            # display the weights as images
            im = ax.pcolor(ordered_weights, edgecolors='gray', linewidths=2)
            # feature names displayed to the left
            ax.yaxis.set_ticks(np.arange(len(ordered_names)))
            ax.set_yticklabels(ordered_names)

            ax2 = ax.twiny()
            ax2.set_xticks([])
            ax2.xaxis.set_ticks_position('top')
            ax.set_xlabel(('Historical' if historical else 'Future') + ' Time-Steps')
            ax2.set_xlabel(('Historical' if historical else 'Future') + ' Time-Steps')

            ax.xaxis.set_major_formatter(FuncFormatter(format_fn))
            fig.colorbar(im, orientation="horizontal", pad=0.05, ax=ax2)
            plt.show(block=False)
