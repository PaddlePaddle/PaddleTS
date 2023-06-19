# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright 2020 Element AI Inc. All rights reserved.
"""
M4 Summary
"""
from collections import OrderedDict
from paddlets.datasets.repository import get_dataset
from paddlets.datasets import UnivariateDataset

import os
import numpy as np
import pandas as pd


class M4Meta:
    seasonal_patterns = [
        'Yearly', 'Quarterly', 'Monthly', 'Weekly', 'Daily', 'Hourly'
    ]
    horizons = [6, 8, 18, 13, 14, 48]
    frequencies = [1, 4, 12, 1, 1, 24]
    horizons_map = {
        'Yearly': 6,  # pred_len
        'Quarterly': 8,
        'Monthly': 18,
        'Weekly': 13,
        'Daily': 14,
        'Hourly': 48
    }  # different predict length
    frequency_map = {
        'Yearly': 1,
        'Quarterly': 4,
        'Monthly': 12,
        'Weekly': 1,
        'Daily': 1,
        'Hourly': 24
    }
    history_size = {
        'Yearly': 1.5,
        'Quarterly': 1.5,
        'Monthly': 1.5,
        'Weekly': 10,
        'Daily': 10,
        'Hourly': 10
    }  # from interpretable.gin


def group_values(values, groups, group_name):
    return np.array([v[~np.isnan(v)] for v in values[groups == group_name]])


def mase(forecast, insample, outsample, frequency):
    return np.mean(np.abs(forecast - outsample)) / np.mean(
        np.abs(insample[:-frequency] - insample[frequency:]))


def smape_2(forecast, target):
    denom = np.abs(target) + np.abs(forecast)
    # divide by 1.0 instead of 0.0, in case when denom is zero the enumerator will be 0.0 anyway.
    denom[denom == 0.0] = 1.0
    return 200 * np.abs(forecast - target) / denom


def mape(forecast, target):
    denom = np.abs(target)
    # divide by 1.0 instead of 0.0, in case when denom is zero the enumerator will be 0.0 anyway.
    denom[denom == 0.0] = 1.0
    return 100 * np.abs(forecast - target) / denom


class M4Summary:
    def __init__(self, file_path, root_path):
        self.file_path = file_path
        self.naive_path = os.path.join(root_path, 'submission-Naive2.csv')
        self.m4_info = pd.read_csv('M4-info.csv')

    def evaluate(self):
        """
        Evaluate forecasts using M4 test dataset.

        :param forecast: Forecasts. Shape: timeseries, time.
        :return: sMAPE and OWA grouped by seasonal patterns.
        """
        grouped_owa = OrderedDict()

        naive2_forecasts = pd.read_csv(self.naive_path).values[:, 1:].astype(
            np.float32)
        naive2_forecasts = np.array(
            [v[~np.isnan(v)] for v in naive2_forecasts])

        model_mases = {}
        naive2_smapes = {}
        naive2_mases = {}
        grouped_smapes = {}
        grouped_mapes = {}
        for group_name in M4Meta.seasonal_patterns:
            dataset_name = 'M4' + group_name

            ts_train_dataset = [get_dataset(dataset_name + 'train')]
            ts_val_dataset = [get_dataset(dataset_name + 'test')]
            label_len = pred_len = M4Meta.horizons_map[dataset_name[
                2:]]  # pred_len is set internally
            seq_len = 2 * pred_len
            history_size = M4Meta.history_size[dataset_name[2:]]
            window_sampling_limit = int(history_size * pred_len)

            train_dataset = UnivariateDataset(ts_train_dataset, seq_len,
                                              pred_len, label_len,
                                              window_sampling_limit)
            valid_dataset = UnivariateDataset(ts_val_dataset, seq_len,
                                              pred_len, label_len,
                                              window_sampling_limit)

            file_name = self.file_path + dataset_name + "_forecast.csv"
            if os.path.exists(file_name):
                model_forecast = pd.read_csv(file_name).values

            naive2_forecast = group_values(naive2_forecasts,
                                           self.m4_info.SP.values, group_name)
            target = valid_dataset.timeseries
            # all timeseries within group have same frequency
            frequency = M4Meta.frequency_map[group_name]
            insample = train_dataset.timeseries

            model_mases[group_name] = np.mean([
                mase(
                    forecast=model_forecast[i],
                    insample=insample[i],
                    outsample=target[i],
                    frequency=frequency) for i in range(len(model_forecast))
            ])
            naive2_mases[group_name] = np.mean([
                mase(
                    forecast=naive2_forecast[i],
                    insample=insample[i],
                    outsample=target[i],
                    frequency=frequency) for i in range(len(model_forecast))
            ])

            naive2_smapes[group_name] = np.mean(
                smape_2(naive2_forecast, target))
            grouped_smapes[group_name] = np.mean(
                smape_2(
                    forecast=model_forecast, target=target))
            grouped_mapes[group_name] = np.mean(
                mape(
                    forecast=model_forecast, target=target))

        grouped_smapes = self.summarize_groups(grouped_smapes)
        grouped_mapes = self.summarize_groups(grouped_mapes)
        grouped_model_mases = self.summarize_groups(model_mases)
        grouped_naive2_smapes = self.summarize_groups(naive2_smapes)
        grouped_naive2_mases = self.summarize_groups(naive2_mases)
        for k in grouped_model_mases.keys():
            grouped_owa[k] = (
                grouped_model_mases[k] / grouped_naive2_mases[k] +
                grouped_smapes[k] / grouped_naive2_smapes[k]) / 2

        def round_all(d):
            return dict(map(lambda kv: (kv[0], np.round(kv[1], 3)), d.items()))

        return round_all(grouped_smapes), round_all(grouped_owa), round_all(
            grouped_mapes), round_all(grouped_model_mases)

    def summarize_groups(self, scores):
        """
        Re-group scores respecting M4 rules.
        :param scores: Scores per group.
        :return: Grouped scores.
        """
        scores_summary = OrderedDict()

        def group_count(group_name):
            return len(np.where(self.m4_info.SP.values == group_name)[0])

        weighted_score = {}
        for g in ['Yearly', 'Quarterly', 'Monthly']:
            weighted_score[g] = scores[g] * group_count(g)
            scores_summary[g] = scores[g]

        others_score = 0
        others_count = 0
        for g in ['Weekly', 'Daily', 'Hourly']:
            others_score += scores[g] * group_count(g)
            others_count += group_count(g)
        weighted_score['Others'] = others_score
        scores_summary['Others'] = others_score / others_count

        average = np.sum(list(weighted_score.values())) / len(
            self.m4_info.SP.values)
        scores_summary['Average'] = average

        return scores_summary


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred