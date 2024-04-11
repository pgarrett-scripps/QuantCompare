import copy
import os
import time
from dataclasses import dataclass
from functools import cached_property, partial
from itertools import groupby
from typing import Any, Dict, List, Tuple, Callable
import argparse
import numpy as np
import pprint
import pandas as pd
import tqdm as tqdm
from scipy.stats import t
from statsmodels.stats.multitest import multipletests

import warnings

# Suppress only the specific RuntimeWarning related to precision loss in scipy
warnings.filterwarnings('ignore', category=RuntimeWarning,
                        message='Precision loss occurred in moment calculation due to catastrophic cancellation.')
np.seterr(divide='ignore', invalid='ignore')


def sum_uncertainty(*std_devs: np.ndarray) -> np.ndarray:
    """
    Calculate the uncertainty (standard deviation) of the sum of multiple values.
    """
    # Calculate the variance of the sum by adding the squares of the standard deviations
    variance_sum = sum([std_dev ** 2 for std_dev in std_devs])

    # The standard deviation is the square root of the variance
    return np.sqrt(variance_sum)


def log2_transformed_uncertainty(ratio: np.ndarray, ratio_std: np.ndarray) -> np.ndarray:
    """
    Calculate the uncertainty (standard deviation) of the log2-transformed value of a given ratio.
    """

    # Calculate the standard deviation of the log2-transformed ratio
    ln2 = np.log(2)
    log2_uncertainty = np.divide(ratio_std, (ratio * ln2))

    return log2_uncertainty


def mean_uncertainty(*std_devs: np.ndarray) -> np.ndarray:
    """
    Calculate the uncertainty (standard deviation) of the mean of multiple values.
    """
    # Calculate the variance of the mean by dividing the sum of the squares of the standard deviations by the number of
    # measurements squared
    variance_mean = sum([std_dev ** 2 for std_dev in std_devs]) / len(std_devs) ** 2

    # The standard deviation is the square root of the variance
    return np.sqrt(variance_mean)


def divide_uncertainty(numerator: np.ndarray, numerator_std: np.ndarray, denominator: np.ndarray,
                       denominator_std: np.ndarray) -> np.ndarray:
    """
    Calculate the uncertainty (standard deviation) of the division of two measurements.
    """
    # Calculate the variance of the division
    variance_division = (np.divide(numerator_std, denominator) ** 2) + (
            np.divide(numerator * denominator_std, denominator ** 2) ** 2)

    # Return the standard deviation of the division (sqrt of variance)
    return np.sqrt(variance_division)


@dataclass(frozen=True)
class Psm:
    peptide: str
    charge: int
    filename: str
    proteins: List[str]
    scannr: int
    intensities: np.array(np.float32)
    norm_intensities: np.array(np.float32)


@dataclass(frozen=True)
class QuantGroup:
    group: Any
    group_indices: List[int]
    psm: Psm

    @property
    def intensities(self) -> np.ndarray:
        return self.psm.intensities[self.group_indices]

    @property
    def norm_intensities(self) -> np.ndarray:
        return self.psm.norm_intensities[self.group_indices]


def group_intensities(intensities: np.ndarray, groups: Dict[Any, List[int]]) -> List[np.ndarray]:
    """
    Group the intensities by the indices in the groups dictionary.
    """
    grouped_intensities = []
    for _, indices in groups.items():
        grouped_intensities.append(intensities[:, indices])
    return grouped_intensities


def add_norm_intensities_to_df(df: pd.DataFrame) -> None:
    """
    Add normalized intensities to the DataFrame.
    """

    intensities = np.vstack(df['reporter_ion_intensity'].values).astype(np.float32)
    norm_intensities = copy.deepcopy(intensities)

    for filename in df['filename'].unique():
        mask = (df['filename'] == filename).values
        file_ints = norm_intensities[mask, :]
        sums = np.sum(file_ints, axis=0, keepdims=True)
        average_sum = np.mean(sums)
        norm_factor = average_sum * sums
        norm_intensities[mask, :] /= norm_factor

    # Normalize across files if needed
    global_sums = np.sum(norm_intensities, axis=0, keepdims=True)
    global_average_sum = np.mean(global_sums)
    norm_intensities /= global_average_sum * global_sums

    df['normalized_reporter_ion_intensity'] = norm_intensities.tolist()


def make_quant_groups(df: pd.DataFrame, groups: Dict[Any, List[int]]) -> List[QuantGroup]:
    """
    Create a list of QuantGroup objects from a DataFrame and a groups dictionary.
    """

    quant_groups = []
    # Create Psm objects for each row in the group
    psms = [Psm(row['peptide'], row['charge'], row['filename'], row['proteins'], row['scannr'],
                np.array(row['reporter_ion_intensity'], dtype=np.float32),
                np.array(row['normalized_reporter_ion_intensity'], dtype=np.float32))
            for i, row in df.iterrows()]

    # Create QuantGroup objects for each Psm and associated groups
    for psm in psms:
        for group, indices in groups.items():
            # Assuming you need to select the intensities by the indices here; otherwise, adjust accordingly.
            quant_groups.append(QuantGroup(group, indices, psm))

    return quant_groups


def calculate_p_value_against_zero(mean, std_dev, n):
    """
    Calculate the p-value for a one-sample t-test comparing the sample mean to 0.
    """

    mu_0 = 0

    # Calculate the t-statistic
    t_statistic = (mean - mu_0) / (std_dev / np.sqrt(n))

    # Calculate the p-value
    p_value = t.sf(np.abs(t_statistic), df=n - 1) * 2  # two-tailed test

    return p_value


"""
def simple_sum_ratio_zscore(arr1: np.array, arr2: np.array) -> Tuple[float, float, float]:
    # Simply sums the intensities of each group and returns the ratio
    group1_mean, group1_std = np.mean(arr1, axis=1), np.std(arr1, axis=1)
    group2_mean, group2_std = np.mean(arr2, axis=1), np.std(arr2, axis=1)

    # take the sum of the means
    group1_mean, group1_mean_std = np.sum(group1_mean), sum_uncertainty(*group1_std)
    group2_mean, group2_mean_std = np.sum(group2_mean), sum_uncertainty(*group2_std)

    # Calculate the ratio of the means and the uncertainties
    ratio, ratio_std = np.divide(group1_mean, group2_mean), divide_uncertainty(group1_mean, group1_mean_std,
                                                                               group2_mean, group2_mean_std)
    log2_ratio, log2_std = np.log2(ratio), log2_transformed_uncertainty(ratio, ratio_std)

    log2_ratio_zscore = (log2_ratio - 0) / log2_std
    log2_ratio_zscore_pvalue = stats.norm.sf(np.abs(log2_ratio_zscore)) * 2

    return log2_ratio, log2_std, log2_ratio_zscore_pvalue


def simple_sum_ratio_ttest_1samp(arr1: np.array, arr2: np.array) -> Tuple[float, float, float, float, float]:
    # Calculate the mean of each row
    group1_mean, group1_std = np.mean(arr1, axis=1), np.std(arr1, axis=1)
    group2_mean, group2_std = np.mean(arr2, axis=1), np.std(arr2, axis=1)

    # Calculate the ratio of the means and the uncertainties
    ratios = group1_mean / group2_mean
    ratio_uncertainties = divide_uncertainty(group1_mean, group1_std, group2_mean, group2_std)

    # Calculate the log2 of the ratios and the uncertainties
    log2_ratios = np.log2(ratios)
    log2_ratio_uncertainties = log2_transformed_uncertainty(ratios, ratio_uncertainties)

    # Calculate the mean of the log2 ratios and uncertainties
    log2_ratio = np.mean(log2_ratios)
    log2_std = np.mean(log2_ratio_uncertainties)

    pvalue = stats.ttest_1samp(log2_ratios, 0)[1]

    return log2_ratio, log2_std, pvalue


def colum_sum_ratio(arr1: np.array, arr2: np.array) -> Tuple[float, float, float]:
    # Sums the intensities of each column and returns the ratio

    group1_sums = np.sum(arr1, axis=0)
    group2_sums = np.sum(arr2, axis=0)

    group1_mean, group1_std = np.mean(group1_sums), np.std(group1_sums)
    group2_mean, group2_std = np.mean(group2_sums), np.std(group2_sums)

    ratio = np.divide(group1_mean, group2_mean)
    std = divide_uncertainty(group1_mean, group1_std, group2_mean, group2_std)

    pvalue = stats.ttest_ind(group1_sums, group2_sums)[1]
    log2_ratio = np.log2(ratio)
    log2_std = log2_transformed_uncertainty(ratio, std)

    return log2_ratio, log2_std, pvalue
"""


def reference_mean_ratio_rollup(arr1: np.array, arr2: np.array, inf_replacement: float = 100) -> Tuple[
    float, float, float]:
    """
    Calculate the ratio of the mean of each row of two arrays.

    .. code-block:: python

        >>> arr1 = np.array([[100, 110, 105], [200, 210, 205]])
        >>> arr2 = np.array([[200, 230, 240, 200], [400, 430, 440, 410]])
        >>> reference_mean_ratio_rollup(arr1, arr2)
        (-0.09307656849087903, 2.5476488778079367, 0.9205955813892215)

        >>> arr1 = np.array([[0, 0, 0], [200, 210, 205]])
        >>> arr2 = np.array([[200, 230, 240, 0], [400, 430, 440, 0]])
        >>> reference_mean_ratio_rollup(arr1, arr2)
        (-inf, nan, nan)

    """

    # Calculate the mean of group1
    group1_means, group1_stds = np.mean(arr1, axis=1).reshape(-1, 1), np.std(arr1, axis=1).reshape(-1, 1)

    # Calculate the ratios
    ratios = np.divide(group1_means, arr2)
    ratios_std = divide_uncertainty(group1_means, group1_stds, arr2, np.zeros_like(arr2))

    # Calculate the log2 of the ratios and the uncertainties
    log2_ratios = np.log2(ratios)
    log2_ratio_uncertainties = log2_transformed_uncertainty(ratios, ratios_std)

    # get index of positive and negative infinities and nan
    pos_inf_idx = np.isposinf(log2_ratios)
    neg_inf_idx = np.isneginf(log2_ratios)
    nan_idx = np.isnan(log2_ratios)

    # replace positive infinities with inf_replacement, and uncertainties with 1/10 of the replacement
    log2_ratios[pos_inf_idx] = inf_replacement
    log2_ratio_uncertainties[pos_inf_idx] = inf_replacement // 10
    # replace negative infinities with -inf_replacement, and uncertainties with 1/10 of the replacement
    log2_ratios[neg_inf_idx] = -inf_replacement
    log2_ratio_uncertainties[neg_inf_idx] = inf_replacement // 10
    # replace nan with 0, and uncertainties with 1/10 of the replacement
    log2_ratios[nan_idx] = 0
    log2_ratio_uncertainties[nan_idx] = inf_replacement // 10

    # remove nans
    ratios = ratios[~np.isnan(ratios)]

    # if all ratios are nan, return nan
    if len(ratios) == 0:
        return np.inf, np.nan, np.nan

    log2_ratios = np.log2(ratios)

    # Replace positive infinities
    log2_ratios[np.isposinf(log2_ratios)] = inf_replacement

    # Replace negative infinities
    log2_ratios[np.isneginf(log2_ratios)] = -inf_replacement

    # Calculate the mean of the log2 ratios and uncertainties
    log2_ratio = np.mean(log2_ratios)
    log2_std = mean_uncertainty(*log2_ratio_uncertainties)[0]
    pvalue = calculate_p_value_against_zero(log2_ratio, log2_std, len(log2_ratios) * min(arr1.shape[1], arr2.shape[1]))

    return float(log2_ratio), float(log2_std), float(pvalue)


def mean_ratio_rollup(arr1: np.array, arr2: np.array, inf_replacement: float = 100) -> Tuple[float, float, float]:
    """
    Calculate the ratio of the mean of each row of two arrays.

    .. code-block:: python

        >>> arr1 = np.array([[100, 110, 105], [200, 210, 205]])
        >>> arr2 = np.array([[200, 230, 240, 200], [400, 430, 440, 410]])
        >>> mean_ratio_rollup(arr1, arr2)
        (-1.0426957456153223, 0.07236359360185755, 3.43566121759087e-07)


        >>> arr1 = np.array([[0, 0, 0], [200, 210, 205], [0, 0, 0]])
        >>> arr2 = np.array([[200, 230, 240, 200], [0, 0, 0, 0], [0, 0, 0, 0]])
        >>> mean_ratio_rollup(arr1, arr2)
        (-1.0426957456153223, 0.07236359360185755, 3.43566121759087e-07)

    """

    # Calculate the means and standard deviations of group of reporter ions
    group1_means, group1_stds = np.mean(arr1, axis=1).reshape(-1, 1), np.std(arr1, axis=1).reshape(-1, 1)
    group2_means, group2_stds = np.mean(arr2, axis=1).reshape(-1, 1), np.std(arr2, axis=1).reshape(-1, 1)

    # Calculate the ratios and uncertainties
    ratios = np.divide(group1_means, group2_means)
    ratio_uncertainties = divide_uncertainty(group1_means, group1_stds, group2_means, group2_stds)

    # Calculate the log2 of the ratios and the uncertainties
    log2_ratios = np.log2(ratios)
    log2_ratio_uncertainties = log2_transformed_uncertainty(ratios, ratio_uncertainties)

    # get index of positive and negative infinities and nan
    pos_inf_idx = np.isposinf(log2_ratios)
    neg_inf_idx = np.isneginf(log2_ratios)
    nan_idx = np.isnan(log2_ratios)

    # replace positive infinities with inf_replacement, and uncertainties with 1/10 of the replacement
    log2_ratios[pos_inf_idx] = inf_replacement
    log2_ratio_uncertainties[pos_inf_idx] = inf_replacement // 10
    # replace negative infinities with -inf_replacement, and uncertainties with 1/10 of the replacement
    log2_ratios[neg_inf_idx] = -inf_replacement
    log2_ratio_uncertainties[neg_inf_idx] = inf_replacement // 10
    # replace nan with 0, and uncertainties with 1/10 of the replacement
    log2_ratios[nan_idx] = 0
    log2_ratio_uncertainties[nan_idx] = inf_replacement // 10

    # Calculate the mean of the log2 ratios and uncertainties
    log2_ratio = np.mean(log2_ratios)
    log2_std = mean_uncertainty(*log2_ratio_uncertainties)[0]
    pvalue = calculate_p_value_against_zero(log2_ratio, log2_std, len(log2_ratios) * min(arr1.shape[1], arr2.shape[1]))

    return float(log2_ratio), float(log2_std), float(pvalue)


@dataclass()
class GroupRatio:
    QuantGroup1: List[QuantGroup]
    QuantGroup2: List[QuantGroup]
    ratio_function: Callable

    # global properties (added later)
    qvalue: float = None
    norm_qvalue: float = None
    centered_log2_ratio: float = None
    centered_norm_log2_ratio: float = None

    @cached_property
    def _log2_ratio(self) -> Tuple[float, float, float]:
        return self.ratio_function(self.group1_intensity_arr, self.group2_intensity_arr)

    @cached_property
    def _log2_norm_ratio(self) -> Tuple[float, float, float]:
        return self.ratio_function(self.group1_norm_intensity_arr, self.group2_norm_intensity_arr)

    @property
    def ratio(self) -> float:
        return 2 ** self.log2_ratio

    @property
    def centered_ratio(self) -> float:
        return 2 ** self.centered_log2_ratio

    @property
    def norm_ratio(self) -> float:
        return 2 ** self.log2_norm_ratio

    @property
    def centered_norm_ratio(self) -> float:
        return 2 ** self.centered_norm_log2_ratio

    @property
    def log2_ratio(self) -> float:
        return self._log2_ratio[0]

    @property
    def log2_ratio_std(self) -> float:
        return self._log2_ratio[1]

    @property
    def log2_ratio_pvalue(self) -> float:
        return self._log2_ratio[2]

    @property
    def log2_norm_ratio(self) -> float:
        return self._log2_norm_ratio[0]

    @property
    def log2_norm_ratio_std(self) -> float:
        return self._log2_norm_ratio[1]

    @property
    def log2_norm_ratio_pvalue(self) -> float:
        return self._log2_norm_ratio[2]

    @cached_property
    def group1_intensity_arr(self) -> np.ndarray:
        group1_array = np.array([qg.intensities for qg in self.QuantGroup1], dtype=np.float32)
        return group1_array

    @cached_property
    def group2_intensity_arr(self) -> np.ndarray:
        group2_array = np.array([qg.intensities for qg in self.QuantGroup2], dtype=np.float32)
        return group2_array

    @cached_property
    def group1_norm_intensity_arr(self) -> np.ndarray:
        group1_array = np.array([qg.norm_intensities for qg in self.QuantGroup1], dtype=np.float32)
        return group1_array

    @cached_property
    def group2_norm_intensity_arr(self) -> np.ndarray:
        group2_array = np.array([qg.norm_intensities for qg in self.QuantGroup2], dtype=np.float32)
        return group2_array

    @property
    def group1(self) -> Any:
        assert all(qg.group == self.QuantGroup1[0].group for qg in self.QuantGroup1)
        return self.QuantGroup1[0].group

    @property
    def group2(self) -> Any:
        assert all(qg.group == self.QuantGroup2[0].group for qg in self.QuantGroup2)
        return self.QuantGroup2[0].group

    @property
    def group1_total_intensity(self) -> np.ndarray:
        return np.sum(self.group1_intensity_arr)

    @property
    def group2_total_intensity(self) -> np.ndarray:
        return np.sum(self.group2_intensity_arr)

    @property
    def group1_total_norm_intensity(self) -> np.ndarray:
        return np.sum(self.group1_norm_intensity_arr)

    @property
    def group2_total_norm_intensity(self) -> np.ndarray:
        return np.sum(self.group2_norm_intensity_arr)

    @property
    def group1_intensity(self) -> np.ndarray:
        return np.sum(self.group1_intensity_arr, axis=0)

    @property
    def group2_intensity(self) -> np.ndarray:
        return np.sum(self.group2_intensity_arr, axis=0)

    @property
    def group1_norm_intensity(self) -> np.ndarray:
        return np.sum(self.group1_norm_intensity_arr, axis=0)

    @property
    def group2_norm_intensity(self) -> np.ndarray:
        return np.sum(self.group2_norm_intensity_arr, axis=0)

    @property
    def group1_norm_average_intensity(self) -> np.ndarray:
        return np.mean(self.group1_norm_intensity)

    @property
    def group2_norm_average_intensity(self) -> np.ndarray:
        return np.mean(self.group2_norm_intensity)

    @property
    def group1_average_intensity(self) -> np.ndarray:
        return np.mean(self.group1_intensity)

    @property
    def group2_average_intensity(self) -> np.ndarray:
        return np.mean(self.group2_intensity)

    def __len__(self) -> float:
        assert len(self.QuantGroup1) == len(self.QuantGroup2)
        return len(self.QuantGroup1)


def get_psm_ratio_data_wide(quant_ratios: List[GroupRatio], pairs: List[Tuple[Any, Any]], groupby_filename: bool) -> \
        (List[str], List[List[Any]]):
    """
    Get the ratio data in wide format for PSMs (groups by peptide, charge, and filename if groupby_filename is True).
    """
    if groupby_filename:
        sort_func = lambda qr: (
            qr.QuantGroup1[0].psm.peptide, qr.QuantGroup1[0].psm.charge, qr.QuantGroup1[0].psm.filename)
        groupby_cols = ['peptide', 'charge', 'filename']

    else:
        sort_func = lambda qr: (qr.QuantGroup1[0].psm.peptide, qr.QuantGroup1[0].psm.charge)
        groupby_cols = ['peptide', 'charge']

    return _get_ratio_data_wide(quant_ratios, pairs, groupby_cols, sort_func)


def get_peptide_ratio_data_wide(quant_ratios: List[GroupRatio], pairs: List[Tuple[Any, Any]],
                                groupby_filename: bool) -> (List[str], List[List[Any]]):
    """
    Get the ratio data in wide format for peptides (groups by peptide if groupby_filename is False).
    """
    if groupby_filename:
        sort_func = lambda qr: (qr.QuantGroup1[0].psm.peptide, qr.QuantGroup1[0].psm.filename)
        groupby_cols = ['peptide', 'filename']

    else:
        sort_func = lambda qr: (qr.QuantGroup1[0].psm.peptide)
        groupby_cols = ['peptide']

    return _get_ratio_data_wide(quant_ratios, pairs, groupby_cols, sort_func)


def get_protein_ratio_data_wide(quant_ratios: List[GroupRatio], pairs: List[Tuple[Any, Any]],
                                groupby_filename: bool) -> (List[str], List[List[Any]]):
    """
    Get the ratio data in wide format for proteins (groups by proteins if groupby_filename is False).
    """
    if groupby_filename:
        sort_func = lambda qr: (qr.QuantGroup1[0].psm.proteins, qr.QuantGroup1[0].psm.filename)
        groupby_cols = ['proteins', 'filename']

    else:
        sort_func = lambda qr: (qr.QuantGroup1[0].psm.proteins,)
        groupby_cols = ['proteins', ]

    return _get_ratio_data_wide(quant_ratios, pairs, groupby_cols, sort_func)


def _get_ratio_data_wide(quant_ratios: List[GroupRatio], pairs: List[Tuple[Any, Any]], groupby_cols: List[str],
                         groupby_func: Callable) -> (List[str], List[List[Any]]):
    """
    Get the ratio data in wide format for a given groupby function. Looks complex, but just aggregates the data.
    """
    pair_keys = set()
    for pair in pairs:
        pair_keys.add(pair[0])
        pair_keys.add(pair[1])
    pair_keys = list(pair_keys)
    pair_keys.sort()

    columns = []
    columns.extend(groupby_cols)
    for key in pair_keys:
        columns.extend(
            [
                f'intensities_{key}',
                f'total_intensity_{key}',
                f'average_intensity_{key}',
                f'norm_intensities_{key}',
                f'total_norm_intensity_{key}',
                f'average_norm_intensity_{key}'
            ]
        )
    for pair in pairs:
        columns.extend(
            [
                f'ratio_{pair[0]}_{pair[1]}',
                f'centered_ratio_{pair[0]}_{pair[1]}',
                f'log2_ratio_{pair[0]}_{pair[1]}',
                f'centered_log2_ratio_{pair[0]}_{pair[1]}',
                f'log2_ratio_std_{pair[0]}_{pair[1]}',
                f'log2_ratio_pvalue_{pair[0]}_{pair[1]}',
                f'log2_ratio_qvalue_{pair[0]}_{pair[1]}',
                f'norm_ratio_{pair[0]}_{pair[1]}',
                f'centered_norm_ratio_{pair[0]}_{pair[1]}',
                f'norm_log2_ratio_{pair[0]}_{pair[1]}',
                f'centered_norm_log2_ratio_{pair[0]}_{pair[1]}',
                f'norm_log2_ratio_std_{pair[0]}_{pair[1]}',
                f'norm_log2_pvalue_{pair[0]}_{pair[1]}',
                f'norm_log2_qvalue_{pair[0]}_{pair[1]}',
            ]
        )
    columns.append('cnt')

    datas = []
    quant_ratios.sort(key=groupby_func)
    for key, group in tqdm.tqdm(groupby(quant_ratios, groupby_func), desc='Generating Wide Data'):
        data = []
        group_to_intensities = {}
        pair_to_ratio_data = {}
        cnt = 0
        for quant_ratio in group:
            cnt += 1
            pair_to_ratio_data[(quant_ratio.QuantGroup1[0].group, quant_ratio.QuantGroup2[0].group)] = \
                (
                    quant_ratio.ratio,
                    quant_ratio.centered_ratio,
                    quant_ratio.log2_ratio,
                    quant_ratio.centered_log2_ratio,
                    quant_ratio.log2_ratio_std,
                    quant_ratio.log2_ratio_pvalue,
                    quant_ratio.qvalue,
                    quant_ratio.norm_ratio,
                    quant_ratio.centered_norm_ratio,
                    quant_ratio.log2_norm_ratio,
                    quant_ratio.centered_norm_log2_ratio,
                    quant_ratio.log2_norm_ratio_std,
                    quant_ratio.log2_norm_ratio_pvalue,
                    quant_ratio.norm_qvalue,
                )

            g1, g2 = quant_ratio.group1, quant_ratio.group2

            if g1 not in group_to_intensities:
                group_to_intensities[g1] = (
                    quant_ratio.group1_intensity,
                    quant_ratio.group1_total_intensity,
                    quant_ratio.group1_average_intensity,
                    quant_ratio.group1_norm_intensity,
                    quant_ratio.group1_total_norm_intensity,
                    quant_ratio.group1_norm_average_intensity)

            if g2 not in group_to_intensities:
                group_to_intensities[g2] = (
                    quant_ratio.group2_intensity,
                    quant_ratio.group2_total_intensity,
                    quant_ratio.group2_average_intensity,
                    quant_ratio.group2_norm_intensity,
                    quant_ratio.group2_total_norm_intensity,
                    quant_ratio.group2_norm_average_intensity)

        if isinstance(key, tuple):
            for val in key:
                data.append(val)
        else:
            data.append(key)  # For case where grouby key is a single value: 'peptide' or 'proteins'

        for pair_key in pair_keys:
            data.extend(group_to_intensities[pair_key])

        for pair in pairs:
            data.extend(pair_to_ratio_data[pair])

        data.append(cnt)

        datas.append(data)

    return columns, datas


def assign_qvalues(group_ratios: List[GroupRatio]) -> None:
    """
    Assign q-values to the group ratios using the Benjamini-Hochberg method.
    """
    # Assuming group_ratios is your list of objects with p-values and you want to update them with q-values
    pvalues = [qr.log2_ratio_pvalue for qr in group_ratios]
    norm_pvalues = [qr.log2_norm_ratio_pvalue for qr in group_ratios]

    # Filter out NaN values and keep track of their original indices
    non_nan_indices = [i for i, pv in enumerate(pvalues) if not np.isnan(pv)]
    non_nan_pvalues = [pv for pv in pvalues if not np.isnan(pv)]

    non_nan_norm_indices = [i for i, pv in enumerate(norm_pvalues) if not np.isnan(pv)]
    non_nan_norm_pvalues = [pv for pv in norm_pvalues if not np.isnan(pv)]

    # Perform multipletests on non-NaN p-values
    qvalues = np.full(len(pvalues), np.nan)  # Initialize full array with NaNs
    norm_qvalues = np.full(len(norm_pvalues), np.nan)

    if non_nan_pvalues:
        qvalues_non_nan = multipletests(non_nan_pvalues, method='fdr_bh')[1]
        qvalues[non_nan_indices] = qvalues_non_nan  # Update only the non-NaN positions

    if non_nan_norm_pvalues:
        norm_qvalues_non_nan = multipletests(non_nan_norm_pvalues, method='fdr_bh')[1]
        norm_qvalues[non_nan_norm_indices] = norm_qvalues_non_nan

    # Update the original objects with the calculated q-values
    for i, qr in enumerate(group_ratios):
        qr.qvalue = qvalues[i]
        qr.norm_qvalue = norm_qvalues[i]


def assign_centered_log2_ratios(group_ratios: List[GroupRatio], center_type: str, inf_replacement: int) -> None:
    """
    Assign centered log2 ratios to the group ratios.
    """
    # Assuming group_ratios is your list of objects with p-values and you want to update them with q-values
    log2_ratios = [qr.log2_ratio for qr in group_ratios]
    log2_norm_ratios = [qr.log2_norm_ratio for qr in group_ratios]

    # Filter out NaN values and keep track of their original indices
    non_nan_indices = [i for i, lr in enumerate(log2_ratios) if not np.isnan(lr)]
    non_nan_log2_ratios = [lr for lr in log2_ratios if not np.isnan(lr)]

    # replace infinities
    non_nan_log2_ratios = [inf_replacement if np.isposinf(lr) else lr for lr in non_nan_log2_ratios]
    non_nan_log2_ratios = [-inf_replacement if np.isneginf(lr) else lr for lr in non_nan_log2_ratios]

    non_nan_norm_indices = [i for i, lr in enumerate(log2_norm_ratios) if not np.isnan(lr)]
    non_nan_log2_norm_ratios = [lr for lr in log2_norm_ratios if not np.isnan(lr)]

    # replace infinities
    non_nan_log2_norm_ratios = [inf_replacement if np.isposinf(lr) else lr for lr in non_nan_log2_norm_ratios]
    non_nan_log2_norm_ratios = [-inf_replacement if np.isneginf(lr) else lr for lr in non_nan_log2_norm_ratios]

    centered_log2_ratios = np.full(len(log2_ratios), np.nan)  # Initialize full array with NaNs
    centered_log2_norm_ratios = np.full(len(log2_norm_ratios), np.nan)

    def _center_log2_ratios(log2_ratios, center_type):

        if center_type == 'mean':
            return log2_ratios - np.mean(log2_ratios)
        elif center_type == 'median':
            return log2_ratios - np.median(log2_ratios)
        else:
            raise ValueError(f'Invalid center_type: {center_type}')

    if non_nan_log2_ratios:
        centered_log2_ratios_non_nan = _center_log2_ratios(non_nan_log2_ratios, center_type)
        centered_log2_ratios[non_nan_indices] = centered_log2_ratios_non_nan  # Update only the non-NaN positions

    if non_nan_log2_norm_ratios:
        centered_log2_norm_ratios_non_nan = _center_log2_ratios(non_nan_log2_norm_ratios, center_type)
        centered_log2_norm_ratios[non_nan_norm_indices] = centered_log2_norm_ratios_non_nan

    # Update the original objects with the calculated q-values
    for i, qr in enumerate(group_ratios):
        qr.centered_log2_ratio = centered_log2_ratios[i]
        qr.centered_norm_log2_ratio = centered_log2_norm_ratios[i]


def _group_quant_groups(quant_groups: List[QuantGroup], pairs: List[Tuple[Any, Any]], group_function: Callable,
                        ratio_function: Callable) -> List[GroupRatio]:
    """
    Group the quant groups by the group_function and calculate the ratios for each pair of groups.
    """
    quant_groups = sorted(quant_groups, key=group_function)

    group_ratios = []
    for key, grouped_quant_groups in tqdm.tqdm(groupby(quant_groups, group_function), desc='Generating Group Ratios'):

        grouped_quant_groups = list(grouped_quant_groups)
        group_to_quant_group = {}
        for quant_group in grouped_quant_groups:
            group_to_quant_group.setdefault(quant_group.group, []).append(quant_group)

        for g1, g2 in pairs:
            group1 = group_to_quant_group[g1]
            group2 = group_to_quant_group[g2]

            group_ratio = GroupRatio(group1, group2, ratio_function)
            group_ratios.append(group_ratio)

            # calculate pvalue and norm_pvalue (cached property)
            _ = group_ratio.log2_ratio_pvalue
            _ = group_ratio.log2_norm_ratio_pvalue

    return group_ratios


def group_by_psms(quant_groups: List[QuantGroup], pairs: List[Tuple[Any, Any]], groupby_filename: bool,
                  ratio_function: Callable) -> List[
    GroupRatio]:
    # sort quant_groups by peptide, charge, filename
    if groupby_filename:
        return _group_quant_groups(quant_groups, pairs, lambda qg: (qg.psm.peptide, qg.psm.charge, qg.psm.filename),
                                   ratio_function)
    return _group_quant_groups(quant_groups, pairs, lambda qg: (qg.psm.peptide, qg.psm.charge), ratio_function)


def group_by_peptides(quant_groups: List[QuantGroup], pairs: List[Tuple[Any, Any]], groupby_filename: bool,
                      ratio_function: Callable) -> \
        List[GroupRatio]:
    # sort quant_groups by peptide, charge, filename
    if groupby_filename:
        return _group_quant_groups(quant_groups, pairs, lambda qg: (qg.psm.peptide, qg.psm.filename), ratio_function)
    return _group_quant_groups(quant_groups, pairs, lambda qg: qg.psm.peptide, ratio_function)


def group_by_proteins(quant_groups: List[QuantGroup], pairs: List[Tuple[Any, Any]], groupby_filename: bool,
                      ratio_function: Callable) -> \
        List[GroupRatio]:
    # sort quant_groups by protein, filename
    if groupby_filename:
        return _group_quant_groups(quant_groups, pairs, lambda qg: (qg.psm.proteins, qg.psm.filename), ratio_function)
    return _group_quant_groups(quant_groups, pairs, lambda qg: qg.psm.proteins, ratio_function)


def filter_quant_groups(quant_groups: List[QuantGroup], filter_type: str) -> List[QuantGroup]:
    """
    Filter the quant groups based on the filter type. Unique will only keep quant groups with unique peptides. All will
    keep all quant groups.
    """
    if filter_type == 'unique':
        quant_groups = [qg for qg in quant_groups if ';' not in qg.psm.proteins]
    elif filter_type == 'all':
        pass
    else:
        raise ValueError(f"Invalid filter type: {filter_type}")

    return quant_groups


def parse_args():
    parser = argparse.ArgumentParser(
        description='Quant Compare: Calcualtes ratios of quant groups and performs statistical tests on the ratios.')
    parser.add_argument('input_file',
                        help='Input parquet file. Must contain the following columns: "reporter_ion_intensity", '
                             '"filename", "peptide", "charge", "proteins", and "scannr"')
    parser.add_argument('output_folder',
                        help='Output folder for writing output files to. This folder will be created if it does not '
                             'exist. File names can be specified with the --psm_file and --peptide_file and '
                             '--protein_file arguments.')
    parser.add_argument('--psm_file',
                        help='The file name for the psms ratios file, will be inside the output_folder dir.',
                        default='psm_ratios')
    parser.add_argument('--peptide_file',
                        help='The file name for the peptide ratios file, will be inside the output_folder dir.',
                        default='peptide_ratios')
    parser.add_argument('--protein_file',
                        help='The file name for the protein ratios file, will be inside the output_folder dir.',
                        default='protein_ratios')

    def parse_pairs(arg):
        try:
            return [tuple(map(int, pair.split(','))) for pair in arg.split(';')]
        except ValueError:
            raise argparse.ArgumentTypeError("Pairs must be in format '1,2;3,4;...'")

    parser.add_argument('--pairs',
                        help='Pairs of groups to compare. Pairs must be in the following format: '
                             '"Group1,Group2;...;Group1:Group3". Each pair must onyl contain 2 values (separated by a '
                             'comma (",")), and these values must match those in the groups argument. Multiple pairs '
                             'must be separated by as semicolon (";").',
                        type=parse_pairs)

    def parse_group(arg):
        try:
            groups = {}
            for group in arg.split(';'):
                key, indices = group.split(':')
                groups[int(key)] = list(map(int, indices.split(',')))
            return groups
        except ValueError:
            raise argparse.ArgumentTypeError("Groups must be in format '1:0,1,2;2:3,4,5;...'")

    parser.add_argument('--groups',
                        help='The group labels mapped to the indecies for their reporter ion channels. '
                             'Groups must be specified in the following format: '
                             '"GroupName1:Index1,...,Index3;GroupName2:Index1,...,Index3;". Group names must be '
                             'unique and can be of any type. Indexes must be separated by a comma (",") and multiple '
                             'groups must be separated by a semicolon (";").',
                        type=parse_group)
    parser.add_argument('--filter', choices=['unique', 'all'], default='all',
                        help='Filter type for peptides. Unique will only keep peptides which map to a single protein. '
                             'All will keep all peptides.')
    parser.add_argument('--groupby_filename', action='store_true',
                        help='Groupby the filename for psm, peptide and proteins. This will add a filename column '
                             'to the output files.')
    parser.add_argument('--output_type', choices=['csv', 'parquet'], default='parquet', help='Output file type.')
    parser.add_argument('--inf_replacement', default=100,
                        help='Infinite values cause many problem with the statistics. This value will be used to '
                             'replace infinite values at the log2 ratio level. Default is 100. (-inf will be replaced '
                             'with -100 and inf will be replaced with 100)')
    parser.add_argument('--no_psms', action='store_true', help='Dont output a PSM ratio file.')
    parser.add_argument('--no_peptides', action='store_true', help='Dont output a Peptide ratio file.')
    parser.add_argument('--no_proteins', action='store_true', help='Dont output a Protein ratio file.')
    parser.add_argument('--center', choices=['mean', 'median'], default='median',
                        help='Center the Log2 Ratios around the mean/median.')
    parser.add_argument('--max_rows', default=-1, type=int,
                        help='(DEBUG OPTION) Maximum number of rows to read from the input file. Default is -1 '
                             '(read all rows).')
    parser.add_argument('--ratio_function', choices=['mean_ratio_rollup', 'reference_mean_ratio_rollup'],
                        default='mean_ratio_rollup')
    parser.add_argument('--keep_decoy', action='store_true',
                        help='Keep decoy proteins in the analysis. Default is to remove decoys.')
    parser.add_argument('--keep_psm', default=1, type=int,
                        help='Keep only the top N PSMs per scan number per file. Default is 1. Set to -1 to keep all '
                             'PSMs.')
    parser.add_argument('--qvalue_threshold', default=0.01, type=float, help='Q-value threshold for significance.')
    parser.add_argument('--qvalue_level', choices=['peptide', 'protein', 'spectrum', 'all'], default='all',
                        help='Q-value level for significance. Default is peptide.')
    return parser.parse_args()


def run():
    # TODO: Possibly fix nan value and inf value replacement

    args = parse_args()

    # Convert args namespace to a dictionary
    args_dict = vars(args)

    print("=" * 30)
    print("Quant Ratio Calculator")
    print("=" * 30)
    print()

    print('Arguments:')
    pprint.pprint(args_dict)
    print()

    os.makedirs(args.output_folder, exist_ok=True)

    print(f'Reading Input File...')
    sage_df = pd.read_parquet(args.input_file, engine='pyarrow')
    if args.max_rows > 0:
        print(f'{"Reading only the first":<20} {args.max_rows} rows')
        sage_df = sage_df.head(args.max_rows)

    # Filter input data
    starting_rows = len(sage_df)
    print(f'{"Total PSMs:":<30} {starting_rows}')

    # Remove decoys "is_decoy" column
    if not args.keep_decoy:
        sage_df = sage_df[~sage_df['is_decoy']].reset_index()

    print(f'{"Filtered PSMs (Decoy):":<30} {starting_rows - len(sage_df)}')
    starting_rows = len(sage_df)

    if args.qvalue_level == 'peptide':
        sage_df = sage_df[(sage_df.peptide_q <= args.qvalue_threshold)].reset_index()
    elif args.qvalue_level == 'protein':
        sage_df = sage_df[(sage_df.protein_q <= args.qvalue_threshold)].reset_index()
    elif args.qvalue_level == 'spectrum':
        sage_df = sage_df[(sage_df.spectrum_q <= args.qvalue_threshold)].reset_index()
    elif args.qvalue_level == 'all':
        sage_df = sage_df[(sage_df.spectrum_q <= args.qvalue_threshold) &
                          (sage_df.peptide_q <= args.qvalue_threshold) &
                          (sage_df.protein_q <= args.qvalue_threshold)].reset_index()
    else:
        raise ValueError(f'Invalid qvalue level: {args.qvalue_level}')

    print(f'{"Filtered PSMs (Qvalue):":<30} {starting_rows - len(sage_df)}')
    starting_rows = len(sage_df)

    # drop duplicates (keep args.keep_psm top PSMs per scan number per file)
    if args.keep_psm > 0:
        sage_df.sort_values(['scannr', 'filename', 'rank'], ascending=[True, True, True], inplace=True)
        sage_df = sage_df.groupby(['scannr', 'filename']).head(args.keep_psm)
        sage_df.reset_index(drop=True, inplace=True)

    print(f'{"Filtered PSMs (Chimeric):":<30} {starting_rows - len(sage_df)}')
    print(f'{"Remaining PSMs:":<30} {len(sage_df)}')
    print()

    add_norm_intensities_to_df(sage_df)
    quant_groups = make_quant_groups(sage_df, args.groups)
    print('Creating Quant Groups...')
    print(f"{'Total Quant Groups:':<30} {len(quant_groups)}")
    print()

    if args.ratio_function == 'mean_ratio_rollup':
        ratio_function = partial(mean_ratio_rollup, inf_replacement=100)
    elif args.ratio_function == 'reference_mean_ratio_rollup':
        ratio_function = partial(reference_mean_ratio_rollup, inf_replacement=100)
    else:
        raise ValueError(f'Invalid ratio function: {args.ratio_function}')

    if not args.no_psms:
        print('Grouping PSMs...')
        psm_quant_ratios = group_by_psms(quant_groups, args.pairs, args.groupby_filename, ratio_function)
        assign_qvalues(psm_quant_ratios)
        assign_centered_log2_ratios(psm_quant_ratios, args.center, args.inf_replacement)
        print(f"{'PSM Quant Groups:':<20} {len(psm_quant_ratios)}")
        time.sleep(0.1)

        cols, data = get_psm_ratio_data_wide(psm_quant_ratios, args.pairs, args.groupby_filename)
        psm_df = pd.DataFrame(data, columns=cols)

        psm_file = os.path.join(args.output_folder, args.psm_file + f'.{args.output_type}')
        if args.output_type == 'csv':
            psm_df.to_csv(os.path.join(args.output_folder, psm_file), index=False)
        elif args.output_type == 'parquet':
            psm_df.to_parquet(os.path.join(args.output_folder, psm_file))
        else:
            raise ValueError(f'Invalid output type: {args.output_type}')
        print(f'PSM Ratios written to {psm_file}')
        print()

        del psm_quant_ratios
        del psm_df
        del data

    if not args.no_peptides:
        print('Grouping Peptides...')
        peptide_quant_ratios = group_by_peptides(quant_groups, args.pairs, args.groupby_filename, ratio_function)
        assign_qvalues(peptide_quant_ratios)
        assign_centered_log2_ratios(peptide_quant_ratios, args.center, args.inf_replacement)
        print(f"{'Peptide Quant Groups:':<20} {len(peptide_quant_ratios)}")
        time.sleep(0.1)

        cols, data = get_peptide_ratio_data_wide(peptide_quant_ratios, args.pairs, args.groupby_filename)
        peptide_df = pd.DataFrame(data, columns=cols)

        peptide_file = os.path.join(args.output_folder, args.peptide_file + f'.{args.output_type}')
        if args.output_type == 'csv':
            peptide_df.to_csv(os.path.join(args.output_folder, peptide_file), index=False)
        elif args.output_type == 'parquet':
            peptide_df.to_parquet(os.path.join(args.output_folder, peptide_file))
        else:
            raise ValueError(f'Invalid output type: {args.output_type}')
        print(f'Peptide Ratios written to {peptide_file}')
        print()

        del peptide_quant_ratios
        del peptide_df
        del data

    if not args.no_proteins:
        print('Grouping Proteins...')
        quant_groups = filter_quant_groups(quant_groups, args.filter)
        protein_quant_ratios = group_by_proteins(quant_groups, args.pairs, args.groupby_filename, ratio_function)
        assign_qvalues(protein_quant_ratios)
        assign_centered_log2_ratios(protein_quant_ratios, args.center, args.inf_replacement)
        print(f"{'Protein Quant Groups:':<20} {len(protein_quant_ratios)}")
        time.sleep(0.1)

        cols, data = get_protein_ratio_data_wide(protein_quant_ratios, args.pairs, args.groupby_filename)
        protein_df = pd.DataFrame(data, columns=cols)

        protein_file = os.path.join(args.output_folder, args.protein_file + f'.{args.output_type}')
        if args.output_type == 'csv':
            protein_df.to_csv(os.path.join(args.output_folder, protein_file), index=False)
        elif args.output_type == 'parquet':
            protein_df.to_parquet(os.path.join(args.output_folder, protein_file))
        else:
            raise ValueError(f'Invalid output type: {args.output_type}')
        print(f'Protein Ratios written to {protein_file}')
        print()

        del protein_quant_ratios
        del protein_df
        del data

    print('Done!')


if __name__ == '__main__':
    run()
