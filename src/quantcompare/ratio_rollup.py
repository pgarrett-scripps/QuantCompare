from typing import Tuple

import numpy as np

from quantcompare.uncertainty_estimation import divide_uncertainty, log2_transformed_uncertainty, mean_uncertainty
from quantcompare.util import calculate_p_value_against_zero


def reference_mean_ratio_rollup(arr1: np.array, arr2: np.array, inf_replacement: float = 100, rtype: str = 'stats') -> \
Tuple[
    float, float, float]:
    """
    Calculate the ratio of the mean of each row of two arrays.

    .. code-block:: python

        >>> arr1 = np.array([[100, 110, 105], [200, 210, 205]])
        >>> arr2 = np.array([[200, 230, 240, 200], [400, 430, 440, 410]])
        >>> reference_mean_ratio_rollup(arr1, arr2)
        (-1.0397599262262949, 0.03151145467192459, 1.1882097621215213e-36)

        >>> arr1 = np.array([[0, 0, 0], [200, 210, 205], [0, 0, 0]])
        >>> arr2 = np.array([[200, 230, 240, 200], [0, 0, 0, 0], [0, 0, 0, 0]])
        >>> reference_mean_ratio_rollup(arr1, arr2)
        (0.0, 5.773502691896258, 1.0)

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

    if rtype == 'array':
        return log2_ratios.flatten(), log2_ratio_uncertainties.flatten(), np.nan

    # Calculate the mean of the log2 ratios and uncertainties
    log2_ratio = np.mean(log2_ratios)
    log2_std = mean_uncertainty(*log2_ratio_uncertainties)[0]

    n = len(log2_ratios) * min(arr1.shape[1], arr2.shape[1])
    pvalue = calculate_p_value_against_zero(log2_ratio, log2_std, n)

    return float(log2_ratio), float(log2_std), float(pvalue)


def mean_ratio_rollup(arr1: np.array, arr2: np.array, inf_replacement: float = 100, rtype: str = 'stats') -> Tuple[
    float, float, float]:
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
        (0.0, 5.773502691896258, 1.0)

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

    if rtype == 'array':
        return log2_ratios.flatten(), log2_ratio_uncertainties.flatten(), np.nan

    # Calculate the mean of the log2 ratios and uncertainties
    log2_ratio = np.mean(log2_ratios)
    log2_std = mean_uncertainty(*log2_ratio_uncertainties)[0]

    n = len(log2_ratios) * min(arr1.shape[1], arr2.shape[1])
    pvalue = calculate_p_value_against_zero(log2_ratio, log2_std, n)

    return float(log2_ratio), float(log2_std), float(pvalue)


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