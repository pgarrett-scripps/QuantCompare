from typing import List

import numpy as np
from scipy.stats import t

from quantcompare.dclasses import Group


def remove_unused_channels(filename: str, intensities: np.ndarray, groups: List[Group]) -> np.ndarray:
    # Find used channel indexes for this file
    used_channel_indexes = []
    for group in groups:
        if group.filename == filename:
            used_channel_indexes.append(group.channel_index)
    used_channel_indexes = sorted(list(set(used_channel_indexes)))

    return intensities[:, used_channel_indexes]


def calculate_p_value_against_zero(mean: np.ndarray, std_dev: np.ndarray, n: np.ndarray) -> float:
    """
    Calculate the p-value for a one-sample t-test comparing the sample mean to 0.
    """

    mu_0 = 0

    # Calculate the t-statistic
    t_statistic = (mean - mu_0) / (std_dev / np.sqrt(n))

    # Calculate the p-value
    p_value = t.sf(np.abs(t_statistic), df=n - 1) * 2  # two-tailed test

    return p_value
