import numpy as np


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
