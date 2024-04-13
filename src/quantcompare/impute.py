from typing import Protocol, Optional, TypeAlias, Literal, Callable

import numpy as np
from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import KNNImputer, IterativeImputer

"""
--input ../tmt_result/results.sage.parquet --output ../output/ --pairs 1,2;1,3 --groups (1,16p_A_1.mzparquet,0,1);(1,16p_A_1.mzparquet,1,1);(1,16p_A_1.mzparquet,2,1);(2,16p_A_1.mzparquet,3,1);(2,16p_A_1.mzparquet,4,1);(2,16p_A_1.mzparquet,5,1);(3,16p_A_1.mzparquet,6,1);(3,16p_A_1.mzparquet,7,1);(3,16p_A_1.mzparquet,8,1) --max_rows 1000 --output_type csv
"""


class Impute(Protocol):
    """
    Protocol for imputation classes.
    """

    def impute(self, arr: np.ndarray, inplace: bool) -> Optional[np.ndarray]:
        pass


class SimpleImpute(Impute):
    def __init__(self, func: Callable, axis: Optional[int]):
        self.func = func
        self.axis = axis

    def impute(self, arr: np.ndarray, inplace: bool) -> Optional[np.ndarray]:

        if not inplace:
            arr = arr.copy()

        all_zero_rows = np.all(arr == 0, axis=1)
        arr[all_zero_rows, :] = np.nan

        if self.axis is None:
            mean_value = self.func(arr[arr != 0])
            arr[arr == 0] = mean_value
        elif self.axis == 0:

            for i in range(arr.shape[1]):
                mean_value = self.func(arr[arr[:, i] != 0, i])
                arr[arr[:, i] == 0, i] = mean_value

        elif self.axis == 1:
            for i in range(arr.shape[0]):
                mean_value = self.func(arr[i, arr[i] != 0])
                arr[i, arr[i] == 0] = mean_value

        else:
            raise ValueError("Axis must be 0 or 1")

        if not inplace:
            return arr


class MeanImpute(SimpleImpute):
    def __init__(self, axis: Optional[int]):
        super().__init__(np.nanmean, axis)

    def impute(self, arr: np.ndarray, inplace: bool) -> Optional[np.ndarray]:
        """
        Impute missing values in the array.

        . code-block:: python

            >>> arr = np.array([[1.0, 2.0, 0.0], [4.0, 0.0, 6.0]])
            >>> imputer = MeanImpute(axis=None)
            >>> imputer.impute(arr, inplace=True)
            >>> arr
            array([[1.  , 2.  , 3.25],
                   [4.  , 3.25, 6.  ]])

            >>> arr = np.array([[1.0, 2.0, 0.0], [4.0, 0.0, 6.0]])
            >>> imputer = MeanImpute(axis=0) # impute along columns
            >>> imputer.impute(arr, inplace=True)
            >>> arr
            array([[1., 2., 6.],
                   [4., 2., 6.]])

            >>> arr = np.array([[1.0, 2.0, 0.0], [4.0, 0.0, 6.0]])
            >>> imputer = MeanImpute(axis=1) # impute along rows
            >>> imputer.impute(arr, inplace=True)
            >>> arr
            array([[1. , 2. , 1.5],
                   [4. , 5. , 6. ]])

        """
        return super().impute(arr, inplace)


class MedianImpute(SimpleImpute):

    def __init__(self, axis: Optional[int]):
        super().__init__(np.nanmedian, axis)

    def impute(self, arr: np.ndarray, inplace: bool) -> Optional[np.ndarray]:
        return super().impute(arr, inplace)


class MinImpute(SimpleImpute):
    def __init__(self, axis: Optional[int]):
        super().__init__(np.nanmin, axis)

    def impute(self, arr: np.ndarray, inplace: bool) -> Optional[np.ndarray]:
        return super().impute(arr, inplace)


class MaxImpute(SimpleImpute):
    def __init__(self, axis: Optional[int]):
        super().__init__(np.nanmax, axis)

    def impute(self, arr: np.ndarray, inplace: bool) -> Optional[np.ndarray]:
        return super().impute(arr, inplace)


class ConstantImpute(SimpleImpute):
    def __init__(self, axis: Optional[int], const: float):
        const_func = lambda _: const
        super().__init__(const_func, axis)

    def impute(self, arr: np.ndarray, inplace: bool) -> Optional[np.ndarray]:
        """
        . code-block:: python

            >>> arr = np.array([[1.0, 2.0, 0.0], [4.0, 0.0, 6.0]])
            >>> imputer = ConstantImpute(axis=None, const=10)
            >>> imputer.impute(arr, inplace=True)
            >>> arr
            array([[ 1.,  2., 10.],
                   [ 4., 10.,  6.]])

        """
        return super().impute(arr, inplace)


class IterativeImpute(Impute):
    def impute(self, arr: np.ndarray, inplace: bool) -> Optional[np.ndarray]:
        """
        . code-block:: python

            >>> arr = np.array([[1.0, 2.0, 0.0], [4.0, 0.0, 6.0]])
            >>> imputer = IterativeImpute()
            >>> imputer.impute(arr, inplace=True)
            >>> arr
            array([[1., 2., 6.],
                   [4., 2., 6.]])

        """

        imputer = IterativeImputer(random_state=0, missing_values=0, keep_empty_features=True)
        imputed_arr = imputer.fit_transform(arr)

        if inplace:
            # find indices of 0 values
            zero_indices = np.where(arr == 0)

            # set 0 values to imputed values
            arr[zero_indices] = imputed_arr[zero_indices]
        else:
            return imputed_arr


class KnnImpute(Impute):
    def __init__(self, n_neighbors: int):
        self.n_neighbors = n_neighbors

    def impute(self, arr: np.ndarray, inplace: bool) -> Optional[np.ndarray]:
        """
        . code-block:: python

            >>> arr = np.array([[1.0, 2.0, 0.0], [4.0, 0.0, 6.0]])
            >>> imputer = KnnImpute(2)
            >>> imputer.impute(arr, inplace=True)
            >>> arr
            array([[1., 2., 6.],
                   [4., 2., 6.]])

        """

        imputer = KNNImputer(n_neighbors=self.n_neighbors, keep_empty_features=True, copy=not inplace, missing_values=0)
        imputed_arr = imputer.fit_transform(arr)

        if not inplace:
            return imputed_arr

