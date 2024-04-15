from typing import Protocol, Optional, Callable, Any, TypeAlias, Literal

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

    def impute(self, arr: np.ndarray, inplace: bool = False, missing_value: Any = 0.0) -> Optional[np.ndarray]:
        pass


class SimpleImpute(Impute):
    def __init__(self, func: Callable, axis: Optional[int]):
        self.func = func
        self.axis = axis

    def impute(self, arr: np.ndarray, inplace: bool = False, missing_value: Any = 0.0) -> Optional[np.ndarray]:
        """
        Impute missing values in the array.

        . code-block:: python

            # Test axis is None
            >>> arr = np.array([[1.0, 2.0, 0.0], [4.0, 0.0, 6.0]])
            >>> imputer = SimpleImpute(np.nanmean, axis=None)
            >>> imputer.impute(arr, inplace=True)
            >>> arr
            array([[1.  , 2.  , 3.25],
                   [4.  , 3.25, 6.  ]])

            # Test axis is 0
            >>> arr = np.array([[1.0, 2.0, 0.0], [4.0, 0.0, 6.0]])
            >>> imputer = SimpleImpute(np.nanmean, axis=0) # impute along columns
            >>> imputer.impute(arr, inplace=True)
            >>> arr
            array([[1., 2., 6.],
                   [4., 2., 6.]])

            # Test axis is 1
            >>> arr = np.array([[1.0, 2.0, 0.0], [4.0, 0.0, 6.0]])
            >>> imputer = SimpleImpute(np.nanmean, axis=1) # impute along rows
            >>> imputer.impute(arr, inplace=True)
            >>> arr
            array([[1. , 2. , 1.5],
                   [4. , 5. , 6. ]])

            # Test inplace
            >>> arr = np.array([[1.0, 2.0, 0.0], [4.0, 0.0, 6.0]])
            >>> imputer = SimpleImpute(np.nanmean, axis=1) # impute along rows
            >>> imputer.impute(arr, inplace=False)
            array([[1. , 2. , 1.5],
                   [4. , 5. , 6. ]])
            >>> arr
            array([[1., 2., 0.],
                   [4., 0., 6.]])

            # test missing value
            >>> imputer.impute(arr, inplace=False, missing_value=np.nan)
            array([[1., 2., 0.],
                   [4., 0., 6.]])

        """
        if not inplace:
            arr = arr.copy()

        all_zero_rows = np.all(arr == missing_value, axis=1)
        arr[all_zero_rows, :] = np.nan

        if self.axis is None:
            mean_value = self.func(arr[arr != missing_value])
            arr[arr == missing_value] = mean_value
        elif self.axis == 0:

            for i in range(arr.shape[1]):
                mean_value = self.func(arr[arr[:, i] != missing_value, i])
                arr[arr[:, i] == missing_value, i] = mean_value

        elif self.axis == 1:
            for i in range(arr.shape[0]):
                mean_value = self.func(arr[i, arr[i] != missing_value])
                arr[i, arr[i] == missing_value] = mean_value

        else:
            raise ValueError("Axis must be 0, 1, or None")

        if not inplace:
            return arr


class NoneImpute(Impute):

    def impute(self, arr: np.ndarray, inplace: bool = False, missing_value: Any = 0.0) -> Optional[np.ndarray]:
        if inplace:
            return None
        else:
            return arr


class MeanImpute(SimpleImpute):
    def __init__(self, axis: Optional[int]):
        super().__init__(np.nanmean, axis)


class MedianImpute(SimpleImpute):

    def __init__(self, axis: Optional[int]):
        super().__init__(np.nanmedian, axis)


class MinImpute(SimpleImpute):
    def __init__(self, axis: Optional[int]):
        super().__init__(np.nanmin, axis)


class MaxImpute(SimpleImpute):
    def __init__(self, axis: Optional[int]):
        super().__init__(np.nanmax, axis)


class ConstantImpute(SimpleImpute):
    def __init__(self, axis: Optional[int], const: Any):
        const_func = lambda _: const
        super().__init__(const_func, axis)

    def impute(self, arr: np.ndarray, inplace: bool = False, missing_value: Any = 0.0) -> Optional[np.ndarray]:
        """
        . code-block:: python

            >>> arr = np.array([[1.0, 2.0, 0.0], [4.0, 0.0, 6.0]])
            >>> imputer = ConstantImpute(axis=None, const=10)
            >>> imputer.impute(arr, inplace=True)
            >>> arr
            array([[ 1.,  2., 10.],
                   [ 4., 10.,  6.]])

        """
        return super().impute(arr, inplace, missing_value)


class IterativeImpute(Impute):
    def impute(self, arr: np.ndarray, inplace: bool = False, missing_value: Any = 0.0) -> Optional[np.ndarray]:
        """
        . code-block:: python

            >>> arr = np.array([[1.0, 2.0, 0.0], [4.0, 0.0, 6.0]])
            >>> imputer = IterativeImpute()
            >>> imputer.impute(arr, inplace=True)
            >>> arr
            array([[1., 2., 6.],
                   [4., 2., 6.]])

        """

        imputer = IterativeImputer(random_state=0, missing_values=missing_value, keep_empty_features=True)
        imputed_arr = imputer.fit_transform(arr)

        if inplace:
            # find indices of 0 values
            zero_indices = np.where(arr == missing_value)

            # set 0 values to imputed values
            arr[zero_indices] = imputed_arr[zero_indices]
        else:
            return imputed_arr


class KnnImpute(Impute):
    def __init__(self, n_neighbors: int):
        self.n_neighbors = n_neighbors

    def impute(self, arr: np.ndarray, inplace: bool = False, missing_value: Any = 0.0) -> Optional[np.ndarray]:
        """
        . code-block:: python

            >>> arr = np.array([[1.0, 2.0, 0.0], [4.0, 0.0, 6.0]])
            >>> imputer = KnnImpute(2)
            >>> imputer.impute(arr, inplace=True)
            >>> arr
            array([[1., 2., 6.],
                   [4., 2., 6.]])

        """

        imputer = KNNImputer(n_neighbors=self.n_neighbors, keep_empty_features=True, copy=not inplace,
                             missing_values=missing_value)
        imputed_arr = imputer.fit_transform(arr)

        if not inplace:
            return imputed_arr


ImputeMethod: TypeAlias = Literal['none', 'mean', 'constant', 'median', 'min', 'max', 'iterative', 'knn']
AxisType: TypeAlias = Literal['all', 'row', 'col']


def get_imputer(strategy: ImputeMethod, constant_value: Any, axis: AxisType, n_neighbors: int) -> Impute:

    if axis == 'row':
        axis = 1
    elif axis == 'col':
        axis = 0
    elif axis == 'all':
        axis = None
    else:
        raise ValueError("Invalid axis value")

    if strategy == 'none':
        return NoneImpute()
    elif strategy == 'mean':
        return MeanImpute(axis=axis)
    elif strategy == 'constant':
        return ConstantImpute(axis=axis, const=constant_value)
    elif strategy == 'median':
        return MedianImpute(axis=axis)
    elif strategy == 'min':
        return MinImpute(axis=axis)
    elif strategy == 'max':
        return MaxImpute(axis=axis)
    elif strategy == 'iterative':
        return IterativeImpute()
    elif strategy == 'knn':
        return KnnImpute(n_neighbors=n_neighbors)
    else:
        raise ValueError(f"Invalid impute strategy: {strategy}")
