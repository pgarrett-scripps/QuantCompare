from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, List, Tuple, Callable, Optional

import numpy as np


@dataclass(frozen=True)
class Group:
    group: Any
    filename: str
    channel_index: int
    scale: float

    def __str__(self):
        return f'Group({self.group}, {self.filename}, {self.channel_index}, {self.scale})'


@dataclass(frozen=True)
class Pair:
    group1: Any
    group2: Any

    # index method
    def __getitem__(self, item: int) -> Any:
        if item == 0:
            return self.group1
        if item == 1:
            return self.group2
        raise IndexError('Pair index out of range')

    def __str__(self):
        return f'Pair({self.group1}, {self.group2})'


@dataclass(frozen=False)
class Psm:
    labels: list
    values: list
    _attributes: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        if len(self.labels) != len(self.values):
            raise ValueError("Labels and values must have the same length.")

        for label, value in zip(self.labels, self.values):
            setattr(self, label, value)
            self._attributes[label] = value

    def __repr__(self):
        attrs = ', '.join(f"{k}={v!r}" for k, v in self._attributes.items())
        return f"{self.__class__.__name__}({attrs})"


@dataclass(frozen=True)
class PsmOld:
    peptide: str
    charge: int
    filename: str
    proteins: List[str]
    scannr: int
    site: int
    bio_rep: Any
    tech_rep: Any
    decoy: bool
    intensities: np.array(np.float32)
    norm_intensities: np.array(np.float32)

    def __str__(self):
        return (f'Psm({self.peptide}, {self.charge}, {self.filename}, {";".join(self.proteins)}, {self.scannr}, '
                f'{list(self.intensities)}, {list(self.norm_intensities)})')


@dataclass(frozen=False)
class QuantGroup:
    group: Any
    group_indices: List[int]
    psm: Psm

    def __str__(self):
        return f'QuantGroup({self.group}, {self.group_indices}, {self.psm})'


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

    @property
    def pair(self) -> Pair:
        return Pair(self.group1, self.group2)

    def __str__(self):
        return f'GroupRatio(Pair=({self.group1}, {self.group2}), log2_ratio={self.log2_ratio}, log2_norm_ratio={self.log2_norm_ratio})'

    @cached_property
    def _log2_ratio(self) -> Tuple[float, float, float]:
        return self.ratio_function(self.group1_intensity_arr, self.group2_intensity_arr)

    @cached_property
    def _log2_norm_ratio(self) -> Tuple[float, float, float]:
        return self.ratio_function(self.group1_norm_intensity_arr, self.group2_norm_intensity_arr)

    @cached_property
    def _log2_ratios(self) -> Tuple[float, float, float]:
        return self.ratio_function(self.group1_intensity_arr, self.group2_intensity_arr, rtype='array')

    @cached_property
    def _log2_norm_ratios(self) -> Tuple[float, float, float]:
        return self.ratio_function(self.group1_norm_intensity_arr, self.group2_norm_intensity_arr, rtype='array')

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
    def log2_ratios(self) -> float:
        return self._log2_ratios[0]

    @property
    def log2_ratio_stds(self) -> float:
        return self._log2_ratios[1]

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
    def log2_norm_ratios(self) -> float:
        return self._log2_norm_ratios[0]

    @property
    def log2_norm_ratio_stds(self) -> float:
        return self._log2_norm_ratios[1]

    @property
    def log2_norm_ratio_pvalue(self) -> float:
        return self._log2_norm_ratio[2]

    @cached_property
    def group1_intensity_arr(self) -> np.ndarray:
        group1_array = np.array([qg.psm.reporter_ion_intensity for qg in self.QuantGroup1], dtype=np.float32)
        return group1_array

    @cached_property
    def group2_intensity_arr(self) -> np.ndarray:
        group2_array = np.array([qg.psm.reporter_ion_intensity for qg in self.QuantGroup2], dtype=np.float32)
        return group2_array

    @cached_property
    def group1_norm_intensity_arr(self) -> np.ndarray:
        group1_array = np.array([qg.psm.normalized_reporter_ion_intensity for qg in self.QuantGroup1], dtype=np.float32)
        return group1_array

    @cached_property
    def group2_norm_intensity_arr(self) -> np.ndarray:
        group2_array = np.array([qg.psm.normalized_reporter_ion_intensity for qg in self.QuantGroup2], dtype=np.float32)
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
    def total_intensity(self) -> np.ndarray:
        return np.sum(self.group1_intensity_arr) + np.sum(self.group2_intensity_arr)

    @property
    def total_norm_intensity(self) -> np.ndarray:
        return np.sum(self.group1_norm_intensity_arr) + np.sum(self.group2_norm_intensity_arr)

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
