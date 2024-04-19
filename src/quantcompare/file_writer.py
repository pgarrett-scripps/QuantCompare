from itertools import groupby
from typing import List, Tuple, Any, Callable, Iterable

import numpy as np
import tqdm as tqdm

from quantcompare.dclasses import GroupRatio


def get_ratio_data_wide(quant_ratios: List[GroupRatio], pairs: List[Tuple[Any, Any]], groupby_cols: List[str],
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
                #f'log2_ratios_{pair[0]}_{pair[1]}',
                #f'log2_ratio_stds_{pair[0]}_{pair[1]}',
                #f'norm_log2_ratios_{pair[0]}_{pair[1]}',
                #f'norm_log2_ratio_stds_{pair[0]}_{pair[1]}',
                f'total_intensity_{pair[0]}_{pair[1]}',
                f'total_norm_intensity_{pair[0]}_{pair[1]}',
            ]
        )

    columns.append('cnt')
    columns.append('log2_ratio_contrast_matrix')
    columns.append('norm_log2_ratio_contrast_matrix')
    #columns.append('log2_ratio_covariance_matrix')
    #columns.append('norm_log2_ratio_covariance_matrix')

    datas = []
    quant_ratios.sort(key=groupby_func)
    for key, group in tqdm.tqdm(groupby(quant_ratios, groupby_func), desc='Generating Wide Data'):
        data = []
        group_to_intensities = {}
        pair_to_ratio_data = {}
        cnt = 0
        for quant_ratio in group:
            cnt += 1
            pair_to_ratio_data[quant_ratio.pair] = \
                (
                    quant_ratio.ratio,  #0
                    quant_ratio.centered_ratio,  #1
                    quant_ratio.log2_ratio,  #2
                    quant_ratio.centered_log2_ratio,  #3
                    quant_ratio.log2_ratio_std,  #4
                    quant_ratio.log2_ratio_pvalue,  #5
                    quant_ratio.qvalue,  #6
                    quant_ratio.norm_ratio,  #7
                    quant_ratio.centered_norm_ratio,  #8
                    quant_ratio.log2_norm_ratio,  #9
                    quant_ratio.centered_norm_log2_ratio,  #10
                    quant_ratio.log2_norm_ratio_std,  #11
                    quant_ratio.log2_norm_ratio_pvalue,  #12
                    quant_ratio.norm_qvalue,  #13
                    #quant_ratio.log2_ratios, #14
                    #quant_ratio.log2_ratio_stds, #15
                    #quant_ratio.log2_norm_ratios, #16
                    #quant_ratio.log2_norm_ratio_stds, #17
                    quant_ratio.total_intensity,  #18
                    quant_ratio.total_norm_intensity,  #19
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

        if isinstance(key, (List, Tuple)):
            for val in key:
                data.append(val)
        else:
            data.append(key)  # For case where grouby key is a single value: 'peptide' or 'proteins'

        for pair_key in pair_keys:
            data.extend(group_to_intensities[pair_key])

        for pair in pairs:
            data.extend(pair_to_ratio_data[pair])

        data.append(cnt)

        log2_contrast_matrix, norm_log2_contrast_matrix = [], []
        for pair in pairs:
            log2_contrast_matrix.append(pair_to_ratio_data[pair][2])
            norm_log2_contrast_matrix.append(pair_to_ratio_data[pair][9])

        log2_contrast_matrix = np.array(log2_contrast_matrix)
        norm_log2_contrast_matrix = np.array(norm_log2_contrast_matrix)

        # must sum to 0
        log2_contrast_matrix = log2_contrast_matrix - np.mean(log2_contrast_matrix)
        norm_log2_contrast_matrix = norm_log2_contrast_matrix - np.mean(norm_log2_contrast_matrix)

        data.append(log2_contrast_matrix)
        data.append(norm_log2_contrast_matrix)

        """
        design_matrix, norm_design_matrix = [], []
        design_matrix_weights, norm_design_matrix_weights = [], []
        for pair, ratio_data in pair_to_ratio_data.items():
            design_matrix.append(ratio_data[14])
            norm_design_matrix.append(ratio_data[16])
            design_matrix_weights.append(ratio_data[18])
            norm_design_matrix_weights.append(ratio_data[19])
        """

        #design_matrix = np.array(design_matrix, dtype=np.float32)
        #norm_design_matrix = np.array(norm_design_matrix, dtype=np.float32)
        #design_matrix_weights = np.array(design_matrix_weights, dtype=np.float32)
        #norm_design_matrix_weights = np.array(norm_design_matrix_weights, dtype=np.float32)

        #covariance_matrix = np.cov(design_matrix, dtype=np.float32) #, aweights=design_matrix_weights)
        #norm_covariance_matrix = np.cov(norm_design_matrix, dtype=np.float32) #, aweights=norm_design_matrix_weights)

        #data.append(design_matrix.tolist())
        #data.append(norm_design_matrix.tolist())
        #data.append(covariance_matrix.tolist())
        #data.append(norm_covariance_matrix.tolist())

        datas.append(data)

    return columns, datas
