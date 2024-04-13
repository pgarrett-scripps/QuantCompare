import copy
import os
import time
from functools import partial
from itertools import groupby
from typing import Any, Dict, List, Tuple, Callable, Set, TypeAlias, Literal
import numpy as np
import pprint
import pandas as pd
import tqdm as tqdm
from statsmodels.stats.multitest import multipletests

import warnings

from quantcompare.args_handler import parse_args
from quantcompare.dclasses import Group, GroupRatio, QuantGroup, Psm
from quantcompare.file_writer import get_ratio_data_wide
from quantcompare.ratio_rollup import mean_ratio_rollup, reference_mean_ratio_rollup

# Suppress only the specific RuntimeWarning related to precision loss in scipy
warnings.filterwarnings('ignore', category=RuntimeWarning,
                        message='Precision loss occurred in moment calculation due to catastrophic cancellation.')
np.seterr(divide='ignore', invalid='ignore')


def make_quant_groups(df: pd.DataFrame, groups: List[Group]) -> List[QuantGroup]:
    """
    Create a list of QuantGroup objects from a DataFrame and a groups dictionary.

    1) Map the groups to their file and channel indices.
    2) Create Psm objects for each row in the DataFrame.
    3) For each created Psm, create a QuantGroups for the groups associated with the Psm.
    4) Return the list of QuantGroups.

    :param df: The DataFrame with the reporter ion intensities.
    :param groups: The list of Group objects, representing channel information.
    :return: The list of QuantGroup objects.

    . code-block:: python

        >>> import pandas as pd
        >>> import numpy as np
        >>> data = {}
        >>> data['peptide'] = ['PEP', 'PEP', 'TID', 'IDE']
        >>> data['charge'] = [2, 3, 2, 2]
        >>> data['filename'] = ['file1', 'file1', 'file1', 'file1']
        >>> data['proteins'] = [['P1'], ['P1'], ['P2'], ['P3']]
        >>> data['scannr'] = [1, 2, 3, 4]
        >>> data['reporter_ion_intensity'] = [[1, 2], [3, 4], [5, 6], [7, 8]]
        >>> data['normalized_reporter_ion_intensity'] = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]
        >>> df = pd.DataFrame(data)

        >>> groups = [Group('1', 'file1', 0, 1), Group('2', 'file1', 1, 1)]
        >>> quant_groups = make_quant_groups(df, groups)
        >>> len(quant_groups) # 4 PSMs and 2 groups = 8 QuantGroups
        8
        >>> str(quant_groups[0]) # Group 1
        'QuantGroup(1, [0], Psm(PEP, 2, file1, P1, 1, [1.0, 2.0], [0.1, 0.2]))'

    """
    # TODO: There may be issues if channels are ever shared between files ( Group 1 has 2 channels in file A and B)?

    groups_dict = map_groups_to_indices(groups)

    quant_groups = []
    # Create Psm objects for each row in the group
    psms = [Psm(row['peptide'], row['charge'], row['filename'], row['proteins'], row['scannr'],
                np.array(row['reporter_ion_intensity'], dtype=np.float32),
                np.array(row['normalized_reporter_ion_intensity'], dtype=np.float32))
            for i, row in df.iterrows()]

    # Create QuantGroup objects for each Psm and associated groups
    for psm in psms:
        for group, file_dict in groups_dict.items():
            if psm.filename not in file_dict:
                continue
            quant_groups.append(QuantGroup(group, file_dict[psm.filename], psm))

    return quant_groups


def _assign_qvalues(pvalues: np.ndarray) -> np.ndarray:
    # TODO: Check if this is the correct way to handle QValues
    """
    . code-block:: python

        >>> pvalues = np.array([0.1, 0.2, 0.3, np.nan, 0.4, 0.5])
        >>> _assign_qvalues(pvalues)
        array([0.5, 0.5, 0.5, nan, 0.5, 0.5])

        >>> pvalues = np.array([0.01, 0.5, 0.6, np.nan, 0.3, 0.1])
        >>> _assign_qvalues(pvalues)
        array([0.05, 0.6 , 0.6 ,  nan, 0.5 , 0.25])

    """
    # Filter out NaN values and keep track of their original indices
    non_nan_indices = np.array([i for i, pv in enumerate(pvalues) if not np.isnan(pv)])
    non_nan_pvalues = np.array([pv for pv in pvalues if not np.isnan(pv)])

    # Perform multipletests on non-NaN p-values
    qvalues = np.full(len(pvalues), np.nan)  # Initialize full array with NaNs

    if len(non_nan_pvalues) > 0:
        qvalues_non_nan = multipletests(non_nan_pvalues, method='fdr_bh')[1]
        qvalues[non_nan_indices] = qvalues_non_nan  # Update only the non-NaN positions

    return qvalues


def assign_qvalues(group_ratios: List[GroupRatio]) -> None:
    """
    Assign q-values to the group ratios using the Benjamini-Hochberg method. Updates the qvalue attribute of the
    GroupRatio objects.

    :param group_ratios: The list of GroupRatio objects.
    :return: None
    """
    # Assuming group_ratios is your list of objects with p-values and you want to update them with q-values
    pvalues = np.array([qr.log2_ratio_pvalue for qr in group_ratios])
    norm_pvalues = np.array([qr.log2_norm_ratio_pvalue for qr in group_ratios])

    qvalues = _assign_qvalues(pvalues)
    norm_qvalues = _assign_qvalues(norm_pvalues)

    # Update the original objects with the calculated q-values
    for i, qr in enumerate(group_ratios):
        qr.qvalue = qvalues[i]
        qr.norm_qvalue = norm_qvalues[i]


def _assign_centered_log2_ratios(ratios: np.ndarray, center_type: str) -> np.ndarray:
    """
    . code-block:: python

        >>> ratios = np.array([1, 1, 2, np.nan, 3, 3])
        >>> center_type = 'mean'

        >>> _assign_centered_log2_ratios(ratios, center_type) # Mean is 2
        array([-1., -1.,  0., nan,  1.,  1.])

        >>> center_type = 'median'
        >>> _assign_centered_log2_ratios(ratios, center_type) # Median is 2
        array([-1., -1.,  0., nan,  1.,  1.])

    """

    # Filter out NaN values and keep track of their original indices
    non_nan_indices = np.array([i for i, lr in enumerate(ratios) if not np.isnan(lr)])
    non_nan_ratios = np.array([lr for lr in ratios if not np.isnan(lr)])
    centered_ratios = np.full(len(ratios), np.nan)  # Initialize full array with NaNs

    def _center_log2_ratios(rs: np.ndarray, ct: str) -> np.ndarray:

        if ct == 'mean':
            return rs - np.mean(rs)
        elif ct == 'median':
            return rs - np.median(rs)
        else:
            raise ValueError(f'Invalid center_type: {ct}')

    if len(non_nan_ratios) > 0:
        centered_log2_ratios_non_nan = _center_log2_ratios(non_nan_ratios, center_type)
        centered_ratios[non_nan_indices] = centered_log2_ratios_non_nan  # Update only the non-NaN positions

    return centered_ratios


def assign_centered_log2_ratios(group_ratios: List[GroupRatio], center_type: str) -> None:
    """
    Assign centered log2 ratios to the group ratios. Updates the centered_log2_ratio attribute of the GroupRatio
    objects.

    :param group_ratios: The list of GroupRatio objects.
    :param center_type: The method to center the log2 ratios. One of 'mean' or 'median'.
    :return: None
    """
    # Assuming group_ratios is your list of objects with p-values and you want to update them with q-values
    log2_ratios = np.array([qr.log2_ratio for qr in group_ratios])
    log2_norm_ratios = np.array([qr.log2_norm_ratio for qr in group_ratios])

    centered_log2_ratios = _assign_centered_log2_ratios(log2_ratios, center_type)
    centered_log2_norm_ratios = _assign_centered_log2_ratios(log2_norm_ratios, center_type)

    # Update the original objects with the calculated q-values
    for i, qr in enumerate(group_ratios):
        qr.centered_log2_ratio = centered_log2_ratios[i]
        qr.centered_norm_log2_ratio = centered_log2_norm_ratios[i]


def group_quant_groups(quant_groups: List[QuantGroup], pairs: List[Tuple[Any, Any]], group_function: Callable,
                       ratio_function: Callable) -> List[GroupRatio]:
    """
    Group the quant groups by the group_function and calculate the ratios for each pair of groups.

    . code-block:: python

        >>> psm1 = Psm('PEP', 2, 'file1', ['P1'], 1, np.array([1.0, 4.0]), np.array([0.1, 0.2]))
        >>> qg1 = QuantGroup('1', [0], psm1)
        >>> qg2 = QuantGroup('2', [1], psm1)
        >>> pairs = [('1', '2')]
        >>> group_function = lambda x: x.psm.peptide
        >>> ratio_function = mean_ratio_rollup

        >>> group_ratios = group_quant_groups([qg1, qg2], pairs, group_function, ratio_function)
        >>> len(group_ratios)
        1
        >>> str(group_ratios[0])
        'GroupRatio(Pair=(1, 2), log2_ratio=-2.0, log2_norm_ratio=-1.0)'

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


def map_groups_to_indices(groups: List[Group]) -> Dict[Any, Dict[str, List[int]]]:
    groups_dict: Dict[Any, Dict[str, List[int]]] = {}
    for group in groups:
        if group.group not in groups_dict:
            groups_dict[group.group] = {}
        if group.filename not in groups_dict[group.group]:
            groups_dict[group.group][group.filename] = []
        groups_dict[group.group][group.filename].append(group.channel_index)

    for group, file_dict in groups_dict.items():
        for filename, indices in file_dict.items():
            groups_dict[group][filename] = sorted(list(set(indices)))

    return groups_dict


QvalueLevel: TypeAlias = Literal["peptide", "protein", "spectrum", "all", "none"]
IntensityNormalizationMethod: TypeAlias = Literal["median", "mean", "none"]
RatioGroupingLevel: TypeAlias = Literal["psm", "peptide", "protein"]
CenteringMethod: TypeAlias = Literal["mean", "median", "none"]
OutputType: TypeAlias = Literal["csv", "parquet"]
DataframeFormat: TypeAlias = Literal["wide", "long"]
ProteinFilterMethod: TypeAlias = Literal["unique", "all", "unique_protein_group"]


def parse_sage_results(df: pd.DataFrame, max_rows: int, keep_decoy: bool, keep_contaminant: bool,
                       qvalue_level: QvalueLevel, qvalue_threshold: float, keep_psm: int) -> pd.DataFrame:
    """
    Filter the input data based on the input arguments.

    1) Read the input file.
    2) Take first max_rows rows, if applicable.
    3) Remove decoy PSMs, if applicable.
    4) Filter based on q-value level and threshold.
    5) Keep only keep_psm PSMs per scan number per file, if applicable.
    6) Return the filtered DataFrame.

    :param df: The DataFrame with the input data.
    :param max_rows: The maximum number of rows to read from the input file.
    :param keep_decoy: Whether to keep decoy PSMs.
    :param qvalue_level: The q-value level to filter on. One of 'peptide', 'protein', 'spectrum', 'all' or 'none'.
    :param qvalue_threshold: The q-value threshold for significance, any PSM with a q-value below this threshold at the
                             specified level(s) will be kept.
    :param keep_psm: The number of PSMs to keep per scan number per file.
    :return: The DataFrame with the normalized input data.

    . code-block:: python

        >>> import pandas as pd
        >>> data = {}
        >>> data['peptide'] = ['PEP', 'PEP', 'TID', 'IDE']
        >>> data['charge'] = [2, 3, 2, 2]
        >>> data['filename'] = ['file1', 'file1', 'file1', 'file1']
        >>> data['proteins'] = ['P1', 'P1;P2', 'P2', 'P3']
        >>> data['scannr'] = [1, 1, 3, 4]  # First 2 are Chimeric
        >>> data['is_decoy'] = [False, False, True, False] # Third is Decoy
        >>> data['peptide_q'] = [0.1, 0.1, 0.3, 0.4] # First 2 pass filter
        >>> data['protein_q'] = [0.1, 0.1, 0.4, 0.5] # First 2 pass filter
        >>> data['spectrum_q'] = [0.1, 0.1, 0.5, 0.6] # First 2 pass filter
        >>> data['rank'] = [1, 2, 1, 1]
        >>> data['reporter_ion_intensity'] = [[1, 2], [3, 4], [5, 6], [7, 8]]
        >>> df = pd.DataFrame(data)

        >>> df = parse_sage_results(df, 3, False, True, 'all', 0.2, 1)
        Reading only the first 3 rows
        Total PSMs:                    3
        Filtered PSMs (Decoy):         1
        Filtered PSMs (Qvalue):        0
        Filtered PSMs (Chimeric):      1
        Remaining PSMs:                1
        <BLANKLINE>

        >>> df[['peptide', 'rank']]
          peptide  rank
        0     PEP     1

    """

    if max_rows > 0:
        print(f'{"Reading only the first":<20} {max_rows} rows')
        df = df.head(max_rows)

    # Filter input data
    starting_rows = len(df)
    print(f'{"Total PSMs:":<30} {starting_rows}')

    # Remove decoys specified by "is_decoy" column
    if not keep_decoy:
        df = df[~df['is_decoy']].reset_index(drop=True)

    if not keep_contaminant:
        contaminant_indices = df['proteins'].str.lower().str.contains('contaminant')
        df = df[~contaminant_indices].reset_index(drop=True)

    print(f'{"Filtered PSMs (Decoy):":<30} {starting_rows - len(df)}')
    starting_rows = len(df)

    # Filter based on q-value at the specified level
    if qvalue_level == 'peptide':
        df = df[(df.peptide_q <= qvalue_threshold)].reset_index(drop=True)
    elif qvalue_level == 'protein':
        df = df[(df.protein_q <= qvalue_threshold)].reset_index(drop=True)
    elif qvalue_level == 'spectrum':
        df = df[(df.spectrum_q <= qvalue_threshold)].reset_index(drop=True)
    elif qvalue_level == 'all':
        df = df[(df.spectrum_q <= qvalue_threshold) &
                (df.peptide_q <= qvalue_threshold) &
                (df.protein_q <= qvalue_threshold)].reset_index(drop=True)
    elif qvalue_level == 'none':
        pass
    else:
        raise ValueError(f'Invalid qvalue level: {qvalue_level}')

    print(f'{"Filtered PSMs (Qvalue):":<30} {starting_rows - len(df)}')
    starting_rows = len(df)

    # drop duplicates (keep args.keep_psm top PSMs per scan number per file)
    if keep_psm > 0:
        df.sort_values(['scannr', 'filename', 'rank'], ascending=[True, True, True], inplace=True)
        df = df.groupby(['scannr', 'filename']).head(keep_psm).reset_index(drop=True)

    print(f'{"Filtered PSMs (Chimeric):":<30} {starting_rows - len(df)}')
    print(f'{"Remaining PSMs:":<30} {len(df)}')
    print()

    # for rows which have all 0 intensities, replace with an array of NaNs
    df['reporter_ion_intensity'] = df['reporter_ion_intensity'].apply(lambda x: [np.nan]*len(x) if np.all(x == 0) else x)

    # count NA
    na_count = df['reporter_ion_intensity'].apply(lambda x: np.sum(np.isnan(x))).sum()
    print(f'{"Total NA values:":<30} {na_count}')


    return df


def normalize_df(df: pd.DataFrame, groups: List[Group], keep_unused_channels: bool,
                 intra_file_normalization: IntensityNormalizationMethod,
                 inter_file_normalization: IntensityNormalizationMethod, row_normalization: bool) -> pd.DataFrame:
    """
    Normalize the DataFrame based on the groups and normalization methods.

    1) Split the DataFrame based on the filename.
    2) Remove unused channels, if applicable (Applies to Intensities and Normalized Intensities).
    3) Normalize for known channel concentration differences (Applies to Normalized Intensities).
    4) Normalize each channel based on the intra_file_normalization method (Applies to Normalized Intensities).
    5) Normalize each row to sum to 1, if applicable (Applies to Normalized Intensities).
    6) Recombine the DataFrames.
    7) Normalize each channel based on the inter_file_normalization method (Applies to Normalized Intensities).

    :param df: The DataFrame with the reporter ion intensities.
    :param groups: The list of Group objects, representing channel information.
    :param keep_unused_channels: Whether to keep channels that are not used in any group.
    :param intra_file_normalization: The normalization method to use within each file.
    :param inter_file_normalization: The normalization method to use across files.
    :param row_normalization: Whether to normalize each row to sum to 1.
    :return: The DataFrame with the normalized reporter ion intensities.

    . code-block:: python

        >>> import pandas as pd
        >>> data = {}
        >>> data['peptide'] = ['PEP', 'PEP', 'TID', 'IDE']
        >>> data['charge'] = [2, 3, 2, 2]
        >>> data['filename'] = ['file1', 'file1', 'file1', 'file1']
        >>> data['proteins'] = ['P1', 'P1;P2', 'P2', 'P3']
        >>> data['scannr'] = [1, 1, 3, 4]
        >>> data['reporter_ion_intensity'] = [[1, 2, 0], [3, 4, 0], [5, 6, 0], [7, 8, 0]]

        >>> groups = [Group('1', 'file1', 0, 1), Group('2', 'file1', 1, 1)]
        >>> df = normalize_df(pd.DataFrame(data), groups, False, 'median', 'median', False)
        >>> np.array(df['normalized_reporter_ion_intensity'].tolist()).shape # One column removed
        (4, 2)
        >>> df = normalize_df(pd.DataFrame(data), groups, True, 'median', 'median', False)
        >>> np.array(df['normalized_reporter_ion_intensity'].tolist()).shape # Column kept
        (4, 3)
        >>> df = normalize_df(pd.DataFrame(data), groups, True, 'none', 'none', True)
        >>> np.sum(np.array(df['normalized_reporter_ion_intensity'].tolist()), axis=1) # floating point error
        array([1.00000003, 1.00000003, 1.00000003, 1.00000003])
        >>> df = normalize_df(pd.DataFrame(data), groups, False, 'median', 'median', False)
        >>> np.array(df['normalized_reporter_ion_intensity'].tolist()) # sol_sums = [16, 20, 0] sum_avg = 18
        array([[1.12499988, 1.79999971],
               [3.37499976, 3.59999943],
               [5.625     , 5.39999962],
               [7.87499952, 7.19999886]])

    """
    # split sage df based on filename
    sage_dfs = {}
    for filename in df['filename'].unique():

        # Find used channel indexes for this file
        used_channel_indexes = []
        for group in groups:
            if group.filename == filename:
                used_channel_indexes.append(group.channel_index)
        used_channel_indexes = sorted(list(set(used_channel_indexes)))

        file_df = df[df['filename'] == filename].reset_index(drop=True)
        intensities = np.vstack(file_df['reporter_ion_intensity'].values).astype(np.float32)

        # Filter unused channels
        if not keep_unused_channels:
            intensities = intensities[:, used_channel_indexes]
            file_df['intensities'] = intensities.tolist()

        # Copy the intensities for normalization
        norm_intensities = copy.deepcopy(intensities)

        # get the channel concentrations for this file
        channel_concentrations = np.ones(intensities.shape[1])
        for group in groups:
            if group.filename == filename:
                channel_concentrations[group.channel_index] = group.scale

        norm_intensities /= channel_concentrations

        sums = np.sum(norm_intensities, axis=0, keepdims=True)
        if intra_file_normalization == 'median':
            norm_intensities /= np.median(sums) * sums
        elif intra_file_normalization == 'mean':
            norm_intensities /= np.mean(sums) * sums
        elif intra_file_normalization == 'none':
            pass
        else:
            raise ValueError(f'Invalid global normalization method: {intra_file_normalization}')

        # Normalize each row to sum to 1
        if row_normalization:
            row_sums = np.sum(norm_intensities, axis=1, keepdims=True)
            norm_intensities /= row_sums

        file_df['normalized_reporter_ion_intensity'] = norm_intensities.tolist()

        sage_dfs[filename] = file_df

    # Combine the dataframes
    df = pd.concat(sage_dfs.values(), ignore_index=True)

    # inter_file_normalization
    norm_intensities = np.vstack(df['normalized_reporter_ion_intensity'].values).astype(np.float32)

    sums = np.sum(norm_intensities, axis=0, keepdims=True)
    if inter_file_normalization == 'median':
        norm_intensities /= np.median(sums) * sums
    elif inter_file_normalization == 'mean':
        norm_intensities /= np.mean(sums) * sums
    elif inter_file_normalization == 'none':
        pass
    else:
        raise ValueError(f'Invalid global normalization method: {intra_file_normalization}')

    df['normalized_reporter_ion_intensity'] = norm_intensities.tolist()

    return df


def write_to_file(df: pd.DataFrame, file_path: str, output_type: OutputType) -> None:
    """
    Write the DataFrame to a file based on the output type.

    :param df: The DataFrame to write.
    :param file_path: The path to write the file.
    :param output_type: The output type. One of 'csv' or 'parquet'.
    """
    if output_type == 'csv':
        df.to_csv(file_path, index=False)
    elif output_type == 'parquet':
        df.to_parquet(file_path)
    else:
        raise ValueError(f'Invalid output type: {output_type}')


def build_ratios_df(quant_groups: List[QuantGroup], pairs: List[Tuple[Any, Any]], ratio_function: Callable,
                    groupby_attributes: List[str], center_method: CenteringMethod,
                    df_format: DataframeFormat) -> pd.DataFrame:
    """
    Build a DataFrame with the ratios for the given groupby attributes.

    1) Create QuantRatios by grouping the QuantGroups and calculating the ratios for each pair of groups.
    2) Assign q-values to the QuantRatios.
    3) Assign centered log2 ratios to the QuantRatios.
    4) Format the DataFrame based on the df_format.
    5) Return the DataFrame.

    :param quant_groups: The list of QuantGroup objects.
    :param pairs: The pairs of groups to compare.
    :param ratio_function: The function to calculate the ratios.
    :param groupby_attributes: The attributes to group by.
    :param center_method: The method to center the ratios.
    :param df_format: The format of the DataFrame.
    :return: The DataFrame with the ratios.
    """

    # QuantGroups, GroupRatios, and PSMs all have peptide, protein, charge, filename, and scannr attributes
    grouping_func = lambda g: [g.__getattribute__(group) for group in groupby_attributes]

    quant_ratios = group_quant_groups(quant_groups=quant_groups,
                                      pairs=pairs,
                                      group_function=grouping_func,
                                      ratio_function=ratio_function)
    assign_qvalues(quant_ratios)
    assign_centered_log2_ratios(quant_ratios, center_method)
    print(f"{'Quant Groups:':<20} {len(quant_ratios)}")
    time.sleep(0.1)

    # Format the DataFrame
    if df_format == 'wide':
        cols, data = get_ratio_data_wide(quant_ratios=quant_ratios,
                                         pairs=pairs,
                                         groupby_cols=groupby_attributes,
                                         groupby_func=grouping_func)
    elif df_format == 'long':
        raise NotImplementedError('Long format not implemented yet.')
    else:
        raise ValueError(f'Invalid DataFrame format: {df_format}')

    return pd.DataFrame(data, columns=cols)


def filter_quant_groups(quant_groups: List[QuantGroup], filter_type: ProteinFilterMethod) -> List[QuantGroup]:
    """
    Filter the quant groups based on the filter type. Unique will only keep quant groups with unique peptides. All will
    keep all quant groups.
    """
    if filter_type == 'unique':
        quant_groups = [qg for qg in quant_groups if ';' not in qg.psm.proteins]
    elif filter_type == 'all':
        pass
    elif filter_type == 'unique_protein_group':
        raise NotImplementedError('Unique Protein Group not implemented yet.')
    else:
        raise ValueError(f"Invalid filter type: {filter_type}")

    return quant_groups



def run():
    # TODO: Possibly fix nan value and inf value replacement

    args = parse_args()
    args_dict = vars(args)

    print("=" * 30)
    print("Quant Ratio Calculator")
    print("=" * 30)
    print()

    print('Arguments:')
    pprint.pprint(args_dict)
    print()

    # Create output folder if it does not exist
    os.makedirs(args.output, exist_ok=True)

    print(f'Reading Input File...')
    sage_df = pd.read_parquet(args.input, engine='pyarrow')

    sage_df = parse_sage_results(df=sage_df,
                                 max_rows=args.max_rows,
                                 keep_decoy=args.keep_decoy,
                                 keep_contaminant=args.keep_contaminant,
                                 qvalue_level=args.qvalue_level,
                                 qvalue_threshold=args.qvalue_threshold,
                                 keep_psm=args.keep_psm)

    sage_df = normalize_df(df=sage_df,
                           groups=args.groups,
                           keep_unused_channels=args.keep_unused_channels,
                           intra_file_normalization=args.intra_file_normalization,
                           inter_file_normalization=args.inter_file_normalization,
                           row_normalization=args.row_normalization)

    quant_groups = make_quant_groups(sage_df, args.groups)
    print('Creating Quant Groups...')
    print(f"{'Total Quant Groups:':<30} {len(quant_groups)}")
    print()

    # TODO: Add a similarity metric to check if normalization worked (probably fine to check mean/median/std of channel ints)
    # TODO: Center in normalization?

    # Create ratio function
    if args.ratio_function == 'mean_ratio_rollup':
        ratio_function = partial(mean_ratio_rollup, inf_replacement=args.inf_replacement)
    elif args.ratio_function == 'reference_mean_ratio_rollup':
        ratio_function = partial(reference_mean_ratio_rollup, inf_replacement=args.inf_replacement)
    else:
        raise ValueError(f'Invalid ratio function: {args.ratio_function}')

    if not args.no_psms:
        print('Grouping PSMs...')
        df = build_ratios_df(quant_groups=quant_groups,
                             pairs=args.pairs,
                             ratio_function=ratio_function,
                             groupby_attributes=args.psm_groupby,
                             center_method=args.center,
                             df_format=args.df_format)

        file_path = str(os.path.join(args.output, args.psm_file + f'.{args.output_type}'))
        write_to_file(df, file_path, args.output_type)
        print(f'PSM Ratios written to {file_path}')
        print()

        del df

    if not args.no_peptides:
        print('Grouping Peptides...')
        df = build_ratios_df(quant_groups=quant_groups,
                             pairs=args.pairs,
                             ratio_function=ratio_function,
                             groupby_attributes=args.peptide_groupby,
                             center_method=args.center,
                             df_format=args.df_format)

        file_path = str(os.path.join(args.output, args.peptide_file + f'.{args.output_type}'))
        write_to_file(df, file_path, args.output_type)
        print(f'Peptide Ratios written to {file_path}')
        print()

        del df

    if not args.no_proteins:
        print('Grouping Proteins...')
        # filter out proteins with less than 2 peptides

        df = build_ratios_df(quant_groups=quant_groups,
                             pairs=args.pairs,
                             ratio_function=ratio_function,
                             groupby_attributes=args.protein_groupby,
                             center_method=args.center,
                             df_format=args.df_format)

        file_path = str(os.path.join(args.output, args.protein_file + f'.{args.output_type}'))
        write_to_file(df, file_path, args.output_type)
        print(f'Protein Ratios written to {file_path}')
        print()

        del df

    print('Done!')


if __name__ == '__main__':
    run()

    """
    compare tmt_result/results.sage.parquet tmt_result/ --pairs '1,2;1,3' --groups '(1,16p_A_1.mzparquet,0,1);(1,16p_A_1.mzparquet,1,1);(1,16p_A_1.mzparquet,2,1);(2,16p_A_1.mzparquet,3,1);(2,16p_A_1.mzparquet,4,1);(2,16p_A_1.mzparquet,5,1);(2,16p_A_1.mzparquet,6,1);(2,16p_A_1.mzparquet,7,1);(2,16p_A_1.mzparquet,8,1)'
    """
