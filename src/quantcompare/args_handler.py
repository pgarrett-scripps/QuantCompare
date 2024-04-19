import argparse
from typing import List, Tuple

from quantcompare.dclasses import Group, Pair


def parse_group(arg) -> List[Group]:
    """
    Parse the group argument into a list of Group objects.
    """
    try:
        groups = []
        for group in arg.split(';'):
            group_elems = group.rstrip(')').lstrip('(').split(',')
            if len(group_elems) == 4:
                group, filename, n, scale = group_elems
                groups.append(Group(group, filename, int(n), float(scale)))
            else:
                raise ValueError

        return groups
    except ValueError:
        raise argparse.ArgumentTypeError("Groups must be in format '(GROUP,FILENAME,CHANNEL_INDEX);(...);'")


def parse_pairs(arg) -> List[Pair]:
    """
    Parse the pairs argument into a list of tuples.
    """
    try:
        return [Pair(*map(str, pair.split(','))) for pair in arg.split(';')]
    except ValueError:
        raise argparse.ArgumentTypeError("Pairs must be in format '1,2;3,4;...'")


def parse_groupby_columns(arg) -> List[str]:
    """
    Parse the groupby columns argument into a list of strings.
    """
    try:
        return arg.split(';')
    except ValueError:
        raise argparse.ArgumentTypeError("Groupby columns must be in format 'column1;column2;...'")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Quant Compare: Calcualtes ratios of quant groups and performs statistical tests on the ratios.')
    parser.add_argument('--config',
                        help='The configuration file for the analysis. Arguments in the configuration file will override '
                                'arguments passed in the command line.')

    parser.add_argument('--input',
                        help='Input parquet file. Must contain the following columns: "reporter_ion_intensity", '
                             '"filename", "peptide", "charge", "proteins", and "scannr"')
    parser.add_argument('--output', default='.',
                        help='Output folder for writing output files to. This folder will be created if it does not '
                             'exist. File names can be specified with the --psm_file and --peptide_file and '
                             '--protein_file arguments.')
    parser.add_argument('--pairs',
                        help='Pairs of groups to compare. Pairs must be in the following format: '
                             '"Group1,Group2;...;Group1:Group3". Each pair must onyl contain 2 values (separated by a '
                             'comma (",")), and these values must match those in the groups argument. Multiple pairs '
                             'must be separated by as semicolon (";").',
                        type=parse_pairs)
    parser.add_argument('--groups',
                        help='The group labels mapped to the indices for their reporter ion channels. Groups must be in '
                             'the following format: "GROUP,FILENAME,CHANNEL_INDEX,SCALE;...". Each group must contain '
                             '4 values separated by a comma (","), and these values must be an string, string, integer, '
                             'and float respectively. Multiple groups must be separated by a semicolon (";").',
                        type=parse_group)

    normalization_options = parser.add_argument_group('Normalization Options')
    normalization_options.add_argument('--filter', choices=['unique', 'all', 'unique_protein_group'], default='all',
                                       help='Filter type for peptides. Unique will only keep peptides which map to a single protein. '
                                            'All will keep all peptides.')
    normalization_options.add_argument('--center', choices=['mean', 'median', 'none'], default='median',
                                       help='Center the Log2 Ratios around the mean/median.')
    normalization_options.add_argument('--ratio_function', choices=['mean_ratio_rollup', 'reference_mean_ratio_rollup'],
                                       default='mean_ratio_rollup')
    normalization_options.add_argument('--inf_replacement', default=100,
                                       help='Infinite values cause many problem with the statistics. This value will be used to '
                                            'replace infinite values at the log2 ratio level. Default is 100. (-inf will be replaced '
                                            'with -100 and inf will be replaced with 100)')
    normalization_options.add_argument('--intra_file_normalization', choices=['mean', 'median', 'none'],
                                       default='median',
                                       help='Normalization method for each file. Default is median.')
    normalization_options.add_argument('--inter_file_normalization', choices=['mean', 'median', 'none'],
                                       default='median',
                                       help='Normalization method between different files (Happens after intra-file normalization). Default is median.')
    normalization_options.add_argument('--row_normalization', action='store_true', help='Ensure each row sums to 1.')

    input_filter_options = parser.add_argument_group('Input Filter Options')
    input_filter_options.add_argument('--max_rows', default=-1, type=int,
                                      help='(DEBUG OPTION) Maximum number of rows to read from the input file. Default is -1 '
                                           '(read all rows).')
    input_filter_options.add_argument('--keep_decoy', action='store_true',
                                      help='Keep decoy proteins in the analysis. Default is to remove decoys.')
    input_filter_options.add_argument('--keep_contaminant', action='store_true',
                                      help='Keep contaminant proteins in the analysis.')
    input_filter_options.add_argument('--keep_psm', default=1, type=int,
                                      help='Keep only the top N PSMs per scan number per file. Default is 1. Set to -1 to keep all '
                                           'PSMs.')
    input_filter_options.add_argument('--qvalue_threshold', default=0.01, type=float,
                                      help='Q-value threshold for significance.')
    input_filter_options.add_argument('--qvalue_level', choices=['peptide', 'protein', 'spectrum', 'all', 'none'],
                                      default='all',
                                      help='Q-value level for significance. Default is peptide.')
    input_filter_options.add_argument('--keep_unused_channels', action='store_true',
                                      help='Keep channels that are not used in any '
                                           'group.')

    output_options = parser.add_argument_group('Output Options')
    output_options.add_argument('--no_psms', action='store_true', help='Dont output a PSM ratio file.')
    output_options.add_argument('--no_peptides', action='store_true', help='Dont output a Peptide ratio file.')
    output_options.add_argument('--no_proteins', action='store_true', help='Dont output a Protein ratio file.')
    output_options.add_argument('--output_type', choices=['csv', 'parquet'], default='parquet',
                                help='Output file type.')
    output_options.add_argument('--df_format', choices=['wide', 'long'], default='wide',
                                help='Output format for the dataframes.')
    output_options.add_argument('--psm_file',
                        help='The file name for the psms ratios file, will be inside the output_folder dir.',
                        default='psm_ratios')
    output_options.add_argument('--peptide_file',
                        help='The file name for the peptide ratios file, will be inside the output_folder dir.',
                        default='peptide_ratios')
    output_options.add_argument('--protein_file',
                        help='The file name for the protein ratios file, will be inside the output_folder dir.',
                        default='protein_ratios')

    # Groupby section
    grouping_options = parser.add_argument_group('Grouping Options')
    grouping_options.add_argument('--psm_groupby', default='peptide;charge;filename', help='Group by columns for PSMs.',
                                  type=parse_groupby_columns)
    grouping_options.add_argument('--peptide_groupby', default='peptide;filename',
                                  help='Group by columns for Peptides.', type=parse_groupby_columns)
    grouping_options.add_argument('--protein_groupby', default='proteins;filename',
                                  help='Group by columns for Proteins.', type=parse_groupby_columns)

    imputing_options = parser.add_argument_group('Imputing Options')
    imputing_options.add_argument('--impute_method', choices=['none', 'mean', 'median', 'min', 'max', 'constant', 'iterative', 'knn'], default='none',
                                  help='Impute missing values in the reporter ion intensities. Default is none.')
    imputing_options.add_argument('--impute_constant', default=0.0, type=float, help='Constant value to impute missing values with. (Only applies to constant imputation)')
    imputing_options.add_argument('--impute_n_neighbors', default=5, type=int, help='Number of neighbors to use in KNN imputation. (Only applies to KNN imputation)')
    imputing_options.add_argument('--impute_axis', choices=['all', 'row', 'col'], default='all', help='Axis to impute along. (Only applies to mean/median/min/max imputation)')
    imputing_options.add_argument('--missing_value', default=0.0, type=float, help='Value to consider as missing. Default is 0.0.')

    return parser.parse_args()
