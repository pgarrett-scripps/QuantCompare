# QuantCompare

This script is designed for calculating quantitative ratios between groups of TMT reporter ion intensities.

## Install

#### From Pip

    pip install quantcompare

## How to Run
The main script is located at quantcompare.main:run, but the 'compare' entry point is provided by the package.

    compare results.sage.parquet output --pairs 1,2;1,3;1,4 --groups 1:0,1,2;2:3,4,5;3:6,7,8;4:9,10
    

## Input File (results.sage.parquet)

### Required Columns
|Columns| Description |
|--|--|
| reporter_ion_intensity | Intensities of TMT channels |
| filename | The PSMs file (can be anything) |
| peptide | The modified peptide sequence |
| charge | The peptides charge state |
| proteins | The protein names/ids (separated by ';') |
| scannr | The peptides scan number |


## Arguments

| Posisitional Arguments         | Description                                                                                                                                                                         |
|------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| input_file       | Input parquet file. Must contain the following columns: "reporter_ion_intensity", "filename", "peptide", "charge", "proteins", and "scannr".                                    |
| output_folder    | Output folder for writing output files to. This folder will be created if it does not exist. File names can be specified with the --psm_file, --peptide_file, and --protein_file arguments. |

| Additional Arguments | Description                                                                                                                                                                                                                       |
|--------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| -h, --help               | show this help message and exit                                                                                                                                                                                                  |
| --psm_file PSM_FILE      | The file name for the psms ratios file, will be inside the output_folder dir.                                                                                                                                                   |
| --peptide_file PEPTIDE_FILE | The file name for the peptide ratios file, will be inside the output_folder dir.                                                                                                                                                |
| --protein_file PROTEIN_FILE | The file name for the protein ratios file, will be inside the output_folder dir.                                                                                                                                                |
| --pairs PAIRS            | Pairs of groups to compare. Pairs must be in the following format: "Group1,Group2;...;Group1:Group3". Each pair must only contain 2 values (separated by a comma), and these values must match those in the groups argument. Multiple pairs must be separated by a semicolon. |
| --groups GROUPS          | The group labels mapped to the indices for their reporter ion channels. Groups must be specified in the following format: "GroupName1:Index1,...,Index3;GroupName2:Index1,...,Index3;". Group names must be unique and can be of any type. Indexes must be separated by a comma, and multiple groups must be separated by a semicolon. |
| --filter {unique,all}   | Filter type for peptides. Unique will only keep peptides which map to a single protein. All will keep all peptides.                                                                                                            |
| --groupby_filename      | Group by the filename for psm, peptide, and proteins. This will add a filename column to the output files.                                                                                                                     |
| --output_type {csv,parquet} | Output file type.                                                                                                                                                                                                               |
| --inf_replacement INF_REPLACEMENT | Infinite values cause many problems with the statistics. This value will be used to replace infinite values at the log2 ratio level. Default is 100. (-inf will be replaced with -100 and inf will be replaced with 100)     |
| --no_psms                | Don't output a PSM ratio file.                                                                                                                                                                                                  |
| --no_peptides            | Don't output a Peptide ratio file.                                                                                                                                                                                              |
| --no_proteins            | Don't output a Protein ratio file.                                                                                                                                                                                              |

## Basic Steps

 1. Read Input File
 2. Generate QuantGroup's (dataclass to store group_name, group_indecies, and the psm info)
 3. Group by PSMs (peptide, charge, Optional[filename])
 4. Group by Peptides (peptide, Optional[filename])
 5. Filter QuantGroup's based on --filter option
 6. Group by Proteins (proteins, Optional[filename])

 
## Grouping Steps

 1. Sort QuantGroups based on grouping criteria.
 2. For each group of QuantGroups, create a dict mapping groupname to a list of QuantGroups.
 3. Loop over the pairs of groups (specified by pairs argument)
 4. Create a GroupRatio object for each pair (dataclass which stores the list of QuantGroups for each groupname of the pair)
 5. Calculate the qvalue for all GroupRatios. The GroupRatio dataclass has properties for pvalue, ratio.... ect, but not for qvalue since this requires knowledge of all GroupRatios. 

## Ratio Calculation (Psudo Code)

This method was chosen since one off groups (groups which contain only one datapoint) cannot generate a pvalue since the degrees of freedom would be 0.

The following values are not accurate at all.

    grp1 = [[100, 110, 105], [250, 245, 255]]
    grp2 = [[200, 210, 205], [500, 490, 0]]
    
    1) Calculate the mean intensity of the reference channel (first groupname in pairs)
    
    grp1_mean = mean(grp1)
    grp1_mean = [100, 250]

	2) Divide the reference mean intensity by the other groups reporter ions. 

	ratios = grp1_mean / grp2 
	ratios = [[100/200, 100/210, 100/205], [250/500, 250/490, 250/0]]
	ratios = [[~0.5, ~0.45, ~0.47], [~0.5, ~0.55, inf]]
	
	3) Flatten the array, and Remove NANs (caused by 0/0)
	ratios = flatten(ratios)
	ratios = [~0.5, ~0.45, ~0.47, ~0.5, ~0.55, ~inf]

	ratios = remove_nan(ratios)
	ratios = [~0.5, ~0.45, ~0.47, ~0.5, ~0.55, ~inf]

	4) Calcualte the log2 ratios, and replace -inf and +inf with the inf_replacement value, default is 100)

	log2_ratios = log2(ratios)
	log2_ratios = [~-2, ~-2, ~-2, ~-2, ~-2, ~-2, inf]

	log2_ratios = replace_inf(log2_ratios)
	log2_ratios = [~-2, ~-2, ~-2, ~-2, ~-2, ~-2, 100]
	
	5) Calculate the mean and std
	
	log2_ratio = mean(log2_ratios)
	log2_ratio = ~14
	log2_ratio_std = std(log2_ratios)
	log2_ratio_std = ~5

	6) Calculate the p-value for a one-sample t-test comparing the sample mean to 0.

	log2_ratio_pvalue = pvalue(log2_ratio, log2_ratio_std, len(grp1))
	log2_ratio_pvalue = ~0.50