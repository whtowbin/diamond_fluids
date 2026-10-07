# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
from typing import Union, List, Any, Dict
from copy import deepcopy
# %%

filepath = Path("Fibrous Diamond LA-ICP-MS/fibrous_LA-ICP-MS_map_121224.xlsx")
df = pd.read_excel(filepath, sheet_name="Fibrous Diamond")
columns = df.columns
element_mean_list = columns[1:-2:2]
element_SD_list = columns[2:-2:2]
element_list = [re.sub(r"\d+", "", col_name).split("_")[0] for col_name in element_mean_list]

element_dict_mean = dict(zip(element_list, element_mean_list))
element_dict_SD = dict(zip(element_list, element_SD_list))


# %%
# %%
# Elements to use for future analysis
# C, Mg, Al, K, Ti, Cr, Fe, Ni, Rb, Sr, Y, Zr, Nb, Ba, La, Ce, Pr, Nd, Hf, Pb, Th, U

df_plot = df[element_dict_mean["Zr"]] / df[element_dict_mean["La"]]
# df_plot = df[element_dict_mean["Pb"]] / df[element_dict_mean["U"]]
# df_plot = df[element_dict_mean["Ba"]] / df[element_dict_mean["Ce"]]
# df_plot = df[element_dict_mean["La"]] / df[element_dict_mean["Ce"]]
# df_plot = df[element_dict_mean["Zr"]] / df[element_dict_mean["Ce"]]
# df_plot = df[element_dict_mean["Zr"]] / df[element_dict_mean["Sr"]]
# df_plot = df[element_dict_mean["Nb"]] / df[element_dict_mean["La"]]
# df_plot = df[element_dict_mean["Al"]] / df[element_dict_mean["La"]]
# df_plot = df[element_dict_mean["U"]] / df[element_dict_mean["Sr"]]
# df_plot = df[element_dict_mean["Mg"]] / df[element_dict_mean["La"]]
# df_plot = df[element_dict_mean["U"]] / df[element_dict_mean["Sr"]]
# df_plot = df[element_dict_mean["Al"]] / df[element_dict_mean["Sr"]]
# df_plot = df[element_dict_mean["Zr"]] / df[element_dict_mean["Y"]]
# df_plot = df[element_dict_mean["Y"]] / df[element_dict_mean["La"]]
# df_plot = df[element_dict_mean["Sr"]] / df[element_dict_mean["La"]]
df_plot = df[element_dict_mean["Ba"]] / df[element_dict_mean["Sr"]]
# df_plot = df[element_dict_mean["Na"]] / df[element_dict_mean["C"]]
# df_plot = df[element_dict_mean["Rb"]] / df[element_dict_mean["La"]]
reshaped_array = df_plot.to_numpy().reshape(16, 17)

xx, yy = np.meshgrid(16, 17)
plt.imshow(np.log10(reshaped_array))
# plt.imshow(reshaped_array)
plt.colorbar()
# %%

# ['C', 'Na', 'Mg', 'Al', 'Si', 'P', 'K', 'Ca', 'Ti', 'Cr', 'Fe', 'Ni', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'Pb', 'Th', 'U']
# columns = df.columns
columns = df.columns
# %%

good_list = [
    "C",
    "Mg",
    "Al",
    "K",
    "Ti",
    # "Cr", # Not actually useful when corrected fro C ablation
    "Fe",
    "Ni",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Sm",
    "Eu",
    "Gd",  # Not great but works at high concentrations
    "Hf",
    "Pb",
    "Th",
    "U",
]
# %%


import pandas as pd
import numpy as np


def split_iolite_data_by_blank_rows(df):
    """Splits a Pandas DataFrame by blank rows.

    Args:
        df: The Pandas DataFrame to split.

    Returns:
        A list of Pandas DataFrames.
    """
    blank_rows = df[df.isnull().all(axis=1)].index.tolist()
    split_dfs = {}

    if not blank_rows:
        key = df.iloc[0, 0]
        split_dfs[key] = df.drop(index=df.index[0])
        return split_dfs

    # split_dfs = []

    start_index = 0

    for index in blank_rows:
        split_df = df.iloc[start_index:index]
        if not split_df.empty:  # Avoid adding empty DataFrames
            split_dfs[split_df.iloc[0, 0]] = split_df.drop(index=split_df.index[0])
        start_index = index + 1

    # Add the remaining part of the DataFrame after the last blank row
    if start_index < len(df):
        split_df = df.iloc[start_index:]
        if not split_df.empty:  # Avoid adding empty DataFrames
            split_dfs[split_df.iloc[0, 0]] = split_df.drop(index=split_df.index[0])

    return split_dfs


def split_iolite_Standard_Values_by_blank_rows(df):
    """Splits a Pandas DataFrame by blank rows.

    Args:
        df: The Pandas DataFrame to split.

    Returns:
        A list of Pandas DataFrames.
    """
    blank_rows = df[df.isnull().all(axis=1)].index.tolist()
    split_dfs = {}

    if not blank_rows:
        key = df.iloc[0, 0]
        split_df = df.drop(index=df.index[0:7])
        split_df.columns = split_df.iloc[0]
        split_df = split_df[1:]
        split_df = split_df.set_index("Measurement")
        split_dfs[key] = split_df
        return split_dfs

    # split_dfs = []

    start_index = 0

    for index in blank_rows:
        split_df = df.iloc[start_index:index]
        if not split_df.empty:  # Avoid adding empty DataFrames
            key = split_df.iloc[0, 0]
            split_df = split_df.drop(index=split_df.index[0:7])
            split_df.columns = split_df.iloc[0]
            split_df = split_df[1:]
            split_df = split_df.set_index("Measurement")
            split_dfs[key] = split_df
        start_index = index + 1

    # Add the remaining part of the DataFrame after the last blank row
    if start_index < len(df):
        split_df = df.iloc[start_index:]
        if not split_df.empty:  # Avoid adding empty DataFrames
            key = split_df.iloc[0, 0]
            split_df = split_df.drop(index=split_df.index[0:7])
            split_df.columns = split_df.iloc[0]
            split_df = split_df[1:]
            split_df = split_df.set_index("Measurement")
            split_dfs[key] = split_df
    return split_dfs


def get_element_column_names(df):
    columns = df.columns
    element_mean_list = columns[1:-2:2]
    element_SD_list = columns[2:-2:2]
    element_list = [re.sub(r"\d+", "", col_name).split("_")[0] for col_name in element_mean_list]

    element_dict_mean = dict(zip(element_list, element_mean_list))
    element_dict_SD = dict(zip(element_list, element_SD_list))
    return {"element_dict_mean": element_dict_mean, "element_dict_SD": element_dict_SD}


def make_calibrations(standard_data_df, standard_values_df, internal_standard=None) -> dict:
    """Calibrate Elements for Single Standard Calibration Curves. This is a very basic calibration
    Slope  = Concentration/Counts
    - A better one would be calculated based on the standard deviation and uncertainty
    - Calibration is not setup for matching standards overtime. This would require exporting data with a time stamp (which must be possible)

    - First attempt at the function does not include internal normalization but that should be an option:
    - To include Internal normalization I would multiply counts for each element but the counts of internal standard, Then in the end I would need to correct the values for a known concentration of that element
    - I could potentially include carbon ad an internal standard even though it will be meaningless for the real concentration of micro-fluid inclusions

    Args:
        standard_data_df (_type_): _description_
        standard_values_df (_type_): _description_
    """
    col_dicts = get_element_column_names(standard_data_df)
    element_dict_mean, element_dict_SD = (
        col_dicts["element_dict_mean"],
        col_dicts["element_dict_SD"],
    )

    calibration_coefficient_dict = {}
    # calibration_rate_check = {}
    # calibration_conc_check = {}

    for element in element_dict_mean.keys():
        if element in standard_values_df.index:
            # Sensitivity = Concentration / Count
            calibration_coefficient_dict[element] = (
                standard_values_df.loc[element]["Value"]
                / standard_data_df[element_dict_mean[element]].mean()
            )
            # Test Points remove in final
            # calibration_rate_check[element] = standard_data_df[element_dict_mean[element]].mean()
            # calibration_conc_check[element] = standard_values_df.loc[element]["Value"]

    if internal_standard != None:
        if internal_standard not in calibration_coefficient_dict.keys():
            raise Exception(
                "Internal Standard Not found in dataset. Make sure element symbol abbreviation is properly capitalized (e.g. 'Fe' not 'fe')"
            )
        else:
            internal_calib_results = {}
            for element in calibration_coefficient_dict.keys():
                # Sensitivity_IS_Norm = Sensitivity_Element/ Sensitivity_InternalStandard
                internal_calib_results[element] = (
                    calibration_coefficient_dict[element]
                    / calibration_coefficient_dict[internal_standard]
                )
            internal_calib_results["internal_standard"] = internal_standard
            calibration_coefficient_dict = internal_calib_results

    return calibration_coefficient_dict  # , calibration_rate_check, calibration_conc_check, internal_calib_results


# Next Step in applying the internal calibration is to correct the count rates of the unknown sample for the calibration rate and concentration of the internal standard.


def appy_calibration(
    sample_data_df: pd.DataFrame,
    calibration_coefficient_dict: Dict,
    use_internal_standard: bool = False,
    internal_standard_known_concentrations: Union[float, List[float]] = None,
):
    col_dicts = get_element_column_names(sample_data_df)
    element_dict_mean, element_dict_SD = (
        col_dicts["element_dict_mean"],
        col_dicts["element_dict_SD"],
    )

    calibrated_conc_dict = {}
    calibrated_SD_dict = {}

    # ToDo fix length logic for float inputs
    # len_int_standard_list = len(internal_standard_known_concentrations)
    # len_rows_dataframe = len(calibration_coefficient_dict["internal_standard"])

    # if (len_int_standard_list < len_rows_dataframe) & (len_int_standard_list > 1):
    #     raise Exception(
    #         f"internal_standard_known_concentrations is the incorrect length. It must be an array the same length as the number of rows of the dataframe or a single value, currently it is {len_int_standard_list} values and the total number of rows is {len_rows_dataframe}"
    #     )

    if use_internal_standard:
        internal_standard = calibration_coefficient_dict["internal_standard"]
        internal_standard_counts = sample_data_df[element_dict_mean[internal_standard]]

    else:
        internal_standard_counts = 1
        internal_standard_known_concentrations = 1

    for element in element_dict_mean.keys():
        if element in calibration_coefficient_dict.keys():
            calibrated_conc_dict[element] = (
                calibration_coefficient_dict[element]
                * sample_data_df[element_dict_mean[element]]
                / internal_standard_counts
                * internal_standard_known_concentrations
            )
            calibrated_SD_dict[element] = (
                calibration_coefficient_dict[element]
                * sample_data_df[element_dict_SD[element]]
                / internal_standard_counts
                * internal_standard_known_concentrations
            )

    return calibrated_conc_dict, calibrated_SD_dict


# %%
# Produce dictionary of sample data
filepath = Path("Fibrous Diamond LA-ICP-MS/fibrous_LA-ICP-MS_map_121224.xlsx")
df_all = pd.read_excel(filepath, sheet_name="Data")
df_dict = split_iolite_data_by_blank_rows(df_all)
df_dict.keys()
# %%
# Produce dictionary of standard reference value dataframes
filepath = Path("Fibrous Diamond LA-ICP-MS/fibrous_LA-ICP-MS_map_121224.xlsx")
df_reference = pd.read_excel(filepath, sheet_name="Reference Material Values")
standards_dict = split_iolite_Standard_Values_by_blank_rows(df_reference)
standards_dict.keys()
# %%
internal_standard = "Ba"
standard_key = "G_NIST612"
calibrations = make_calibrations(
    df_dict[standard_key], standards_dict[standard_key], internal_standard=internal_standard
)

sample_key = "TE doped graphite"  # "macs-3 georem" #"G_NIST612"

data_calib = appy_calibration(
    df_dict[sample_key], calibrations, True, standards_dict[sample_key].loc[internal_standard].Value
)

df_calibrated = pd.DataFrame(data_calib[0])

standards_dict[sample_key]
# %%
df_calibrated.to_excel(
    f"standard-{sample_key} concentrations_internalstandard-{internal_standard}.xlsx"
)
standards_dict[sample_key].T[df_calibrated.columns].to_excel(
    f"standard-{sample_key} known concentrations_internalstandard-{internal_standard}.xlsx"
)
# %%


internal_standard = "Ba"
standard_key = "G_NIST612"
calibrations = make_calibrations(
    df_dict[standard_key], standards_dict[standard_key], internal_standard=internal_standard
)

sample_key = "TE doped graphite"  # "macs-3 georem" #"G_NIST612"

data_calib = appy_calibration(
    df_dict[sample_key], calibrations, True, standards_dict[sample_key].loc[internal_standard].Value
)

df_calibrated = pd.DataFrame(data_calib[0])

standards_dict[sample_key]
# %%
df_calibrated.to_excel(
    f"standard-{sample_key} concentrations_internalstandard-{internal_standard}.xlsx"
)
standards_dict[sample_key].T[df_calibrated.columns].to_excel(
    f"standard-{sample_key} known concentrations_internalstandard-{internal_standard}.xlsx"
)
# %%
