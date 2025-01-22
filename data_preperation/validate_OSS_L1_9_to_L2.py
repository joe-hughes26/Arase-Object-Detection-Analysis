import datetime
import glob
import os
import json

import netCDF4
import numpy as np

"""
A collection of utility functions to validate that the generated L2 files from L2 files are the same as the originally created L2 files.
"""

# Where the original L2 files are stored.
OG_L2_SAVE_DIR = "data/Arase_L2_original"

"""
Indicates if the local L2 files use the intermiediary week directories in the form:

    OG_L2_SAVE_DIR/0429-0505/OSSLevel2_Arase_perigee_20180429040254.nc

If False, assumes the naming convention:
    
    OG_L2_SAVE_DIR/OSSLevel2_Arase_perigee_20180429040254.nc
"""
OG_L2_FILES_USE_WEEK_SUBFLODERS = True

# NOTE: The L2 files created by these scripts DO NOT use the week directory naming convention.

# Where the newly created L2 files are stored: 
NEW_L2_SAVE_DIR = "data/Arase_L2"

# Hard coded cutoff dates for OSS internal L2 folder structure.
cutoff_dates = [
    datetime.datetime(2018, 4, 29),
    datetime.datetime(2018, 5, 6),
    datetime.datetime(2018, 5, 13),
    datetime.datetime(2018, 5, 20),
    datetime.datetime(2018, 5, 27),
    datetime.datetime(2018, 6, 3),
    datetime.datetime(2018, 6, 10),
    datetime.datetime(2018, 6, 17),
    datetime.datetime(2018, 6, 24),
    datetime.datetime(2018, 7, 1),
    datetime.datetime(2018, 7, 8),
    datetime.datetime(2018, 7, 15),
    datetime.datetime(2018, 7, 22),
    datetime.datetime(2018, 7, 29),
    datetime.datetime(2018, 8, 5),
    datetime.datetime(2018, 8, 12),
    datetime.datetime(2018, 8, 19),
    datetime.datetime(2018, 8, 26),
    datetime.datetime(2018, 9, 2),
    datetime.datetime(2018, 9, 2),
    datetime.datetime(2018, 9, 16),
    datetime.datetime(2018, 9, 24),
]

def compare_two_nc_files(original_path: str, new_path: str) -> dict:
    """
    Function to compare the original OSS L2 .nc file to the new L2 .nc file which is created from the L1.9 dataset.

    Args:
        original_path (str): The path to the original L2 .nc file.
        new_path (str): The path to the new L2 .nc file.

    Returns:
        dict: A JSON serializable dictionary with information about if the two datasets are equivilant, and any discrepencies.
    """
    individual_comparison_report = {}
    individual_comparison_report["original_L2_path"] = original_path
    individual_comparison_report["new_L2_path"] = new_path
    individual_comparison_report["dataset_vars_are_all_close"] = True

    differences = {}
    vars_in_both = []
    vars_in_only_old = []
    vars_that_are_not_close = []
    exceptions = []
    with netCDF4.Dataset(original_path) as og_dset:
        with netCDF4.Dataset(new_path) as new_dset:
            og_vars = og_dset.variables.keys()
            new_vars = new_dset.variables.keys()
            for var in og_vars:
                try:
                    if var not in new_vars:
                        vars_in_only_old.append(var)
                        individual_comparison_report["dataset_vars_are_all_close"] = False
                        continue
                    vars_in_both.append(var)
                    og_var_arr = np.array(og_dset[var])
                    new_var_arr = np.array(new_dset[var])
                    var_is_equal = False
                    if var == "generator_name":
                        var_is_equal = np.all(og_var_arr == new_var_arr)
                    else:
                        if og_var_arr.dtype == "object":
                            og_var_arr = og_var_arr.astype(np.datetime64).astype(np.int64)
                            new_var_arr = new_var_arr.astype(np.datetime64).astype(np.int64)
                        if np.allclose(og_var_arr, new_var_arr, equal_nan=True):
                            var_is_equal = True
                    if not var_is_equal:
                        vars_that_are_not_close.append(var)
                        individual_comparison_report["dataset_vars_are_all_close"] = False
                except Exception as e:
                    individual_comparison_report["dataset_vars_are_all_close"] = False
                    exceptions.append({var: str(e)})
    differences["vars_not_in_new_file"] = vars_in_only_old
    differences["vars_that_are_not_close"] = vars_that_are_not_close
    differences["exceptions"] = exceptions
    if not individual_comparison_report["dataset_vars_are_all_close"]:
        individual_comparison_report["differences"] = differences
    else:
        individual_comparison_report["differences"] = None

    return individual_comparison_report


def main_tester(report_save_dir: str = "OG_and_new_L2_comparison_report", print_info: bool = True) -> None:
    """
    The main function to evaluate the equivilance of all orignal an dnewly generated L2 files,

    Args:
        report_save_dir (str, optional): Directory to save the JSON report to. Defaults to "OG_and_new_L2_comparison_report".
        print_info (bool, optional): Indicator if information should be printed to the terminal. Defaults to True.
    """

    original_L2_files = glob.glob(os.path.join(OG_L2_SAVE_DIR, "*/*.nc" if OG_L2_FILES_USE_WEEK_SUBFLODERS else "*.nc"), recursive=True)

    new_L2_files = glob.glob(os.path.join(NEW_L2_SAVE_DIR, "*.nc"), recursive=True)
    full_comparison_report = {}
    old_files_missing_new_pair = []
    L2_pairs = {}
    for original_L2_file in original_L2_files:
        fname = os.path.split(original_L2_file)[1]
        has_pair = False
        for ind in range(len(new_L2_files)):
            if fname ==  os.path.split(new_L2_files[ind])[1]:
                new_fpath = new_L2_files.pop(ind)
                has_pair = True
                break
        
        if has_pair:
            L2_pairs[original_L2_file] = new_fpath
        else:
            old_files_missing_new_pair.append(original_L2_file)

    full_comparison_report["old_files_missing_new_pair"] = old_files_missing_new_pair
    if print_info:
        print(f"Old Files without a new pair:\n\t{old_files_missing_new_pair}")
    individual_comparison_reports = []
    for old_file, new_file in L2_pairs.items():
        if print_info:
            print(f"\nComparing: \n\t{old_file}\n\t{new_file}")
        comparison = compare_two_nc_files(old_file, new_file)
        if comparison["differences"] is not None and print_info:
            print("Found differences:")
            print(json.dumps(comparison["differences"], indent=1))
        individual_comparison_reports.append(comparison)
    full_comparison_report["individual_comparison_reports"] = individual_comparison_reports

    save_path = os.path.join(report_save_dir, datetime.datetime.now().strftime("comparison_%Y%m%d_%H%M%S.json"))
    if not os.path.exists(report_save_dir):
        os.makedirs(report_save_dir)
    with open(save_path, "w") as fp:
        json.dump(full_comparison_report, fp, indent=1)

if __name__ == "__main__":
    main_tester()