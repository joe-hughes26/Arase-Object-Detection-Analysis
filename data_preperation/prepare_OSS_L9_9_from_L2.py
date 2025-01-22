import os
import subprocess
import glob

import numpy as np
import netCDF4

from data_preperation.Create_OSS_L2_perigee_product_from_OSS_L1_9 import ARASE_NET_CDF_VARS, ARASE_DERIVED_NETCDF_VARS, ARASE_L2_SAVE_DIRECTORY

"""
Utility functions for the OSS team to create the Initial L1.9 Dataset from the L2 dataset. Do not use.

Warning: `nccopy_dataset` uses subprocess, the default arguments assume that the script is running from PS or CMD and that an installation of wsl exists with `nccopy` available.
"""

def nccopy_dataset(
    original_path: str,
    new_path: str,
    use_wsl: bool = True,
    use_shell: bool = True,
    compression_level: int =4,
) -> subprocess.CompletedProcess[bytes]:
    """
    Copy a netcdf file to a new path using nccopy and subprocess. 

    Args:
        original_path (str): The original .nc file path.
        new_path (str): The new .nc file path.
        use_wsl (bool, optional): Indicates if subprocess command should use wsl as the initial Arg. Defaults to True.
        use_shell (bool, optional): Indicates if subprocess should set `shell=True`. Defaults to True.
        compression_level (int, optional): the compression level used by `nccopy`. Defaults to 4.

    Returns:
        subprocess.CompletedProcess[bytes]: The result of the nccopy `subprocess.run(...)` command.
    """
    # Warning: uses subprocess, the default arguments assume that the script is running from PS or CMD and that an installation of wsl exists with `nccopy` available.
    if use_wsl:
        command = ["wsl"]
    else:
        command = []
    nc_command = ["nccopy", "-d", f"{compression_level}", f"{original_path}", f"{new_path}"]
    command.extend(nc_command)
    return subprocess.run(command, shell=use_shell, capture_output=True)

def purge_arase_data_from_L2_dataset(netcdf_L2_dataset_path: str, save_dir: str = None, delete_original: bool = False):
    """
    Removes Arase data from an OSS L2 Arase perfigee file.

    Args:
        netcdf_L2_dataset_path (str): The file path to the L2 dataset.
        save_dir (str, optional): the directory to save the new file to. Defaults to None.
        delete_original (bool, optional): Indicates if th eoriginal L2 file should be deleted. Defaults to False.
    """

    # Assumes current_fname conforms to LEVEL_2_FILENAME_TEMPLATE
    current_dir, current_fname = os.path.split(netcdf_L2_dataset_path)
    new_fname = "_".join(["OSSLevel1_9_Arase_perigee", current_fname.split("_")[-1]])
    full_new_path = os.path.join(current_dir, new_fname) if save_dir is None else os.path.join(save_dir, new_fname)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    copy_result = nccopy_dataset(netcdf_L2_dataset_path, full_new_path, use_wsl=True, use_shell=True, compression_level=4)
    if copy_result.returncode != 0:
        print(f"Failed to copy: {netcdf_L2_dataset_path} to {full_new_path} after purging due to subprocess error: {copy_result.stderr}")
    dataset = netCDF4.Dataset(full_new_path, 'r+')
    for var in [*ARASE_NET_CDF_VARS, *ARASE_DERIVED_NETCDF_VARS]:
        if np.dtype(dataset[var]) == '<U0':
            fill_arr = np.zeros_like(np.array(dataset[var]))
            fill_arr[:] = ""
            # continue
        else:
            fill_arr = np.zeros_like(np.array(dataset[var]))
            fill_arr[:] = np.nan
        dataset[var][:] = fill_arr[:]
    dataset.close()
    
    if delete_original:
        os.remove(netcdf_L2_dataset_path)

def create_l1_9_datasets_from_l2_datasets() -> None:
    """
    Creates all L1.9 files from the L2 files in ARASE_L2_SAVE_DIRECTORY
    """
    glob_path=os.path.join(ARASE_L2_SAVE_DIRECTORY, "*/*.nc")
    l2_file_paths = glob.glob(glob_path, recursive=True)
    for l2_file_path in l2_file_paths:
        print(f"Rewriting L2 file {l2_file_path} now.")
        purge_arase_data_from_L2_dataset(os.path.normpath(l2_file_path).replace("\\", "/"), "data/arase_L1_9/")
        print(f"Finished rewriting L2 file {l2_file_path}.")
