import os
import glob
import datetime
import functools
import asyncio
from itertools import chain

import aiohttp
import netCDF4
import numpy as np
import numpy.typing as npt
import cdflib

from typing import Union, Any

NumpyNumeric = Union[np.floating, np.integer]

# Directory where the arase raw data should be downloaded to. Relative to the root of the repo.
ARASE_DOWNLOAD_DIRECTORY = "data/arase_raw_data"

# Directory where the OSS L2 perigee data product is to be stored. Relative to the root of the repo.
ARASE_L2_SAVE_DIRECTORY = "data/arase_L2"

# Directory where the OSS L1.9 data product is stored. Relative to the root of the repo.
ARASE_L1_9_SAVE_DIRECTORY = "data/arase_L1_9/arase_L1_9_archive"

# The naming convention for OSS L2 perigee files.
LEVEL_2_FILENAME_TEMPLATE = "OSSLevel2_Arase_perigee_{year:04d}{month:02d}{day:02d}{hour:02d}{minute:02d}{second:02d}.nc"

# The naming convention for OSS L1.9 perigee files.
LEVEL_1_9_FILENAME_TEMPLATE = "OSSLevel1_9_Arase_perigee_{year:04d}{month:02d}{day:02d}{hour:02d}{minute:02d}{second:02d}.nc"

# Variables within the OSS L2 (and L1.9) Arase Perigee netcdf files that are derived directly from the Arase data.
ARASE_NET_CDF_VARS = [
    "Arase_ECEF_pos",
    "Arase_LLA",
    "Efreq",
    "Etime_OFACOMPLEX",
    "E_OFACOMPLEX",
    "Ex_OFACOMPLEX",
    "Ey_OFACOMPLEX",
    "Eflag_OFACOMPLEX",
    "Bfreq",
    "Btime_OFACOMPLEX",
    "B_OFACOMPLEX",
    "Bx_OFACOMPLEX",
    "By_OFACOMPLEX",
    "Bz_OFACOMPLEX",
    "Bflag_OFACOMPLEX",
    "Etime_OFASPEC",
    "E_OFASPEC",
    "Echannel_OFASPEC",
    "Eflag_OFASPEC",
    "Btime_OFASPEC",
    "B_OFASPEC",
    "Bchannel_OFASPEC",
    "Bflag_OFASPEC",
    "Etime_OFAwave",
    "E_OFAwave",
    "Eflag_OFAwave",
    "Eobscal_OFAwave",
    "Btime_OFAwave",
    "B_OFAwave",
    "Bflag_OFAwave",
    "Bobscal_OFAwave",
    "kvec_polar",
    "kvec_polar_masked",
    "kvec_azimuth",
    "kvec_azimuth_masked",
    "polarization",
    "polarization_masked",
    "planarity",
    "planarity_masked",
    "Ptime_OFAwave",
    "Pvec_angle",
    "Pvec_angle_masked",
    "HFA_time",
    "UHRfreq_HFA",
    "Ne_HFA",
]

# Variables within the L2 (and L1.9) netcdf files that are indirectly derived from Arase Data
ARASE_DERIVED_NETCDF_VARS = [
    'dx',
]

# Base URLs for retrieving Arase data:
ARASE_BASE_URLS = {
    "l2_complex":  "https://ergsc.isee.nagoya-u.ac.jp/data/ergsc/satellite/erg/pwe/ofa/l2/complex/",
    "l2_spec":     "https://ergsc.isee.nagoya-u.ac.jp/data/ergsc/satellite/erg/pwe/ofa/l2/spec/",
    "l3_property": "https://ergsc.isee.nagoya-u.ac.jp/data/ergsc/satellite/erg/pwe/ofa/l3/property/",
    "hfa_l3":      "https://ergsc.isee.nagoya-u.ac.jp/data/ergsc/satellite/erg/pwe/hfa/l3/",
    "orb_def":     "https://ergsc.isee.nagoya-u.ac.jp/data/ergsc/satellite/erg/orb/def/",
}

# Latest versions available at time of creating the OSS level 2 dataset. We cannot guarentee that any version changes iniitated by the Arase team will result in compatibility.
# We support multiple "iterations" of a version for select datasets below; However, at the time the versions must remain fixed.
ARASE_EXTENDED_URLS = {
    "l2_complex":  "https://ergsc.isee.nagoya-u.ac.jp/data/ergsc/satellite/erg/pwe/ofa/l2/complex/{year:04d}/{month:02d}/erg_pwe_ofa_l2_complex_sgi_{year:04d}{month:02d}{day:02d}_v01_{iteration:02d}.cdf",
    "l2_spec":     "https://ergsc.isee.nagoya-u.ac.jp/data/ergsc/satellite/erg/pwe/ofa/l2/spec/{year:04d}/{month:02d}/erg_pwe_ofa_l2_spec_{year:04d}{month:02d}{day:02d}_v02_{iteration:02d}.cdf",
    "l3_property": "https://ergsc.isee.nagoya-u.ac.jp/data/ergsc/satellite/erg/pwe/ofa/l3/property/{year:04d}/{month:02d}/erg_pwe_ofa_l3_property_dsi_{year:04d}{month:02d}{day:02d}_v01_{iteration:02d}.cdf",
    "hfa_l3":      "https://ergsc.isee.nagoya-u.ac.jp/data/ergsc/satellite/erg/pwe/hfa/l3/{year:04d}/{month:02d}/erg_pwe_hfa_l3_1min_{year:04d}{month:02d}{day:02d}_v03_{iteration:02d}.cdf",
    "orb_def":     "https://ergsc.isee.nagoya-u.ac.jp/data/ergsc/satellite/erg/orb/def/{year:04d}/erg_orb_l2_{year:04d}{month:02d}{day:02d}_v03.cdf",
}

ARASE_EXTENDED_URLS_VALID_ITERATIONS = {
    "l2_complex": [3],
    "l2_spec": [3],
    "l3_property": [3],
    "hfa_l3": [5, 6], # Currently, this is the only Dataset that requires access to multiple iterations.
    "orb_def": [None], # Arase team does not have multiple iterations for this version 3 dataset.
}

# A map from the Arase dataset "name" to the cdf keys/vars we want to extract from that dataset.
ARASE_DATA_VAR_NAMES = {
    "l2_complex": [
        "Etotal_132",
        "Btotal_132",
        "Ex_132",
        "Ey_132",
        "Bx_132",
        "By_132",
        "Bz_132",
        "quality_flag_e132",
        "quality_flag_b132",
        "freq_e132",
        "freq_b132",
        "epoch_e132",
        "epoch_b132",
    ],
    "l2_spec": [
        "E_spectra_132",
        "B_spectra_132",
        "quality_flag_e132",
        "quality_flag_b132",
        "ch_e132",
        "ch_b132",
        "epoch_e132",
        "epoch_b132",
    ],
    "hfa_l3": [
        "Fuhr",
        "ne_mgf",
        "Epoch",
    ],
    "l3_property": [
        "kvec_polar_132",
        "kvec_azimuth_132",
        "polarization_132",
        "planarity_132",
        "Pvec_angle_132",
        "kvec_polar_masked_132",
        "kvec_azimuth_masked_132",
        "polarization_masked_132",
        "planarity_masked_132",
        "Pvec_angle_masked_132",
        "quality_flag_e132",
        "quality_flag_b132",
        "obs_cal_e132",
        "obs_cal_b132",
        "lpf_e132",
        "lpf_b132",
        "E_spectra_132",
        "B_spectra_132",
        "epoch_p132",
        "epoch_e132",
        "epoch_b132",
    ],
    "orb_def": [
        "pos_llr",
        "epoch",
    ]
}

def convert_arase_epoch_time_to_np_dt(epoch_time: npt.NDArray[NumpyNumeric]) -> npt.NDArray[np.datetime64]:
    """
    Convert Arase nanoseconds since 2000 (terestial time), to np.datetime64 (UT). Accounts for aproximately 69.1 seconds of leapsecond drift from the gregorian calandar.
    See https://en.wikipedia.org/wiki/%CE%94T_(timekeeping) for more information about this drift.

    Args:
        epoch_time (npt.NDArray[np.int64]): Time data from an arase dataset (nanoseconds since start of year 2000).

    Returns:
        npt.NDArray[np.datetime64]: Arase time converted to UT as np.datetime64. Same dimentions as input array.
    """
    return np.datetime64('2000-01-01T12:00:00') + epoch_time.astype('timedelta64[ns]') - np.array(69.1*1e9).astype('timedelta64[ns]')

def convert_np_dt_to_float(
        np_datetime64: npt.NDArray[np.datetime64], 
        reference_date: np.datetime64 = np.datetime64('2000-01-01T12:00:00'), 
        time_scale: np.timedelta64 = np.timedelta64(1, 'ns')
    ) -> npt.NDArray[np.floating]: 
    """
    Converts the input datetime to a float which represents the number of `time_scale`s since the `reference_date`.
    Default Args perform the inverse operation of `convert_arase_epoch_time_to_np_dt`.

    Args:
        np_datetime64 (npt.NDArray[np.datetime64]): The datetime array to convert to floats.
        reference_date (np.datetime64 , optional): The reference date compute the offser with respect to. Defaults to np.datetime64('2000-01-01T12:00:00').
        time_scale (np.timedelta64, optional): The time scale for which resulting float is calculated. Defaults to np.timedelta64(1, 'ns').

    Returns:
        npt.NDArray[np.float64]: The number of `time_scales` since `reference_date`.
    """
    return (np_datetime64 - reference_date) / time_scale

# Simple variable preprocessing that can be completed at data ingest.
ARASE_OSS_PREPROCESS_FUNCS = {
    "epoch_e132": lambda epoch_e132: convert_arase_epoch_time_to_np_dt(epoch_e132),
    "epoch_b132": lambda epoch_b132: convert_arase_epoch_time_to_np_dt(epoch_b132),
    "epoch_p132": lambda epoch_p132: convert_arase_epoch_time_to_np_dt(epoch_p132),
    "Epoch":      lambda Epoch:      convert_arase_epoch_time_to_np_dt(Epoch),
    "epoch":      lambda epoch:      convert_arase_epoch_time_to_np_dt(epoch),
}

def ecef_to_geodetic_vectorized(pos: npt.NDArray[NumpyNumeric]) -> tuple[npt.NDArray[NumpyNumeric], npt.NDArray[NumpyNumeric], npt.NDArray[NumpyNumeric]]:
    """
    Convert an ECEF position vector into geodetic Latitude, Longitude, Altitude.

    Args:
        pos (npt.NDArray[NumpyNumeric]): A vector of ECEF coordinates of shape (..., 3) [km]

    Returns:
        tuple[npt.NDArray[NumpyNumeric], npt.NDArray[NumpyNumeric], npt.NDArray[NumpyNumeric]]: (latitude [deg], longitude [deg], height [km])

    Notes:
        Uses equations given in Misra and Enge. #TODO: include proper citation.
    """
      
    a = 6378137.0 # Semi-major axis
    f = 1/298.257223563 # Flattening
    b = a*(1-f) # Semi-minor axis

    pos = pos*1000
    x = pos[..., 0]
    y = pos[..., 1]
    z = pos[..., 2]
    lon = np.arctan2(y, x)
    # Latitude and height are calculated iteratively
    p = np.sqrt(x**2 + y**2)
    lat = np.arctan2(z, p)
    # Fix issues with special case along equator
    lat[lat==0.0] = 0.1
    h = z/np.sin(lat)
    # Initialize relative changes in h and lat
    dh = 1.0
    dlat = 1.0
    while (dh > 1e-2) or (dlat > 1e-8):
        N = (a**2)/np.sqrt(a**2*np.cos(lat)**2 + b**2*np.sin(lat)**2)
        N1 = N*(b/a)**2
        # Calculate values for current iteration
        h_temp = p/np.cos(lat) - N
        lat_temp = np.arctan2(z/(N1+h), p/(N+h))
        # calculate difference from previous iteration (stopping condition)
        dh = np.max(np.abs(h_temp-h))
        dlat = np.max(np.abs(lat_temp-lat))
        # Update lat and h
        lat = lat_temp
        h = h_temp
    
    h = h/1000
    lat = np.degrees(lat)
    lon = np.degrees(lon)
    
    return lat, lon, h

def concatenate_arase_numpy_dicts(numpy_dicts: list[dict[Any, npt.NDArray[NumpyNumeric | np.datetime64]]], axis = 0) -> dict[Any, npt.NDArray[NumpyNumeric | np.datetime64]]:
    """
    Takes a dictionary and returns all the elements of that dictionary concatentated along an axis. Particular to intermediary Arase dictionaries in that have been loaded. Ignores kwargs containing "freq_*" and simply returns one element of the array. 

    Args:
        numpy_dicts (list[dict[Any, npt.NDArray[NumpyNumeric  |  np.datetime64]]]): An array of dictionaries that arae loaded from an arase datasetand should be concatentated with each other.
        axis (int, optional): The common axis along all arrays to perform concatenation. Defaults to 0.

    Returns:
        dict[Any, npt.NDArray[NumpyNumeric | np.datetime64]]: A dictionary with all arrays conceatenated (except arrays containing frequencey inforamation)
    """
    out_dict = {}
    for k in numpy_dicts[0].keys():
        arrs = []
        for numpy_dict in numpy_dicts:
            arrs.append(numpy_dict[k])
        out_dict[k] = np.concatenate(arrs, axis=axis) if "freq_" not in k else arrs[0]
    return out_dict

# NOTE: For OSS Reviewers, the orb_def `jumps` variable is uneeded at this time, due to the pergee ranges already having been calculated within each L2_file.

def get_valid_arase_time_mask(oss_master_time: npt.NDArray[np.datetime64], arase_epoch_time: npt.NDArray[np.datetime64]) -> npt.NDArray[np.bool_]:
    """
    Find the Arase epoch times that are within the specified time range given by `oss_master_time`.

    Args:
        oss_master_time (npt.NDArray[np.datetime64]): The `master_time` from the OSS perigee dataset.
        arase_epoch_time (npt.NDArray[np.datetime64]): The epoch time from an Arase dataset.

    Returns:
        npt.NDArray[np.bool_]: A boolean mask of the arase_epoch_time that are within the start and end time of the `oss_master_time` array.
    """
    return (oss_master_time[0] <= arase_epoch_time) * (arase_epoch_time <= oss_master_time[-1])

def get_OSS_L2_data_from_arase_dict_l2_complex(oss_master_time: npt.NDArray[np.datetime64], arase_data_l2_complex: dict[str, npt.NDArray]) -> dict[str, npt.NDArray[NumpyNumeric | np.datetime64]]:
    """
    Extract the OSS level 2 data fields from the Arase `l2_complex` dataset.

    Args:
        oss_master_time (npt.NDArray[np.datetime64]): The `master_time` from the OSS perigee dataset.
        arase_data_l2_complex (dict[str, npt.NDArray]): The relevant data extracted from the Arase `l2_complex` .cdf file.

    Returns:
        dict[str, npt.NDArray[NumpyNumeric | np.datetime64]]: A dictionary with keys and values drectly correspond to those in the variables and data in the OSS L2 Arase Perigee product.
            The keys of the output dictionary are ["Efreq", "Etime_OFACOMPLEX", "E_OFACOMPLEX", "Ex_OFACOMPLEX", "Ey_OFACOMPLEX", "Eflag_OFACOMPLEX", "Bfreq", "Btime_OFACOMPLEX", "B_OFACOMPLEX", "Bx_OFACOMPLEX", "By_OFACOMPLEX", "Bz_OFACOMPLEX", "Bflag_OFACOMPLEX"].
    """
    epoch_e132 = arase_data_l2_complex["epoch_e132"]
    epoch_b132 = arase_data_l2_complex["epoch_b132"]
    e_time_inds = get_valid_arase_time_mask(oss_master_time, epoch_e132)
    b_time_inds = get_valid_arase_time_mask(oss_master_time, epoch_b132)

    return {
        "Efreq":                arase_data_l2_complex["freq_e132"],
        "Etime_OFACOMPLEX":     arase_data_l2_complex["epoch_e132"][e_time_inds],
        "E_OFACOMPLEX":         arase_data_l2_complex["Etotal_132"][e_time_inds] * 1e6,
        "Ex_OFACOMPLEX": np.sum(arase_data_l2_complex["Ex_132"]**2, axis=-1)[e_time_inds] * 1e6,
        "Ey_OFACOMPLEX": np.sum(arase_data_l2_complex["Ey_132"]**2, axis=-1)[e_time_inds] * 1e6,
        "Eflag_OFACOMPLEX":     arase_data_l2_complex["quality_flag_e132"][e_time_inds],
        "Bfreq":                arase_data_l2_complex["freq_b132"],
        "Btime_OFACOMPLEX":     arase_data_l2_complex["epoch_b132"][b_time_inds],
        "B_OFACOMPLEX":         arase_data_l2_complex["Btotal_132"][b_time_inds],
        "Bx_OFACOMPLEX": np.sum(arase_data_l2_complex["Bx_132"]**2, axis=-1)[b_time_inds],
        "By_OFACOMPLEX": np.sum(arase_data_l2_complex["By_132"]**2, axis=-1)[b_time_inds],
        "Bz_OFACOMPLEX": np.sum(arase_data_l2_complex["Bz_132"]**2, axis=-1)[b_time_inds],
        "Bflag_OFACOMPLEX":     arase_data_l2_complex["quality_flag_b132"][b_time_inds],
    }

def get_OSS_L2_data_from_arase_dict_l2_spec(oss_master_time: npt.NDArray[np.datetime64], arase_data_l2_spec: dict[str, npt.NDArray]) -> dict[str, npt.NDArray[NumpyNumeric | np.datetime64]]:
    """
    Extract the OSS level 2 data fields from the Arase `l2_spec` dataset.

    Args:
        oss_master_time (npt.NDArray[np.datetime64]): The `master_time` from the OSS perigee dataset.
        arase_data_l2_spec (dict[str, npt.NDArray]): The relevant data extracted from the Arase `l2_spec` .cdf file.

    Returns:
        dict[str, npt.NDArray[NumpyNumeric | np.datetime64]]: A dictionary with keys and values drectly correspond to those in the variables and data in the OSS L2 Arase Perigee product.
            The keys of the output dictionary are ["Etime_OFASPEC", "E_OFASPEC", "Echannel_OFASPEC", "Eflag_OFASPEC", "Btime_OFASPEC", "B_OFASPEC", "Bchannel_OFASPEC", "Bflag_OFASPEC"].
    """
    epoch_e132 = arase_data_l2_spec["epoch_e132"]
    epoch_b132 = arase_data_l2_spec["epoch_b132"]
    e_time_inds = get_valid_arase_time_mask(oss_master_time, epoch_e132)
    b_time_inds = get_valid_arase_time_mask(oss_master_time, epoch_b132)
    
    return {
        "Etime_OFASPEC":    arase_data_l2_spec["epoch_e132"][e_time_inds],
        "E_OFASPEC":        arase_data_l2_spec["E_spectra_132"][e_time_inds] * 1e6,
        "Echannel_OFASPEC": arase_data_l2_spec["ch_e132"][e_time_inds],
        "Eflag_OFASPEC":    arase_data_l2_spec["quality_flag_e132"][e_time_inds],
        "Btime_OFASPEC":    arase_data_l2_spec["epoch_b132"][b_time_inds],
        "B_OFASPEC":        arase_data_l2_spec["B_spectra_132"][b_time_inds],
        "Bchannel_OFASPEC": arase_data_l2_spec["ch_b132"][b_time_inds],
        "Bflag_OFASPEC":    arase_data_l2_spec["quality_flag_b132"][b_time_inds],
    }

def get_OSS_L2_data_from_arase_dict_hfa_l3(oss_master_time: npt.NDArray[np.datetime64], arase_data_hfa_l3: dict[str, npt.NDArray]) -> dict[str, npt.NDArray[NumpyNumeric | np.datetime64]]:
    """
    Extract the OSS level 2 data fields from the Arase `hfa_l3` dataset.

    Args:
        oss_master_time (npt.NDArray[np.datetime64]): The `master_time` from the OSS perigee dataset.
        arase_data_hfa_l3 (dict[str, npt.NDArray]): The relevant data extracted from the Arase `hfa_l3` .cdf file.

    Returns:
        dict[str, npt.NDArray[NumpyNumeric | np.datetime64]]: A dictionary with keys and values drectly correspond to those in the variables and data in the OSS L2 Arase Perigee product.
            The keys of the output dictionary are ["HFA_time", "UHRfreq_HFA", "Ne_HFA"]
    """
    Epoch = arase_data_hfa_l3["Epoch"]
    time_inds = get_valid_arase_time_mask(oss_master_time, Epoch)

    return {
        "HFA_time": arase_data_hfa_l3["Epoch"][time_inds],
        "UHRfreq_HFA": arase_data_hfa_l3["Fuhr"][time_inds],
        "Ne_HFA": arase_data_hfa_l3["ne_mgf"][time_inds] * 1e6,
    }

def get_OSS_L2_data_from_arase_dict_l3_property(oss_master_time: npt.NDArray[np.datetime64], arase_data_l3_property: dict[str, npt.NDArray]) -> dict[str, npt.NDArray[NumpyNumeric | np.datetime64]]:
    """
    Extract the OSS level 2 data fields from the Arase `l3_property` dataset.

    Args:
        oss_master_time (npt.NDArray[np.datetime64]): The `master_time` from the OSS perigee dataset.
        arase_data_l3_property (dict[str, npt.NDArray]): The relevant data extracted from the Arase `l3_property` .cdf file.

    Returns:
        dict[str, npt.NDArray[NumpyNumeric | np.datetime64]]: A dictionary with keys and values drectly correspond to those in the variables and data in the OSS L2 Arase Perigee product.
            The keys of the output dictionary are ["Etime_OFAwave", "E_OFAwave", "Eflag_OFAwave", "Eobscal_OFAwave", "Btime_OFAwave", "B_OFAwave", "Bflag_OFAwave", "Bobscal_OFAwave", "kvec_polar", "kvec_polar_masked", "kvec_azimuth", "kvec_azimuth_masked", "polarization", "polarization_masked", "planarity", "planarity_masked", "Ptime_OFAwave", "Pvec_angle", "Pvec_angle_masked"]
    """
    epoch_e132 = arase_data_l3_property["epoch_e132"]
    epoch_b132 = arase_data_l3_property["epoch_b132"]
    epoch_p132 = arase_data_l3_property["epoch_p132"]
    e_time_inds = get_valid_arase_time_mask(oss_master_time, epoch_e132)
    b_time_inds = get_valid_arase_time_mask(oss_master_time, epoch_b132)
    p_time_inds = get_valid_arase_time_mask(oss_master_time, epoch_p132)

    return {
        "Etime_OFAwave":       arase_data_l3_property["epoch_e132"][e_time_inds],
        "E_OFAwave":           arase_data_l3_property["E_spectra_132"][e_time_inds] * 1e6,
        "Eflag_OFAwave":       arase_data_l3_property["quality_flag_e132"][e_time_inds],
        "Eobscal_OFAwave":     arase_data_l3_property["obs_cal_e132"][e_time_inds],
        "Btime_OFAwave":       arase_data_l3_property["epoch_b132"][b_time_inds],
        "B_OFAwave":           arase_data_l3_property["B_spectra_132"][b_time_inds],
        "Bflag_OFAwave":       arase_data_l3_property["quality_flag_b132"][b_time_inds],
        "Bobscal_OFAwave":     arase_data_l3_property["obs_cal_b132"][b_time_inds],
        "kvec_polar":          arase_data_l3_property["kvec_polar_132"][b_time_inds],
        "kvec_polar_masked":   arase_data_l3_property["kvec_polar_masked_132"][b_time_inds],
        "kvec_azimuth":        arase_data_l3_property["kvec_azimuth_132"][b_time_inds],
        "kvec_azimuth_masked": arase_data_l3_property["kvec_azimuth_masked_132"][b_time_inds],
        "polarization":        arase_data_l3_property["polarization_132"][b_time_inds],
        "polarization_masked": arase_data_l3_property["polarization_masked_132"][b_time_inds],
        "planarity":           arase_data_l3_property["planarity_132"][b_time_inds],
        "planarity_masked":    arase_data_l3_property["planarity_masked_132"][b_time_inds],
        "Ptime_OFAwave":       arase_data_l3_property["epoch_p132"][p_time_inds],
        "Pvec_angle":          arase_data_l3_property["Pvec_angle_132"][p_time_inds],
        "Pvec_angle_masked":   arase_data_l3_property["Pvec_angle_masked_132"][p_time_inds],
    }

def get_OSS_L2_data_from_arase_dict_orb_def(oss_master_time: npt.NDArray[np.datetime64], oss_generator_ECEF_pos: npt.NDArray, arase_data_orb_def: dict[str, npt.NDArray]) -> dict[str, npt.NDArray[NumpyNumeric | np.datetime64]]:
    """
    Extract the OSS level 2 data fields from the Arase `orb_def` dataset.

    Args:
        oss_master_time (npt.NDArray[np.datetime64]): The `master_time` from the OSS perigee dataset.
        oss_generator_ECEF_pos (npt.NDArray): The L1_9 generator (satellite) positions array from the OSS perigee dataset.
        arase_data_orb_def (dict[str, npt.NDArray]): The relevant data extracted from the Arase `orb_def` .cdf file.

    Returns:
        dict[str, npt.NDArray[NumpyNumeric | np.datetime64]]: A dictionary with keys and values drectly correspond to those in the variables and data in the OSS L2 Arase Perigee product.
            The keys of the output dictionary are ["Arase_ECEF_pos", "Arase_LLA"].
    """
    # Unpack arase latitude, longitude and radius.
    lat, lon, rad = arase_data_orb_def["pos_llr"][:, 0], arase_data_orb_def["pos_llr"][:, 1], arase_data_orb_def["pos_llr"][:, 2]
    # Arase definitive oribits are provided in spherical coordinates (not WGS84 LatLonAlt). We convert here:
    x_ = rad * np.cos(np.radians(lat)) * np.cos(np.radians(lon))
    y_ = rad * np.cos(np.radians(lat)) * np.sin(np.radians(lon))
    z_ = rad * np.sin(np.radians(lat))

    epoch = arase_data_orb_def["epoch"]
    # interpolate onto the oss_master_time grid. Fill with nans.
    x, y, z = [
        np.interp(
            convert_np_dt_to_float(oss_master_time),
            convert_np_dt_to_float(epoch),
            coord,
            left=np.nan,
            right=np.nan
            ) 
            for coord in [x_, y_, z_]
    ]
    arase_ecef_pos = np.stack([x, y, z], axis=1)
    
    # convert ECEF position to geodedic/WGS84 LLA
    lat_geod, lon_geod, alt_geod = ecef_to_geodetic_vectorized(arase_ecef_pos)
    arase_lla = np.stack([lat_geod, lon_geod, alt_geod], axis=1)

    # no convert arase_ecef_pos to km:
    arase_ecef_pos = arase_ecef_pos * 1000

    # Finally, extract generator `dx` from 
    dx = np.sqrt(np.sum(np.square(arase_ecef_pos[:, :, None] - oss_generator_ECEF_pos), axis=1))

    return {
        "Arase_ECEF_pos": arase_ecef_pos,
        "Arase_LLA": arase_lla,
        "dx": dx
    }

def populate_full_arase_urls(year: int, month: int, day: int) -> dict[str, str]:
    """
    Generate a dictionary of URLs for downloading the Arase data files that are relevent to our L2 dataset.

    Args:
        year (int): year of Arase data files.
        month (int): month of Arase data files.
        day (int): day of Arase data files.

    Returns:
        dict[str, str]: Keys give a brief description of the Arase data type, and values are the full urls to download each data file for that type.
    """
    out_urls = {}
    fmap = {
        "year": year,
        "month": month,
        "day": day,
    }
    for k, v in ARASE_EXTENDED_URLS.items():
        local_out_urls = []
        for iteration in ARASE_EXTENDED_URLS_VALID_ITERATIONS[k]:
            local_out_urls.append(v.format_map({**fmap, "iteration": iteration}))
        out_urls[k] = local_out_urls
    return out_urls

def get_arase_raw_data_local_filepaths(year: int, month: int, day: int) -> dict[str, str]:
    """
    Retrieve the local downloaded filepaths for the Arase data on a given day.

    Args:
        year (int): year of interest.
        month (int): month of interest.
        day (int): day of interest.

    Returns:
        dict[str, str]: A dictionary where each value corresponds to the key of the arase .cdf file at a given date. 
    """
    arase_urls = populate_full_arase_urls(year, month, day)
    local_filepaths = {}
    for k, v in arase_urls.items():
        local_filepaths[k] = [os.path.join(ARASE_DOWNLOAD_DIRECTORY, k, os.path.split(a_url)[1]) for a_url in arase_urls[k]]
    return local_filepaths

async def download_file(url: str, filepath: str, session: aiohttp.ClientSession, semaphore: asyncio.Semaphore, overwrite_existing: bool = False) -> None:
    """
    Downloads a file at the given url to the given filename. Leverages a semaphore and session object to download data asyncronously.

    Args:
        url (str): The download url.
        filepath (str): The name of the file to save the downloaded data as.
        session (aiohttp.ClientSession): The session manager for downloading data.
        semaphore (asyncio.Semaphore): The semaphore to limit concurrent downloads and manage the async context.
        overwrite_existing (bool, optional): Indicates if the data should overwrite an existing file at `filepath`. If False, will not download any data. Defaults to False.
    """
    if not overwrite_existing and os.path.exists(filepath):
        print(f"File {filepath} already exists, not overwriting.")
        return 200
    async with semaphore:
        print(f"starting to download {url}")
        save_dir, save_name = os.path.split(filepath)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        async with session.get(url) as response:
            if response.status == 200:
                with open(filepath, 'wb') as f:
                    f.write(await response.read())
                print(f"Downloaded {filepath}")
            else:
                print(f"Failed to download {filepath}:\n\tWith status code {response.status}")
            return response.status

async def download_first_available_file(urls: list[str], save_paths: list[str], session: aiohttp.ClientSession, semaphore: asyncio.Semaphore) -> None:
    """
    Downloads the first available file from the list of urls, saves them to the assocated save_path in `save_paths`.

    Args:
        urls (list[str]): A list of urls to download data from. The first file that can succesfully be downloaded is.
        save_paths (list[str]): A collecion of save paths to save the data to. Should be the same length as the `urls`, and the i-th entry is where the i-th url is to be saved locally.
        session (aiohttp.ClientSession): The download session manager.
        semaphore (asyncio.Semaphore): The download semaphore.
    """
    for url, save_path in zip(urls, save_paths):
        download_status = await download_file(url, save_path, session, semaphore)
        if download_status == 200:
            break

async def download_day_of_arase_raw_data(year: int, month: int, day: int, session: aiohttp.ClientSession, semaphore: asyncio.Semaphore) -> None:
    """
    Asyncronously download a day of arase data.

    Args:
        year (int): Year of interest.
        month (int): Month of interest.
        day (int): Day of interest.
        session (aiohttp.ClientSession): The session manager for downloading data.
        semaphore (asyncio.Semaphore): The semaphore to limit concurrent downloads and manage the async context.
    """
    urls = populate_full_arase_urls(int(year), int(month), int(day))
    save_paths = get_arase_raw_data_local_filepaths(year, month, day)
    coros = []
    for dataset, url_list in urls.items():
        save_path_list = save_paths[dataset]
        coros.append(download_first_available_file(url_list, save_path_list, session, semaphore))
    await asyncio.gather(*coros)

def delete_day_of_arase_raw_data(dt: datetime.datetime) -> None:
    """
    Helper function to delete one day of locally downloaded Arase data.

    Args:
        dt (datetime.datetime): the date of data to delete.
    """
    year, month, day = dt.year, dt.month, dt.day
    save_paths = get_arase_raw_data_local_filepaths(year, month, day)
    for dset_name, save_path_list in save_paths.items():
        for save_path in save_path_list:
            try:
                os.remove(save_path)
                print(f"Deleted: {save_path}")
            except Exception as e:
                print(f"Could not delete {save_path} due to the Exception: {e}.")
    print(f"Deleted day of Arase raw data: {year}-{month}-{day}")


def daterange(date1: datetime.datetime, date2: datetime.datetime) -> list[datetime.datetime]:
    """
    Calculate the list of datetimes between date1 and date2, inclusive on both sides.

    Args:
        date1 (datetime.datetime): lower datetime in date range
        date2 (datetime.datetime): upper datetime for date range (inclusive)

    Returns:
        list[datetime.datetime]: list of datetimes between date1 and date2 (inclusive).
    """
    return [date1 + datetime.timedelta(days=x) for x in range((date2-date1).days + 1)]

async def download_days_of_Arase_data(year: int, first_month: int, first_day: int, second_month: int, second_day: int, session: aiohttp.ClientSession, semaphore):
    """
    Utilitiy function to download the days of arase data between two days in the same year. Only retrieves datasets that are relevant to the OSS L2 dataset.

    Args:
        year (int): Year of interst.
        first_month (int): Initial month.
        first_day (int): Initial day of month.
        second_month (int): Final month.
        second_day (int): Final day of month.
        session (_type_): _description_
        semaphore (_type_): _description_

    Returns:
        asyncio.Future[list[asyncio._T]]: _description_
    """
    datetimes_to_download = daterange(datetime.datetime(year, first_month, first_day), datetime.datetime(year, second_month, second_day))
    coroutines = []
    for dt in datetimes_to_download:
        coroutines.append(download_day_of_arase_raw_data(dt.year, dt.month, dt.day, session, semaphore))
    await asyncio.gather(*coroutines)

def get_date_time_from_oss_filename(file_name: str) -> datetime.datetime:
    """
    Retrieve the date and time from a L1_9 or L2 OSS dataset.

    Args:
        file_name (str): The file_name (or path) of the OSS dataset. Assumes the filename is of the form 

    Returns:
        datetime.datetime: the datetime object corresponding to the OSS filename.
    """
    date_str = file_name.split("_")[-1].strip(".nc")
    return datetime.datetime.strptime(date_str, "%Y%m%d%H%M%S")

class CDFReader:
    """
    A simple interface to safely asyncronously read a variable from a cdf file by implementing a lock on the dataset.
    """
    def __init__(self, file_path: str) -> None:
        """
        Initialize the reader object.

        Args:
            file_path (str): The .cdf of the file to be read in.
        """
        self.file_path = file_path
        self.cdf_file = cdflib.CDF(file_path)  # Open the file once
        self.lock = asyncio.Lock()  # Create a lock

    async def read_variable(self, variable_name: str) -> NumpyNumeric:
        """
        A mutex locked function for reading in a variable from a dataset.

        Args:
            variable_name (str): The name of the variable to read in.

        Returns:
            NumpyNumeric: the data from the .cdf file for the particular variable.
        """
        # Ensure that only one task can access this block of code at a time
        async with self.lock:
            data = await asyncio.to_thread(self.cdf_file.varget, variable_name)
            return data

class CDFManager:
    """
    A simple manager to interact with a collection of .cdf datasets. Simply maintains the reference to each CDFReader in a dictionary.
    """
    def __init__(self, file_paths):
        self.cdf_readers = {
            fp: CDFReader(fp) for fp in file_paths
        }

    def get_cdf_reader(self, file_path: str) -> CDFReader:
        """
        Return an associated CDFReader based on file path. If there is not existing referecnce to the reader, create one. (potentially unsafe if two readers are )

        Args:
            file_path (str): The .cdf of the file to be read in.

        Returns:
            CDFReader: The reader to safely asynchrounously read in CDF variables.
        """
        if file_path in self.cdf_readers.keys():
            return self.cdf_readers[file_path]
        else:
            self.cdf_readers[file_path]  = CDFReader(file_path)
            return self.cdf_readers[file_path]
        
    def remove_cdf_reader(self, file_path) -> None:
        del self.cdf_readers[file_path]

async def load_arase_data(dt_s: datetime.datetime | list[datetime.datetime], cdf_manager: CDFManager) -> dict[str, dict[str, npt.NDArray[NumpyNumeric | np.datetime64]]]:
    """
    Read in a continuous collection of arase data. Accounts for the possibility of different "iterations" of Arase data.

    Args:
        dt_s (datetime.datetime | list[datetime.datetime]): If single day, load in all the Arase data for that day. If list, assumes the list is an ordered list of consecutive days, then concatenates all the arase data into a single dinctionary.
        cdf_manager (CDFManager): the CDFManager that allows for asynchronous acces to the Arase data.

    Returns:
        dict[str, dict[str, npt.NDArray[NumpyNumeric | np.datetime64]]]: The dictionary of all Arase data.
    """
    if isinstance(dt_s, list):
        # simple method to concatentate multiple days of data, assumes the dt_s are in order. (typically only used for two consecutive days of data)
        days_of_data = []
        for dt in dt_s:
            days_of_data.append(await load_arase_data(dt, cdf_manager))
        output_data = {}
        for k in days_of_data[0].keys():
            output_data[k] = concatenate_arase_numpy_dicts([day_of_data[k] for day_of_data in days_of_data])
        return output_data
        
    arase_paths = get_arase_raw_data_local_filepaths(dt_s.year, dt_s.month, dt_s.day)
    data = {}
    for dataset_name, file_path_list in arase_paths.items():
        file_path = None
        for test_file_path in file_path_list:
            if os.path.exists(test_file_path):
                file_path = test_file_path
                break
        var_data = {}
        vars = ARASE_DATA_VAR_NAMES[dataset_name]
        cdf_reader = cdf_manager.get_cdf_reader(file_path)
        for var in vars:
            var_data[var] = await cdf_reader.read_variable(var)
            if var in ARASE_OSS_PREPROCESS_FUNCS.keys():
                var_data[var] = ARASE_OSS_PREPROCESS_FUNCS[var](var_data[var])
        data[dataset_name] = var_data
    return data

async def add_arase_data_to_l1_9_file(l1_9_file_path: str, cdf_manager: CDFManager) -> None:
    """
    Given an L1.9 OSS file, load the arase data, from local memory, and populate the L1.9 dataset with that data. If successful, renames the l1.9 file as an L2 file and moves to the L2 directory.

    Args:
        l1_9_file_path (str): The L1.9 file to populate with Arase data.s
        cdf_manager (CDFManager): The CDFManager that allows for asynchronous access to Arase data.
    """
    dt = get_date_time_from_oss_filename(l1_9_file_path)
    dt_next_day = dt + datetime.timedelta(days=1)
    try:
        with netCDF4.Dataset(l1_9_file_path, "r+") as l1_9_dataset:
            master_time = np.array(l1_9_dataset['master_time']).astype(np.datetime64)
            generator_ECEF_pos = np.array(l1_9_dataset["generator_ECEF_pos"])
            min_time = master_time.min().astype(datetime.datetime)
            max_time = master_time.max().astype(datetime.datetime)

            if max_time.day == min_time.day:
                arase_data = await load_arase_data(dt, cdf_manager)
            else:
                arase_data = await load_arase_data([dt, dt_next_day], cdf_manager)
            
            orb_def_remapped_data = get_OSS_L2_data_from_arase_dict_orb_def(master_time, generator_ECEF_pos, arase_data["orb_def"])
            l2_complex_remapped_data = get_OSS_L2_data_from_arase_dict_l2_complex(master_time, arase_data["l2_complex"])
            l2_spec_remapped_data = get_OSS_L2_data_from_arase_dict_l2_spec(master_time, arase_data["l2_spec"])
            hfa_l3_remapped_data = get_OSS_L2_data_from_arase_dict_hfa_l3(master_time, arase_data["hfa_l3"])
            l3_property_remapped_data = get_OSS_L2_data_from_arase_dict_l3_property(master_time, arase_data["l3_property"])

            for var, l2_data in chain(
                orb_def_remapped_data.items(), 
                l2_complex_remapped_data.items(), 
                l2_spec_remapped_data.items(), 
                hfa_l3_remapped_data.items(), 
                l3_property_remapped_data.items()
            ):
                # print(f"writing var to L1_9: {var}")
                l1_9_dataset[var][:] = l2_data.astype(l1_9_dataset[var].dtype)

        dt_map = {k: getattr(dt, k) for k in ["year", "month", "day", "hour", "minute", "second"]}
        l2_file_path = os.path.join(ARASE_L2_SAVE_DIRECTORY, LEVEL_2_FILENAME_TEMPLATE.format_map(dt_map))
        os.rename(l1_9_file_path, l2_file_path)
    except Exception as e:
        print(f"Failed to add data to the file {l1_9_file_path} due to the exception:\n\t{e}")
        """
     

    Args:
        current_day (_type_): 
        l1_9_files (_type_): 
        session (_type_): 
        semaphore (_type_): 
        cdf_manager (CDFManager): 
        outer_semaphore (_type_): _description_
    """

async def handle_one_day_of_l1_9_to_l2_files(
        current_day: datetime.datetime, 
        l1_9_files: list[str], 
        session: aiohttp.ClientSession, 
        semaphore: asyncio.Semaphore, 
        cdf_manager: CDFManager, 
        outer_semaphore: asyncio.Semaphore
    ) -> None:
    """
    Helper function to populate all L1.9 files for one day. Allows for the simple workflow:
        Download day of Arase data -> populate L1.9 files with Arase data -> delete day of Arase data (with exceptions for files that strech through the night).

    Args:
        current_day (datetime.datetime): The day of L1.9 files to populate.
        l1_9_files (list[str]): The collection of L1.9 files for the day of interest.
        session (aiohttp.ClientSession): The session to manage downloading data.
        semaphore (asyncio.Semaphore): The semaphore that manages Asynchronous downloads of Arase data.
        cdf_manager (CDFManager): The CDFManager that manages asynchronous access to Arase files.
        outer_semaphore (asyncio.Semaphore): A semaphore that manages the number of days that are populated at one time. Helps to maintain order amongst the days so that data is not downloaded for all dayas at once.
    """
    async with outer_semaphore:
        print(f"\n\nStarting day: {current_day}")
        print(f"Starting download for day: {current_day}")
        next_day = current_day + datetime.timedelta(days=1)
        await asyncio.gather(
            download_day_of_arase_raw_data(current_day.year, current_day.month, current_day.day, session=session, semaphore=semaphore), 
            download_day_of_arase_raw_data(next_day.year, next_day.month, next_day.day, session=session, semaphore=semaphore)
        )
        print(f"Finished download for day: {current_day}")
        print(f"Starting L2 file writing for day: {current_day}")
        data_coros = [add_arase_data_to_l1_9_file(l1_9_f, cdf_manager=cdf_manager) for l1_9_f in l1_9_files]
        print(f"Finished L2 file writing for day: {current_day}")
        await asyncio.gather(*data_coros)

async def add_arase_data_to_all_l1_9_files(
        session: aiohttp.ClientSession, 
        semaphore: asyncio.Semaphore, 
        cdf_manager: CDFManager, 
        delete_raw_arase_data_when_done: bool = True
    ) -> None:
    """
    Populates all arase files in the L1.9 Directory.

    Args:
        session (aiohttp.ClientSession): The data downloading session.
        semaphore (asyncio.Semaphore): the semaphore to manage async downloads.
        cdf_manager (CDFManager): local async data access manager.
        delete_raw_arase_data_when_done (bool, optional): Indicator to delete Arase data when finished. If True, removes each day of day of Arase .cdf files once they are not needed, and all lingering Arase .cdf files at the end. If False, removes no Arase data, resulting in ~50GB of necessary local storage. Defaults to True.
    """
    outer_semaphore = asyncio.Semaphore(5)
    l1_9_files = glob.glob(os.path.join(ARASE_L1_9_SAVE_DIRECTORY, "**.nc"), recursive=True)
    dts = [get_date_time_from_oss_filename(f).date() for f in l1_9_files]
    l1_9_files_by_date = {dt: [] for dt in set(dts)}
    for dt, l1_9_f in zip(dts, l1_9_files):
        l1_9_files_by_date[dt].append(l1_9_f)
    all_tasks = []
    for dt in sorted(list(set(dts))):
        task = asyncio.create_task(handle_one_day_of_l1_9_to_l2_files(dt, l1_9_files_by_date[dt], session, semaphore, cdf_manager, outer_semaphore))
        if delete_raw_arase_data_when_done:
            task.add_done_callback(functools.partial(lambda d, x: delete_day_of_arase_raw_data(d), dt))
        all_tasks.append(task)
    await asyncio.gather(*all_tasks)
    print("\n\nFinised, just wrapping things up.\n\n")
    if delete_raw_arase_data_when_done:
        # remove any lingering files that may have gotten lost.
        fpath_match = os.path.join(ARASE_DOWNLOAD_DIRECTORY, "*/*.cdf")
        for file in glob.glob(fpath_match, recursive=True):
            os.remove(file)
            print(f"Deleted: {file}")
    
async def main(download_semaphore: asyncio.Semaphore = asyncio.Semaphore(9)) -> None:
    """
    Main loop to create all L2 files.

    Args:
        download_semaphore (asyncio.Semaphore, optional): Async donwload manager. Defaults to asyncio.Semaphore(9).
    """
    cdf_manager = CDFManager([])
    #session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(verify_ssl=False))
    # await add_arase_data_to_l1_9_file("data/arase_L1_9/OSSLevel1_9_Arase_perigee_20180429040254.nc", cdf_manager)
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(verify_ssl=False)) as session:
        await add_arase_data_to_all_l1_9_files(session, download_semaphore, cdf_manager)

if __name__ == "__main__":
    asyncio.run(main())
