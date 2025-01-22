#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from netCDF4 import Dataset
import os
import itertools
import datetime

def load_minutes(d, nc_var, t0=None):
    '''Load datetime64 from level2 netCDF file and covert to minutes relative to t0'''
    t_dt64 = np.array(d[nc_var]).astype('datetime64[ms]')
    
    # If relative to itself
    if t0 is None:
        t0 = t_dt64[0]

    t_minutes = (t_dt64 - t0)/np.timedelta64(1000, 'ms')/60
    return t_minutes, t0


def spectral_time_middle(t):
    '''Convert spectral timestamps, defined as the start of each FFT window, to the middle of the window'''
    if len(t)==0:
        return t
    dt = np.min(np.diff(t))
    return t + dt/2


def load_data_for_stats(files, EorB, meas):
    '''
    This function loads variables of interest for statistical analysis from the OSS Level 2 data for Arase.
    Only one observation type (E-field or B-field; and OFA wave, SPEC, or COMPLEX) is loaded. All time
    series variables are interpolated to the timescale of the chosen observation.
    
    DIMENSIONS:
        nt: number of total times
        ng: number of generators
            Note that a given generator index does not consistently refer to the same generator (i.e., NORAD ID).
            Please use data['NORAD_ID'] if trying to filter for specific generators.
        nf: number of frequencies in spectral measurement
        nflag: number of quality flags (32)
    
    INPUTS: 
        files: A list of full filepaths for the Level 2 netCDF files to be loaded
        EorB: Must be 'E' or 'B' to specify which observation type to load
        meas: Must be 'wave', 'COMPLEX', or 'SPEC' to specify with data type to load
    
    OUTPUTS:
        data: a dictionary containing the following numpy arrays...
        data['filename']: Level 2 netCDF filename used for the data sample, shape (nt,)
        data['t']: Time of Arase measurements [minutes from start of perigee], shape (nt,)
        data['f']: Frequencies in Arase spectrogram [kHz], shape (nf,)
        data['flag']: Quality flag of Arase measurement, boolean array shape (nt, nflag)
        data['spectra']: Arase spectrogram [same units as level 2], shape (nt,nf)
        data['lat']: latitude of Arase [deg N], shape (nt,)
        data['lon']: longitude of Arase [deg E], shape (nt,)
        data['alt']: altitude of Arase [km], shape (nt,)
        data['LT']: local time of Arase [hours], shape (nt,)
        data['Ne']: IRI electron density [electrons/m^3], shape (nt,)
        data['NORAD_ID']: NORAD ID of the generator, shape (nt,ng)
        data['RCS']: Radar cross-section of the generator [m^2], shape (nt,ng)
        data['dx']: Distance between Arase and generator [km], shape (nt,ng)
        data['mach_angle']: Angle of Arase from center of generator's mach cone [deg], shape (nt,ng)
        data['B_mag']: Magnetic field strength at Arase from IGRF [T], shape (nt,)
        data['B_angle']: Generator velocity angle w.r.t. magnetic field [deg], shape (nt,ng)
        data['B_perp']: Smallest distance from generator to Arase's field line [km], shape (nt,ng)
        data['B_par']: Generator position projected onto Arase's field line [km], shape (nt,ng)
        
    '''
    # Load data from all of the files
    master_to_obs_interp_vars = ['lat', 'lon', 'alt', 'LT', 'dx', 'mach_angle', 'B_par', 'B_perp', 'B_angle', 'B_mag']

    l2_list = []

    for i, file in enumerate(files):
        filename = os.path.basename(file)
        print(f"Working on {filename} (file {i+1:03d} of {len(files):03d})", end='\r')
        l2 = {}

        ### Load variables needed for this analysis
        d = Dataset(file)

        ### Load ancillary variables
        lla = np.array(d['Arase_LLA'])
        l2['lat'] = lla[:,0]
        l2['lon'] = lla[:,1]
        l2['alt'] = lla[:,2]

        ### Get master time info and local time
        master_t, t0 = load_minutes(d, 'master_time')
        t0_timestamp = t0.astype(str)
        hour = float(t0_timestamp[11:13])
        minute = float(t0_timestamp[14:16])
        second = float(t0_timestamp[17:19])
        UT_hour = (hour + minute/60 + second/60/60) + master_t/60
        l2['LT'] = np.mod(UT_hour + l2['lon']/15, 24)

        ### Get generator characteristics
        norad_id = np.array(d['NORAD_ID'])
        rcs = np.array(d['generator_RCS'])

        ### Get dx
        l2['dx'] = np.array(d['dx'])/1000

        ### Get mach cone angle
        rDG = np.array(d['generator_ECEF_pos']) - np.array(d['detector_ECEF_pos'])[:,:,None]
        vG = np.array(d['generator_ECEF_vel'])
        l2['mach_angle'] = np.degrees(np.arccos(np.sum(rDG*vG, axis=1)/(np.sqrt(np.sum(rDG**2, axis=1)) * 
                                                                        np.sqrt(np.sum(vG**2, axis=1)))))

        ### Get B_perp and B_parallel
        Bg, Bd, Rd, Rg = [np.array(d[x]) for x in ('generator_ECEF_B', 
                          'detector_ECEF_B', 'Arase_ECEF_pos', 'generator_ECEF_pos')]
        R = (Rd[:,:,None] - Rg)/1000
        bg = np.sqrt(np.sum(Bg**2, axis = 1))
        bd = np.sqrt(np.sum(Bd**2, axis = 1))
        rg = np.sqrt(np.sum(Rg**2, axis = 1))
        rd = np.sqrt(np.sum(Rd**2, axis = 1))
        r = np.sqrt(np.sum(R**2, axis = 1))
        Bdu = Bd/bd[:,None]
        Bgu = Bg/bg[:,None,:]

        l2['B_par'] = np.sum(R * Bdu[:,:,None], axis = 1)
        l2['B_perp'] = r - np.abs(l2['B_par'])

        ### Get generator velocity angle w.r.t. B
        vg = np.array(d['generator_ECEF_vel']) 
        vgu = vg/np.sqrt(np.sum(vg**2, axis=1))[:,None,:]
        dp = np.sum(Bgu*vgu, axis=1)
        ang = np.degrees(np.arccos(dp))
        ang[ang>90] = 180 - ang[ang>90]
        l2['B_angle'] = ang

        ### Also store magnitude of detector B-field (converted from nT to T) for possible computation of plasma frequencies
        l2['B_mag'] = bd/10**9

        ### Get electron densities
        # Just IRI, users can add HFA-derived electron density if they really want it. But it's pretty gappy in time.
        t_iri = load_minutes(d, f'IRI_time', t0=t0)[0]
        ne = np.array(d['Ne_IRI'])

        ### Load Arase observation vars
        l2['spectra'] = np.array(d[f'{EorB}_OFA{meas}'])
        f = np.array(d[f'{EorB}freq'])
        l2['t'] = spectral_time_middle(load_minutes(d, f'{EorB}time_OFA{meas}', t0=t0)[0])
        flag_mask = np.array(d[f'{EorB}flag_OFA{meas}']).astype(int)

        d.close()

        nt = len(l2['t'])
        if len(master_t)==0 or nt==0:
            continue

        ### Deal with the flag bitmask
        # Get the Boolean flag array of shape (n_times, n_flags), where n_flags=32. This tells you which flags are flipped on at which time.
        # The indices of the flag dimension match the table here: https://ergsc.isee.nagoya-u.ac.jp/mw/index.php/ErgSat/Pwe/Ofa
        # This is a vectorized version of this solution: https://stackoverflow.com/questions/37580272/numpy-boolean-array-representation-of-an-integer
        l2['flag'] = (flag_mask[:,None] & (1 << np.arange(32)[None,:])) > 0

        ### Interpolate everything that's on master time to the time scale of the observation
        for var in master_to_obs_interp_vars:
            interpolator = interp1d(master_t, l2[var], axis=0, bounds_error=False, fill_value=np.nan)
            l2[var] = interpolator(l2['t'])

        ### Interpolate IRI electron densities onto the time scale of the observation
        interpolator = interp1d(t_iri, ne, axis=0, bounds_error=False, fill_value=np.nan)
        l2['Ne'] = interpolator(l2['t'])

        ### Add NORAD ID and RCS, inflated to (nt,ng)
        # NORAD ID needs to be converted to a float to allow padding along generator dimension with NaNs later on
        l2['NORAD_ID'] = np.repeat(norad_id[None,:], nt, axis=0).astype(float)
        l2['RCS'] = np.repeat(rcs[None,:], nt, axis=0)

        ### Add filename, inflated to (nt,)
        l2['filename'] = np.full((nt,), filename)

        l2_list.append(l2)

    # Get the largest generator dimension
    ng_values = np.array([l2['dx'].shape[1] for l2 in l2_list])
    ng_max = np.max(ng_values)

    # Inflate all data with generator dimension to ng_max so we can concatenate
    for i, l2 in enumerate(l2_list):
        for var in l2.keys():
            if l2[var].ndim==2 and var not in ['spectra','flag']:
                l2[var] = np.pad(l2[var], ((0,0),(0,ng_max-ng_values[i])), mode='constant', constant_values=np.nan)

    # Concatenate everything together!
    data = {key:np.concatenate([l2_list[i][key] for i in range(len(l2_list))], axis=0) for key in l2_list[0].keys()}
    data['f'] = f

    print("\nDone!")
    return data


def filter_data(data, score, acceptable_flags=[], other_filter=None):
    '''
    This function filters all data and the chosen score along time in three ways:
    
    1. Any samples that have NaN score are removed
    2. Any samples that are flagged (ignoring user-specified "acceptable_flags" are removed
    3. The user can apply additional filtering with boolean array "other_filter" (True=keep the sample)
    
    INPUTS:
        data: The dictionary returned from load_data_for_stats()
        score: User-defined score, shape (nt,)
        acceptable_flags: List of acceptable flags (as integers)
            See https://ergsc.isee.nagoya-u.ac.jp/mw/index.php/ErgSat/Pwe/Wfc
            For example, acceptable_flags=[4,5] will allow "eclipse" and "magnetorquer operated"
        other_filter: User-specified boolean filter, shape (nt,)
    
    OUTPUTS:
        data: The dictionary where all time series data have been filtered
        score: The filtered score, shape (nt_new,)
        
    '''
    
    unacceptable_flags = list(set(np.arange(32)) - set(acceptable_flags))
    no_unacceptable_flags = ~np.any(data['flag'][:,unacceptable_flags], axis=1)
    
    no_nans = ~np.isnan(score)
    
    full_filter = (no_unacceptable_flags & no_nans)
    
    if other_filter is not None:
        full_filter = (full_filter & other_filter)
    
    # Filter the score
    score = score[full_filter]
    
    # Filter all of the time series data
    for var in data.keys():
        if (var!='f'):
            data[var] = data[var][full_filter]
    
    return data, score


def pick_populations_v2_1(
        data,
        score,
        max_population_sizes, 
        lvs=('lat', 'alt'),
        num=50,
        score_experiment_cutoff=None,
        score_control_cutoff=None,
        perigee_overlap = -1,
        experiment_candidates=None,
        control_candidates=None,
        return_match_indxs=False,
    ):
    '''
    This function returns the indices of two populations: the exeriment and 
    the control. The experimental population has LOW values of the score, and 
    the control has HIGH values of score. Additionally, this function attempts 
    to mirror the distributions of the latent variables (lvs) between these two 
    populations

    This function is an altered version of `pick_populations` that attempts to 
    maximize the distance between the two populations.

    This function is updated from pick_populations_v2 to allow for upper and lower cuttofs in score.
    
    DIMENSIONS:
        nt: total times
        ng: number of generators
    
    INPUTS: 
        dic: a dictionary with arrays of size [nt] with names matching lvs
        score: a size [nt] array with numerical values for the experimental variable
        max_populations_sizes: maximum number of samples in the two poluations
        lvs: tuple or list of names for the latent variables. any length
        num: argument passed to `bin` within the np.histogrammdd splitting of latent variables
        score_experiment_cutoff: a float cutoff score value for the experiment distribution
        score_control_cutoff: a float cuttoff score value for the control distribution
        perigee_overlap: integer number of overlapping perigee files allowed to pair the distriubtions
        control_candidates: a size [nt] boolean array to pre-select samples that are possibly selected within the control distribution. If None, then all samples are considered.
        experiment_candidates: a size [nt] boolean array to pre-select samples that are possibly selected within the experiment distrubution. If None, then all samples are considered.
        return_match_indxs: Boolean value to indicate if the indices of paired samples should be returned to the user

    
    OUTPUTS:
        experiment: a size [nt] boolean array to filter for experiment population
        control: a size [nt] boolean array to filter for control population
    
    
    '''
    nt = len(score)
    nlv = len(lvs)
    lv = np.vstack([data[l] for l in lvs]).T
    

    #
    timestamp = []
    f_t = [] 
    for i in range(len(data['filename'])):
        f_t.append(datetime.datetime.strptime(data['filename'][i], 'OSSLevel2_Arase_perigee_%Y%m%d%H%M%S.nc'))
        timestamp.append(datetime.datetime.strptime(data['filename'][i], 'OSSLevel2_Arase_perigee_%Y%m%d%H%M%S.nc') + datetime.timedelta(minutes=data['t'][i]))
    timestamp = np.array(timestamp)
    vtm = np.vectorize(datetime.timedelta.total_seconds) 
    # data_total_minutes = vtm(timestamp - timestamp.min()) / 60
    data_perigee_start_minute = vtm(np.array(f_t) - f_t[0]) / 60
    perigee_minutes_diff = 11 * 60
    if score_experiment_cutoff is None:
        score_experiment_cutoff = np.nanquantile(score, .1)
    if score_control_cutoff is None:
        score_control_cutoff = np.nanquantile(score, .9)
    perigee = data_perigee_start_minute
    if experiment_candidates is None:
        experiment_candidates = np.ones(nt, dtype=bool)
    if control_candidates is None:
        control_candidates = np.ones(nt, dtype=bool)
        
    experiment_candidates = (score <= score_experiment_cutoff) & experiment_candidates
    control_candidates = (score >= score_control_cutoff) & control_candidates

    u_perigees = np.unique(perigee)
    overlapping_perigees = []
    if perigee_overlap >= 0:
        for i in range(len(u_perigees)):
            ops = []
            for j in range(i, len(u_perigees)):
                if abs(u_perigees[i] - u_perigees[j]) <= ((perigee_overlap - 1) * perigee_minutes_diff):
                    ops.append(u_perigees[j])
            overlapping_perigees.append(ops)
    else:
        overlapping_perigees = [None]

    index_pairs = []
    index_pairs_score_distance = []


    #histogram of entire population
    counts, edges = np.histogramdd(lv, bins=num)
    #trick to avoid having to explicitly loop over all lvs
    inds_temp = [range(lv_bin_size) for lv_bin_size in counts.shape]
    inds = itertools.product(*inds_temp)
    for pvals in overlapping_perigees:
        if perigee_overlap >= 0:
            p_bool = np.zeros_like(score, dtype=bool)
            for pval in pvals:
                p_bool = p_bool | (perigee == pval)
        else:
            p_bool = np.ones_like(score, dtype=bool)
        p_experiment = (p_bool & experiment_candidates)
        p_control = (p_bool & control_candidates)

        for ii in inds:
            cii = int(counts[ii])
            if cii < 2:
                continue

            in_bin = np.ones(nt, dtype = bool)
            for k in range(nlv):
                ek = edges[k]
                in_bin = in_bin & (ek[ii[k]] <= lv[:, k]) & (lv[:, k] < ek[ii[k]+1])
            
            in_bin_control_candidates    = in_bin & p_control
            in_bin_experiment_candidates = in_bin & p_experiment
            in_bins_control = np.nonzero(in_bin_control_candidates)[0]
            in_bins_experiment = np.nonzero(in_bin_experiment_candidates)[0]

            #scores in this bin
            scoreii_control = score[in_bin_control_candidates]
            scoreii_experiment = score[in_bin_experiment_candidates]

            control_inds_sorted = np.argsort(scoreii_control)
            experiment_inds_sorted = np.argsort(scoreii_experiment)

            max_inds = max(len(control_inds_sorted), len(experiment_inds_sorted))
            for control_i, experiment_i in zip(control_inds_sorted[:max_inds], experiment_inds_sorted[:max_inds]):
                score_diff = scoreii_control[control_i] - scoreii_experiment[experiment_i]
                true_control_i = in_bins_control[control_i]
                true_experiment_i = in_bins_experiment[experiment_i]

                index_pairs.append([true_experiment_i, true_control_i])
                index_pairs_score_distance.append([score_diff])

    global_score_diff = np.array(index_pairs_score_distance).squeeze()
    global_e_inds = np.array([ei for [ei, _] in index_pairs], dtype=np.int64).squeeze()
    global_c_inds = np.array([ci for [_, ci] in index_pairs], dtype=np.int64).squeeze()
    score_sorted_inds = np.argsort(global_score_diff)[::-1]
    global_score_diff = global_score_diff[score_sorted_inds]
    global_e_inds = global_e_inds[score_sorted_inds]
    global_c_inds = global_c_inds[score_sorted_inds]

    final_e_inds = []
    final_c_inds = []
    while (len(global_e_inds) != 0) and (len(global_c_inds) != 0) and len(final_e_inds) < max_population_sizes:
        e_i = global_e_inds[0]
        c_i = global_c_inds[0]
        final_e_inds.append(e_i)
        final_c_inds.append(c_i)
        e_mask = global_e_inds == e_i
        c_mask = global_c_inds == c_i
        mask = ~(e_mask | c_mask)
        global_e_inds = global_e_inds[mask]
        global_c_inds = global_c_inds[mask]
    experiment = np.zeros(nt, dtype=bool)
    control = np.zeros(nt, dtype=bool)
    experiment[final_e_inds] = 1
    control[final_c_inds] = 1
    if return_match_indxs:
        return experiment, control, final_e_inds, final_c_inds
    return experiment, control


def plot_population_summary(data, score, experiment, control, lvs, score_label, log_score_axis=False, save = None, show_plot=True):
    '''Plot distributions of control versus experiment over score and latent variables'''
    
    nlvs = len(lvs)
    fig, ax = plt.subplots(1, 1+nlvs, figsize=(3*(1+nlvs), 3))
    lv_label_map = {
        "LT": "Local Time [hr]",
        "lon": "Longitude [deg]",
        "lat": "Latitude [deg]",
        "file_index": "File Index []",
        "alt": "Altitude [km]"
    }
    
    ### First plot: control versus experiment versus all data for score
    if log_score_axis:
        ax[0].set_xscale('symlog')
        if np.nanmin(score) < 0 and np.nanmax(score) < 0:
            bins = -np.logspace(np.log10(-np.nanmax(score)), np.log10(-np.nanmin(score)), 100)[::-1]
        else: 
            bins = np.logspace(np.log10(np.nanmin(score)), np.log10(np.nanmax(score)), 100)
    else:
        bins = np.linspace(np.nanmin(score), np.nanmax(score), 100)
    ax[0].hist(score[control], bins=bins, label="Control", zorder=11, alpha=.8)
    ax[0].hist(score[experiment], bins=bins, label="Experiment", zorder=10, alpha=.8)
    ax[0].hist(score, bins=bins, label="All", color='gray', alpha=0.3)
    ax[0].legend()
    ax[0].set_ylabel("Counts")
    ax[0].set_xlabel(score_label)
    ax[0].set_title("Score")
    
    ### Rest of the plots: control versus experiment distribution across latent variables
    for i, lv in enumerate(lvs):
        bins = np.linspace(np.nanmin(data[lv]), np.nanmax(data[lv]), 50)
        ax[i+1].hist(data[lv][control], bins=bins, alpha=0.5, label="Control")
        ax[i+1].hist(data[lv][experiment], bins=bins, alpha=0.5, label="Experiment")
        ax[i+1].set_ylabel("Counts")
        ax[i+1].set_xlabel(lv_label_map[lv])
        ax[i+1].legend()
        ax[i+1].set_title(f"latent variable {i+1}: {lv}")
    
    fig.suptitle(f"Number of Samples: {experiment.sum()}")
 
    plt.tight_layout()

    if save is not None:
        plt.savefig(save, dpi=600, bbox_inches='tight', pad_inches=0.2, transparent=False)
    
    if show_plot:
        plt.show()
    
    plt.close()
    
def plot_population_lv_joint_distributions(data, experiment, control, lvs):
    
    nlvs = len(lvs)
    
    for lv_1 in range(nlvs):
        bins_1 = np.linspace(np.nanmin(data[lvs[lv_1]]), np.nanmax(data[lvs[lv_1]]), 20)
        for lv_2 in range(lv_1 + 1, nlvs):
            bins_2 = np.linspace(np.nanmin(data[lvs[lv_2]]), np.nanmax(data[lvs[lv_2]]), 20)
            fig, axs = plt.subplots(1, 2, figsize = (6, 3), sharey=True, sharex=True)
            h0 = axs[0].hist2d(data[lvs[lv_1]][control], data[lvs[lv_2]][control], label="Control", bins=[bins_1, bins_2])
            h1 = axs[1].hist2d(data[lvs[lv_1]][experiment], data[lvs[lv_2]][experiment], label="Experiment", bins=[bins_1, bins_2])
            axs[0].set_xlabel(f"{lvs[lv_1]}")
            axs[0].set_ylabel(f"{lvs[lv_2]}")
            axs[1].set_xlabel(f"{lvs[lv_1]}")
            axs[1].set_ylabel(f"{lvs[lv_2]}")
            axs[0].set_title("Control")
            axs[1].set_title("Experiment")
            fig.suptitle("Latent Variable Joint Distribution")
            plt.colorbar(h0[3], ax=axs[0])
            plt.colorbar(h1[3], ax=axs[1])
            fig.tight_layout()
            plt.show()
    plt.close()
    return


def plot_spectrohistogram_comparison(experiment, control, obs, f, title, log_bin_edges=np.arange(-4, 4.01, 0.25)):
    '''Plot "spectrohistograms" for experiment, control, and the ratio between them'''
    
    log_obs = np.log10(obs)
    
    nf = len(f)
    df_low = f[1] - f[0]
    df_high = f[-1] - f[-2]
    f_edges = np.concatenate(([f[0] - df_low], 0.5*(f[:-1] + f[1:]), [f[-1] + df_high]))

    experiment_counts = np.zeros([nf, log_bin_edges.size-1])
    control_counts = np.zeros([nf, log_bin_edges.size-1])
    control_median = np.zeros(nf)
    experiment_median = 0 * control_median
    
    for i in range(nf):
        experiment_counts[i,:], _ = np.histogram(log_obs[experiment,i], bins=log_bin_edges)
        control_counts[i,:], _ = np.histogram(log_obs[control,i], bins=log_bin_edges)
        control_median[i] = np.nanmedian(log_obs[control,i])
        experiment_median[i] = np.nanmedian(log_obs[experiment,i])
        
    experiment_counts[experiment_counts == 0.] = np.nan
    control_counts[control_counts == 0.] = np.nan
    
    fig, axs = plt.subplots(3,2, sharex='col', gridspec_kw={'width_ratios':[100, 2]},
                          figsize = (6,8))
    (ax1, ax2, ax3) = axs[:, 0]
    (cax1, cax2, cax3) = axs[:,1]
    for a in (ax1, ax2, ax3): a.grid()
    
    vmax = np.max([3*np.nanstd(experiment_counts), 3*np.nanstd(control_counts)])
    kwd = {'cmap':'inferno', 'vmax':vmax}
    
    im = ax1.pcolormesh(f_edges, log_bin_edges, experiment_counts.T, **kwd)
    ax1.plot(f, experiment_median, 'k--', label='Experiment Median')
    cb = plt.colorbar(im, cax=cax1, extend='max')
    cb.set_label('Counts []')
    ax1.set_title(f'Experiment ({experiment.sum()} spectra)')
    ax1.legend(loc = 'upper left')
    
    im = ax2.pcolormesh(f_edges, log_bin_edges, control_counts.T, **kwd)
    ax2.plot(f, control_median, 'k--', label='Control Median')
    cb = plt.colorbar(im, cax=cax2, extend='max')
    cb.set_label('Counts []')
    ax2.set_title(f'Control ({control.sum()} spectra)')
    ax2.legend(loc = 'upper left')
    
    rat = experiment_counts.T/control_counts.T
    mincts = np.minimum(experiment_counts.T, control_counts.T)
    rat[mincts <= 5] = np.nan
    
    im = ax3.pcolormesh(f_edges, log_bin_edges, rat, cmap = 'bwr', vmin=0.25, vmax=1.75)
    ax3.plot(f, control_median, 'k--', label='Control Median')
    
    cb = plt.colorbar(im, cax=cax3, extend='both')
    cb.set_label('Counts Top/Counts Bottom []')
    ax3.set_title(f'Ratio (experiment/control)')
    ax3.legend(loc = 'upper left')
    
    ax3.set_xscale('log')
    ax3.set_xlim([0.5, 20])
    ax3.set_xticks([1, 2, 4, 8, 16], labels = ['1', '2', '4', '8', '16'])
    ax3.set_xlabel('Frequency [kHz]')
    fig.supylabel('log$_{10}$(Power [mv$^2$/m$^2$/Hz])')
    fig.suptitle(title)
    
    plt.tight_layout()
    plt.show()
    plt.close()
    return

def plot_spectrohistogram_and_ks(experiment, control, obs, f, title, pvalue_hypothesis, fmin=0.5, log_bin_edges=np.arange(-4, 4.01, 0.25), min_for_comparison=5, EorB='E', save=None, show_plot=True):
    '''Plot "spectrohistograms" for experiment, control, and the ratio between them'''
    
    log_obs = np.log10(obs)
    
    nf = len(f)
    df_low = f[1] - f[0]
    df_high = f[-1] - f[-2]
    f_edges = np.concatenate(([f[0] - df_low], 0.5*(f[:-1] + f[1:]), [f[-1] + df_high]))

    experiment_counts = np.zeros([nf, log_bin_edges.size-1])
    control_counts = np.zeros([nf, log_bin_edges.size-1])
    control_median = np.zeros(nf)
    experiment_median = 0 * control_median
    
    for i in range(nf):
        experiment_counts[i,:], _ = np.histogram(log_obs[experiment,i], bins=log_bin_edges)
        control_counts[i,:], _ = np.histogram(log_obs[control,i], bins=log_bin_edges)
        control_median[i] = np.nanmedian(log_obs[control,i])
        experiment_median[i] = np.nanmedian(log_obs[experiment,i])
        
    experiment_counts[experiment_counts == 0.] = np.nan
    control_counts[control_counts == 0.] = np.nan
    
    fig, ((ax_e, cax_e, sp1, ax_r, cax_r), (ax_c, cax_c, sp2, ax_ks, cax_ks)) = plt.subplots(2,5, sharex='col', gridspec_kw={'width_ratios':[100, 2, 35, 100, 2]}, figsize = (10,5))
    fig.subplots_adjust(wspace=0.1)
    
    for ax in (sp1, sp2, cax_ks):
        ax.axis('off')
    
    for ax in (ax_e, ax_c, ax_r, ax_ks): 
        ax.grid()
        ax.set_xscale('log')
        ax.set_xlim([fmin, 20])
        ax.set_xticks([0.5, 1, 2, 4, 8, 16], labels = ['0.5', '1', '2', '4', '8', '16'])
        
    for ax in (ax_e, ax_c, ax_r):
        if EorB=='E':
            ax.set_ylabel('log$_{10}$(Power [mV$^2$/m$^2$/Hz])')
        elif EorB=='B':
            ax.set_ylabel('log$_{10}$(Power [nT$^2$/Hz])')
        else:
            print("EorB must be 'E' or 'B', of course!")
    
    for ax in (ax_c, ax_ks):
        ax.set_xlabel("Frequency [kHz]")
    
    vmax = np.max([3*np.nanstd(experiment_counts), 3*np.nanstd(control_counts)])
    kwd = {'cmap':'inferno', 'vmax':vmax}
    
    # Plot experiment population spectrohistogram
    im = ax_e.pcolormesh(f_edges, log_bin_edges, experiment_counts.T, **kwd)
    ax_e.plot(f, experiment_median, 'k--', label='Experiment Median')
    cb = plt.colorbar(im, cax=cax_e, extend='max')
    cb.set_label('Counts []')
    ax_e.set_title(f'Experiment ({experiment.sum()} spectra)')
    ax_e.legend(loc = 'upper left')
    
    # Plot control population spectrohistogram
    im = ax_c.pcolormesh(f_edges, log_bin_edges, control_counts.T, **kwd)
    ax_c.plot(f, control_median, 'k--', label='Control Median')
    cb = plt.colorbar(im, cax=cax_c, extend='max')
    cb.set_label('Counts []')
    ax_c.set_title(f'Control ({control.sum()} spectra)')
    ax_c.legend(loc = 'upper left')
    
    # Plot ratio of experiment and control
    rat = experiment_counts.T/control_counts.T
    mincts = np.minimum(experiment_counts.T, control_counts.T)
    rat[mincts <= min_for_comparison] = np.nan
    
    im = ax_r.pcolormesh(f_edges, log_bin_edges, rat, cmap = 'bwr', vmin=0.25, vmax=1.75)
    ax_r.plot(f, control_median, 'k--', label='Control Median')
    cb = plt.colorbar(im, cax=cax_r, extend='both')
    cb.set_label('Counts ratio []')
    ax_r.set_title(f'Ratio (experiment/control)')
    ax_r.legend(loc='upper left')
    
    # Plot KS test results
    ax_ks.plot(f, pvalue_hypothesis, lw=2.0, color='red', zorder=3)
    ax_ks.axhline(0.01, linestyle='--', lw=2.0, color='black')
    ax_ks.set_yscale('log')
    ax_ks.set_ylabel('p value')
    ax_ks.set_ylim([1e-5, 1])
    ax_ks.set_title(f'KS test (p min={pvalue_hypothesis.min():0.2e})')
    
    fig.suptitle(title)
    
    if save is not None:
        plt.savefig(save, dpi=600, bbox_inches='tight', pad_inches=0.2, transparent=False)
    
    if show_plot:
        plt.show()
    plt.close()
    return

def plot_spectrohistogram_and_score(experiment, control, obs, f, title, score, s0, s1, pop_hist_ymax=None, fmin=0.5, log_bin_edges=np.arange(-4, 4.01, 0.25), min_for_comparison=5, EorB='E', save=None, show=False):
    '''Plot "spectrohistograms" for experiment, control, and the ratio between them'''
    
    log_obs = np.log10(obs)
    
    nf = len(f)
    df_low = f[1] - f[0]
    df_high = f[-1] - f[-2]
    f_edges = np.concatenate(([f[0] - df_low], 0.5*(f[:-1] + f[1:]), [f[-1] + df_high]))

    experiment_counts = np.zeros([nf, log_bin_edges.size-1])
    control_counts = np.zeros([nf, log_bin_edges.size-1])
    control_median = np.zeros(nf)
    experiment_median = 0 * control_median
    
    for i in range(nf):
        experiment_counts[i,:], _ = np.histogram(log_obs[experiment,i], bins=log_bin_edges)
        control_counts[i,:], _ = np.histogram(log_obs[control,i], bins=log_bin_edges)
        control_median[i] = np.nanmedian(log_obs[control,i])
        experiment_median[i] = np.nanmedian(log_obs[experiment,i])
        
    experiment_counts[experiment_counts == 0.] = np.nan
    control_counts[control_counts == 0.] = np.nan
    
    fig, ((ax_e, cax_e, sp1, ax_r, cax_r), (ax_c, cax_c, sp2, ax_s, cax_s)) = plt.subplots(2,5, gridspec_kw={'width_ratios':[100, 2, 35, 100, 2]}, figsize = (10,5))
    fig.subplots_adjust(wspace=0.1, hspace=0.6)
    
    for ax in (sp1, sp2, cax_s):
        ax.axis('off')
    
    for ax in (ax_e, ax_c, ax_r): 
        ax.grid()
        ax.set_xscale('log')
        ax.set_xlim([fmin, 20])
        ax.set_xticks([0.5, 1, 2, 4, 8, 16], labels = ['0.5', '1', '2', '4', '8', '16'])
        ax.set_xlabel("Frequency [kHz]")
        
    for ax in (ax_e, ax_c, ax_r):
        if EorB=='E':
            ax.set_ylabel('log$_{10}$(Power [mV$^2$/m$^2$/Hz])')
        elif EorB=='B':
            ax.set_ylabel('log$_{10}$(Power [nT$^2$/Hz])')
        else:
            print("EorB must be 'E' or 'B', of course!")
            
    vmax = np.max([3*np.nanstd(experiment_counts), 3*np.nanstd(control_counts)])
    kwd = {'cmap':'inferno', 'vmax':vmax}
    
    # Plot experiment population spectrohistogram
    im = ax_e.pcolormesh(f_edges, log_bin_edges, experiment_counts.T, **kwd)
    ax_e.plot(f, experiment_median, 'k--', label='Experiment Median')
    cb = plt.colorbar(im, cax=cax_e, extend='max')
    cb.set_label('Counts []')
    ax_e.set_title(f'Experiment ({experiment.sum()} spectra)')
    ax_e.legend(loc = 'upper left')
    
    # Plot control population spectrohistogram
    im = ax_c.pcolormesh(f_edges, log_bin_edges, control_counts.T, **kwd)
    ax_c.plot(f, control_median, 'k--', label='Control Median')
    cb = plt.colorbar(im, cax=cax_c, extend='max')
    cb.set_label('Counts []')
    ax_c.set_title(f'Control ({control.sum()} spectra)')
    ax_c.legend(loc = 'upper left')
    
    # Plot ratio of experiment and control
    rat = experiment_counts.T/control_counts.T
    mincts = np.minimum(experiment_counts.T, control_counts.T)
    rat[mincts <= min_for_comparison] = np.nan
    
    im = ax_r.pcolormesh(f_edges, log_bin_edges, rat, cmap = 'bwr', vmin=0.25, vmax=1.75)
    ax_r.plot(f, control_median, 'k--', label='Control Median')
    cb = plt.colorbar(im, cax=cax_r, extend='both')
    cb.set_label('Counts ratio []')
    ax_r.set_title(f'Ratio (experiment/control)')
    ax_r.legend(loc='upper left')
    
    # Plot KS test results
    ### First plot: control versus experiment versus all data for score
    bins = np.linspace(np.nanmin(score), np.nanmax(score), 100)
    ax_s.hist(score[control], bins=bins, label="Control", color='red', zorder=11)
    ax_s.hist(score[experiment], bins=bins, label="Exp.", color='green', zorder=10)
    ax_s.hist(score, bins=bins, label="All", color='gray', alpha=0.3)
    ax_s.axvline(s0, ymax=0.75, lw=1.0, color='black', linestyle='dashed', zorder=12)
    ax_s.axvline(s1, ymax=0.75, lw=1.0, color='black', linestyle='dashed', zorder=12)
    ax_s.legend(ncols=3, framealpha=0.7)
    ax_s.set_ylabel("Counts")
    ax_s.set_xlabel(title)
    ax_s.set_title("Score")
    
    if pop_hist_ymax is not None:
        ax_s.set_ylim([0.0, pop_hist_ymax])
    
    # fig.suptitle(title)
    
    # Label (a)-(d) for the paper
    for ax, label in zip([ax_e, ax_c, ax_r, ax_s], ['(a)', '(b)', '(c)', '(d)']):
        ax.text(0.0, 1.15, label, transform=ax.transAxes, ha='left', va='top', size=14, fontweight='bold')
    
    if save is not None:
        plt.savefig(save, dpi=600, bbox_inches='tight', pad_inches=0.2, transparent=False)
    if show:  
        plt.show()
    plt.close()
    
    return

def plot_ks_comparison(f, p_values, labels, monkey_p):
    fig, ax = plt.subplots(1, 1, figsize=(8,4))

    # Plot the monkeys (P value curves for random scores)
    for i in range(monkey_p.shape[0]):
        ax.plot(f, monkey_p[i,:], alpha=0.2, lw=0.8, color='black')
    
    for p, label in zip(p_values, labels):
        ax.plot(f, p, lw=2.0, label=label)
    
    ax.axhline(0.01, linestyle='--', lw=2.0, color='black')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim([0.5, 20])
    ax.set_xticks([1, 2, 4, 8, 16], labels = ['1', '2', '4', '8', '16'])
    ax.set_xlabel('Frequency [kHz]')
    ax.set_ylabel('p value')
    ax.legend(loc='lower right')

    plt.savefig("./statistical_experiments/KS_comparison.png", dpi=600, bbox_inches='tight', pad_inches=0.2, transparent=False)
    plt.close()

    return




### Legacy population pickers ###

def pick_populations(data, score, s0, lvs=('lat', 'alt'), num=50):
    '''
    This function returns the indices of two populations: the exeriment and 
    the control. The experimental population has LOW values of the score, and 
    the control has HIGH values of score. Additionally, this function attempts 
    to mirror the distributions of the latent variables (lvs) between these two 
    populations
    
    DIMENSIONS:
        nt: total times
        ng: number of generators
    
    INPUTS: 
        dic: a dictionary with arrays of size [nt] with names matching lvs
        score: a size [nt] array with numerical values for the experimental variable
        s0: cutoff score for the experimental population
        lvs: tuple or list of names for the latent variables. any length
        num: number of bins to use for histograms
    
    OUTPUTS:
        experiment: a size [nt] boolean array to filter for experiment population
        control: a size [nt] boolean array to filter for control population
    
    
    '''
    
    nt = len(score)
    nlv = len(lvs)
    lv = np.vstack([data[l] for l in lvs]).T

    #define our experimental population
    experiment = score < s0

    #preallocate our control poplulation
    control = np.zeros(nt, dtype=bool)

    #experimental latent variables
    lv_exp = lv[experiment, :]

    #histogram of the experimental latent variables
    counts, edges = np.histogramdd(lv_exp, bins=num)

    #R. Kelly's trick to avoid having to explicitly loop over all lvs
    inds_temp = [range(num) for _ in range(nlv)]
    inds = itertools.product(*inds_temp)

    for ii in inds:
        cij = int(counts[ii])
        if cij == 0: continue

        in_bin = np.ones(nt, dtype = bool)
        for k in range(nlv):
            ek = edges[k]
            in_bin = in_bin & (ek[ii[k]] <= lv[:, k]) & (lv[:, k] < ek[ii[k]+1])
        
        in_bins = np.nonzero(in_bin)[0]
        
        #scores in this bin
        scoreij = score[in_bin]
        
        #find the cij larges scores in this bin
        high_inds = np.argsort(scoreij)[-cij:]

        #get the inds that correspond to the main time
        inds = in_bins[high_inds]
        
        #now add these members to the control population
        control[inds] = True

    return experiment, control

def pick_populations_v2(data, score, population_sizes, lvs=('lat', 'alt'), num=50):
    '''
    This function returns the indices of two populations: the exeriment and 
    the control. The experimental population has LOW values of the score, and 
    the control has HIGH values of score. Additionally, this function attempts 
    to mirror the distributions of the latent variables (lvs) between these two 
    populations

    This function is an altered version of `pick_populations` that attempts to 
    maximize the distance between the two populations.
    
    DIMENSIONS:
        nt: total times
        ng: number of generators
    
    INPUTS: 
        dic: a dictionary with arrays of size [nt] with names matching lvs
        score: a size [nt] array with numerical values for the experimental variable
        populations_sizes: number of samples in the two poluations
        lvs: tuple or list of names for the latent variables. any length
        num: number of bins to use for histograms
    
    OUTPUTS:
        experiment: a size [nt] boolean array to filter for experiment population
        control: a size [nt] boolean array to filter for control population    
    '''
    nt = len(score)
    nlv = len(lvs)
    lv = np.vstack([data[l] for l in lvs]).T
    #histogram of entire population
    counts, edges = np.histogramdd(lv, bins=num)

    #preallocate control and experiment populations
    experiment = np.zeros(nt, dtype=bool)
    control = np.zeros(nt, dtype=bool)

    #initialize index_pairs: indices that are in ths same bins, but are far from each other in terms of score
    index_pairs = []
    index_pairs_score_distance = []

    #trick to avoid having to explicitly loop over all lvs
    inds_temp = [range(num) for _ in range(nlv)]
    inds = itertools.product(*inds_temp)

    for ii in inds:
        cii = int(counts[ii])
        if cii < 2:
            continue

        in_bin = np.ones(nt, dtype = bool)
        for k in range(nlv):
            ek = edges[k]
            in_bin = in_bin & (ek[ii[k]] <= lv[:, k]) & (lv[:, k] < ek[ii[k]+1])
        
        in_bins = np.nonzero(in_bin)[0]
        #scores in this bin
        scoreii = score[in_bin]

        #pair indices according to difference in score
        ind_scores_sorted = np.argsort(scoreii)
        for i in range(len(ind_scores_sorted) // 2):
            lower_i = ind_scores_sorted[i]
            upper_i = ind_scores_sorted[-i]

            diff = scoreii[upper_i] - scoreii[lower_i]
            outer_upper_i = in_bins[upper_i]
            outer_lower_i = in_bins[lower_i]

            index_pairs.append([outer_lower_i, outer_upper_i])
            index_pairs_score_distance.append(diff)

    index_pairs = np.array(index_pairs)
    index_pairs_score_distance = np.array(index_pairs_score_distance)

    sorted_score_diffs = np.argsort(index_pairs_score_distance)
    largest_diff_index_pairs = index_pairs[sorted_score_diffs[-population_sizes:]]

    experiment[largest_diff_index_pairs[:, 0]] = True
    control[largest_diff_index_pairs[:,1]] = True

    return experiment, control

def pick_populations_v3(
        data,
        score,
        score_experiment_cutoff=None,
        score_control_cutoff=None,
        perigee=None,
        perigee_overlap = 1,
        lvs=('lat', 'alt'),
        lvs_scale_parameters=None,
        lv_distance_cutoff = .01,
        experiment_candidates=None,
        control_candidates=None,
        return_match_indxs=False,
    ):
    '''
    This function returns the indices of two populations: the exeriment and 
    the control. The experimental population has LOW values of the score, and 
    the control has HIGH values of score. Additionally, this function attempts 
    to mirror the distributions of the latent variables (lvs) between these two 
    populations.

    This function is an altered version of `pick_populations_v2` that attempts to 
    maximize the distance between the two populations while enforcing score
    cutoffs and relaxing/rethinking the latent distribution symmetrization 
    process.
    
    DIMENSIONS:
        nt: total times
        ng: number of generators
    
    INPUTS: 
        data: a dictionary with arrays of size [nt] with names matching lvs
        score: a size [nt] array with numerical values for the experimental variable
        score_experiment_cutoff: maximum value for experiment population score
        score_control_cutoff: minimum value for control population score
        lvs: tuple or list of names for the latent variables
        perigee: Depricated. size [nt] array indicating which perigee each sample is in.
            The values can be either integers, filenames, or any other value, as
            long as the values are sensibly sortable. 
        perigee_overlap: integer to spicify own many consecutive perigees are used to
            pick twin samples. Severly impacts the speed of the algorithm.
        lvs_scale_parameters: Scale parameters to enforce the `metric` in `latent 
            space`. If None, defaults to (max - min) for each lv. List of
            length lvs.
        lv_distance_cutoff: distance between two points in `latent space`
            to consider adding two paired points to the experimetn and control pops.
        experiment_candidates: Optional boolean array of length [nt] indicating 
            which samples can be included in the experiment population. May be 
            provided to allow further user customization, such as splitting based 
            on mindx before creating distributions from score.
        control_candidates: Optional boolean array of length [nt] indicating 
            which samples can be included in the control population. May be 
            provided to allow further user customization, such as splitting based 
            on mindx before creating distributions from score.
        return_matcch_indexs: boolean indicating if the index pairs for experiment
            and control matches should be returned.
    
    OUTPUTS:
        experiment: a size [nt] boolean array to filter for experiment population
        control: a size [nt] boolean array to filter for control population
    '''
    perigee_minutes_diff = 11 * 60
    nt = len(score)
    nlv = len(lvs)
    lv = np.vstack([data[l] for l in lvs]).T
    if score_experiment_cutoff is None:
        score_experiment_cutoff = np.nanquantile(score, .1)
    if score_control_cutoff is None:
        score_control_cutoff = np.nanquantile(score, .9)
    if perigee is None:
        perigee = data['perigee_start_min']
    if experiment_candidates is None:
        experiment_candidates = np.ones(nt, dtype=bool)
    if control_candidates is None:
        control_candidates = np.ones(nt, dtype=bool)
        
    experiment_candidates = (score < score_experiment_cutoff) & experiment_candidates
    control_candidates = (score > score_control_cutoff) & control_candidates

    lv_mins = []
    lv_maxs = []
    for lvi in range(nlv):
        lv_mins.append(np.nanmin(lv[:, lvi]))
        lv_maxs.append(np.nanmax(lv[:, lvi]))

    if lvs_scale_parameters is None:
        lvs_scale_parameters = []
        for lvi in range(nlv):
            min_lv = lv_mins[lvi]
            max_lv = lv_maxs[lvi]
            lvs_scale_parameters.append(max_lv - min_lv)
    lvs_scale_parameters = np.array(lvs_scale_parameters)
    lv_bins = []
    lv_edges = []
    for lvi in range(nlv):
        lv_bins.append(int(np.floor((lv_maxs[lvi] - lv_mins[lvi]) / lvs_scale_parameters[lvi] / lv_distance_cutoff)))
        lv_edges.append(np.linspace(lv_mins[lvi], lv_maxs[lvi], lv_bins[lvi]))
    lv_edges_lh = []
    for lvi in range(nlv):
        lv_edges_lh.append(np.stack([lv_edges[lvi][:-2], lv_edges[lvi][2:]], axis=-1))
    bin_bools = []
    for lv_lh_edges in itertools.product(*lv_edges_lh):
        bin_bool = np.ones(nt, dtype=bool)
        for lv_l, lv_h in lv_lh_edges:
            in_lv_bin = (lv[:, lvi] >= lv_l) & (lv[:, lvi] <= lv_h)
            bin_bool = bin_bool & in_lv_bin
        bin_bools.append(bin_bool)

    global_score_diff = []
    global_e_inds = []
    global_c_inds = []

    def compute_latent_distance(lv_control, lv_experiment, lv_scale_params):
        lv_control = np.squeeze(lv_control)
        lv_experiment = np.squeeze(lv_experiment)
        return np.sqrt(np.sum(np.square((lv_control - lv_experiment) / lv_scale_params)))

    u_perigees = np.unique(perigee)
    overlapping_perigees = []
    if perigee_overlap != -1:
        for i in range(len(u_perigees)):
            ops = []
            for j in range(i, len(u_perigees)):
                if abs(u_perigees[i] - u_perigees[j]) <= ((perigee_overlap - 1) * perigee_minutes_diff):
                    ops.append(u_perigees[j])
            overlapping_perigees.append(ops)
    else:
        overlapping_perigees = [None]
    
    for bin_bool in bin_bools:
        for pvals in overlapping_perigees:
            if perigee_overlap != -1:
                p_bool = np.zeros_like(score, dtype=bool)
                for pval in pvals:
                    p_bool = p_bool | (perigee == pval)
            else:
                p_bool = np.ones_like(score, dtype=bool)
            p_experiment = (p_bool & experiment_candidates & bin_bool)
            p_control = (p_bool & control_candidates & bin_bool)

            p_e_index = np.nonzero(p_experiment)[0]
            p_c_index = np.nonzero(p_control)[0]

            for e_i, c_i in itertools.product(p_e_index, p_c_index):
                if (e_i != c_i) and (compute_latent_distance(lv[e_i], lv[c_i], lvs_scale_parameters) < lv_distance_cutoff):
                    global_e_inds.append(e_i)
                    global_c_inds.append(c_i)
                    global_score_diff.append(score[c_i] - score[e_i])
    global_score_diff = np.array(global_score_diff)
    global_e_inds = np.array(global_e_inds, dtype=np.int64)
    global_c_inds = np.array(global_c_inds, dtype=np.int64)
    score_sorted_inds = np.argsort(global_score_diff)[::-1]
    global_score_diff = global_score_diff[score_sorted_inds]
    global_e_inds = global_e_inds[score_sorted_inds]
    global_c_inds = global_c_inds[score_sorted_inds]

    final_e_inds = []
    final_c_inds = []
    while (len(global_e_inds) != 0) and (len(global_c_inds) != 0) and True:
        e_i = global_e_inds[0]
        c_i = global_c_inds[0]
        final_e_inds.append(e_i)
        final_c_inds.append(c_i)
        e_mask = global_e_inds == e_i
        c_mask = global_c_inds == c_i
        mask = ~(e_mask | c_mask)
        global_e_inds = global_e_inds[mask]
        global_c_inds = global_c_inds[mask]
    experiment = np.zeros(nt, dtype=bool)
    control = np.zeros(nt, dtype=bool)
    experiment[final_e_inds] = 1
    control[final_c_inds] = 1
    if return_match_indxs:
        return experiment, control, final_e_inds, final_c_inds
    return experiment, control

def pick_populations_v4(
        data,
        score,
        max_population_sizes,
        score_experiment_cutoff=None,
        score_control_cutoff=None,
        post_score_scaler=1,
        perigee=None,
        lvs=('lat', 'alt'),
        lvs_scale_parameters=None,
        experiment_candidates=None,
        control_candidates=None,
        return_match_indxs=False
    ):
    '''
    This function returns the indices of two populations: the exeriment and 
    the control. The experimental population has LOW values of the score, and 
    the control has HIGH values of score. Additionally, this function attempts 
    to mirror the distributions of the latent variables (lvs) between these two 
    populations.

    This function is an altered version of `pick_populations_v2` that attempts to 
    maximize the distance between the two populations while enforcing score
    cutoffs and relaxing/rethinking the latent distribution symmetrization 
    process. It is distinct from `pick_populations_v3` in that the latent space 
    distance is combined with the score, and then the populations of a fixed size
    that maximize the difference in combined score are found.
    
    DIMENSIONS:
        nt: total times
        ng: number of generators
    
    INPUTS: 
        data: a dictionary with arrays of size [nt] with names matching lvs
        score: a size [nt] array with numerical values for the experimental variable
        score_experiment_cutoff: maximum value for experiment population score
        score_control_cutoff: minimum value for control population score
        lvs: tuple or list of names for the latent variables
        perigee: size [nt] array indicating which perigee each sample is in.
            The values can be either integers, filenames, or any other value.
        lvs_scale_parameters: Scale parameters to enforce the `metric` in `latent 
            space`. If None, defaults to (max - min) / 10 for each lv. List of
            length lvs.
        lv_distance_cutoff: distance between two points in `latent space`
            to consider adding two paired points to the experimetn and control pops.
        experiment_candidates: Optional boolean array of length [nt] indicating 
            which samples can be included in the experiment population. May be 
            provided to allow further user customization, such as splitting based 
            on mindx before creating distributions from score.
        control_candidates: Optional boolean array of length [nt] indicating 
            which samples can be included in the control population. May be 
            provided to allow further user customization, such as splitting based 
            on mindx before creating distributions from score.
        return_matcch_indexs: boolean indicating if the index pairs for experiment
            and control matches should be returned.
    
    OUTPUTS:
        experiment: a size [nt] boolean array to filter for experiment population
        control: a size [nt] boolean array to filter for control population
    '''
    nt = len(score)
    nlv = len(lvs)
    lv = np.vstack([data[l] for l in lvs]).T
    if score_experiment_cutoff is None:
        score_experiment_cutoff = np.nanquantile(score, .1)
    if score_control_cutoff is None:
        score_control_cutoff = np.nanquantile(score, .9)
    # if perigee is None:
    #     perigee = filtered_data['filename']
    if experiment_candidates is None:
        experiment_candidates = np.ones(nt, dtype=bool)
    if control_candidates is None:
        control_candidates = np.ones(nt, dtype=bool)
        
    experiment_candidates = (score < score_experiment_cutoff) & experiment_candidates
    control_candidates = (score > score_control_cutoff) & control_candidates

    if lvs_scale_parameters is None:
        lvs_scale_parameters = []
        for lvi in range(nlv):
            min_lv = np.nanmin(lv[:, lvi])
            max_lv = np.nanmax(lv[:, lvi])
            lvs_scale_parameters.append(max_lv - min_lv / 10)
    lvs_scale_parameters = np.squeeze(np.array(lvs_scale_parameters))

    global_score_diff = []
    global_e_inds = []
    global_c_inds = []

    def compute_latent_distance(lv_control, lv_experiment):
        lv_control = np.squeeze(lv_control)
        lv_experiment = np.squeeze(lv_experiment)
        return np.sqrt(np.sum(np.square((lv_control - lv_experiment) / lvs_scale_parameters)))

    for pval in np.unique(perigee):
        pbool = (perigee == pval)
        p_experiment = (pbool & experiment_candidates)
        p_control = (pbool & control_candidates)

        p_e_index = np.nonzero(p_experiment)[0]
        p_c_index = np.nonzero(p_control)[0]

        for e_i, c_i in itertools.product(p_e_index, p_c_index):
            global_e_inds.append(e_i)
            global_c_inds.append(c_i)
            global_score_diff.append((score[c_i] - score[e_i]) * post_score_scaler - compute_latent_distance(lv[e_i], lv[c_i]))
    global_score_diff = np.array(global_score_diff)
    global_e_inds = np.array(global_e_inds, dtype=np.int64)
    global_c_inds = np.array(global_c_inds, dtype=np.int64)
    score_sorted_inds = np.argsort(global_score_diff)[::-1]
    global_score_diff = global_score_diff[score_sorted_inds]
    global_e_inds = global_e_inds[score_sorted_inds]
    global_c_inds = global_c_inds[score_sorted_inds]

    final_e_inds = []
    final_c_inds = []
    count = 0
    while (len(global_e_inds) != 0) and (len(global_c_inds) != 0) and count < max_population_sizes:
        e_i = global_e_inds[0]
        c_i = global_c_inds[0]
        final_e_inds.append(e_i)
        final_c_inds.append(c_i)
        e_mask = global_e_inds == e_i
        c_mask = global_c_inds == c_i
        mask = ~(e_mask | c_mask)
        global_e_inds = global_e_inds[mask]
        global_c_inds = global_c_inds[mask]
        count += 1
    experiment = np.zeros(nt, dtype=bool)
    control = np.zeros(nt, dtype=bool)
    experiment[final_e_inds] = 1
    control[final_c_inds] = 1
    if return_match_indxs:
        return experiment, control, final_e_inds, final_c_inds
    return experiment, control