from netCDF4 import Dataset
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D


def ne2pfs(ne, B, lam = 50, kT = 0.1, o2hratio=0.3):
    # Compute the various plasma frequencies
    pfs = {}
    
    #naming convention: e = electron, o=oxygen, h = hydrogen
    me = 9.11e-31
    ep0 = 8.85e-12
    mu0 = 1.267e-6
    q0 = 1.602e-19
    amu = 1.66e-27
    c = 3e8
    Z = 16
    mio = amu * Z
    mih = amu
    
    #plasma frequencies
    pfs['fe'] = (1/2/np.pi)*np.sqrt(ne*q0**2/(me*ep0))
    pfs['fo'] = (1/2/np.pi)*np.sqrt(ne*q0**2/(mio*ep0))
    pfs['fh'] = (1/2/np.pi)*np.sqrt(ne*q0**2/(mih*ep0))
    
    #gyrofrequencies
    pfs['feg'] = (1/2/np.pi)* q0 * B/me
    pfs['fog'] = (1/2/np.pi)* q0 * B/mio
    pfs['fhg'] = (1/2/np.pi)* q0 * B/mih
    
    #lower hybrid
    pfs['folh'] = (1/(pfs['feg']*pfs['fog']) + 1/pfs['fo']**2)**(-1/2)
    pfs['fhlh'] = (1/(pfs['feg']*pfs['fhg']) + 1/pfs['fh']**2)**(-1/2)
    
    #ion acoustic
    pfs['vso'] = 2 * np.sqrt(kT*q0/mio)
    pfs['vsh'] = 2 * np.sqrt(kT*q0/mih)
    # foia = vso / lam
    # fhia = vsh / lam
    
    #alfven, parallel to B
    rho = ne * (o2hratio * mio + (1-o2hratio) * mih)
    pfs['va'] = B / np.sqrt(mu0 * rho)
    # fa = va/lam
    
    #magnetosonic, perp to B
    pfs['foM'] = (c/lam) * np.sqrt((pfs['vso']**2 + pfs['va']**2)/(c**2 + pfs['va']**2))/(2*np.pi)
    pfs['fhM'] = (c/lam) * np.sqrt((pfs['vso']**2 + pfs['va']**2)/(c**2 + pfs['va']**2))/(2*np.pi)
    
    return pfs


def load_level2(level2_file):

    d = Dataset(level2_file)
    data = {}
    data['filename'] = level2_file.split('/')[-1]
    
    ### Read in the time variables
    time_vars = ['master_time', 'IRI_time', 'Etime_OFACOMPLEX', 'Btime_OFACOMPLEX', 'Etime_OFASPEC', 'Btime_OFASPEC',
                 'Etime_OFAwave', 'Btime_OFAwave', 'Ptime_OFAwave', 'HFA_time']
    
    data['t'] = {}
    for var in time_vars:
        data['t'][var] = {}
    
    # Load in numpy datetime64 version of all time vars
    for var in time_vars:
        # Load in numpy datetime64 version of all time vars
        data['t'][var]['dt64'] = np.array(d[var]).astype('datetime64[ms]')
        # Get elapsed minutes version of all time vars
        data['t'][var]['minutes'] = (data['t'][var]['dt64'] - data['t']['master_time']['dt64'][0])/np.timedelta64(1000, 'ms')/60
        # Get pcolormesh plotting version of all time vars
        # This doesn't work for HFA time when it only has one sample
        try:
            data['t'][var]['mesh'] = np.concatenate((data['t'][var]['minutes'], [data['t'][var]['minutes'][-1] + 
                                                                             (data['t'][var]['minutes'][-1] - data['t'][var]['minutes'][-2])]))
        except:
            continue

    ### Read all of the other variables
    other_vars = ['NORAD_ID', 'dx_TLE', 'detector_ECEF_pos', 'detector_ECEF_vel', 'generator_ECEF_pos', 'generator_ECEF_vel',
                  'generator_hill_pos', 'generator_hill_vel', 'generator_LLA', 'generator_name', 'generator_RCS', 'dx', 
                  'Ne_IRI', 'detector_ECEF_B', 'generator_ECEF_B', 'Arase_LLA', 'Arase_ECEF_pos',
                  'Efreq', 'Bfreq', 'E_OFACOMPLEX', 'Ex_OFACOMPLEX', 'Ey_OFACOMPLEX', 'Eflag_OFACOMPLEX',
                  'B_OFACOMPLEX', 'Bx_OFACOMPLEX', 'By_OFACOMPLEX', 'Bz_OFACOMPLEX', 'Bflag_OFACOMPLEX',
                  'E_OFASPEC', 'Echannel_OFASPEC', 'Eflag_OFASPEC', 'B_OFASPEC', 'Bchannel_OFASPEC', 'Bflag_OFASPEC',
                  'E_OFAwave', 'Eflag_OFAwave', 'Eobscal_OFAwave', 'B_OFAwave', 'Bflag_OFAwave', 'Bobscal_OFAwave',
                  'kvec_polar', 'kvec_polar_masked', 'kvec_azimuth', 'kvec_azimuth_masked',
                  'polarization', 'polarization_masked', 'planarity', 'planarity_masked', 'Pvec_angle', 'Pvec_angle_masked',
                  'UHRfreq_HFA', 'Ne_HFA']
    
    for var in other_vars:
        data[var] = np.array(d[var])
        
    d.close()
    
    # Get pcolormesh plotting version of frequency vars
    df_low = data['Efreq'][1] - data['Efreq'][0]
    df_high = data['Efreq'][-1] - data['Efreq'][-2]
    data['Efreq_mesh'] = np.concatenate(([data['Efreq'][0] - df_low], 0.5*(data['Efreq'][:-1] + data['Efreq'][1:]), [data['Efreq'][-1] + df_high]))
    df_low = data['Bfreq'][1] - data['Bfreq'][0]
    df_high = data['Bfreq'][-1] - data['Bfreq'][-2]
    data['Bfreq_mesh'] = np.concatenate(([data['Bfreq'][0] - df_low], 0.5*(data['Bfreq'][:-1] + data['Bfreq'][1:]), [data['Bfreq'][-1] + df_high]))

    # Compute angle between generator velocity and B-field
    bmag0 = np.sqrt(np.sum(data['generator_ECEF_B']**2, axis=1))
    bhat = data['generator_ECEF_B']/bmag0[:,None,:]
    vhat = data['generator_ECEF_vel']/np.sqrt(np.sum(data['generator_ECEF_vel']**2, axis=1))[:,None,:]
    dp = np.sum(bhat*vhat, axis = 1)
    ang = np.degrees(np.arccos(dp))
    ang[ang>90] = 180 - ang[ang>90]
    data['B_angle'] = ang
    
    ### Compute IRI and HFA plasma frequencies
    # Magnetic field magnitude at detector, converted from nT to T
    bmag0d = np.sqrt(np.sum(data['detector_ECEF_B']**2, axis=1))/10**9
    
    # Compute various plasma frequencies for the IRI electron densities
    bmag0d_IRI = np.interp(data['t']['IRI_time']['minutes'], data['t']['master_time']['minutes'], bmag0d)
    data['pfs_IRI'] = ne2pfs(data['Ne_IRI'], bmag0d_IRI)
    
    # Compute various plasma frequencies for the HFA electron densities
    # Interpolate magnetic field magnitude at detector to HFA time
    bmag0d_HFA = np.interp(data['t']['HFA_time']['minutes'], data['t']['master_time']['minutes'], bmag0d)
    # Compute plasma frequencies
    data['pfs_HFA'] = ne2pfs(data['Ne_HFA'], bmag0d_HFA)
    
    empty_vars = []
    for var in data.keys():
        if isinstance(data[var], np.ndarray):
            if data[var].shape[0]==0:
                empty_vars.append(var)
    if len(empty_vars)>0:
        print("Empty Arase data for this perigee:")
        print(*empty_vars, sep=', ')
        print('\n')
    
    return data


def determine_mode_string(channel, channel_type):
    e_modes = {1:'Eu', 2:'Ev', 3:'Eu+Ev', 4:'Ev1', 5:'Ev2', 6:'E1+Ev2'}
    b_modes = {1:'Balpha', 2:'Bbeta', 3:'Bgamma', 4:'Balpha+Bbeta+Bgamma'}
    
    if np.any(channel != channel[0]):
        mode_str = "Mode changes during perigee"
    
    else:
        if channel_type=="E_OFASPEC":
            mode_str = e_modes[channel[0]]
        elif channel_type=="B_OFASPEC":
            mode_str = b_modes[channel[0]]
    
    return mode_str


def plot_level2(data, params=None):

    # Default plot parameters
    plot_params = {
        "display": True, # Toggle plot display on and off
        "save": False, # If True, save the plot (600 dpi) at "save_filepath"
        "save_filepath": data['filename'].split('.')[0] + '.png', # Full filepath for saving the plot ("save" must be True)
        "mindx_threshold": None, # Mark closest approach for all generators that get within 'mindx_threshold'. Choose None to mark nothing.
        "print_close_approaches": False, # Print time, ID, name, sqrt(RCS), and dx for closest approach of generators that get within "mindx_threshold"
        "use_log_scale": False, # Turn log scale on or off for frequency axis of time-frequency plots
        "wave_property_mask": True, # Toggle between masked and unmasked version of kvec, polarization, planarity, Poynting
        "obs": ["E_OFAwave", "B_OFAwave"], # Choose which observation plots to show
        "plot_plasma_lines": [], # Plot plasma lines on all of these observation plots
        "hide_plasma_line_legend": False, # If True, hide the plasma line legend (it sometimes covers up useful measurements!)
        "xmin": data['t']['master_time']['minutes'][0], # Start of x-axis, in minutes relative to first master_time
        "xmax": data['t']['master_time']['minutes'][-1], # End of x-axis, in minutes relative to first master_time
        "ymin": 0.05, # Start of y-axis for time-frequency plots [kHz]
        "ymax": 20, # End of y-axis for time-frequency plots [kHz]
        "logE_min": -2, # Start of E-field colorbar, in log(magnitude)
        "logE_max": 1, # End of E-field colorbar, in log(magnitude)
        "logB_min": -2, # Start of B-field colorbar, in log(magnitude)
        "logB_max": -1.5, # End of E-field colorbar, in log(magnitude)
        "hide_generators":False, # Choose True to hide conjunctions with generators
        "dx_ymax": 300 # End of y-axis for the generator conjunction plots [km]
    }

    # Overwrite default parameters with anything user specified
    if params is not None:
        for key in params.keys():
            if key not in plot_params.keys():
                print(f'Plot parameter "{key}" does not exist')
            else:
                plot_params[key] = params[key]

    ### Plot settings for all Arase observations
    plot_helper = {}
    plot_helper['E_OFACOMPLEX'] = {
        't_name':'Etime_OFACOMPLEX',
        'f':data['Efreq_mesh'],
        'vmin':plot_params['logE_min'],
        'vmax':plot_params['logE_max'],
        'flag':data['Eflag_OFACOMPLEX'],
        'label':'OFA-COMPLEX\n$E$ total\n [log10(mV$^2$/m$^2$/Hz)]',
        'log':True
    }
    plot_helper['Ex_OFACOMPLEX'] = {
        't_name':'Etime_OFACOMPLEX',
        'f':data['Efreq_mesh'],
        'vmin':plot_params['logE_min'],
        'vmax':plot_params['logE_max'],
        'flag':data['Eflag_OFACOMPLEX'],
        'label':'OFA-COMPLEX\n$E_u$\n [log10(mV$^2$/m$^2$/Hz)]',
        'log':True
    }
    plot_helper['Ey_OFACOMPLEX'] = {
        't_name':'Etime_OFACOMPLEX',
        'f':data['Efreq_mesh'],
        'vmin':plot_params['logE_min'],
        'vmax':plot_params['logE_max'],
        'flag':data['Eflag_OFACOMPLEX'],
        'label':'OFA-COMPLEX\n$E_v$\n [log10(mV$^2$/m$^2$/Hz)]',
        'log':True
    }
    plot_helper['B_OFACOMPLEX'] = {
        't_name':'Btime_OFACOMPLEX',
        'f':data['Bfreq_mesh'],
        'vmin':plot_params['logB_min'],
        'vmax':plot_params['logB_max'],
        'flag':data['Bflag_OFACOMPLEX'],
        'label':'OFA-COMPLEX\n$B$ total\n [log10(nT$^2$/Hz)]',
        'log':True
    }
    plot_helper['Bx_OFACOMPLEX'] = {
        't_name':'Btime_OFACOMPLEX',
        'f':data['Bfreq_mesh'],
        'vmin':plot_params['logB_min'],
        'vmax':plot_params['logB_max'],
        'flag':data['Bflag_OFACOMPLEX'],
        'label':'OFA-COMPLEX\n' + r'$B_{\alpha}$' + '\n [log10(nT$^2$/Hz)]',
        'log':True
    }
    plot_helper['By_OFACOMPLEX'] = {
        't_name':'Btime_OFACOMPLEX',
        'f':data['Bfreq_mesh'],
        'vmin':plot_params['logB_min'],
        'vmax':plot_params['logB_max'],
        'flag':data['Bflag_OFACOMPLEX'],
        'label':'OFA-COMPLEX\n' + r'$B_{\beta}$' + '\n [log10(nT$^2$/Hz)]',
        'log':True
    }
    plot_helper['Bz_OFACOMPLEX'] = {
        't_name':'Btime_OFACOMPLEX',
        'f':data['Bfreq_mesh'],
        'vmin':plot_params['logB_min'],
        'vmax':plot_params['logB_max'],
        'flag':data['Bflag_OFACOMPLEX'],
        'label':'OFA-COMPLEX\n' + r'$B_{\gamma}$' + '\n [log10(nT$^2$/Hz)]',
        'log':True
    }
    plot_helper['E_OFASPEC'] = {
        't_name':'Etime_OFASPEC',
        'f':data['Efreq_mesh'],
        'vmin':plot_params['logE_min'],
        'vmax':plot_params['logE_max'],
        'flag':data['Eflag_OFASPEC'],
        'label':'OFA-SPEC\n$E$\n [log10(mV$^2$/m$^2$/Hz)]',
        'log':True,
        'mode_var':'Echannel_OFASPEC'
    }
    plot_helper['B_OFASPEC'] = {
        't_name':'Btime_OFASPEC',
        'f':data['Bfreq_mesh'],
        'vmin':plot_params['logB_min'],
        'vmax':plot_params['logB_max'],
        'flag':data['Bflag_OFASPEC'],
        'label':'OFA-SPEC\n$B$\n [log10(nT$^2$/Hz)]',
        'log':True,
        'mode_var':'Bchannel_OFASPEC'
    }
    plot_helper['E_OFAwave'] = {
        't_name':'Etime_OFAwave',
        'f':data['Efreq_mesh'],
        'vmin':plot_params['logE_min'],
        'vmax':plot_params['logE_max'],
        'flag':data['Eflag_OFAwave'],
        'label':'OFA wave\n$E$ total\n [log10(mV$^2$/m$^2$/Hz)]',
        'log':True
    }
    plot_helper['B_OFAwave'] = {
        't_name':'Btime_OFAwave',
        'f':data['Bfreq_mesh'],
        'vmin':plot_params['logB_min'],
        'vmax':plot_params['logB_max'],
        'flag':data['Bflag_OFAwave'],
        'label':'OFA wave\n$B$ total\n [log10(nT$^2$/Hz)]',
        'log':True
    }
    plot_helper['kvec_polar'] = {
        't_name':'Btime_OFAwave',
        'f':data['Bfreq_mesh'],
        'vmin':0,
        'vmax':90,
        'label':'OFA wave \n k_vec polar angle\n [deg.]',
        'cmap':'jet',
        'cbar_ticks':[0,45,90]
    }
    plot_helper['kvec_azimuth'] = {
        't_name':'Btime_OFAwave',
        'f':data['Bfreq_mesh'],
        'vmin':-180,
        'vmax':180,
        'label':'OFA wave \n k_vec azimuth angle\n [deg.]',
        'cmap':'twilight',
        'cbar_ticks':[-180, -90, 0, 90, 180]
    }
    plot_helper['polarization'] = {
        't_name':'Btime_OFAwave',
        'f':data['Bfreq_mesh'],
        'vmin':-1,
        'vmax':1,
        'label':'OFA wave \n Polarization \n []',
        'cmap':'coolwarm',
        'cbar_ticks':[-1, -0.5, 0.0, 0.5, 1]
    }
    plot_helper['planarity'] = {
        't_name':'Btime_OFAwave',
        'f':data['Bfreq_mesh'],
        'vmin':0,
        'vmax':1,
        'label':'OFA wave \n Planarity \n []',
        'cmap':'plasma',
        'cbar_ticks':[0.0, 0.5, 1.0]
    }
    plot_helper['Pvec_angle'] = {
        't_name':'Ptime_OFAwave',
        'f':data['Bfreq_mesh'],
        'vmin':0,
        'vmax':90,
        'label':'OFA wave \n Poynting vector angle\n [deg.]',
        'cmap':'jet',
        'cbar_ticks':[0, 45, 90]
    }

    nt = len(data['t']['master_time']['minutes'])
    ng = len(data['NORAD_ID'])
    N_obs = len(plot_params['obs'])
    N_panels = N_obs + 2

    ### Make the figure

    f = plt.figure(figsize=(10,2*N_panels+1), constrained_layout=True, facecolor="w")

    gs = GridSpec(N_panels, 2, figure=f, width_ratios=(100, 2))

    f.suptitle(data['filename'])

    ### Create the axes
    axes = [None]*N_panels
    caxes = [None]*N_panels
    for i in range(N_panels):
        axes[i] = f.add_subplot(gs[i,0])
        axes[i].grid()
        axes[i].set_xlim([plot_params['xmin'], plot_params['xmax']])
        caxes[i] = f.add_subplot(gs[i,1])

    ### Plot the generators

    # Colors for B-field angle
    norm = matplotlib.colors.Normalize(vmin=0, vmax=90)
    cmap = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.jet)
    cmap.set_array([])

    tmins = []
    close_approach_messages = []

    for ii in range(ng):
        idi = data['NORAD_ID'][ii]
        namei = data['generator_name'][ii]
        if len(namei)==0:
            namei="unknown object"
        rcsi = np.sqrt(data['generator_RCS'][ii])
        if rcsi==np.nan:
            rcs_string = "unknown"
        else:
            rcs_string = f"{rcsi:.2f} m"
        mean_ang = np.nanmean(data['B_angle'][:,ii])
        if np.isnan(rcsi):
            color = 'gray'
        elif rcsi < 0.4:
            color = 'green'
        elif (rcsi >= 0.4) and (rcsi < 1):
            color = 'gold'
        else:
            color = 'red'
        if not plot_params['hide_generators']:
            axes[0].plot(data['t']['master_time']['minutes'][:], data['dx'][:, ii]/1000, c=color)
            axes[1].plot(data['t']['master_time']['minutes'][:], data['dx'][:, ii]/1000, c=cmap.to_rgba(mean_ang))

        if plot_params['mindx_threshold'] is not None:
            axes[0].axhline(plot_params['mindx_threshold'], color='black', linewidth=1.0, linestyle='dashed')
            axes[1].axhline(plot_params['mindx_threshold'], color='black', linewidth=1.0, linestyle='dashed')
            mindx = np.nanmin(data['dx'][:, ii])
            tmin = data['t']['master_time']['minutes'][data['dx'][:,ii]==mindx][0]
            if (mindx/1000 < plot_params['mindx_threshold']):
                close_approach_messages.append(f"At {tmin:.2f} minutes: NORAD ID {idi} ({namei}, {rcs_string} size) with min dx={mindx/1000:.1f} km")
                tmins.append(tmin)

    tmins = np.array(tmins)
    tmins_in_range = tmins[(tmins > plot_params['xmin']) & (tmins < plot_params['xmax'])]

    if plot_params['print_close_approaches']:
        for ii in np.argsort(tmins):
            print(close_approach_messages[ii])

    # Manually create a legend to show RCS categories
    line1 = Line2D([0], [0], label=r"$\sqrt{RCS}< 0.4$ m", color='green')
    line2 = Line2D([0], [0], label=r"$0.4 \leq \sqrt{RCS} < 1$ m", color='gold')
    line3 = Line2D([0], [0], label=r"$\sqrt{RCS} \geq 1$ m", color='red')
    line4 = Line2D([0], [0], label="No data", color='gray')
    handles = [line1, line2, line3, line4]
    caxes[0].axis('off')
    axes[0].legend(handles=handles, loc='upper right', ncol=4, fancybox=True, framealpha=0.9, fontsize=9)

    # Plot generator B-field angle colorbar
    cb = plt.colorbar(cmap, cax=caxes[1], ticks=np.arange(0, 91, 15))
    cb.set_label(r'$\angle \vec{v}_G, \vec{B}$'+'\n[deg]')

    for ax in [axes[0], axes[1]]:
        ax.set_ylim([0, plot_params['dx_ymax']])
        ax.set_ylabel('Generator-Detector\nDistance [km]')

    for i in range(2, N_panels):

        axes[i].set_ylim([plot_params['ymin'], plot_params['ymax']])
        if plot_params['use_log_scale']:
            axes[i].set_yscale('log')
        axes[i].set_ylabel('Frequency [kHz]')

        var = plot_params['obs'][i-2]
        
        # Handle wave property mask
        if plot_params['wave_property_mask'] and var in ['kvec_polar', 'kvec_azimuth', 'polarization', 'planarity', 'Pvec_angle']:
            plot_var_name = var + '_masked'
        else:
            plot_var_name = var

        # Choose the colormap
        if 'cmap' in plot_helper[var].keys():
            cmap = plot_helper[var]['cmap']
        else:
            cmap = 'viridis'

        # Handle log scale that applies to some variables
        if 'log' in plot_helper[var].keys():
            plot_var = np.log10(data[plot_var_name]).T
        else:
            plot_var = data[plot_var_name].T
            
        # Skip plotting and inform user if the requested observation is empty for this perigee
        if len(data['t'][plot_helper[var]['t_name']]['dt64'])==0:
            print(f"Observation {var} is empty for this perigee.")
            continue
        
        im = axes[i].pcolormesh(data['t'][plot_helper[var]['t_name']]['mesh'], plot_helper[var]['f'], plot_var, 
                                vmin=plot_helper[var]['vmin'], vmax=plot_helper[var]['vmax'], shading='flat', cmap=cmap)

        # Plot the colorbar
        cb = plt.colorbar(im, cax=caxes[i])
        cb.set_label(plot_helper[var]['label'], fontsize = 10)
        if 'cbar_ticks' in plot_helper[var].keys():
            cb.set_ticks(plot_helper[var]['cbar_ticks'])

        # Plot quality flags, when applicable
        if 'flag' in plot_helper[var].keys():
            t = data['t'][plot_helper[var]['t_name']]['minutes']
            bad_qual = ((plot_helper[var]['flag'] > 0) & (t > plot_params['xmin']) & (t < plot_params['xmax']))
            if np.any(bad_qual):
                axes[i].scatter(t[bad_qual], np.ones((bad_qual.sum(),))*plot_params['ymin'], color="red", s=10, clip_on=False, zorder=100)

        # Add mode string for OFA-SPEC measurements
        if 'mode_var' in plot_helper[var].keys():
            mode_str = determine_mode_string(data[plot_helper[var]['mode_var']], var)
            axes[i].text(0.02, 0.95, "Mode: " + mode_str, transform=axes[i].transAxes, va='top', ha='left', color='black',
                         bbox=dict(facecolor='white', edgecolor='black', pad=1.0))

        # Plot plasma frequencies on the specified plots
        if var in plot_params['plot_plasma_lines']:
            axes[i].plot(data['t']['IRI_time']['minutes'], data['pfs_IRI']['fo']/1000, 'm-', label = '(IRI) O+ plasma ')
            axes[i].plot(data['t']['HFA_time']['minutes'], data['pfs_HFA']['fo']/1000, 'm.', label ='(obs) O+ plasma ')

            axes[i].plot(data['t']['IRI_time']['minutes'], data['pfs_IRI']['fog']/1000, 'b-', label='O+ gyro')
            axes[i].plot(data['t']['HFA_time']['minutes'], data['pfs_HFA']['fog']/1000, 'b.', label='(obs) O+ gyro')

            axes[i].plot(data['t']['IRI_time']['minutes'], data['pfs_IRI']['fhg']/1000, 'c-', label='H+ gyro')
            axes[i].plot(data['t']['HFA_time']['minutes'], data['pfs_HFA']['fhg']/1000, 'c.', label='(obs) H+ gyro')

            axes[i].plot(data['t']['IRI_time']['minutes'], data['pfs_IRI']['folh']/1000, 'y-', label='(IRI) O+ LH')
            axes[i].plot(data['t']['HFA_time']['minutes'], data['pfs_HFA']['folh']/1000, 'y.', label='(obs) O+ LH')

            axes[i].plot(data['t']['IRI_time']['minutes'], data['pfs_IRI']['fhlh']/1000, 'r-', label='(IRI) H+ LH')
            axes[i].plot(data['t']['HFA_time']['minutes'], data['pfs_HFA']['fhlh']/1000, 'r.', label='(obs) H+ LH')

        if len(plot_params['plot_plasma_lines'])>0:
            if var==plot_params['plot_plasma_lines'][-1]:
                if not plot_params['hide_plasma_line_legend']:
                    axes[i].legend(loc='lower left', ncol=5, fontsize = 6)

        # Mark times of close conjunctions
        if plot_params['mindx_threshold'] is not None and not plot_params['hide_generators']:
            axes[i].scatter(tmins_in_range, np.ones(tmins_in_range.shape)*plot_params['ymin'], marker="^",
                            facecolor="black", edgecolor="white", s=100, clip_on=False, zorder=101)
            axes[i].scatter(tmins_in_range, np.ones(tmins_in_range.shape)*plot_params['ymax'], marker="v",
                            facecolor="black", edgecolor="white", s=100, clip_on=False, zorder=101)

    axes[-1].set_xlabel(f"Time [minutes since {data['t']['master_time']['dt64'][0]}]")

    if plot_params['save']:
        f.savefig(plot_params['save_filepath'], dpi=600)
    
    if plot_params['display']:
        plt.show()
    else:
        plt.close()

    return


def bin2msgs(x, meanings):
    #takes a list of strings in
    nf = len(x)
    msg = [None] * nf
    for i in range(nf):
        s = x[i]
        flagNums = np.nonzero(np.array([ss == '1' for ss in s[::-1]]))[0]
        msg[i] = np.array(meanings)[flagNums - 1]
    allmsg = np.concatenate(msg)
    umsg = np.unique(allmsg)
    return msg, umsg


def plot_quality_flags(data, display=True, save_filepath=None):
    emeanings = [None] * 31
    emeanings[:6] = ['DC-CAL signal ON', 'AC-CAL(E) signal ON', 'AC-CAL(B) signal ON', 
                 'eclipse', 'magnetorquer operated', 'ambiguous UTC label']
    emeanings[15:17] = ['Eu Saturated', 'Ev Saturated']
    emeanings[18] = 'time syncronization failed'
    bmeanings = [None] * 31
    bmeanings[:6] = ['DC-CAL signal ON', 'AC-CAL(E) signal ON', 'AC-CAL(B) signal ON', 
                 'eclipse', 'magnetorquer operated', 'ambiguous UTC label']
    bmeanings[15:18] = ['Balpha Saturated', 'Bbeta Saturated', 'Bgamma Saturated']
    bmeanings[18:20] = ['time syncronization failed', 'Balpha Contaminated by broadband noise']
    for i in range(len(emeanings)):
        if emeanings[i] is None: 
            emeanings[i] = 'reserved'
        if bmeanings[i] is None:
            bmeanings[i] = 'reserved'

    emeanings = np.array(emeanings)
    bmeanings = np.array(bmeanings)
    eflagWave = [bin(x)[2:] for x in data['Eflag_OFAwave'].astype(int)]
    bflagWave = [bin(x)[2:] for x in data['Bflag_OFAwave'].astype(int)]
    eflagCom = [bin(x)[2:] for x in data['Eflag_OFACOMPLEX'].astype(int)]
    bflagCom = [bin(x)[2:] for x in data['Bflag_OFACOMPLEX'].astype(int)]
    eflagSpe = [bin(x)[2:] for x in data['Eflag_OFASPEC'].astype(int)]
    bflagSpe = [bin(x)[2:] for x in data['Bflag_OFASPEC'].astype(int)]

    msgew, umsgew = bin2msgs(eflagWave, emeanings)
    msgbw, umsgbw = bin2msgs(bflagWave, bmeanings)
    msgec, umsgec = bin2msgs(eflagCom, emeanings)
    msgbc, umsgbc = bin2msgs(bflagCom, bmeanings)
    msges, umsges = bin2msgs(eflagSpe, emeanings)
    msgbs, umsgbs = bin2msgs(bflagSpe, bmeanings)
    
    f, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6,1, sharex=True, figsize = (6,8))

    for j in range(umsgew.size):
        gi = np.nonzero(np.array([umsgew[j] in m for m in msgew]))[0]
        ax1.plot(data['t']['Etime_OFAwave']['minutes'][gi], j * np.ones(gi.size), 'r.')
    for j in range(umsgbw.size):
        gi = np.nonzero(np.array([umsgbw[j] in m for m in msgbw]))[0]
        ax2.plot(data['t']['Btime_OFAwave']['minutes'][gi], j * np.ones(gi.size), 'r.')

    for j in range(umsgec.size): 
        gi2 = np.nonzero(np.array([umsgec[j] in m for m in msgec]))[0]
        ax3.plot(data['t']['Etime_OFACOMPLEX']['minutes'][gi2], j * np.ones(gi2.size), 'r.')
    for j in range(umsgbc.size): 
        gi2 = np.nonzero(np.array([umsgbc[j] in m for m in msgbc]))[0]
        ax4.plot(data['t']['Btime_OFACOMPLEX']['minutes'][gi2], j * np.ones(gi2.size), 'r.')

    for j in range(umsges.size): 
        gi2 = np.nonzero(np.array([umsges[j] in m for m in msges]))[0]
        ax5.plot(data['t']['Etime_OFASPEC']['minutes'][gi2], j * np.ones(gi2.size), 'r.')
    for j in range(umsgbc.size): 
        gi2 = np.nonzero(np.array([umsgbs[j] in m for m in msgbs]))[0]
        ax6.plot(data['t']['Btime_OFASPEC']['minutes'][gi2], j * np.ones(gi2.size), 'r.')
            
    ax1.set_yticks(np.arange(umsgew.size), labels=umsgew)
    ax2.set_yticks(np.arange(umsgbw.size), labels=umsgbw)
    ax3.set_yticks(np.arange(umsgec.size), labels=umsgec)
    ax4.set_yticks(np.arange(umsgbc.size), labels=umsgbc)
    ax5.set_yticks(np.arange(umsges.size), labels=umsges)
    ax6.set_yticks(np.arange(umsgbs.size), labels=umsgbs)
    ax1.set_ylabel('OFA wave E')
    ax2.set_ylabel('OFA wave B')
    ax3.set_ylabel('OFA\nCOMPLEX E')
    ax4.set_ylabel('OFA\nCOMPLEX B')
    ax5.set_ylabel('OFA-SPEC E')
    ax6.set_ylabel('OFA-SPEC B')
    ax1.set_title(data['filename'])
    ax6.set_xlabel('Minutes since ' + str(data['t']['master_time']['dt64'][0]))
    for a in (ax1, ax2, ax3, ax4, ax5, ax6):
        a.grid()
        a.set_xlim([data['t']['master_time']['minutes'][0], data['t']['master_time']['minutes'][-1]]) 
    
    if save_filepath is not None:
        f.savefig(save_filepath, bbox_inches='tight', dpi=600)
    
    if display:
        plt.show()
    else:
        plt.close()
    
    return