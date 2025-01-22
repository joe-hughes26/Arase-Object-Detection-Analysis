#!/usr/bin/env python
# coding: utf-8

# In[1]:


from level2_utils import load_level2, plot_level2, plot_quality_flags


# In[2]:


# Load the data for one example perigee: 2018/05/07 starting at 20:12:12 UT.
# Other perigees can be found in the OSS_L2_2018 directory of this deliverable
level2_file = "OSSLevel2_Arase_perigee_20180507201212.nc"
data = load_level2(level2_file)


# In[3]:


# Make the default plot
plot_level2(data)


# In[4]:


# Make a more detailed plot and print information about close approaches
params = {
    'obs':["E_OFASPEC", "E_OFAwave", "B_OFASPEC", "B_OFAwave", "kvec_polar", "kvec_azimuth", "polarization", "planarity"],
    'plot_plasma_lines':["E_OFASPEC", "E_OFAwave"],
    'wave_property_mask':True,
    'mindx_threshold':100,
    'print_close_approaches':True
}

plot_level2(data, params=params)


# In[5]:


# Create and save plots with different Arase data types (OFA-wave, OFA-COMPLEX, and OFA-SPEC)

# OFA Wave
plot_level2(data, params=dict(obs=["E_OFAwave","B_OFAwave"], plot_plasma_lines=["E_OFAwave", "B_OFAwave"], mindx_threshold=75, 
                              save=True, save_filepath="test_OFAwave.png"))

# OFA-COMPLEX
plot_level2(data, params=dict(obs=["E_OFACOMPLEX","B_OFACOMPLEX"], plot_plasma_lines=["E_OFACOMPLEX", "B_OFACOMPLEX"], mindx_threshold=75,
                              save=True, save_filepath="test_OFACOMPLEX.png"))

# OFA-SPEC
plot_level2(data, params=dict(obs=["E_OFASPEC","B_OFASPEC"], plot_plasma_lines=["E_OFASPEC", "B_OFASPEC"], mindx_threshold=75,
                              save=True, save_filepath="test_OFASPEC.png"))


# In[6]:


# Make a detailed plot of the quality flags
# Nothing is flagged in this example perigee, but this is useful for other perigees.
plot_quality_flags(data)

