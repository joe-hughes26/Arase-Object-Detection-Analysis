# Overview

This repo is intended to allow users to recreate the analysis perform by the Orion Space Solutions team for the paper [CITE].

The datasets that are leveraged for the statistical analysis include:

1. [Raw Arase satellite observation data](https://ergsc.isee.nagoya-u.ac.jp/data_info/erg.shtml.en)
1. [Propagated Space-Track TLE data](https://www.space-track.org/)
1. [PyIRI estimated electron densities](https://github.com/victoriyaforsythe/PyIRI)
1. [CelesTrack satellite data](https://celestrak.org/satcat/satcat-format.php)

For a full list of data sources and information about the OSS Level 2 dataset, see `OSSLevel2_Arase_DataDictionary.xlsx`.

Due to the requirement that Arase data not be redistributed, we provide a collection of the OSS Level 2 files with the Arase data removed, and a set of utility function to repopulate the Arase data from their original sources.
We refer to the stripped data sets as the OSS Level 1.9 dataset. To recreate the Level 2 dataset:

1. Download the zipped level 1.9 Dataset from [INSERT LINK].
1. Move the unzipped netcdf files to the top level of `data/arase_L1_9`, relative to the root of this repo.
1. Install dependencies with `pip install -r requirements.txt` (Create a new conda environment with python3.11 if needed)
1. From the root of the repo, run `python data_preperation/create_OSS_L2_perigee_product_from_OSS_L1_9.py`, which will:

    a. Download Arase data to `data/arase_raw_data`

    b. Populate the Level 1.9 netcdf files with the Arase data, 

    c. Rename the fully populated Level 1.9 file as a Level 2 file, and then move it to `data/arase_L2`

Notes about `data_preperation/create_OSS_L2_perigee_product_from_OSS_L1_9.py`:

1. The Arase team regularly updates their datasets, it is possible that some of the L1.9 files will not be fully populated after the script runs.
1. The default configuretion will delete the Arase data after it has been used to populate the L1.9 datasets, to change this configuration set the global variable `RETAIN_RAW_ARASE_DATA`.
1. In order to create the datasets in a reasonable amont of time (about 30 minutes with ~ 100Mbps download speed), the Arase data is downloaded asyncronously which may use a large amount of internet. If you want to limit this download speed, change the global variable `MAX_ASYNC_DOWNLOADS`.

Once the Arase L2 dataset has been crated, you can recreate the statistical analysis by first creating an intermediary dictionary for ease of processing, then by running the statistical analysis code:

1. Run `python data_preperation/create_pickle_file_for_stats.py`, which will create a small pickle file in `data/arase_L2_analysis_pickle`
1. Run `python perform_statistical_analysis.py` which create a collection of figures in `statistical_experiments`


