# HR-tools
preprocessing tools and information theory measures for heart rate (HR) data manipulation and data analysis

this set of files provides Python functions to read and format properly HR data, which can be:
- raw ECG measurements
- R-peaks times
- RR-intervals (RRI)
- HR data (in bpm)
either event-driven (R-peaks times) or sampled at a fixed frequency (ECG, RRI or HR).

# usage
import the module(s) as wanted, to use the corresponding function(s). See the quick description below.


# file HR_tools.py

routines to convert peaks timestamps data into RRI or HR data, and to detect physiologicaly relevant values (and exclude non-physiologicaly relevant ones)

# file HR_entropy.py

routines to compute multi-scale entropies (mainly differrential entropy rate, and usual complexities (ApEn and SampEn: Approximate and Sample Entropies), as a function of a provided vector of time-scales.

# file ECG_tools.py

functions to manipulate ECG data and extract peaks timestamps (using NeuroKit 2). Note that these functions have been somehow deprecated and other repositories offer better extraction of R-peaks from an ECG dataset.

# files HR_load.py and HR_load_Felicity 

functions to load data from the Felicity 1 and Felicity 2 databases.

# examples

in the directory examples/ are several example Jupyter notebooks:

- ECG_RRI_HR.ipynb : shows how to:
  * load RRI data or raw ECG data
  * convert raw ECG data into peaks timestamps data
  * convert peaks data into RRI data
  * convert RRI data into HR data
    
- b (to-do)
