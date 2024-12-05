# HR-tools
preprocessing tools and information theory measures for heart rate (HR) data manipulation and data analysis

this set of files provides Python functions to read and format properly HR data, which can be:
- raw ECG measurements
- R-peaks times
- RR-intervals (RRI)
- HR data (in bpm)
either event-driven (R-peaks times) or sampled at a fixed frequency (ECG, RRI or HR).

# usage
import the module(s) as wanted, to use the function(s) 

- ECG_tools : functions to manipulate ECG data and extract peaks timestamps (using NeuroKit 2)
- HR_tools  : functions to manipulate peaks timestamps and RRI / HR (convert peaks to RRI or HR)
- HR_load, HR_load_Felicity : functions to load data from the Felicity 1 and Felicity 2 databases

# examples

in the directory examples are several example Jupyter notebooks:

- ECG_RRI_HR.ipynb : shows how to:
  * load RRI data or raw ECG data
  * convert raw ECG data into peaks timestamps data
  * convert peaks data into RRI data
  * converrt RRI data into HR data
    
- b (to-do)
