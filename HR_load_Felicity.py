# -*- coding: ascii -*-
# set of functions to load and anippulate Felicity 1 data

import numpy
from scipy.io import loadmat # to load Matlab files
import neurokit2 as nk       # not really used

# Import raw peaks from the mat files
data_path ='/Users/ngarnier/Documents/research/data/2021-mother-fetus' # macos
#data_path ="/home/ngarnier/codes/TE"                                   # phycal2
f_filepath_peaks=data_path+"/raw_peaks/f" # fetal raw peaks mat files;
m_filepath_peaks=data_path+"/raw_peaks/m" # maternal;

f_peaks_files = [f for f in sorted(Path(f_filepath_peaks).iterdir())  #create a list of relevant files in directory
            if f.suffix == '.mat']
m_peaks_files = [f for f in sorted(Path(m_filepath_peaks).iterdir())  #create a list of relevant files in directory
            if f.suffix == '.mat']

# a global variable containing indices of bad/discarded data:
discarded_couples= numpy.array([3, 7, 15, 18, 21, 23, 29, 35, 39, 43, 52, 68, 70, 73, 74, 84, 88, 89, 93, 100, 101, 110, 111, 112, 113, 117, 118, 123, 125, 131, 138, 142, 147, 153, 157, 158, 160, 162]) 
# 2022-04-07: discarded because fHR is too small (in absolute <110bpm and relative to mHR, ratio<=1)



# this function returns a list of all valid couples
def set_couples(do_discard=0, verbosity=0):
    couples_full    = numpy.arange(1,168)             # all expected couples
    missing_couples = numpy.array([12, 58, 96, 120, 136, 139, 141, 150, 163])  # 12, 58, 96... are not there!
    missing_couples = missing_couples-1               # index in Python
    couples_full    = numpy.delete(couples_full, missing_couples)

    if (do_discard==1): # then we discard bad data (couples which are "rejected" or "discarded")
         couples_full = project_array(couples_full, couples_full, discarded_couples)
    
    if (verbosity>0): 
        print("[set_couples] %d couples under consideration" %couples_full.shape)
    return couples_full



# the functions below manipulate subsets of couples (stressed or not, and sex)
# 2022-02-17: added sorbitol data (some kind of stress of the mother)
# 2022-04-08: refactored function
def set_groups(verbosity=-1, restrict="sex", do_discard=0):
    ''' restrict="sex" or "cortisol" depending on which data to keep 
    (data with cortisol values is a subset of data with sex information)
    '''
    # data with HR data:
    couples_full    = set_couples(do_discard=do_discard)

    # data with stress (and sex) information:
    missing_stress  = numpy.array([4, 23, 43, 131])
    couples_stress  = couples_full.copy()
    for i in missing_stress:
        couples_stress = couples_stress[couples_stress!=i]

    missing_sex     = numpy.array([124, 144])
    couples_sex     = couples_stress.copy()
    for i in missing_sex:
        couples_sex = couples_sex[couples_sex!=i]

    missing_cortisol= numpy.array([1, 3, 5, 6, 7, 11, 14, 25, 40, 56, 57, 63, 66, 69, 70, 76, 77, 80, 87, 93, 94, 102, 104, 107, 108, 111, 113, 128, 132, 151, 152, 154, 155, 157, 162])
    couples_cortisol= couples_sex.copy()
    for i in missing_cortisol:
        couples_cortisol = couples_cortisol[couples_cortisol!=i]

    if (verbosity>0):
        print("[set_groups] ", couples_full.shape[0], "couples with HR data")
        #print(couples_full)
        print("[set_groups] ", couples_stress.shape[0], "couples with mother stress data")
        #print(couples_stress)
        print("[set_groups] ", couples_sex.shape[0], "couples with foetus sex data")
        print("[set_groups] ", couples_cortisol.shape[0], "couples with cortisol data")

    # home made csv file, with 5 columns:
    # - ID
    # - stress score (boolean) <=> (PSS >= 19)
    # - fetal sex
    # - PSS score 
    # - PDQ score 
    groups_full = numpy.loadtxt(data_path+'/scores.csv', skiprows=0, delimiter=' ', dtype=int)
    groups_full = groups_full.transpose()
    
    # extra csv file with sorbitol data:
    cortisol    = numpy.loadtxt(data_path+'/cortisol.csv', skiprows=0, delimiter=' ', dtype=int)
    cortisol    = cortisol.transpose()
    
    if (restrict=="stress"):
        missing_ind = missing_stress
        groups_full = -1 # we cannot return anything... edit the .csv file to have something when no sex
        print("set_groups : please use restrict=""sex"" or restrict=""cortisol""")
    elif (restrict=="sex"): # default, corresponds exactly to the .csv file
        missing_ind = numpy.append(missing_stress, missing_sex)         
    elif (restrict=="cortisol"):
        missing_ind = numpy.append(missing_stress, missing_sex)
        missing_ind = numpy.append(missing_ind, missing_cortisol)
        groups_full = project_array(groups_full , groups_full[0], missing_ind) # remove couples without cortisol value
        groups_full = numpy.concatenate((groups_full, cortisol[1,:].reshape(1,cortisol.shape[1])), axis=0) # consolidate all groups/scores

    if (do_discard==1):
        if (verbosity>0): print("[set_groups] I will discard bad couples. ", groups_full.shape[1], end="")
        missing_ind = numpy.append(missing_ind, discarded_couples)
        groups_full = project_array(groups_full , groups_full[0], missing_ind) # remove couples with crappy fHR data
        if (verbosity>0): print("->", groups_full.shape[1])
        
    if (verbosity>0):
        print("\t=> groups_full contains : %d variables for %d couples" %(groups_full.shape[0], groups_full.shape[1]))
    return groups_full, missing_ind



# function to keep only couples for which we have stress and sex information:
# see the other function below for a more versatile function
# 2022-04-08: added possibility for x to be a multi-dimensional array
def project_array(x, x_indices, missing_indices):
    y   = x.copy()
    ind = x_indices.copy()
    for i in missing_indices:
        tmp = (ind!=i)
        ind = ind[tmp]
        y   = y  [...,tmp]
#        print("indice", i, "at", tmp)
    return y

# 2022-04-13: same as function "project_array" above, but much more elegant
def project_array_new(x, x_indices, missing_indices):
    mask = numpy.isin(x_indices, missing_indices, assume_unique=True, invert=True)
    y = x[...,mask]
    return y


# 2022-04-13: function to do the opposite as the function "project_array" above
# (which can thus be rewritten much more efficiently!)
def project_array_keep(x, x_indices, valid_indices):
    mask = numpy.isin(x_indices, valid_indices, assume_unique=True, invert=False)
    y = x[...,mask]
    return y



# given "indices" (couple numbers), this function prepares a mask corresponding to indices locations
# "total" indicates if the mask should have the size of (and should correspond to) all 158 valid couples
#         for which TE or another quantity has been computed, of if it should have the size of only the
#         152 couples for which there is sex data.
#         if "valid", then all couples (with peaks data) are considered
#         if "sex", then only the subset of couples which have an indication on the fetus sex are considered
#         if do_discard=1, then only subset of couples with sex info, and with valid fHR data are considered
def create_mask_from_indices(indices, total="valid", do_discard=0):
    if (total=="valid"):
        couples_full = set_couples(do_discard)            # all 158 valid couples, or 120 with correct fHR
    if (total=="sex"):
        couples_full = set_couples(do_discard)            # all 158 valid couples, or 120 with correct fHR
        groups_full, missing_ind = set_groups() # all 152 couples with stress and sex information
#        print("missing", missing_ind)
        for i in missing_ind:
#            tmp = tmp[couples_full!=i,...]
            couples_full = couples_full[couples_full!=i]
    if (do_discard==1):
        couples_full = set_couples(do_discard=1) # all 120 couples with valid fHR data
#        groups_full, missing_ind = set_groups(do_discard=1) # all 152 couples with stress and sex information
#        print("missing", missing_ind)
#        for i in missing_ind:
#            tmp = tmp[couples_full!=i,...]
#            couples_full = couples_full[couples_full!=i]
        
#    print("[create_mask_from_indices] size of couples_full =", couples_full.shape)
    mask_ind  = numpy.zeros(couples_full.shape, dtype=bool)
#    print("indices to keep:", indices)
    for i in indices:
        mask_ind[couples_full==i] = True 
 
    return(mask_ind)
    

# loads a couple (foetus,mother) HR data (in that order: first mother, second foetus)
# and returns it downsampled at the specified frequency (in Hz)
# NBG 2021/10/05
#
# 2022-03-09 : new parameter "noise_level" (default=0.25)
#              to add a little noise on the HR data, to avoid discretization effects in IT tools.
# 2022-03-22 : new parameter "do_fix_peaks" (default=1)
#              to (for tests) disable the fix_peaks processing stage
# 2022-08-25 : "do_fix_peaks" has now default value 0
# 2022-08-25 : new parameter "quantity" that can be set to "HR" or "RRI", or eventually "peaks"
def load_couple(i, fs=1000, noise_level=0.25, do_fix_peaks=0, do_filter=1, quantity="HR"):
    foetus_filename=f_filepath_peaks+"/fetal_Rpeaks_"+str(i)+".mat"
    mother_filename=m_filepath_peaks+"/mother_Rpeaks_"+str(i)+".mat"
    
    f_file_PEAK_raw=loadmat(foetus_filename)
    m_file_PEAK_raw=loadmat(mother_filename)

    f_peaks=f_file_PEAK_raw['fetal_Rpeaks'][4] #this is my 5th row ECG-SAVER-extracted peaks channel
    m_peaks=m_file_PEAK_raw['mother_Rpeaks'][4] #this is my 5th row
          
    # Trim trailing zeros
    # https://numpy.org/doc/stable/reference/generated/numpy.trim_zeros.html
    fp_trimmed=numpy.trim_zeros(f_peaks, trim='b')
    mp_trimmed=numpy.trim_zeros(m_peaks, trim='b')
   
    # Artifact removal [see next section for details]
    # https://neurokit2.readthedocs.io/en/dev/functions.html?highlight=signal_fixpeaks#neurokit2.signal.signal_fixpeaks
    if (do_fix_peaks==1):
        fp_clean=nk.signal_fixpeaks(fp_trimmed, sampling_rate=1000, iterative=False, show=False,interval_min=0.33,interval_max=0.75, method="kubios")[1] #allow 80-180 bpm
        mp_clean=nk.signal_fixpeaks(mp_trimmed, sampling_rate=1000, iterative=False, show=False,interval_min=0.4,interval_max=1.5, method="kubios")[1] #allow 40-150 bpm
    else:
        fp_clean=fp_trimmed
        mp_clean=mp_trimmed
    
    # compute optimal time range:
#    print("[load_couple]: last peak date =", fp_clean[-1], mp_clean[-1])
#    print("[load_couple]: starting times =", fp_clean[0],  mp_clean[0])
    starting_time = numpy.max([fp_clean[0],  mp_clean[0]])
    ending_time   = numpy.min([fp_clean[-1], mp_clean[-1]])
#    print("[load_couple]: t in [%d; %d]" %(starting_time, ending_time))
    
    # Convert to HR or RRI
    if (quantity=="HR"):
        fhr = peaks_to_HR(fp_clean, sampling_rate=1000, data_length=ending_time)
        mhr = peaks_to_HR(mp_clean, sampling_rate=1000, data_length=ending_time)
#    print("[load_couple]: fHR has shape", fhr.shape)
        full_output=numpy.array([fhr[starting_time:], mhr[starting_time:]], dtype=float)
    elif (quantity=="RRI"):
        frri = peaks_to_RRI(fp_clean, sampling_rate=1000, data_length=ending_time, interpolate=True)
        mrri = peaks_to_RRI(mp_clean, sampling_rate=1000, data_length=ending_time, interpolate=True)
#    print("[load_couple]: fRRI has shape", frri.shape)
        full_output=numpy.array([frri[starting_time:], mrri[starting_time:]], dtype=float)
    
    if (do_filter==1):
        if ((1000%fs)!=0):
            print("warning! I couldn't get simply your required sampling rate %d", fs)
            print("\t use a divisor of the initial value 1000")
        else:
            full_output=causality_tools.filter_FIR(full_output, 1000//fs)
        
    full_output += rng.standard_normal(full_output.shape)*noise_level
    
    if (quantity=="peaks"):
        return numpy.array(fp_clean)-starting_time, numpy.array(mp_clean)-starting_time
    else:
        return full_output
  


