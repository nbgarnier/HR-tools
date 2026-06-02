# -*- coding: ascii -*-
# version 2026-02-19
import numpy

# compute RRI signal from a series of peaks timestamps    
# NBG, 2021/10/05
# input parameters:
#    peaks           : contains peaks timestamps
#    sampling_rate   : the sampling rate of peaks 
#    unit            : 's' for seconds, or "ms" (default) for milliseconds (new 2025/10/20)
#    data_length     : max size (in time) of the output
#    do_clean        : (boolean) removes spurious zeros at the beginning
#    do_interpolate  : (boolean) returns a RRI with a constant sampling rate
# returns:
#     RRI     (either time-driven or with constant sampling, depending on "do_interpolate")
#             RRI unit is either 's' or 'ms' depending on the parameter chosen
def peaks_to_RRI(peaks, sampling_rate=1000, unit="ms", data_length=-1, do_clean=False, do_interpolate=False):
    ''' compute RRI signal from a series of peaks timestamps    
    
    input parameters:
      peaks           : contains peaks timestamps
      sampling_rate   : the sampling rate of peaks 
      data_length     : max size (in time) of the output
      do_clean        : (boolean) removes spurious zeros at the beginning (but introduce a time shift)
                        set it to False if you want to keep original time units from the peaks data,
                        e.g., to compare with RRI or peaks time-traces (default: False)
      do_interpolate  : (boolean) returns a RRI with a constant sampling rate (default: False)
    returns:
      RRI     (either time-driven or with constant sampling, depending on "do_interpolate")
    '''
    
    rri = numpy.diff(peaks) / sampling_rate # in seconds
    if unit=="ms":
        rri = rri * 1000
    if do_interpolate is False:
        return rri
    else:
        if (data_length==-1): 
            if do_clean is True: peaks = peaks-peaks[0]
            data_length = numpy.max(peaks) # same as peaks[-1]
        RRI = numpy.zeros(data_length, dtype=float)
        
        for i in numpy.arange(peaks.size-1):
            RRI[peaks[i]:peaks[i+1]] = rri[i]  # in seconds or milliseconds
       
        return RRI



# compute HR signal from a series of peaks timestamps    
# NBG, 2021/10/05
#    peaks          : array of indexes where a peak is observed/measured
#    sampling_rate  : obvious, in Hertz
#    data_length    : max size (in time) of the output, removed 2025/10/20, and replaced with parameter below:
#    do_clean       : (boolean) removes spurious zeros at the beginning (default True)
#                      set it to False if you want to keep original time units from the peaks data,
#                      e.g., to compare with RRI or peaks time-traces
# returns: 
#    HR, an array (of appropriate length) with the HR signal, in bpm, and with constant sampling frequency
def peaks_to_HR(peaks, sampling_rate=1000, do_clean=True):
    ''' compute HR signal from a series of peaks timestamps
    
    input parameters:
      peaks          : array of indexes where a peak is observed/measured
      sampling_rate  : of the peaks data, in Hertz
      do_clean       : (boolean) removes spurious zeros at the beginning (default True)
                       set it to False if you want to keep original time units from the peaks data,
                       e.g., to compare with RRI or peaks time-traces
    returns: 
      HR, an array (of appropriate length) with the HR signal, in bpm, and with constant sampling frequency
      equal to the sampling frequency of the peaks
    '''
    if do_clean==True: peaks = peaks-peaks[0]   # shift the time axis
    data_length = numpy.max(peaks) # same as peaks[-1]
    rri = numpy.diff(peaks)/sampling_rate
    
    hr = numpy.zeros(data_length, dtype=float)
    for i in numpy.arange(peaks.size-1):
        hr[peaks[i]:peaks[i+1]] = 60/rri[i] if rri[i]>0 else numpy.nan # in bpm
    return hr


# converts RRI data into HR data
# NBG, 2024/03/20
# input parameters:
#    rri                 : the RRI data
#    sampling_rate       : the HR (output) sampling rate (default 20)
#    rri_sampling_rate   : the RRI sampling rate (default 1000)
#    data_length         : how long the output will be (-1 for all data)
# returns:
#    HR  
def RRI_to_HR(rri, sampling_rate=20, rri_sampling_rate=1000, rri_unit="ms", data_length=-1):
    ''' converts RRI data into HR data
    
    input parameters:
      rri                 : the RRI data
      sampling_rate       : the HR (output) sampling rate (in Hz, default 20)
      rri_sampling_rate   : the RRI sampling rate (in Hz, default 1000)
      rri_unit            : unit in which the RRi is expressed (default 'ms')
      data_length         : how long the output will be (-1 for all data)
    returns:
      HR, in bpm, sampled at sampling_rate
    '''
    if (data_length==-1): 
        data_length = (int)(numpy.size(rri)*sampling_rate/rri_sampling_rate)
    
    if rri_unit=="ms":    # added 2025/10/20
        rri = rri/1000    # in seconds
    hr = numpy.zeros(data_length, dtype=float)
    for i in numpy.arange(numpy.size(rri)-1):
        hr[(int)(i*sampling_rate/rri_sampling_rate):int((i+1)*sampling_rate/rri_sampling_rate)] = 60/rri[i]  # in bpm
    return hr



# converts RRI data into another RRI data, with (usually larger) new sampling rate
# NBG, 2024/08/29
# input parameters:
#    rri                 : the input RRI data
#    sampling_rate       : the output sampling rate (default 20)
#    rri_sampling_rate   : the RRI sampling rate (default 1000)
#    data_length         : how long the output will be (-1 for all data)
# returns:
#    rri
def RRI_to_RRI(rri, sampling_rate=20, rri_sampling_rate=1000, data_length=-1):
    ''' converts RRI data into another RRI data, with (usually larger) new sampling rate
    
    input parameters:
      rri                 : the input RRI data
      sampling_rate       : the output sampling rate (default 20)
      rri_sampling_rate   : the RRI sampling rate (default 1000)
      data_length         : how long the output will be (-1 for all data)
    returns:
      rri
    '''
    if (data_length==-1): 
        data_length = (int)(numpy.size(rri)*sampling_rate/rri_sampling_rate)
    
    rri_new = numpy.zeros(data_length, dtype=float)
    for i in numpy.arange(numpy.size(rri)-1):
        rri_new[(int)(i*sampling_rate/rri_sampling_rate):int((i+1)*sampling_rate/rri_sampling_rate)] = rri[i]  # in milliseconds
    return rri_new



# cleans RRI data 
# NBG, 2024/03/20
def clean_RRI(x):
    ''' clean RRI data by removing spurious ou duplicated values

    this function is aimed to be used with Bittium device RRI data (labeled "HR" but RRI)
    '''
    y = x.copy()
    Npts = y.size
    for i in numpy.arange(Npts-1):
        if (y[i]==0):
            j=i+1
            while (y[j]==0) and (j<Npts-1): j+=1
#            if (i<50): print(i, j)
            y[i:j] = x[j]
    return y



def check_HR(x, HR_min=40, HR_max=180):
    ''' examines HR data for erratic values
    this function does not touch the data x (HR) but returns a new array with only correct values 
    and NaNs where values are not correct
    the default values are fine for an adult (mother), but not for a foetus: adapt parameters for foetus.

    input parameters:
      x        : the HR data in bpm
      HR_min   : the minimum acceptable value for HR (default 40bpm)
      HR_max   : the maximum acceptable value for HR (default 180bpm)
    returns:
      a checked version of x    
    '''
    y = x.copy()

    # check for bad values:
    bad_ind = numpy.where((x>HR_max) | (x<HR_min))
    y[bad_ind] = numpy.nan

    return y



# examines HR data for erratic values
# NBG, 2024/03/20            
# this function does not touch the data x (HR) but returns a mask where it thinks values are correct   
# the default values are fine for an adult (mother), but not for a foetus: use parameters to adapt.
# input parameters:
#    x        : the HR data in bpm
#    HR_min   : the minimum acceptable value for HR (default 40bpm)
#    HR_max   : the maximum acceptable value for HR (default 180bpm)
#    do_check_bad_values : if True, the mask will ignore HR values outisde the range [HR_min; HR_max]
# returns:
#    a mask     
#
# 2025/10/20: now checking for NaNs
def mask_HR(x, HR_min=40, HR_max=180, do_check_bad_values=True):
    ''' examines HR data for erratic values
    this function does not touch the data x (HR) but returns a mask where it thinks values are correct   
    the default values are fine for an adult (mother), but not for a foetus: adapt parameters for foetus.

    input parameters:
      x        : the HR data in bpm
      HR_min   : the minimum acceptable value for HR (default 40bpm)
      HR_max   : the maximum acceptable value for HR (default 180bpm)
      do_check_bad_values : a boolean (default=True). 
                 If True, the mask will ignore HR values outside the range [HR_min; HR_max]
                 If False, the mask will only ignore NaNs

    returns:
      a mask for the data x (with 1 for a good value and 0 for a bad value)
    '''
    mask = numpy.ones(x.shape, dtype='int8')
    
    # check for bad values:
    if do_check_bad_values:
        bad_ind = numpy.where((x>HR_max) | (x<HR_min))
        mask[bad_ind]=0

    # check for NaN of Inf:
    wrong_ind = numpy.where(numpy.isnan(x))
    mask[wrong_ind]=0

    return mask



# examines RRI data for erratic values
# NBG, 2024/08/29            
# this function does not touch the data x (input RRI) but returns a mask where it thinks values are correct   
# the default values are fine for an adult (mother), but not for a foetus: use parameters to adapt.
# input parameters:
#    x        : the RRI data in milliseconds
#    RRI_min  : the minimum acceptable value for RRI (default 333 ms, ie 180 bpm)
#    RRI_max  : the maximum acceptable value for HR (default 1500 ms, ie 40 bpm)
# returns:
#    a mask     
def mask_RRI(x, RRI_min=333, RRI_max=1500):
    ''' examines RRI data for erratic values
    this function does not touch the data x (input RRI) but returns a mask where it thinks values are correct   
    the default values are fine for an adult (mother), but not for a foetus: use parameters to adapt.
    
    input parameters:
      x        : the RRI data in milliseconds
      RRI_min  : the minimum acceptable value for RRI (default 333 ms, ie 180 bpm)
      RRI_max  : the maximum acceptable value for HR (default 1500 ms, ie 40 bpm)
    returns:
      a mask  
    '''
    mask = numpy.ones(x.shape, dtype='int8')
    bad_ind = numpy.where((x>RRI_max) | (x<RRI_min))
    mask[bad_ind]=0
    
    return mask
    


# this usefull function insures that the time dimension is the first dimension
# if we are dealing with a 1-d ndarray, it is cast into a 2-d array
# 2023-10-26: now also insures that data is C-contiguous
def reorder(x):
    """
    makes any nd-array compatible with any function of the code for temporal-like signals
    
    :param x: any nd-array
    :returns: a well aligned and ordered nd-array containing the same data as input x
    """
    if (x.ndim==1):    # ndarray of dim==1
        x=x.reshape((1,x.size))
    elif (x.ndim==2):  # regular matrix (for multivariate 1-d data)
        if (x.shape[0]>x.shape[1]):
            x=x.transpose()
    else:
        print("please use reorder_2d for multivariate images")
    if x.flags['C_CONTIGUOUS']: return x
    else:                       return x.copy()
    

   
# low pass filter signal(s) along time (and return a 2-d array)
# 2024-04-15, adapted from "causality_tools.py" 
# this improves the dynamics (in terms of nb of non-redondant points)
# and this reduces the signal size (so better for ANN algorithms)
#
# data         : 1-d or 2-d signal
# tau_LP       : nb of pts to average
# f_resampling : how many point per set of tau_LP to keep (oversampling)
#
# 2022-10-24 : added parameter f_resampling (old default was indeed 1)
#              tested OK (see notebook "causality_couples_2022-10-21_tests_decel")
# 2024-04-15 : now with parameter mask (and using NaN and nanmean)
# 2024-04-17 : now returning a correct nb of points (too any before!)
# 2024-04-19 : parameter mask_strict (default=False):
#                False : the nb of NaN is reduced
#                True  : the nb of NaN is increased over all perturbed measurements
#
def filter_FIR(data, tau_LP, f_resampling=1, mask=None, mask_strict=False, return_time=False):
    ''' low pass filter signal(s) along time (and return a well-ordered 2-d array)
    -> improves the dynamics (in terms of nb of non-redondant points) of piecewise-constant data
    -> (eventually) reduces the signal size (so better for ANN algorithms)

    input parameters:
      data         : 1-d or 2-d array with the data
      tau_LP       : (int) nb of pts to average
      f_resampling : (int) how many points per set of tau_LP to keep (oversampling)
      mask         : (bool) use only values given in array mask
      mask_strict  : (bool) False : the nb of NaN is reduced (default)
                            True  : the nb of NaN is increased over all perturbed measurements
      return_time  : (bool) return an extra array with exact timestamps of the filtered data
    '''
    signals = reorder(data)        
    Npts = signals.shape[1] # along time
    Nout = Npts*f_resampling//tau_LP - (tau_LP-1)
#    print("[filter_LP]", Npts, "pts in data,", Nout, "pts expected in output")
    
    if (tau_LP==1): # no filtering
        if (return_time==True): return data, np.arange(Npts)
        else:                   return data
    
    if isinstance(mask, numpy.ndarray):
        bad_ind = numpy.where(mask==0)  # mask = 0 if data is missing
        signals[:,bad_ind] = numpy.nan  # => we put a NaN there
    
#    print(Npts, "->", (Npts*f_resampling)//tau_LP, "pts in time")
    signals_LP=numpy.zeros((signals.shape[0], (Npts*f_resampling)//tau_LP), dtype="float")
    shift = tau_LP//f_resampling
    for i in range(signals.shape[0]):
        for j in numpy.arange(f_resampling):
            tmp = signals[i,j*shift:]
            Npts_decimated = tmp.size//tau_LP
#            print(Npts_decimated, "(new Npts)")
            tmp = numpy.reshape(tmp[:(Npts_decimated*tau_LP)], (Npts_decimated,tau_LP))
#            print(tmp.shape, "->", numpy.mean(tmp, axis=1).shape, "vs", signals_LP[i,j::f_resampling].shape)
            if (mask_strict==True):
                signals_LP[i,j:Npts_decimated*f_resampling:f_resampling] = numpy.mean(tmp, axis=1)
            else:
                signals_LP[i,j:Npts_decimated*f_resampling:f_resampling] = numpy.nanmean(tmp, axis=1)
#        print("%2d : %3d -> %5d non-duplicates" %(i, count_non_duplicates(signals[i,:]), count_non_duplicates(signals_LP[i,:])))
    
    if (return_time==True): return signals_LP[:,:-tau_LP+1], (numpy.arange(Nout)*tau_LP/f_resampling+(tau_LP-1)//2)
    else:                   return signals_LP[:,:-tau_LP+1]


    