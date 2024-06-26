# -*- coding: ascii -*-
import numpy


# compute RRI signal from a series of peaks timestamps    
# NBG, 2021/10/05
# input parameters:
#    peaks           : contains peaks timestamps
#    sampling_rate   : the sampling rate of peaks 
#    data_length     : max size (in time) of the output
#    do_interpolate  : (boolean) returns a RRI with a constant sampling rate
# returns:
#     RRI     (either time-driven or with constant sampling, depending on "do_interpolate")
def peaks_to_RRI(peaks, sampling_rate=1000, data_length=-1, interpolate=False):
    if (data_length==-1): data_length = numpy.max(peaks)
    
    rri = numpy.diff(peaks) / sampling_rate 
    if interpolate is False:
        return rri
    else:
        RRI = numpy.zeros(data_length, dtype=float)
        
        for i in numpy.arange(peaks.size-1):
            RRI[peaks[i]:peaks[i+1]] = rri[i]  # in seconds
       
        return RRI



# compute HR signal from a series of peaks timestamps    
# NBG, 2021/10/05
#    peaks          : array of indexes where a peak is observed/measured
#    sampling_rate  : obvious, in Hertz
#    data_length    : max size (in time) of the output
# returns: 
#    HR, an array (of appropriate length) with the HR signal, in bpm, and with constant sampling frequency
def peaks_to_HR(peaks, sampling_rate=1000, data_length=-1):
#    print("[peaks_to_HR] : ",data_length, numpy.max(peaks))
    if (data_length==-1): data_length = numpy.max(peaks)
    rri = numpy.diff(peaks)/sampling_rate
#    print("[peaks_to_HR] : length =",data_length)
    
    hr = numpy.zeros(data_length, dtype=float)
    for i in numpy.arange(peaks.size-1):
        hr[peaks[i]:peaks[i+1]] = 60/rri[i]  # in bpm
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
def RRI_to_HR(rri, sampling_rate=20, rri_sampling_rate=1000, data_length=-1):
#    print("[peaks_to_HR] : ",data_length, numpy.max(peaks))
    if (data_length==-1): 
        data_length = (int)(numpy.size(rri)*sampling_rate/rri_sampling_rate)
#    rri = numpy.diff(peaks)/ECG_sampling_rate
#    print("[peaks_to_HR] : length =",data_length)
    
    hr = numpy.zeros(data_length, dtype=float)
    for i in numpy.arange(numpy.size(rri)-1):
        hr[(int)(i*sampling_rate/rri_sampling_rate):(i+1)*int(sampling_rate/rri_sampling_rate)] = 60*1000/rri[i]  # in bpm
    return hr



# cleans RRI data 
# NBG, 2024/03/20
def clean_RRI(x):
    y = x.copy()
    Npts = y.size
    for i in numpy.arange(Npts-1):
        if (y[i]==0):
            j=i+1
            while (y[j]==0) and (j<Npts-1): j+=1
#            if (i<50): print(i, j)
            y[i:j] = x[j]
    return y
        


# examines HR data for erratic values
# NBG, 2024/03/20            
# this function does not touch the data x (HR) but returns a mask where it thinks values are correct   
# the default values are fine for an adult (mother), but not for a foetus: use parameters to adapt.
# input parameters:
#    x        : the HR data
#    HR_min   : the minimum acceptable value for HR (default 40bpm)
#    HR_max   : the maximum acceptable value for HR (default 180bpm)
# returns:
#    a mask     
def mask_HR(x, HR_min=40, HR_max=180):
    mask = numpy.ones(x.shape, dtype='int8')
    bad_ind = numpy.where((x>HR_max) | (x<HR_min))
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
# v2024-04-15, adapted from "causality_tools.py" 
# this improves the dynamics (in terms of nb of non-redondant points)
# and this reduces the signal size (so better for ANN algorithms)
#
# data         : 1-d or 2-d signal
# tau_LP       : nb of pts to average
# f_resampling : how many point per set of tau_LP to keep (oversampling)
#
# 2022-10-24 : added parameter f_resampling (old default was indeed 1)
#            : tested OK (see notebook "causality_couples_2022-10-21_tests_decel")
# 2024-04-15 : now with parameter mask (and using NaN and nanmean)
# 2024-04-17 : now returning a correct nb of points (too any before!)
# 2024-04-19 : parameter mask_strict (default=False):
#                False : the nb of NaN is reduced
#                True  : the nb of NaN is increased over all perturbed measurements
#
def filter_FIR(data, tau_LP, f_resampling=1, mask=None, mask_strict=False, return_time=False):
    ''' low pass filter signal(s) along time (and return a well-ordered 2-d array)
        this improves the dynamics (in terms of nb of non-redondant points)
        and this reduces the signal size (so better for ANN algorithms)

        data         : 1-d or 2-d array with the data
        tau_LP       : (int) nb of pts to average
        f_resampling : (int) how many point per set of tau_LP to keep (oversampling)
        mask         : (bool) use only values given in array mask
        mask_strict  : (bool) False : the nb of NaN is reduced (default)
                              True  : the nb of NaN is increased over all perturbed measurements
        return_time  : (bool) return an extra array with corresponding times for filtered data
    '''
    signals = reorder(data)        
    Npts = signals.shape[1] # along time
    Nout = Npts*f_resampling//tau_LP - (tau_LP-1)
#    print("[filter_LP]", Npts, "pts in data,", Nout, "pts expected in output")
    
    if (tau_LP==1): # no filtering
        if (return_time==True): return data, np.arange(Npts)
        else:                   return data
    
    if isinstance(mask, numpy.ndarray):
        bad_ind = numpy.where(mask==0)  # mask = 0 if data is mmissing
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
    
    if (return_time==True): return signals_LP[:,:-tau_LP+1], numpy.arange(Nout)*tau_LP/f_resampling+(tau_LP-1)//2
    else:                   return signals_LP[:,:-tau_LP+1]


    