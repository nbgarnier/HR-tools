# -*- coding: ascii -*-
# version 2026-02-19
import numpy as np
import HR_tools as HRt
import entropy.entropy as entropy
import entropy.tools as tools    


# to compute the entropy rate from HR data, over a range of time-scales
# data      : a 1d numpy array with the data
# stride    : array of stride values to consider
# N_shuffles: nb of shuffles to perform and average over (default=0)
# do_filter : filter (FIR) the signal or not
# fs_in     : sampling frequency of input data
#             fs_in = 1000 for ECG-derived HR
#             fs_in = 5    for device produced RRI, and infered HR
# fs_out    : effective sampling frequency after filtering
#             fs_out = 20 is a good compromise to have enough points
#
# returns 3 or 4 arrays in the following order:
# h, h_std (the std), h_bias (the bias)
# h0 (the entropy normalization due to the standard deviation)
#
# N.B.G. 2024/04/05, edited 2025/10/22
# 2026/05/26 : new parameter cutoff_timescale (for experimenting)
def compute_HR_entropy_rate(data, stride_values, mask=None, 
                            N_shuffles=0, do_filter=False, cutoff_timescale=1,
                            fs_in=1000, fs_out=20, verbosity=1):
    ''' compute entropy rate of (HR or any type of) data, over a range of time-scales

    input parameters:
      data      : a 1d NumPy array containing the data
      stride    : array of stride values (timescales) to consider (in points)
      N_shuffles: nb of shuffles to perform and average over (default=0)
      do_filter : filter (FIR) the signal or not
      cutoff_timescale : an integer to give the ratio stride/tau_LP for the filtering cutoff tau_LP,
                    i.e., the nb of pts that are averaged is given by tau_LP = stride / cutoff_timescale
                    (default 1, i.e., tau_LP = stride)
                    Experiment: set it to 2 to filter "less" and see the effect.
      fs_in     : sampling frequency of input data
                  fs_in = 1000 for ECG-derived HR
                  fs_in = 5    for device produced RRI, and infered HR
      fs_out    : effective sampling frequency after filtering
                  fs_out = 20 is a good compromise to have enough points

    returns 4 arrays in the following order:
      h, h_std (the std), h_bias (the bias) and h0 (the entropy normalization due to the standard deviation)
    '''
    
    s = data.shape
    if len(s)==1: x = tools.reorder(data)
    else:         x = tools.reorder(data[0])   

    h     =np.zeros(stride_values.shape, dtype=float)
    h_std =np.zeros(stride_values.shape, dtype=float)
    h_bias=np.zeros(stride_values.shape, dtype=float)
    h0    =np.zeros(stride_values.shape, dtype=float)  # the effect of the std / normalization
    
    i=0
    for stride in stride_values:   
        # filtering 
        if (do_filter==True):
#            fs_in = fs
#            fs_out= fs
            tau_LP = stride // cutoff_timescale # added 2026/05/26
            if (tau_LP<1): tau_LP = 1
            if isinstance(mask, np.ndarray):
                data2 = HRt.filter_FIR(x, tau_LP*fs_in//fs_out, f_resampling=tau_LP, mask=mask)
#                mask2 = mask[(stride-1)//2:(-stride+1)//2]
                mask2 = HRt.mask_HR(data2, do_check_bad_values=False)   # 2026-02-18, added parameter "do_check_bad_values"
                                                    # and set it to False to de-activate checking for bad values,
                                                    # because if data is normalized, values are completely off
                mask2 = mask2[0,:]
            else:
                data2 = HRt.filter_FIR(x, tau_LP*fs_in//fs_out, f_resampling=tau_LP)
                mask2 = mask
            if (verbosity>1):
                print("stride", stride, "tau_LP", tau_LP, "initial data shape", x.shape, "filtered into", data2.shape, "with", np.sum(mask2), "good values")
        else:
            data2 = x 
            mask2 = mask
        std = np.nanstd(x)     # 2026-05-12: replaced std by nanstd
        h0[i] = np.log(std)
          
        if isinstance(mask, np.ndarray):
            if (verbosity>1): 
                print("using masked data of shape", data2.shape, "with mask of shape", mask2.shape )
            h[i]   = entropy.compute_entropy_rate(data2, stride=stride, mask=mask2)
        else:
            h[i]   = entropy.compute_entropy_rate(data2, stride=stride)
                
        [h_std[i]] = entropy.get_last_info()[:1]
    
        # bias estimate with shuffling: cannot work with masking!!!
        if (N_shuffles>0):
            for i_shuffle in np.arange(N_shuffles):
                x_shuffled = entropy.surrogate(data2)
        
                if isinstance(mask2, np.ndarray):
                    h_bias[i]+=entropy.compute_entropy_rate(x_shuffled, stride=stride, mask=mask2)
#                    h_bias[i]-=entropy.compute_entropy(x_shuffled, stride=stride, mask=mask2)
                else:
                    h_bias[i]+=entropy.compute_entropy_rate(x_shuffled, stride=stride)
#                    h_bias[i]-=entropy.compute_entropy(x_shuffled, stride=stride)

            h_bias[i]/=N_shuffles
            
        i+=1
    
    return h, h_std, h_bias, h0 


# to compute the entropy rate from HR data, in sliding windows, over a single time-scale
# data      : a 1d numpy array with the HR data
# stride    : the stride value to consider
# T         : the window size (in points)
# overlap   : the overlap between 2 consecutive windows (in points)
# fs        : the sampling frequency (Hz) for data
# N_shuffles: nb of shuffles to perform and average over (default=0)
# do_return_time : if ==1, then returns an extra vector with dates of centers of time windows
#
# returns 3 arrays in the following order:
# h, h_std (the std), h_bias (the bias)
# and an extra 4th array with confidence index, if asked for (parameter "do_confidence" set to 1)
#
def compute_window_HR_entropy_rate(data, stride, T=300, overlap=0, fs=20, mask=None, N_shuffles=0, 
                                   do_what="entropy", do_return_time=1, do_filter=False, do_confidence=False):
    ''' compute entropy rate of (HR or any type of) data, in sliding windows, over a single time-scale

    input parameters:
      data      : a 1d numpy array with the HR data
      stride    : the stride value, i.e., the timescale to consider (in points)
      T         : the window size (in points)
      overlap   : the overlap between 2 consecutive windows (in points)
      fs        : the sampling frequency (Hz) for data
      N_shuffles: nb of shuffles to perform and average over (default=0)
      do_what   : "entropy" for entropy rate
                  "complex" for ApEn and SampEn
                  "std"  
      do_return_time : if ==1, then returns an extra vector with timestamps of the centers of time windows
    
    returns 3 arrays in the following order:
        h, h_std (the std), h_bias (the bias)
    and if asked for (parameter "do_confidence" set to 1) an extra 4th array with confidence index
    '''
    
    s = data.shape
    if len(s)==1: x = tools.reorder(data)
    else:         x = tools.reorder(data[0])
    
    N_pts = x.shape[1]
    N_windows = (N_pts-T)//(T-overlap) + 1 
    print(N_windows, "time-windows")
    
    h      = np.zeros(N_windows, dtype=float)
    h2     = np.zeros(N_windows, dtype=float) # for a second measure (SampEn or top of ApEn if "complexities)
    h_std  = np.zeros(N_windows, dtype=float)
    h_bias = np.zeros(N_windows, dtype=float)
    h2_bias= np.zeros(N_windows, dtype=float)
    t      = np.zeros(N_windows, dtype=float)
    confid = np.zeros(N_windows, dtype=float)

    # filtering (stride is constant in this function)
    if (do_filter==True):
        fs_in = fs
        fs_out= fs
        if isinstance(mask, np.ndarray):
            data2 = HRt.filter_FIR(x, stride*fs_in//fs_out, f_resampling=stride, mask=mask)
            mask = mask[(stride-1)//2:(-stride+1)//2] # 2025/10/22: works only if no resampling (ie fs_in=fs_out)
        else:
            data2 = HRt.filter_FIR(x, stride*fs_in//fs_out, f_resampling=stride)
        x = data2.copy()
    
    for i in np.arange(N_windows):   
        i_start = i*(T-overlap)
        i_end   = i_start + T
        t[i]    = (i_start + T/2)/fs
        
        if do_confidence:
            my_x      = x[:,i_start:i_end]
            my_mask   = HRt.mask_HR(my_x)
            confid[i] = np.sum(my_mask)/T  # fraction of pts with no problem
        
        if isinstance(mask, np.ndarray):
            print(x[:,i_start:i_end].shape, np.sum(mask[i_start:i_end]), "valid pts")
            if do_what=="entropy":
                h[i]   = entropy.compute_entropy_rate(x[:,i_start:i_end], stride=stride, mask=mask[i_start:i_end])
            elif do_what=="complex":
                a,b  = entropy.compute_complexities(x[:,i_start:i_end], stride=stride, mask=mask[i_start:i_end])
                h[i]  = a[-1] # ApEn
                h2[i] = b[-1] # SampEn
        else:
            if do_what=="entropy":
                h[i]   = entropy.compute_entropy_rate(x[:,i_start:i_end], stride=stride)
            elif do_what=="complex":
                a,b = entropy.compute_complexities(x[:,i_start:i_end], stride=stride)
                h[i]  = a[-1] 
                h2[i] = b[-1]
            
        [h_std[i]] = entropy.get_last_info()[:1]
    
        # bias estimate with shuffling: (as of 2024/08/29, bias is only computed for entropy rate, not for complexities)
        if (N_shuffles>0):
            for i_shuffle in np.arange(N_shuffles):
                x_shuffled = entropy.surrogate(x[:,i_start:i_end])  # Shuffle 
        
                if isinstance(mask, np.ndarray):
                    if do_what=="entropy":
                        h_bias[i]+=entropy.compute_entropy_rate(x_shuffled, stride=stride, mask=mask[i_start:i_end])
                        h_bias[i]-=entropy.compute_entropy(x_shuffled, stride=stride, mask=mask[i_start:i_end])
                    elif do_what=="complex":
                        a,b  = entropy.compute_complexities(x[:,i_start:i_end], stride=stride, mask=mask[i_start:i_end])
                        h_bias[i]  += a[-1] # ApEn
                        h2_bias[i] += b[-1] # SampEn

                else:
                    if do_what=="entropy":
                        h_bias[i]+=entropy.compute_entropy_rate(x_shuffled, stride=stride) 
                        h_bias[i]-=entropy.compute_entropy(x_shuffled, stride=stride)
                    elif do_what=="complex":
                        a,b = entropy.compute_complexities(x[:,i_start:i_end], stride=stride)
                        h_bias[i]  = a[-1] 
                        h2_bias[i] = b[-1]

            h_bias[i] /=N_shuffles
            h2_bias[i]/=N_shuffles
    
    if do_what=="entropy":
        returned_variables = [h, h_std, h_bias]
    elif do_what=="complex":    # as of 2024/08/29, bias is only computed for entropy rate, not for complexities
        returned_variables = [h, h2, h_std, h_bias, h2_bias]
            
    if (do_return_time==1): 
        returned_variables.append(t)
    if do_confidence:   
        returned_variables.append(confid)
    
    return returned_variables



# to compute ApEn and SampEn from HR data, over a range of time-scales
# data      : a 1d numpy array with the data
# stride    : array of stride values to consider
# N_shuffles: nb of shuffles to perform and average over (default=0)
# do_filter : filter (FIR) the signal or not
# fs_in     : sampling frequency of input data
#             fs_in = 1000 for ECG-derived HR
#             fs_in = 5    for device produced RRI, and infered HR
# fs_out    : effective sampling frequency after filtering
#             fs_out = 20 is a good compromise to have enough points
#
# returns 3 or 4 arrays in the following order:
# h, h_std (the std), h_bias (the bias)
# h0 (the entropy normalization due to the standard deviation)
#
# N.B.G. 2025/11/04
def compute_HR_complexities(data, stride_values, mask=None, 
                            N_shuffles=0, do_filter=False, cutoff_timescale=1,
                            fs_in=1000, fs_out=20, verbosity=1):
    ''' compute entropy rate of (HR or any type of) data, over a range of time-scales

    input parameters:
      data      : a 1d numpy array with the data
      stride    : array of stride values (timescales) to consider (in points)
      N_shuffles: nb of shuffles to perform and average over (default=0)
      do_filter : filter (FIR) the signal or not
      cutoff_timescale : an integer to give the ratio stride/tau_LP for the filtering cutoff tau_LP,
                    i.e., the nb of pts that are averaged is given by tau_LP = stride / cutoff_timescale
                    (default 1, i.e., tau_LP = stride)
                    Experiment: set it to 2 to filter "less" and see the effect.
      fs_in     : sampling frequency of input data
                  fs_in = 1000 for ECG-derived HR
                  fs_in = 5    for device produced RRI, and infered HR
      fs_out    : effective sampling frequency after filtering
                  fs_out = 20 is a good compromise to have enough points

    returns 4 arrays in the following order:
      h, h_std (the std), h_bias (the bias) and h0 (the entropy normalization due to the standard deviation)
    '''
    
    s = data.shape
    if len(s)==1: x = tools.reorder(data)
    else:         x = tools.reorder(data[0])   
    
    AE     =np.zeros(stride_values.shape, dtype=float)
    AE_std =np.zeros(stride_values.shape, dtype=float)
    AE_bias=np.zeros(stride_values.shape, dtype=float)
    SE     =np.zeros(stride_values.shape, dtype=float) 
    SE_std =np.zeros(stride_values.shape, dtype=float)
    SE_bias=np.zeros(stride_values.shape, dtype=float)
    
    i=0
    for stride in stride_values:   
        # filtering 
        if (do_filter==True):
#            fs_in = fs
#            fs_out= fs
            tau_LP = stride // cutoff_timescale # added 2026/05/26
            if (tau_LP<1): tau_LP = 1
            if isinstance(mask, np.ndarray):
                data2 = HRt.filter_FIR(x, tau_LP*fs_in//fs_out, f_resampling=tau_LP, mask=mask)
#                mask2 = mask[(stride-1)//2:(-stride+1)//2]
                mask2 = HRt.mask_HR(data2, do_check_bad_values=False)   
                mask2 = mask2[0,:]
            else:
                data2 = HRt.filter_FIR(x, tau_LP*fs_in//fs_out, f_resampling=tau_LP)
                mask2 = mask
            if (verbosity>1):
                print("stride", stride, "tau_LP", tau_LP, "initial data shape", x.shape, "filtered into", data2.shape, "with", np.sum(mask2), "good values")
        else:
            data2 = x 
            mask2 = mask
#        std = np.nanstd(x)     # 2026-05-12: replaced std by nanstd
          
        if isinstance(mask, np.ndarray):
            if (verbosity>1): 
                print("using masked data of shape", data2.shape, "with mask of shape", mask2.shape )
            ApEn, SampEn = entropy.compute_complexities(data2, stride=stride, mask=mask2)
        else:
            ApEn, SampEn = entropy.compute_complexities(data2, stride=stride)
#            print("for stride=", stride, "ApEn :", ApEn, "SampEn :", SampEn)
        
        AE[i] = ApEn[-1]
        SE[i] = SampEn[-1]
        [AE_std[i]] = entropy.get_last_info()[:1]
        [SE_std[i]] = entropy.get_last_info()[1:2]
    
        # bias estimate with shuffling: cannot work with masking!!!
        if (N_shuffles>0):
            for i_shuffle in np.arange(N_shuffles):
                x_shuffled = entropy.surrogate(data2)
        
                if isinstance(mask2, np.ndarray):
                    ApEn, SampEn = entropy.compute_complexities(x_shuffled, stride=stride, mask=mask2)
                else:
                    ApEn, SampEn = entropy.compute_complexities(x_shuffled, stride=stride)
                AE_bias[i] += ApEn[-1] 
                SE_bias[i] += SampEn[-1] 
            
            AE_bias[i] /=N_shuffles
            SE_bias[i] /=N_shuffles
            
        i+=1
    
    return AE, SE, AE_std, SE_std, AE_bias, SE_bias




# average over a band of time-scales values:
# x             the quantity to be averaged
# timescale     the set of available timescales (in points, not in seconds)
# fs            the sampling rate (to convert timescales from points to seconds)
# t_min, t_max  the bounds of the time-interval to use
def time_average(x, timescale, fs, t_min=0.5, t_max=2.5):
    t           = timescale/fs  # timescales in seconds
    index_range = np.array(np.where((t_min<=t) & (t<=t_max)))
    mean        = x[:,index_range[0]]   # retain only a band of lags
    
    return mean


