# -*- coding: ascii -*-
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
# N.B.G. 2024/04/05
def compute_HR_entropy_rate(data, stride_values, mask=None, N_shuffles=0, do_filter=False, fs_in=1000, fs_out=20):
    ''' compute entropy rate of (HR or any type of) data, over a range of time-scales

    input parameters:
      data      : a 1d numpy array with the data
      stride    : array of stride values (timescales) to consider (in points)
      N_shuffles: nb of shuffles to perform and average over (default=0)
      do_filter : filter (FIR) the signal or not
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
            fs_in = fs
            fs_out= fs
            if isinstance(mask, np.ndarray):
                data2 = HRt.filter_FIR(x, stride*fs_in//fs_out, f_resampling=stride, mask=mask)
                mask = mask[(stride-1)//2:(-stride+1)//2]
            else:
                data2 = HRt.filter_FIR(x, stride*fs_in//fs_out, f_resampling=stride)
#            print("stride", stride, x.shape, "vs", data2.shape)
            x = data2.copy()
        std = np.std(x)
        h0[i] = np.log(std)
          
        if isinstance(mask, np.ndarray):
            h[i]   = entropy.compute_entropy_rate(x, stride=stride, mask=mask)
        else:
            h[i]   = entropy.compute_entropy_rate(x, stride=stride)
                
        [h_std[i]] = entropy.get_last_info()[:1]
    
        # bias estimate with shuffling:
        if (N_shuffles>0):
            for i_shuffle in np.arange(N_shuffles):
                x_shuffled = entropy.surrogate(x)
        
                if isinstance(mask, np.ndarray):
                    h_bias[i]+=entropy.compute_entropy_rate(x_shuffled, stride=stride, mask=mask)
                    h_bias[i]-=entropy.compute_entropy(x_shuffled, stride=stride, mask=mask)
                else:
                    h_bias[i]+=entropy.compute_entropy_rate(x_shuffled, stride=stride)
                    h_bias[i]-=entropy.compute_entropy(x_shuffled, stride=stride)

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
            mask = mask[(stride-1)//2:(-stride+1)//2]
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



