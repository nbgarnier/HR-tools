import numpy as np
import neurokit2 as nk


def ECG_to_peaks(ECG, sampling_rate, do_filtering=True, do_cleaning=True, lowcut=0.5, highcut=40, method='butterworth', order=4, verbosity=0):
    ''' find and return R peaks in an ECG time-trace.
    
    ECG is a nd-array of shape (number of ECG channels, length)
    if "do_filtering" is True, some filgtering (with parameters provided) is performe on ECG
    if "do_cleaning" is True, some filtering (non controlable) is performed on the ECG
    
    returns a list of nd-arrays, one per ECG channel
    '''
    N_channels=ECG.shape[0]
    found_peaks=[]

    # Filter the data to remove noise (e.g., using a low-pass filter)
    # https://neuropsychology.github.io/NeuroKit/functions/signal.html#neurokit2.signal_filter
    filtered_ECG = ECG
    if do_filtering:
        for i in range(N_channels):
            filtered_ECG[i,:] = nk.signal_filter(ECG[i,:], lowcut=lowcut, highcut=highcut, method=method, order=order, sampling_rate=sampling_rate)
            
    # Clean the data (that's what NK fanboyz are suggesting)
    # https://neuropsychology.github.io/NeuroKit/functions/ecg.html#neurokit2.ecg.ecg_clean
    clean_ECG = filtered_ECG
    if do_cleaning:
        for i in range(N_channels):
            clean_ECG[i,:] = nk.ecg_clean(filtered_ECG[i,:], sampling_rate=sampling_rate)

    for i in range(N_channels):
        try:
            peaks = (nk.ecg_peaks(clean_ECG[i,:], sampling_rate=sampling_rate)[1])['ECG_R_Peaks']
            # https://neuropsychology.github.io/NeuroKit/functions/ecg.html#ecg-peaks
            mean_HR = peaks.size*1000*60/clean_ECG.shape[1]
            if (verbosity>0): print("ECG channel", i+1, ":", peaks.size, "peaks found, i.e., average HR about %2.2f bpm" %(mean_HR))
            if (verbosity>2): print("   peaks:", peaks)
            found_peaks.append(peaks)
            
        except Exception:
            print("    problem with ECG channel", i+1)
            found_peaks.append([np.nan])

    return found_peaks