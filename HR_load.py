# -*- coding: ascii -*-
import numpy as np

import pyedflib # https://pyedflib.readthedocs.io/en/latest/

# for Felicity 2:
def edf_to_arr(edf_path, verbosity=1):
    f = pyedflib.EdfReader(edf_path)
    n = f.signals_in_file
    if (verbosity>0): print("found", n, "signals")
#    print("datarecord_duration", f.datarecord_duration)
#    print("datarecords_in_file", f.datarecords_in_file)
    signal_labels = f.getSignalLabels()
    if (verbosity>1): print(signal_labels)
    if (verbosity>1): print(f.equipment)
    
    ECG = np.zeros((n, f.getNSamples()[0]))
    for i in np.arange(n):
        if (verbosity>1): print("converting signal", i+1,"/",n, "\t", signal_labels[i], end="\t")
        x = f.readSignal(i)
        if (verbosity>1): print("shape", x.shape, "units", f.getPhysicalDimension(i))
        if (0<=i<=2): ECG[i, :] = x
        if (signal_labels[i]=="HRV"): HRV = x
        
    f.close()
    if (n<9): return ECG, np.array(np.nan)
    else:     return ECG, HRV



