# -*- coding: ascii -*-
import numpy as np
import pyedflib # https://pyedflib.readthedocs.io/en/latest/


# for Felicity 2:
def edf_to_arr(edf_path, verbosity=1):
    f = pyedflib.EdfReader(edf_path)
    n = f.signals_in_file
    if (verbosity>1): print("found", n, "signals/channels")
    if (verbosity>2):
        print("  datarecord_duration", f.datarecord_duration)
        print("  datarecords_in_file", f.datarecords_in_file)
    signal_labels = f.getSignalLabels()
    if (verbosity>1): print("  channels names:", signal_labels)
    if (verbosity>2): print("  equipment     :", f.equipment)
    
    ECG = np.zeros((n, f.getNSamples()[0]))     # create tmp variable for ECG
    nb_ECG = 0                                  # nb of ECG channels found
    found_HRV = 0                               # is there an HRV channel?
    for i in np.arange(n):
        if (verbosity>1): print("  converting signal", i+1,"/",n, "\t", signal_labels[i], end="\t")
        x = f.readSignal(i)
        if (verbosity>1): print("shape", x.shape, "units", f.getPhysicalDimension(i), end=" ")
#        if (0<=i<=2): ECG[i, :] = x
        if (signal_labels[i][0:3]=="ECG"): 
            if (verbosity>2): print("(this is an ECG channel)", end=" ") 
            ECG[nb_ECG, :] = x
            nb_ECG +=1
        if (signal_labels[i]=="HRV"): 
            HRV = x
            found_HRV = 1
        if (verbosity>1): print()
        
    f.close()
    if (verbosity>0): 
        print("  found", nb_ECG, "ECG channels", end=" ")
        if found_HRV:   print("and an HRV channel", end="")
        print()
        
    if found_HRV:   return ECG[:nb_ECG,:], HRV              # there is RRI from device
    else        :   return ECG[:nb_ECG,:], np.array(np.nan) # HRV/RRI from device is missing
    


