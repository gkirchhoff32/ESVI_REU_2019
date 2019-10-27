# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 12:56:53 2019

@author: Grant
"""

# plot my values against the other sensors

import matplotlib.pyplot as plt
#from new_sensor import saline
import rms
from rms import N_fn
import numpy as np

def saline_calibration(vrms_list):
    salinity = []
    # newly calibrated using 7.0 and 35.0 salinity values
    for i in vrms_list:
        m = 0.02118707
        b = 0.09562055
        S = (i - b)/m
        salinity.append(S)
        
    return salinity


n_fn1 = np.arange(2, N_fn, 2)
n_fn2 = np.arange(1, N_fn, 2)

# calls RMS function from rms.py, returns list of rms values from tests

vrms_list = rms.RMS(n_fn1,n_fn2)

real_salinity = [3.0, 7.0, 12.0, 15.0, 20.0, 25.0, 30.0, 35.0, 1.0]
atlas = [0.84, 3.22, 7.5, 12.7, 15.7, 20.82, 25.9, 31.0, 36.1]
aanderaa = [0.94, 3.11, 6.96, 12.2, 15.2, 20.1, 25.8, 30.98, 36.11]
saline_cal = saline_calibration(vrms_list)
print(saline_cal)

plt.plot(np.unique(real_salinity), np.unique(real_salinity), label='Actual Salinity')
plt.plot(np.unique(real_salinity), atlas, 'o',markersize=4, label='Atlas')
plt.plot(np.unique(real_salinity), aanderaa, 'o',markersize=4, label='Aanderaa')
plt.plot(np.unique(real_salinity), np.unique(saline_cal), 'o',markersize=4, label='Prototype')
plt.legend()
plt.xlabel('Predicted Output (ppt)')
plt.ylabel('Observed Output (ppt)')
plt.title('Sensor Accuracy')
plt.savefig('three_sensor_plot.png')
