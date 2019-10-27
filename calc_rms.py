# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 16:43:33 2019

Generate RMS voltage as a function of the salinity.
Inputs: csv file containing signals from oscope from the increasing saline solutions.
Outputs: Plots rms voltage relationship and prints linear best fit equation.


@author: Grant
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Find RMS Voltage of the signal
def V_rms(x1, x2, dt):
    x3 = []
    for i in range(len(x1)):
           x3.append(x1[i]-x2[i])
    rms = np.sqrt((1./len(x3))*np.sum(np.square(x3-np.mean(x3)))) # RMS calculations  
    return rms

# Build arrays of file numbers to loop over
N_fn = 22
n_fn1 = np.arange(2, N_fn, 2)
n_fn2 = np.arange(1, N_fn, 2)
v_rms, v_rms_1, v_rms_2 = [],[],[]
flist = []

### USE THIS IF 'BAD' SIGNAL IN TEST ###
# List of file numbers to exclude from the final data set

# test3:
#n_double = [(1,2),(3,4),(41,42),(47,48),(49,50)]

# test4
#n_double = []

# test5
n_double = [(1,2)] # In this case, my first saline solution had a 'bad' signal\
                   # so I exlude it from my calculations.

# Calculate v_rms for pair of files
def RMS(n_fn1,n_fn2):
    for (n1, n2) in zip(n_fn1, n_fn2):
        if not((n2, n1) in n_double):     
            data_fn1 = r'D:\test5\NewFile{0:d}.csv'.format(n1)
            data_fn2 = r'D:\test5\NewFile{0:d}.csv'.format(n2)
            
            # Read in the data and the acquisition params
            df_pars = pd.read_csv(data_fn2, names=["units", "start", "inc"],
                                  delimiter=",",
                                  skiprows=1,
                                  nrows=1,
                                  header=None)
            df1 = pd.read_csv(data_fn2,
                             skiprows=2,
                             usecols=[0],
                             delimiter=",",
                             names=["x"],
                             header=None)
            
            df2 = pd.read_csv(data_fn1,
                             skiprows=2,
                             usecols=[0],
                             delimiter=",",
                             names=["x"],
                             header=None)
            
            v = V_rms(df1["x"], df2["x"], df_pars["inc"][0])
            print("V_rms = {0:f}".format(v))
            
            N_pts = df1.shape[0]
            v1 = V_rms(df1["x"][0:int(N_pts/2)], \
                       df2["x"][0:int(N_pts/2)], \
                       df_pars["inc"][0])
            v2 = V_rms(df1["x"][int(N_pts/2):].values,\
                       df2["x"][int(N_pts/2):].values, \
                       df_pars["inc"][0])
            v_rms_1.append(v1)
            v_rms_2.append(v2)
            v_rms.append(v)
            
            # Keep track of the files for each calculation
            flist.append((data_fn1, data_fn2))
    
    return v_rms
    
if __name__ == "__main__":
    
    RMS(n_fn1,n_fn2)
    
    # test3
    #sals = [0.5, 0.9, 1.6, 3.0, 4.5, 8.0, 12.0, 16.0, 20.0, 28.0, 35.0, 2.5, \
    #        3.5, 6.0, 10.0, 14.0, 18.0, 24.0, 31.0, 33.0, 22.0, 26.0, 34.0]
    
    # test4
    #sals = [1, 3, 7, 12]
    
    # test5
    sals = [3, 7, 12, 15, 20, 25, 30, 35, 1]
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_title('Output Voltage Relationship with Salinity')
    ax.set_xlabel('Salinity (ppt)')
    ax.set_ylabel('RMS Voltage (V)')
    ax.plot(sals, v_rms, 'b.')
    ax.plot(sals, np.poly1d(np.polyfit(sals, v_rms, 1))(sals), 'r')
    
    m,b=np.polyfit(sals,v_rms,1)
    yp=np.polyval([m,b],sals)
    equation='y = ' + str(round(m,4)) + 'x' ' + ' + \
    str(round(b,4))
    print(equation)
    
    plt.savefig('output_voltage_vs_salinity.png')
    