# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 15:37:22 2019

@author: Grant
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_time_freq(x, dt):
    # Put together a time vec
    t = np.linspace(0, x.size*dt, num=x.size)

    # Try a rolling window to minimize spectral leakage
    # win =
    
    # Dip into the frequency domain
    X = np.fft.fft(x)
    f = np.fft.fftfreq(x.size, dt)
    
    f_0 = 100e3 # Freq (Hz)
    
    # Try to inject some ringing
    #X[np.abs(f) > f_0] = 0
    #X[np.abs(f) < f_0] = 0
    #x = np.fft.ifft(X)

    # The freq domain vals to actually plot
    X_plot = np.fft.fftshift(X)/len(x)
    f_plot = np.fft.fftshift(f)

    # Plot!
    fig = plt.figure()
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3, sharex=ax2)
    
    ax1.plot(t, x)
    ax2.plot(f_plot, abs(X_plot))
    ax3.plot(f_plot, np.angle(X_plot, deg=True))

    return fig


data_fn = r'D:\test5\NewFile20.csv'

# Read in the data and the acq. params
# data = np.loadtxt
df_pars = pd.read_csv(data_fn, names=["units", "start", "inc"],
                      delimiter=",",
                      skiprows=1,
                      nrows=1,
                      header=None)
df = pd.read_csv(data_fn,
                 skiprows=2,
                 usecols=[0],
                 delimiter=",",
                 names=["x"],
                 header=None)


# Plot the raw data in time and freq
print(max(df["x"]))
fig = plot_time_freq(df["x"], df_pars["inc"][0])
fig.show()
fig.savefig('plot_scope.png')

# L
