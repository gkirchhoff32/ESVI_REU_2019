
'''
Conductivity sensor circuit simulation
Written by Taylor Viti July 2019
'''

from scipy import signal as sig
from matplotlib import pyplot as plt
import numpy as np


def Z_par(Z1, Z2):
    Z_out = np.zeros(Z2.shape, dtype=np.complex)

    Z_out = 1.0/(1.0/Z1 + 1.0/Z2)

    return Z_out


# Construct an input signal
f_0 = 100e3  # Signal freq. (Hz)
f_s = 100*(2*f_0)  # Sample freq
T = 10.0/f_0  # Window length
t = np.arange(-T, T, step=1.0/f_s)  # Time
dt = t[1] - t[0]

# Cosine input
# A = 10
# v_i = A*np.cos(2*np.pi*f_0*t)

# Square wave input
A = 1.76
v_i = A*sig.square(2*np.pi*f_0*(t - 1/4/f_0))

# Go into the freq. domain
V_i = np.fft.fft(v_i)
f = np.fft.fftfreq(len(v_i), d=1.0/f_s)
omega = 2*np.pi*f

# Try to inject some ringing
#V_i[np.abs(f) > 20*f_0] = 0
#v_i = np.fft.ifft(V_i)

# Shifted signal, strictly for plotting (note the factor
# 1/N on the signal, to approximate a CTFT)
f_shifted = np.fft.fftshift(f)
V_i_shifted = np.fft.fftshift(V_i)/len(v_i)

# Plot the input signal in time and frequency
plt.close("all")
fig = plt.figure()
ax1 = fig.add_subplot(3, 1, 1)
ax2 = fig.add_subplot(3, 1, 2)
ax3 = fig.add_subplot(3, 1, 3, sharex=ax2)
ax1.plot(t*1e3, v_i)
ax1.set_xlabel("time [ms]")
ax1.set_xlim(0, 2*1e3/f_0)
ax2.plot(f_shifted*1e-3, np.abs(V_i_shifted))
ax3.plot(f_shifted*1e-3, np.angle(V_i_shifted))
ax3.set_xlabel("freq. [kHz]")
ax1.set_title("input signal")


def TF(omega, n_4, n_1, R_A, R_w, L):
    # Construct the transfer func. and output signal
    # Note the small value added to epsilon, to avoid singularities
    return n_4/n_1/(1 + n_4**2*R_w*(1.0/R_A + 1.0/1j/(omega + 1e-16)/L))


def ac_couple(omega, R, C):
    # Transfer function for AC coupling to diff amp inputs
    return 1j*R*omega*C/(1j*R*omega*C + 1)


def diff_amp(omega, Rg):
    """ Mimick an instrumentation amplifier using a butterworth lowpass filter"""
    f_c = 3.4e6  # cutoff freq
    G = (1 + 1e4/Rg)
    b, a = sig.butter(4, f_c, 'low', analog=True)
    w, H = sig.freqresp((b, a), w=omega)
        
    return G*H
    
        
# Transfer function over a couple of different salinities (PSU?)
n_4 = 40
n_1 = 40
R_A = 100
#R_A = Z_par(100, 1.0/1j/(omega + 1e-12)/(5e-9))
#R_A = Z_par(100, 1.0/1j/(omega + 1e-12)/(10e-6))
L = 150*1e-6
# sal = np.logspace(1, np.log10(35), 5)
# K_c = 110.0
# R_w = K_c/(sal*0.1)
#R_w = np.logspace(-1, np.log10(22), 10)
#R_w = np.array([22, 50, 100, 1000, 10000, np.inf])
#R_w = np.array([1e12])
R_w = np.array([22])
H = np.zeros((len(R_w), len(f)), dtype=np.complex)
for (i, r) in enumerate(R_w):
    H[i, :] = TF(omega, n_4, n_1, R_A, r, L)

# Compute time domain TF signal, and fftshifted, for plotting
#h = np.fft.ifft(H)
#H_shifted = np.fft.fftshift(H)

# Pass the input signal through the sensor
V_o = H*np.tile(V_i, reps=(H.shape[0], 1))
#V_o_shifted = np.fft.fftshift(V_o)/len(v_i)
#v_o = np.fft.ifft(V_o, axis=1)

# Pass the signal through the AC coupling stage
H_1 = np.tile(ac_couple(omega, 445, 5e-9), reps=(V_o.shape[0], 1))
V_o = V_o*H_1
H = H*H_1
#V_o_shifted = np.fft.fftshift(V_o)/len(v_i)
#v_o = np.fft.ifft(V_o, axis=1)

# Pass the signal through a mock "diff-amp" stage
#H_2 = np.tile(diff_amp(omega, 1), reps=(V_o.shape[0], 1))
#V_o = V_o*H_2
#H = H*H_2
#V_o_shifted = np.fft.fftshift(V_o)/len(v_i)
#v_o = np.fft.ifft(V_o, axis=1)

# One final amplification stage
H_3 = 1
V_o = V_o*H_3
H = H*H_3
V_o_shifted = np.fft.fftshift(V_o)/len(v_i)
v_o = np.fft.ifft(V_o, axis=1)

# Compute time domain TF signal, and fftshifted, for plotting
h = np.fft.ifft(H)
H_shifted = np.fft.fftshift(H)

# Try to do QAM in time to extract the out imaginary part of the output
v_m_op = np.sin(2*np.pi*f_0*t)  # Multiplier signal
v_m_ip = np.cos(2*np.pi*f_0*t)  # Multiplier signal
q_ip = v_o*np.tile(v_m_ip, reps=(v_o.shape[0], 1))  # Demodulate
q_op = v_o*np.tile(v_m_op, reps=(v_o.shape[0], 1))  # Demodulate
# Boxcar filter in freq to remove aliases
Q_ip = np.fft.fft(q_ip, axis=1)
Q_op = np.fft.fft(q_op, axis=1)
Q_ip[:, np.abs(f) >= f_0] = 0
Q_op[:, np.abs(f) >= f_0] = 0
# Multiplier to recover amplitude (is this necessary?)
Q_ip = 2*Q_ip
Q_op = 2*Q_op
q_ip = np.fft.ifft(Q_ip, axis=1)
q_op = np.fft.ifft(Q_op, axis=1)

# Shift for plotting
Q_ip_shifted = np.fft.fftshift(Q_ip)/len(v_i)
Q_op_shifted = np.fft.fftshift(Q_op)/len(v_i)

# See if we can recover the resistivities from the QAM demod. signal
# Note that this is only gonna work for the sinusoidal input signal
# (since in that case, we can do all the freq. domain maths in time domain)
# q_ip_mean = np.mean(q_ip, axis=1).real
# R_w_hat = R_A/n_4**2*(n_4/n_1*A/q_ip_mean - 1)

# Plot the output in time and frequency
fig = plt.figure()
ax1 = fig.add_subplot(3, 1, 1)
ax2 = fig.add_subplot(3, 1, 2)
ax3 = fig.add_subplot(3, 1, 3, sharex=ax2)
ax1.set_title("Predicted Output Signal", fontsize=45)
ax1.plot(t*1e3, v_o.T, linewidth=4)
ax1.set_xlabel("time (ms)", fontsize=32)
ax1.set_ylabel("voltage (V)", fontsize=32)
ax1.set_xlim(0, 2*1e3/f_0)
ax2.plot(f_shifted*1e-3, np.abs(V_o_shifted.T))
ax3.plot(f_shifted*1e-3, np.angle(V_o_shifted.T))
ax3.set_xlabel("freq. [kHz]")
ax3.legend(R_w)

# Plot the qam signal in time and frequency
fig = plt.figure()
ax1 = fig.add_subplot(3, 2, 1)
ax2 = fig.add_subplot(3, 2, 3)
ax3 = fig.add_subplot(3, 2, 5, sharex=ax2)
ax1.set_title("in phase")
ax1.plot(t*1e3, q_ip.T)
ax1.set_xlabel("time [ms]")
ax1.set_xlim(0, 2*1e3/f_0)
ax2.plot(f_shifted*1e-3, np.abs(Q_ip_shifted.T))
ax3.plot(f_shifted*1e-3, np.angle(Q_ip_shifted.T))
ax3.set_xlabel("freq. [kHz]")
ax3.legend(R_w)

ax1 = fig.add_subplot(3, 2, 2)
ax2 = fig.add_subplot(3, 2, 4)
ax3 = fig.add_subplot(3, 2, 6, sharex=ax2)
ax1.set_title("quad")
ax1.plot(t*1e3, q_op.T)
ax1.set_xlabel("time [ms]")
ax1.set_xlim(0, 2*1e3/f_0)
ax2.plot(f_shifted*1e-3, np.abs(Q_op_shifted.T))
ax3.plot(f_shifted*1e-3, np.angle(Q_op_shifted.T))
ax3.set_xlabel("freq. [kHz]")

# Plot the transfer function
fig = plt.figure()
ax1 = fig.add_subplot(3, 1, 1)
ax2 = fig.add_subplot(3, 1, 2)
ax3 = fig.add_subplot(3, 1, 3)
ax1.plot(t*1e3, h.T)
ax1.set_xlabel("time [ms]")
ax1.set_xlim(0, 2*1e3/f_0)
ax2.plot(f_shifted*1e-3, np.abs(H_shifted.T))
ax3.plot(f_shifted*1e-3, np.angle(H_shifted.T))
ax3.set_xlabel("freq. [kHz]")
ax1.set_title("response function")

# See if we successfully recovered R_w
# fig = plt.figure()
# ax = fig.add_subplot()
# ax.scatter(R_w, R_w_hat)
# ax.plot([R_w[0], R_w[-1]], [R_w[0], R_w[-1]])
# ax.set_ylabel("R_w obs.")
# ax.set_xlabel("R_w true.")

plt.show()
