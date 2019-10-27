# ESVI_REU_2019
Data analysis of conductivity sensor prototype developed for Dr. Brian Glazer's lab of the University of Hawaii at Manoa.

'calc_rms.py'

Calculates RMS voltage of signal. Set 'data_fn1' and 'data_fn2' variables to file path of oscope csv file. 
Run to generate 'V_rms' values for each salinity test, best-fit line, and plot.

'plot_three_sensors.py'

Remember to perform Calibrate step. Use the best-fit slope (m) and intercept (b) in 'saline_calibration' function.
Adjust 'real_salinity', 'atlas', and 'aanderaa' list values.
Run to generate calculated salinity levels post-calibration for prototype and plot comparing its performance with the other sensors.

'circuit_sim.py'

Taylor Viti's code to predict the circuit performance.

'plot_oscope_data.py'

For plotting oscope data.
