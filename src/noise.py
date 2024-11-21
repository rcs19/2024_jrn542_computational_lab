from scipy import signal
import numpy as np
import matplotlib.pyplot as plt # Matplotlib plotting library

file = np.loadtxt("savedata/1000_12.566370614359172_20.txt", delimiter=",")
xdata = np.array(file[0])
ydata = np.array(file[1])

def Get_Signal_and_Noise():
    # Make a semilog plot to see exponential damping
    plt.figure()

    # (Scatter) Plotting the peaks of the first harmonic 
    peaks_xindex, _ = signal.find_peaks(ydata)
    # iteratres through peaks, if next peak is larger than previous then remove peak. 
    peaks_value = ydata[peaks_xindex]
    for i, val in enumerate(peaks_value):
        if peaks_value[i+1]>peaks_value[i]: # Compares if next peak is greater than this peak
            last_signal = peaks_xindex[i]   # Last peak before the signal is dominated by noise
            break
        else:
            last_signal = peaks_xindex[-1]  # If there is no noise
        
    plt.axvline(x=xdata[last_signal], c="black", linestyle="--",label="Last peak before noise-dominated")
    plt.plot(xdata, ydata,label="Signal")
    plt.xlabel("Time [Normalised]")
    plt.ylabel("First harmonic amplitude [Normalised]")
    plt.yscale('log')    
    plt.ioff() # This so that the windows stay open
    plt.legend()
    plt.show()

Get_Signal_and_Noise()