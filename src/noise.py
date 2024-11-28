from scipy import signal
from scipy.fft import rfft, rfftfreq
import numpy as np
import matplotlib.pyplot as plt # Matplotlib plotting library

file = np.loadtxt("savedata/1000_12.566370614359172_20.txt", delimiter=",")
xdata = np.array(file[0])
ydata = np.array(file[1])

def fft(x,y):
    """Perform a Fast Fourier Transform given (x,y) arrays of data."""
    simulation_time = x.max()-x.min()
    sample_rate = len(x)/simulation_time 
    N = len(x)                # number of samples
    fft_y = rfft(y)     
    fft_x = rfftfreq(N, d = 1 / sample_rate) # (N = number of samples, d = sample spacing)
    return fft_x, np.abs(fft_y)

def Get_Signal_and_Noise():
    #                                               #
    # Finding the peaks of the first harmonic plot  #
    #                                               #
    peaks_xindex, _ = signal.find_peaks(ydata)
    peaks_value = ydata[peaks_xindex]
    for i, val in enumerate(peaks_value):         # iteratres through peaks, if next peak is larger than previous then remove peak. 
        if peaks_value[i+1]>peaks_value[i]:       # Compares if next peak is greater than this peak
            last_signal_index = peaks_xindex[i]   # Last peak before the signal is dominated by noise
            break
        else:
            last_signal_index = peaks_xindex[-1]  # If there is no noise

    xdata_noise = xdata[last_signal_index:]       # x and y arrays of noise dominated data
    ydata_noise = ydata[last_signal_index:]

    plt.figure()        
    plt.axvline(x=xdata[last_signal_index], c="black", linestyle="--",label="Last peak before noise-dominated")
    plt.plot(xdata[:last_signal_index+1], ydata[:last_signal_index+1],label="Signal")
    plt.plot(xdata_noise, ydata_noise,label="Noise", color="red")
    plt.xlabel("Time [Normalised]")
    plt.ylabel("First harmonic amplitude [Normalised]")
    plt.yscale('log')    
    plt.title("Damping of First Harmonic Amplitude")
    plt.ioff() # This so that the windows stay open
    plt.legend()
    plt.show()

    # # FFT of the whole range 0 to 20 seconds
    # all = fft(xdata,ydata)
    # plt.plot(all[0],all[1])
    # plt.show()
    
    #                                                                       #
    #  Fast Fourier transform of signal dominated part and noise dominated part  #
    #                                                                       #
    fft_signal =  fft(xdata[:last_signal_index+1], ydata[:last_signal_index+1])
    fft_noise = fft(xdata_noise,ydata_noise)

    # Plotting FFT
    fig, ax = plt.subplots(2,1, figsize = (8,4), sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0)
    ax[0].plot(fft_signal[0],fft_signal[1])
    ax[0].set_ylabel("Amplitude")
    ax[0].text(0.9,0.8,"Signal Dominated", horizontalalignment='right', transform=ax[0].transAxes)
    ax[1].plot(fft_noise[0],fft_noise[1])
    ax[1].text(0.9,0.8,"Noise Dominated", horizontalalignment='right', transform=ax[1].transAxes)
    ax[1].set_xlabel("Frequency (Hz)")
    ax[1].set_ylabel("Amplitude")

    ax[0].set_title("Fast Fourier Transform\nSignal Dominated vs Noise Dominated")

    plt.show()

Get_Signal_and_Noise()