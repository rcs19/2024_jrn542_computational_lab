from scipy import signal
from scipy.fft import rfft, rfftfreq
from scipy.optimize import curve_fit
from scipy.stats import linregress
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # Matplotlib plotting library

def fft(x,y):
    """Perform a Fast Fourier Transform given (x,y) arrays of data."""
    simulation_time = x.max()-x.min()
    sample_rate = len(x)/simulation_time 
    N = len(x)  # number of samples, depends on how much signal we have
    # print(f"Sample Rate: {sample_rate:.3f}, N: {N}")
    fft_y = rfft(y)     
    fft_x = rfftfreq(N, d = 1 / sample_rate) # (N = number of samples, d = sample spacing)
    return fft_x, np.abs(fft_y)

def Signal_vs_Noise():
    # Linear fitting the peaks
    #
    peaks_xindex, _ = signal.find_peaks(ydata)
    peaks_xindex = np.insert(peaks_xindex,0,0)                      # insert index 0 as a peak
    peaks_xindex = peaks_xindex[peaks_xindex<last_signal_index+1]   # obtain indices for all peaks below last signal dominated peak
    y_signal = ydata[peaks_xindex]
    x_signal = xdata[peaks_xindex]

    # Transforming to log before fitting linear regression.
    log_y = np.log(y_signal)
    slope, intercept, r_val, p_val, std_err = linregress(x_signal, log_y)
    # for i, val in enumerate([slope, intercept, r_val, p_val, std_err]):
    #     print("Parameter {}: {:.5f}".format(i,val)) # Print the linear regression parameters
    print(f"Slope = {slope:.5f} ± {std_err:.5f}")
    log_y_line = slope * x_signal + intercept
    y_line = np.exp(log_y_line)                     # Transform back to original y-scale

    # Semilog plot for exponential damping
    plt.figure()    

    plt.axvline(x=xdata[last_signal_index], c="black", linestyle="--",label="Last peak before noise-dominated")
    plt.plot(xdata[:last_signal_index+1], ydata[:last_signal_index+1],label="Signal")
    plt.plot(xdata_noise, ydata_noise,label="Noise", color="red")
    plt.plot(x_signal, y_line, label="Linear-Log Fit")

    plt.xlabel("Time [Normalised]")
    plt.ylabel("First harmonic amplitude [Normalised]")
    plt.yscale('log')    
    plt.title("Damping of First Harmonic Amplitude")
    plt.ioff() # This so that the windows stay open
    plt.legend()
    plt.show()
    
    #                                                                            #
    #  Fast Fourier transform of signal dominated part and noise dominated part  #
    #                                                                            #
    fft_signal =  fft(xdata[:last_signal_index+1], ydata[:last_signal_index+1])
    fft_noise = fft(xdata_noise,ydata_noise)

    # Plotting FFT
    signal_peak_indexes = signal.find_peaks(fft_signal[1])[0]
    noise_peak_indexes = signal.find_peaks(fft_noise[1])[0]
    fig, ax = plt.subplots(2,1, figsize = (8,4), sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0)

    # Signal-to-Noise Ratio (SNR)
    signal_firstpeak_val = fft_signal[1][signal_peak_indexes[0]]
    try:
        noise_firstpeak_val = fft_noise[1][noise_peak_indexes[0]]
        SNR = signal_firstpeak_val/noise_firstpeak_val
        print("SNR =",SNR)

    except:
        print("No Noise to FFT")
    
    ax[0].plot(fft_signal[0],fft_signal[1])
    ax[0].scatter(fft_signal[0][signal_peak_indexes],fft_signal[1][signal_peak_indexes], c="r")
    ax[0].set_ylabel("Amplitude")
    ax[0].text(0.96,0.8,"Signal Dominated", horizontalalignment='right', transform=ax[0].transAxes)
    ax[0].set_title("Fast Fourier Transform\nSignal Dominated vs Noise Dominated")
    try:
        ax[1].plot(fft_noise[0],fft_noise[1])
        ax[1].scatter(fft_noise[0][noise_peak_indexes],fft_noise[1][noise_peak_indexes], c="r")
        ax[1].text(0.96,0.8,"Noise Dominated", horizontalalignment='right', transform=ax[1].transAxes)
        ax[1].set_xlabel("Frequency (Hz)")
        ax[1].set_ylabel("Amplitude")
    except:
        print("No Noise To Plot")

    plt.show()

def Find_Frequency():
    """
    Finds the most significant frequency in the signal. 
    """
    x_fft, y_fft =  fft(xdata[:last_signal_index+1], ydata[:last_signal_index+1])
    peaks, properties = signal.find_peaks(y_fft)
    print(f"Peaks: {peaks}")

    maxpeak_index = peaks[np.argmax(y_fft[peaks])]
    signal_maxpeak_freq = x_fft[maxpeak_index]
    signal_maxpeak_amp = y_fft[maxpeak_index]
    condition = (x_fft>0.1) & (x_fft<2)
    x_fft = x_fft[condition]
    y_fft = y_fft[condition]

    print(f"Signal Max Peak Frequency: {signal_maxpeak_freq:.3f} Hz, Signal Max Peak Amplitude: {signal_maxpeak_amp:.3f} ")

    plt.plot(x_fft,y_fft)
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.show()
    def gaussian(x, *params):   
        A, x0, c, y0, m = params
        return m*x + y0 + A * np.exp(-((x - x0) / (np.sqrt(2) * c))**2) # Gaussian with linear baseline

    # Initial guesses
    guess = [signal_maxpeak_amp, signal_maxpeak_freq, 0.1, 0, 0]

    try:
        popt, pcov = curve_fit(gaussian, x_fft, y_fft, p0=guess, maxfev=5000)
    except RuntimeError as e:
        print("Fit did not converge:", e)
        return

    # Print Parameters
    # for i in range(len(popt)):
    #     print(f"Parameter {i}: {popt[i]:.5f} ± {np.sqrt(pcov[i][i]):.5f}")
    print(f"Frequency = {popt[1]:.5f} ± {popt[2]:.5f}")
    print(f"Angular Frequency = {popt[1]*np.pi:.5f} ± {popt[2]*np.pi:.5f}")

    # Plot the results
    x_extended = np.linspace(x_fft.min(), x_fft.max(), 500)
    yfit = gaussian(x_extended, *popt)

    plt.figure(figsize=(10, 4))
    plt.scatter(x_fft, y_fft, label="Data", s=5)
    plt.plot(x_extended, yfit, label="Fit", color="red")
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()

if __name__ == "__main__":

    file = np.loadtxt("savedata/session2_repeats/1000_12.566370614359172_20-1.txt", delimiter=",")
    # file = np.loadtxt("savedata/1000_12.566370614359172_20.txt", delimiter=",")
    xdata = np.array(file[0])
    ydata = np.array(file[1])

    # Finding the peaks of the first harmonic plot  #
    peaks_xindex, _ = signal.find_peaks(ydata)
    peaks_value = ydata[peaks_xindex]
    for i, val in enumerate(peaks_value):         # iteratres through peaks, if next peak is larger than previous then remove peak. 
        try:
            if peaks_value[i+1]>peaks_value[i]:       # Compares if next peak is greater than this peak
                last_signal_index = peaks_xindex[i]   # Last peak before the signal is dominated by noise
                break
            else:
                continue
        except:
            last_signal_index = peaks_xindex[-1]  # If there is no noise
    xdata_noise = xdata[last_signal_index:]       # x and y arrays of noise dominated data
    ydata_noise = ydata[last_signal_index:]

    Signal_vs_Noise()
    Find_Frequency()