from scipy import signal
from scipy.fft import rfft, rfftfreq
from scipy.optimize import curve_fit
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

def Signal_vs_Noise():
    # Semilog plot for exponential damping
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
    noise_firstpeak_val = fft_noise[1][noise_peak_indexes[0]]
    SNR = signal_firstpeak_val/noise_firstpeak_val
    print("SNR =",SNR)
    
    ax[0].plot(fft_signal[0],fft_signal[1])
    ax[0].scatter(fft_signal[0][signal_peak_indexes],fft_signal[1][signal_peak_indexes], c="r")
    ax[0].set_ylabel("Amplitude")
    ax[0].text(0.96,0.8,"Signal Dominated", horizontalalignment='right', transform=ax[0].transAxes)

    ax[1].plot(fft_noise[0],fft_noise[1])
    ax[1].scatter(fft_noise[0][noise_peak_indexes],fft_noise[1][noise_peak_indexes], c="r")
    ax[1].text(0.96,0.8,"Noise Dominated", horizontalalignment='right', transform=ax[1].transAxes)
    ax[1].set_xlabel("Frequency (Hz)")
    ax[1].set_ylabel("Amplitude")
    ax[0].set_title("Fast Fourier Transform\nSignal Dominated vs Noise Dominated")

    plt.show()

def Find_Frequency():

    # States the frequency at which 
    x_fft, y_fft =  fft(xdata[:last_signal_index+1], ydata[:last_signal_index+1])
    signal_peak_indexes = signal.find_peaks(y_fft)[0]
    signal_firstpeak_freq = x_fft[signal_peak_indexes[0]]#
    signal_firstpeak_amp = y_fft[signal_peak_indexes[0]]
    print("Frequency of First Peak =",signal_firstpeak_freq)

    x_fft = x_fft[3:signal_peak_indexes[0]+7]
    y_fft = y_fft[3:signal_peak_indexes[0]+7]

    def gaussian(x, *params):
        A, x0, c, y0 = params
        return y0 + A * np.exp(-((x - x0) / (np.sqrt(2) * c))**2)

    # Initial guesses
    guess = [signal_firstpeak_amp, signal_firstpeak_freq, 0.1, 0]

    try:
        popt, pcov = curve_fit(gaussian, x_fft, y_fft, p0=guess, maxfev=5000)
    except RuntimeError as e:
        print("Fit did not converge:", e)
        return

    # Print Parameters
    for i in range(len(popt)):
        print(f"Parameter {i}: {popt[i]:.5f} ± {np.sqrt(pcov[i][i]):.5f}")

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

    # Finding the peaks of the first harmonic plot  #
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

    Signal_vs_Noise()
    # Find_Frequency()