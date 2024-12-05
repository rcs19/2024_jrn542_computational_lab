from scipy import signal
from scipy.fft import rfft, rfftfreq
from scipy.optimize import curve_fit
from scipy.stats import linregress
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # Matplotlib plotting library
import os

def Unpack_data(filepath):
    file = np.loadtxt(filepath, delimiter=',')

    xdata = np.array(file[0])
    ydata = np.array(file[1])

    # Finding the peaks of the first harmonic plot  #
    peaks_xindex, peaks_properties = signal.find_peaks(ydata)
    peaks_xindex = np.insert(peaks_xindex,0,0)      # insert index 0 as a peak
    peaks_value = ydata[peaks_xindex]
    # Iteratres through peaks, if next peak is larger than previous then remove peak #
    for i, val in enumerate(peaks_value):         
        try:
            if peaks_value[i+1]>peaks_value[i]:       # Compares if next peak is greater than this peak
                last_signal_index = peaks_xindex[i]   # Last peak before the signal is dominated by noise
                noise_present = True
                break
            else:
                continue
        except:
            last_signal_index = len(xdata)-1  # If there is no noise
            noise_present = False
            print("NoisePresent=",noise_present)
    return xdata, ydata, peaks_xindex, last_signal_index, noise_present

def fft(x,y):
    '''Perform a Fast Fourier Transform given (x,y) arrays of data.
    
    Returns:
    - fft_x: x-values representing frequencies
    - fft_y: y-values representing amplitude at frequency x'''
    simulation_time = x.max()-x.min()
    sample_rate = len(x)/simulation_time 
    N = len(x)  # number of samples, depends on how much signal we have
    # print(f'Sample Rate: {sample_rate:.3f}, N: {N}')
    fft_y = rfft(y)     
    fft_x = rfftfreq(N, d = 1 / sample_rate) # (N = number of samples, d = sample spacing)
    return fft_x, np.abs(fft_y)

def Signal_vs_Noise(filepath, show_plot=False):
    '''
    Returns:
    - slope: damping rate
    - std_error: damping rate error (from linear regression)
    - noise_level: noise level. `np.nan` if completely signal dominated
    '''
    xdata, ydata, peaks_xindex, last_signal_index, noise_present = Unpack_data(filepath)
    # Linear fitting the signal peaks 
    signal_peaks_xindex = peaks_xindex[peaks_xindex<last_signal_index+1]   # obtain indices for all peaks below last signal dominated peak
    y_peaks_signal = ydata[signal_peaks_xindex]
    x_peaks_signal = xdata[signal_peaks_xindex]

    # Noise level
    ydata_noise = ydata[last_signal_index:]
    noise_level = np.mean(ydata_noise)

    # Transforming to log before fitting linear regression.
    log_y = np.log(y_peaks_signal)
    slope, intercept, r_val, p_val, std_err = linregress(x_peaks_signal, log_y)
    log_y_line = slope * x_peaks_signal + intercept
    y_line = np.exp(log_y_line)                     # Transform back to original y-scale
    print(f'Linear-Log Fit Slope = {slope:.3f} ± {std_err:.3f}')

    if show_plot:
        # Semi-log plot for exponential damping
        fig, ax = plt.subplots(figsize=(8,5))

        ax.axvline(x=xdata[last_signal_index], c='black', linestyle='--',label='Last peak before noise-dominated')
        ax.plot(xdata[:last_signal_index+1], ydata[:last_signal_index+1],label='Signal')
        ax.plot(xdata[last_signal_index:], ydata[last_signal_index:],label='Noise', color='red')
        ax.plot(xdata[last_signal_index:],np.ones_like(xdata[last_signal_index:])*noise_level, color='black', alpha=0.5, label=f'Noise Level: {noise_level:.2g}')
        ax.plot(x_peaks_signal, y_line, color='orange', label=f'Linear-Log Fit\nSlope = {slope:.3f} ± {std_err:.3f}')
        ax.scatter(xdata[peaks_xindex],ydata[peaks_xindex],color="r")
        ax.set_xlabel('Time [Normalised]')
        ax.set_ylabel('First harmonic amplitude [Normalised]')
        ax.set_yscale('log')    
        ax.set_title('Damping of First Harmonic Amplitude')
        ax.legend()

        plt.show()

    if noise_present:
        return slope, std_err, noise_level
    else:
        print('No noise detected (data is completely signal dominated), cannot find noise level.')
        return slope, std_err, np.nan

def Find_Frequency(filepath, show_plot=False):
    '''
    Finds the most significant frequency in the signal by fitting a gaussian to largest peak. 
    Returns:
    - omega: angular frequency
    - omega_std: angular frequency error (standard deviation of gaussian)
    '''
    xdata, ydata, peaks_xindex, last_signal_index, noise_present = Unpack_data(filepath)

    x_fft, y_fft =  fft(xdata[:last_signal_index+1], ydata[:last_signal_index+1])
    peaks, properties = signal.find_peaks(y_fft)

    maxpeak_index = peaks[np.argmax(y_fft[peaks])]
    signal_maxpeak_freq = x_fft[maxpeak_index]
    signal_maxpeak_amp = y_fft[maxpeak_index]

    condition = (x_fft>0.1) # & (x_fft<1.2) 
    x_fft_masked = x_fft[condition]
    y_fft_masked = y_fft[condition]
    # print(f'Signal Max Peak Frequency: {signal_maxpeak_freq:.3f} Hz\nSignal Max Peak Amplitude: {signal_maxpeak_amp:.3f} ')

    # Gaussian fitting to most significant (max) peak
    def gaussian(x, *params):   
        A, x0, c, y0, m = params
        return m*x + y0 + A * np.exp(-((x - x0) / (np.sqrt(2) * c))**2) # Gaussian with linear baseline

    # Initial Guesses
    guess = [signal_maxpeak_amp, signal_maxpeak_freq, 0.1, 0, 0]

    try:
        popt, pcov = curve_fit(gaussian, x_fft_masked, y_fft_masked, p0=guess, maxfev=5000)
    except RuntimeError as e:
        print('Fit did not converge:', e)
        return

    # Print Parameters
    # for i in range(len(popt)):
    #     print(f'Parameter {i}: {popt[i]:.5f} ± {np.sqrt(pcov[i][i]):.5f}')
    print(f'Frequency = {popt[1]:.5f} ± {popt[2]:.5f} Hz')
    print(f'Angular Frequency = {popt[1]*np.pi:.5f} ± {popt[2]*np.pi:.5f} s^-1')

    # Plot the results
    if show_plot:
        fig, ax = plt.subplots(figsize=(6, 4))
        x_extended = np.linspace(x_fft.min(), x_fft.max(), 500)
        yfit = gaussian(x_extended, *popt)

        ax.scatter(x_fft, y_fft, label='Data', s=10, color='red', zorder=3)
        ax.plot(x_extended, yfit, label='Fit', zorder=2)
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Amplitude')
        ax.set_title('FFT: Gaussian Fit about Greatest Peak')
        ax.axvspan(x_fft_masked.min(), x_fft_masked.max(), alpha=0.05, color='blue', label='Mask Region', zorder=1)
        ax.legend()
        plt.show()

    # Return the angular frequency (frequency*pi) and its error (standard deviation of the gaussian fit)
    omega = popt[1]*np.pi  
    omega_std = popt[2]*np.pi
    return omega, omega_std

if __name__ == '__main__':

    filepath = 'savedata/session2_variations_05-12-2024/N5000_L4.0pi_Ncells180_T45.7858.txt'
    
    Signal_vs_Noise(filepath, show_plot=True)
    Find_Frequency(filepath, show_plot=True)
