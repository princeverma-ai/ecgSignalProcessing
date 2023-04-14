import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.io import loadmat 
import scipy as scipy

# Load the ECG signal
ecg_signal = loadmat('./ecg_signal.mat')

# Plot the ECG signal
plt.plot(ecg_signal)
plt.title('Original ECG signal')
plt.xlabel('Sample number')
plt.ylabel('Amplitude')
plt.show()

# Preprocessing: Apply high-pass filtering
b, a = scipy.signal.butter(4, 0.5, 'highpass', fs=1000)
filtered_ecg_signal = scipy.signal.filtfilt(b, a, ecg_signal)

# Choose a denoising algorithm: Wavelet denoising
wavelet = 'db4'
level = 4
mode = 'soft'
denoised_ecg_signal = pywt.wavedec(filtered_ecg_signal, wavelet, level=level)
denoised_ecg_signal[1:] = (pywt.threshold(i, np.std(i)/10, mode) for i in denoised_ecg_signal[1:])
denoised_signal = pywt.waverec(denoised_ecg_signal, wavelet)

# Evaluate the denoising performance
snr = 20*np.log10(np.linalg.norm(ecg_signal)/np.linalg.norm(ecg_signal - denoised_signal))

# Visualize the denoised signal
plt.plot(denoised_signal)
plt.title(f'Denoised ECG signal (SNR={snr:.2f} dB)')
plt.xlabel('Sample number')
plt.ylabel('Amplitude')
plt.show()
