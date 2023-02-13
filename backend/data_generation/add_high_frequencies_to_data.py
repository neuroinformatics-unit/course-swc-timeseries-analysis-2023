"""
Add high frequency noise to the original data. Make it in the shape of a
smile using a biexponential function for the smile and gaussian for the eyes.
These functions are chosen to keep the frequency spectrum smooth - sharp
discontinuities will create ripples in the time domain after iDFT.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


# Load Data
repo_path = Path(r"C:\Users\Joe\work\git-repos\swc-timeseries-analysis-course-2023\backend\data_generation")
raw_data = pd.read_csv(repo_path / "sub-001_drug-cch_origdata.csv")

signal = raw_data["current_pa"].to_numpy()
time_s = raw_data["time_s"].to_numpy()

ts = time_s[1]
N = signal.size


# Take the FFT
signal_mean = np.mean(signal)
signal = signal - signal_mean

freqs = np.fft.fftfreq(N, d=ts)
X = np.fft.fft(signal)


# Make a smooth smile and add it to the FFT
# at pos and neg freqs
new_spec = X.copy()

n = np.linspace(0, 1, 50000)
left_smile = 150*(1 - np.exp(-n / 0.05)) * np.exp(-n / 0.2) + 1

plt.plot(left_smile)
plt.show()

n = np.linspace(-1, 1, 5000)
eye = 50 * np.exp(-0.5 * ((n - 0) / 0.1)**2) + 1

plt.plot(eye)
plt.show()

new_spec[500000:550000] *= left_smile
new_spec[550000:600000] *= np.flip(left_smile)

new_spec[540000:545000] *= eye
new_spec[555000:560000] *= eye

new_spec[600000:650000] *= left_smile
new_spec[650000:700000] *= np.flip(left_smile)

new_spec[640000:645000] *= eye
new_spec[655000:660000] *= eye

plt.plot(freqs, np.abs(new_spec))
plt.plot(freqs,  np.abs(X))
plt.show()


# Take the iFFT of the smile spectrum and save
reconstruct = np.real(np.fft.ifft(new_spec))
reconstruct += signal_mean  # add the mean back

plt.plot(time_s, reconstruct)
plt.show()

to_save = pd.DataFrame(np.c_[time_s, reconstruct], columns=["time_s", "current_pa"])
to_save.to_csv(repo_path / "sub-001_drug-cch_rawdata.csv")
