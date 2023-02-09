from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy

show_all_plots = False

# Load the Data
repo_path = Path(r"C:\fMRIData\git-repo\swc_timeseries_course")
pandas_dataframe = pd.read_csv(repo_path / "sub-001_drug-cch_rawdata.csv")

raw_signal = pandas_dataframe["im_pa"].to_numpy()
time_s = pandas_dataframe["time_s"].to_numpy()

# Plot the Data
if show_all_plots:
    plt.plot(time_s, raw_signal)
    plt.ylabel("Im (pA)")
    plt.xlabel("Time (s)")
    plt.title(" Original Loaded Data")
    plt.show()

# Play with time, try and reconstruct the time array
N = raw_signal.size
ts = time_s[1] - time_s[0]  # 0.001
fs = 1/ts                   # 10,000
signal_length_s = 120

time_array1 = np.linspace(start_time, stop_time, num_samples)  # method 1
time_array2 = np.arange(num_samples) * a_value                 # method 2

# checks every sample matches exactly
assert np.array_equal(first_array, second_array)

# checks every sample matches up to 5 decimal places
assert np.allclose(first_array, second_array, rtol=0, atol=0.00001)


reconstruct_time1 = np.linspace(0, 120 - ts, N)
reconstruct_time2 = np.arange(N) * ts

breakpoint()

# this will fail, floating point issue
# assert np.array_equal(time_s, reconstruct_time1)
# assert np.array_equal(time_s, reconstruct_time2)

assert np.allclose(time_s, reconstruct_time1, rtol=0, atol=0.00000001)
assert np.allclose(time_s, reconstruct_time2, rtol=0, atol=1e-08)

# Take the FFT


# signal_demean += 1 optional sanity check on the magnitude scaling

freqs = np.fft.fftfreq(num_samples, d=sample_spacing)

signal_demean = raw_signal - np.mean(raw_signal)
signal_fft = np.fft.fft(signal_demean)

magnitude_fft = np.abs(signal_fft)

# note this scales the 0 Hz frequency by 2x what it should be.
scale_magnitude_fft = magnitude_fft * 2 / N

def show_line_plot(x, y, ylabel="", xlabel="", title=""):
    """ Make and show a matplotlib lineplot """
    plt.plot(x, y)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.show()


# Plot the FFT (TODO: separate module)

import plotters  # import at top of file

plotters.show_line_plot(x=freqs,
                        y=scale_magnitude_fft,
                        ylabel="Magnitude (pA)",
                        xlabel="Frequency (Hz)",
                        title="Demeaned Signal Frequency Spectrum")


if show_all_plots:
    plt.stem(freqs[:1000], magnitude_fft[:1000],  markerfmt=" ")
    plt.ylabel("Magnitude (pA)")
    plt.xlabel("Frequency (Hz)")
    plt.title("Demeaned Signal Frequency Spectrum")
    plt.show()

if show_all_plots:


# signal_ = np.fft.ifft(signal_fft)

# Filter the Signal
b, a = scipy.signal.butter(8, 
                     Wn=4000,
                     btype='low',
                     analog=False,
                     output='ba',
                     fs=sampling_rate)
signal_filtered = scipy.signal.filtfilt(b, a, signal)

filt_signal_fft = np.fft.fft(signal_filtered)
filt_magnitude_fft = np.abs(filt_signal_fft) * 2 / N

show_line_plot(time_s, signal_filtered, "Im (pA", "Time (s)", "Filtered Signal")

show_line_plot(freqs, filt_magnitude_fft, "Magnitude (pA)", "Frequency (Hz)", "Filtered Signal")

plt.plot(time_s, signal_filtered)  # make own function
plt.ylabel("Im (pA)")
plt.xlabel("Time (s)")
plt.title("Filtered Signal")
plt.show()


# Plot the FFT (TODO: separate module)
if show_all_plots:
    plt.plot(freqs, filt_magnitude_fft)
    plt.ylabel("Magnitude (pA)")
    plt.xlabel("Frequency (Hz)")
    plt.title("Demeaned Signal Frequency Spectrum")
    plt.show()

    plt.plot(time_s, signal_filtered)  # make own function
    plt.ylabel("Im (pA)")
    plt.xlabel("Time (s)")
    plt.title("Filtered Signal")
    plt.show()

# Find the mid-idx
mid_idx = np.where(time_s == 60)[0][0]  # note tuple index
pre_drug = signal_filtered[:mid_idx]
post_drug = signal_filtered[mid_idx:]

pre_drug_time = time_s[:mid_idx]
post_drug_time = time_s[mid_idx:]

assert pre_drug.size == post_drug.size


plt.plot(pre_drug_time, pre_drug * -1)
plt.show()

plt.plot(post_drug_time, post_drug * -1)
plt.show()

# Find peaks!
peaks_idx_pre_drug = scipy.signal.find_peaks(pre_drug * -1,
                                             prominence=20)[0]


pre_drug_peak_times = pre_drug_time[peaks_idx_pre_drug]
pre_drug_peak_im = pre_drug[peaks_idx_pre_drug]

plt.plot(pre_drug_time, pre_drug)
plt.scatter(pre_drug_peak_times, pre_drug_peak_im, c="red")
plt.show()

pre_drug_event_amplitudes = peak_drug_peak_im - 0  # TODO: calculate baseline in a better way

# 1) make a function for this last bit
# 2) make seaborn plots


