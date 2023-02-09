from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import seaborn as sns

import plotters

show_all_plots = False

# Load the data
repo_path = Path(r"C:\fMRIData\git-repo\swc_timeseries_course")
pandas_dataframe = pd.read_csv(repo_path / "sub-001_drug-cch_rawdata.csv")

raw_signal = pandas_dataframe["current_pa"].to_numpy()
time_s = pandas_dataframe["time_s"].to_numpy()

# Plot the data
if show_all_plots:
    plt.plot(time_s, raw_signal)
    plt.ylabel("Current (pA)")
    plt.xlabel("Time (s)")
    plt.title(" Raw Signal")
    plt.show()

# Play with time, try and reconstruct the time array
num_samples= raw_signal.size
sample_spacing = time_s[1] - time_s[0]  # 0.001
sampling_rate = 1 / sample_spacing      # 10,000

method_1_time = np.linspace(0, 120 - sample_spacing, num_samples)  # method 1
method_2_time = np.arange(num_samples) * sample_spacing            # method 2

# checks every sample matches exactly. This will always fail
# assert np.array_equal(time_s, method_1_time) 

bad_m1_index = np.where(method_1_time != time_s)[0]
print(method_1_time[bad_m1_index][0])    # 9.999999999999999e-05  (i.e. 0.0000999...)
print(time_s[bad_m1_index][0])           # 0.0001

# Use all-close to compare floating-point numbers
#assert np.allclose(time_s, method_1_time, rtol=0, atol=1e-10)  # OK
#assert np.allclose(time_s, method_1_time, rtol=0, atol=1e-15)  # FAIL

freqs = np.fft.fftfreq(num_samples, d=sample_spacing)  # this outputs all positive, then all negative freqs

signal_demean = raw_signal - np.mean(raw_signal)
signal_fft = np.fft.fft(signal_demean)

magnitude_fft = np.abs(signal_fft)

# note this scales the 0 Hz frequency by 2x what it should be.
# In our case, we don't need to worry as it is zero'd out.
scale_magnitude_fft = magnitude_fft * 2 / num_samples

if show_all_plots:
    plt.plot(freqs, scale_magnitude_fft)
    plt.ylabel("Scaled Magnitude")
    plt.xlabel("Frequency (Hz)")
    plt.title("Demeaned Signal Frequency Spectrum")
    plt.show()

plt.stem(freqs[:1000], scale_magnitude_fft[:1000], markerfmt=" ", basefmt="k")
plt.ylabel("Scaled Magnitude")
plt.xlabel("Frequency (Hz)")
plt.title("Demeaned Signal Frequency Spectrum")
plt.show()

# Create the filter and apply
b, a = scipy.signal.butter(16,
                     Wn=4000,
                     btype='low',
                     analog=False,
                     output='ba',
                     fs=sampling_rate)

signal_filtered = scipy.signal.filtfilt(b, a, signal_demean)

# Take the FFT of the filtered signal, plot FFT and original signal
filt_signal_fft = np.fft.fft(signal_filtered)
filt_magnitude_fft = np.abs(filt_signal_fft) * 2 / num_samples

if show_all_plots:
    plotters.show_line_plot(freqs, filt_magnitude_fft,
                            xlabel="Frequency (Hz)", title="2nd'order filtered Signal")

    plotters.show_line_plot(time_s, signal_filtered,
                            "Current (pA)", "Time (s)", "2nd order filtered Signal")

    # Check the frequency response of the filter
    filter_response_freqs, filter_response_h = scipy.signal.freqz(b, a, fs=sampling_rate, worN=5000)

    plotters.show_line_plot(filter_response_freqs,
                            np.abs(filter_response_h),
                            xlabel="Frequency (Hz)",
                            title="2nd order butter response")  # 20 * np.log10(filter_response_h) for decibels


# Improve the filter
sos = scipy.signal.butter(16,
                          Wn=4000,
                          btype='low',
                          analog=False,
                          output='sos',
                          fs=sampling_rate)

signal_filtered = scipy.signal.sosfiltfilt(sos, signal_demean)

filter_response_freqs, filter_response_h = scipy.signal.sosfreqz(sos, fs=sampling_rate, worN=5000)

if show_all_plots:
    plotters.show_line_plot(filter_response_freqs,
             np.abs(filter_response_h), xlabel="Frequency (Hz)", title="16th order butter response")

    filt_signal_fft = np.fft.fft(signal_filtered)
    filt_magnitude_fft = np.abs(filt_signal_fft) * 2 / num_samples

    plotters.show_line_plot(freqs, filt_magnitude_fft, "Magnitude (pA)", "Frequency (Hz)", "16th order Filtered Signal")

    plotters.show_line_plot(time_s, signal_filtered, "Im (pA", "Time (s)", "16th order filtered Signal")


# Split the dataset into pre-drug and drug sections
mid_idx = np.where(time_s == 60)[0][0]  # note tuple index

pre_drug = signal_filtered[:mid_idx]
post_drug = signal_filtered[mid_idx:]

pre_drug_time = time_s[:mid_idx]
post_drug_time = time_s[mid_idx:]

assert pre_drug.size == post_drug.size


# This is a copy
signal_filtered_copy = signal_filtered.copy()

signal_filtered_copy[0] = np.NaN
print(signal_filtered_copy[0])
print(signal_filtered[0])

# This is a View
signal_filtered_view = signal_filtered_copy[:mid_idx]

signal_filtered_copy[0] = 1
print(signal_filtered_copy[0])
print(signal_filtered_view[0])

# Lets protect our signal
signal_filtered.setflags(write=False)
# signal_filtered[0] = np.NaN

if show_all_plots:
    plotters.show_line_plot(pre_drug_time,
                            pre_drug * -1,
                            "Time (s)",
                            "Current (pA)")

    plotters.show_line_plot(pre_drug_time,
                            pre_drug * -1,
                            "Time (s)",
                            "Current (pA)")

results = {}

peaks_idx_pre_drug = scipy.signal.find_peaks(pre_drug * -1,
                                             prominence=20)[0]

if show_all_plots:
    plt.plot(pre_drug_time, pre_drug)
    plt.scatter(pre_drug_peak_times, pre_drug_peak_im, c="red")  # TODO:
    plt.show()


peaks_idx_post_drug = scipy.signal.find_peaks(post_drug * -1,
                                             prominence=20)[0]

pre_drug_results = {
    "peak_times":  pre_drug_time[peaks_idx_pre_drug],
    "peak_im": pre_drug[peaks_idx_pre_drug],
    "inter_event_interval": np.diff(pre_drug_time[peaks_idx_pre_drug]),
}

post_drug_results = {
    "peak_times":  post_drug_time[peaks_idx_post_drug],
    "peak_im": post_drug[peaks_idx_post_drug],
    "inter_event_interval": np.diff(post_drug_time[peaks_idx_post_drug]),
}

# Initialise dataframe in this way to handle uneven column lengths
pre_drug_pd = pd.DataFrame.from_dict(pre_drug_results, orient="index").T
post_drug_pd = pd.DataFrame.from_dict(post_drug_results, orient="index").T

pre_drug_pd["condition"] = "pre_drug"
post_drug_pd["condition"] = "post_drug"
results = pd.concat([pre_drug_pd, post_drug_pd])

sns.set_theme()

sns.ecdfplot(data=results[["inter_event_interval", "condition"]].dropna(), x="inter_event_interval", hue="condition", palette="pastel")
plt.ylim(0, 1.1)
plt.ylabel("Cumulative Probability")
plt.xlabel("Inter-event interval (s)")
plt.show()

sns.set_style("whitegrid")

sns.barplot(data=results, x="condition", y="peak_im", errorbar="se")
plt.ylabel("Peak Current (pA)")
plt.show()

sns.set_style("dark")

sns.violinplot(data=results, x="condition", y="peak_im", width=0.4)
plt.ylabel("Peak Current (pA)")
plt.show()

results.to_csv(repo_path / "sub-001_drug-cch_results.csv")

# Plot the Amplitudes





















