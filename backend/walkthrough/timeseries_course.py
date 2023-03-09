from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import seaborn as sns

show_all_plots = True
repo_path = Path("/home/joe/git-repos/swc-timeseries-analysis-course-2023")  #  Path(r"C:\Users\Joe\work\git-repos\swc-timeseries-analysis-course-2023")

# ======================================================================================================================
# Load the data
# ======================================================================================================================

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

# ======================================================================================================================
# Play with time, try and reconstruct the time array
# ======================================================================================================================

num_samples = raw_signal.size
sample_spacing = time_s[1] - time_s[0]  # 0.001
sampling_rate = 1 / sample_spacing      # 10,000

method_1_time = np.linspace(0, 120 - sample_spacing, num_samples)  # method 1
method_2_time = np.arange(num_samples) * sample_spacing            # method 2

# checks every sample matches exactly. This will always fail
#assert np.array_equal(time_s, method_1_time)

bad_m1_index = np.where(method_1_time != time_s)[0]
print(method_1_time[bad_m1_index][0])    # 9.999999999999999e-05  (i.e. 0.0000999...)
print(time_s[bad_m1_index][0])           # 0.0001

# Use all-close to compare floating-point numbers
# assert np.allclose(time_s, method_1_time, rtol=0, atol=1e-10)  # OK
# assert np.allclose(time_s, method_1_time, rtol=0, atol=1e-15)  # FAIL

# ======================================================================================================================
# Plotting Function
# ======================================================================================================================

def show_line_plot(x, y, xlabel="", ylabel="", title=""):
   """ Make and show a matplotlib lineplot """
   plt.plot(x, y)
   plt.ylabel(ylabel)
   plt.xlabel(xlabel)
   plt.title(title)
   plt.show()

# ======================================================================================================================
# Take the FFT
# ======================================================================================================================

freqs = np.fft.fftfreq(num_samples, d=sample_spacing)
# freqs order: all positive (0 to most positive)
#              then all negative (most negative to zero)

signal_demean = raw_signal - np.mean(raw_signal)
signal_fft = np.fft.fft(signal_demean)

magnitude_fft = np.abs(signal_fft)

# Scaling: note this scales 0 Hz freq by 2x what it should be.
# In our case, we don't need to worry as it is zero'd out.
# Don’t scale if planning on taking the inverse DFT to get back your original signal.
scale_magnitude_fft = magnitude_fft * 2 / num_samples

if show_all_plots:

    show_line_plot(x=freqs,
                   y=scale_magnitude_fft,
                   xlabel="Frequency (Hz)",
                   ylabel="Scaled Magnitude",
                   title="Demeaned Signal Frequency Spectrum")

    plt.stem(freqs[:1000],
             scale_magnitude_fft[:1000],
             markerfmt=" ",
             basefmt="k")
    plt.ylabel("Scaled Magnitude")
    plt.xlabel("Frequency (Hz)")
    plt.title("Demeaned Signal Frequency Spectrum")
    plt.show()

# ======================================================================================================================
# Create the filter and apply
# ======================================================================================================================

b, a = scipy.signal.butter(2,
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
    show_line_plot(freqs,
                   filt_magnitude_fft,
                   "Frequency (Hz)", "Scaled Magnitude", "2nd order filtered signal")

    show_line_plot(time_s,
                   signal_filtered, "Time (s)", "Current (pA)", "2nd order filtered signal")

    # Check the frequency response of the filter
    filter_response_freqs, filter_response_h = scipy.signal.freqz(b, a, fs=sampling_rate, worN=5000)

    show_line_plot(filter_response_freqs, np.abs(filter_response_h), # 20 * np.log10(np.abs(filter_response_h)) for dB
                   "Frequency (Hz)", "Frequency Attenuation", "2nd order butter response")


# ======================================================================================================================
# Improve the filter
# ======================================================================================================================

sos = scipy.signal.butter(16,
                          Wn=4000,
                          btype='low',
                          analog=False,
                          output='sos',
                          fs=sampling_rate)

signal_filtered = scipy.signal.sosfiltfilt(sos, signal_demean)

filter_response_freqs, filter_response_h = scipy.signal.sosfreqz(sos, fs=sampling_rate, worN=5000)

if show_all_plots:
    show_line_plot(filter_response_freqs, np.abs(filter_response_h),
                   "Frequency (Hz)", "Frequency Attenuation", "16th order butter response")

    # Check the DFT
    filt_signal_fft = np.fft.fft(signal_filtered)
    filt_magnitude_fft = np.abs(filt_signal_fft) * 2 / num_samples

    show_line_plot(freqs, filt_magnitude_fft,
                   "Frequency (Hz)", "Scaled Magnitude", "16th order Filtered Signal")

    show_line_plot(time_s,  signal_filtered,
                   "Time (s)", "Current (pA)", "16th order filtered signal")


# ======================================================================================================================
# Split the dataset into pre-drug and drug sections
# ======================================================================================================================

mid_idx = np.where(time_s == 60)[0][0]  # note tuple index

pre_drug = signal_filtered[:mid_idx]
post_drug = signal_filtered[mid_idx:]

pre_drug_time = time_s[:mid_idx]
post_drug_time = time_s[mid_idx:]

assert pre_drug.size == post_drug.size

# ======================================================================================================================
# Copy vs. View
# ======================================================================================================================

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

# ======================================================================================================================
# Peak detection
# ======================================================================================================================

# Look at the flipped signal
if show_all_plots:
    show_line_plot(pre_drug_time,
                   pre_drug * -1,
                   "Time (s)",
                   "Current (pA)",
                   "Flipped pre-drug signal")

    show_line_plot(post_drug_time,
                   post_drug * -1,
                   "Time (s)",
                   "Current (pA)",
                   "Flipped post-drug signal")

results = {}

# Find the peak indexes
distance_between_peaks_in_samples = int(0.030 * sampling_rate)  # 30 ms * samples per second

peaks_idx_pre_drug = scipy.signal.find_peaks(
    pre_drug * -1,
    distance=distance_between_peaks_in_samples,
    height=20,
    prominence=20
)[0]

peaks_idx_post_drug = scipy.signal.find_peaks(post_drug * -1,
                                              distance=distance_between_peaks_in_samples,
                                              height=20,
                                              prominence=20)[0]

# Plot the results
if show_all_plots:
    plt.plot(pre_drug_time, pre_drug)
    plt.scatter(pre_drug_time[peaks_idx_pre_drug],
                pre_drug[peaks_idx_pre_drug],
                c="red")
    plt.title("Pre-drug peaks")
    plt.xlabel("Time")
    plt.ylabel("Current (pA)")
    plt.show()

    plt.plot(post_drug_time, post_drug)
    plt.scatter(post_drug_time[peaks_idx_post_drug],
                post_drug[peaks_idx_post_drug],
                c="red")
    plt.title("Post-drug peaks")
    plt.xlabel("Time")
    plt.ylabel("Current (pA)")
    plt.show()

# ======================================================================================================================
# Format Results
# ======================================================================================================================
# Note we could also set the column titles to appear as we want them in our Seaborn plots

headers = ["peak_time", "amplitude", "inter_event_interval"]

pre_drug_event_times = pre_drug_time[peaks_idx_pre_drug]
pre_drug_event_amplitudes = pre_drug[peaks_idx_pre_drug]
pre_drug_inter_event_intervals = np.diff(pre_drug_time[peaks_idx_pre_drug])

pre_drug_pd = pd.DataFrame([pre_drug_event_times,
                            pre_drug_event_amplitudes,
                            pre_drug_inter_event_intervals],
                           index=headers).T

post_drug_event_times = post_drug_time[peaks_idx_post_drug]
post_drug_event_amplitudes = post_drug[peaks_idx_post_drug]
post_drug_inter_event_intervals = np.diff(post_drug_time[peaks_idx_post_drug])


post_drug_pd = pd.DataFrame([post_drug_event_times,
                             post_drug_event_amplitudes,
                             post_drug_inter_event_intervals],
                            index=headers).T

pre_drug_pd["condition"] = "pre_drug"
post_drug_pd["condition"] = "post_drug"
results = pd.concat([pre_drug_pd, post_drug_pd])

# ======================================================================================================================
# Seaborn Plots of results
# ======================================================================================================================

if show_all_plots:
    sns.set_theme()

    sns.barplot(data=results, x="condition", y="amplitude", errorbar="se")
    plt.ylabel("Peak Current (pA)")
    ax = plt.gca()  # this is an example of matplotlib's annoying syntax. plt has no equivalent  function to set_xticklabels()
                    # (.xticks() is closest but requires passing the existing ticks back into the function which is not nice).
                    # So, we have to get the axis instance in order to use the axis function.
    ax.set_xticklabels(labels=["Pre-drug", "Post-drug"])
    plt.show()

    sns.set_style("dark")


    sns.violinplot(data=results, x="condition", y="amplitude", width=0.4)
    plt.ylabel("Peak Current (pA)")
    ax = plt.gca()
    ax.set_xticklabels(labels=["Pre-drug", "Post-drug"])
    plt.show()

    results.to_csv(repo_path / "sub-001_drug-cch_results.csv")


    sns.ecdfplot(data=results[["inter_event_interval", "condition"]].dropna(),
                 x="inter_event_interval",
                 hue="condition", palette="pastel")
    plt.ylim(0, 1.1)
    plt.ylabel("Cumulative Probability")
    plt.xlabel("Inter-event interval (s)")
    plt.show()

    sns.set_style("whitegrid")
