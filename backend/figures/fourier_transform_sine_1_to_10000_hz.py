"""
Note need to manually remove 1e6 scale
"""
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

N = 1200000
n = np.arange(N)

fix, axes = plt.subplots(4, 2)

for pos, freq in zip([[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1], [3, 0], [3, 1]],
                     [1,      2,      3,      4,      5,     6,      8,        10]):

    axes[pos[0], pos[1]].plot(n, np.sin(2 * np.pi * n * freq / N))

    title_ = f"{freq} Cycle" if freq == 1 else f"{freq} Cycles"
    axes[pos[0], pos[1]].set_title(title_)
    axes[pos[0], pos[1]].set_xlim([0, N])

    if pos[0] != 3:
        axes[pos[0], pos[1]].get_xaxis().set_visible(False)
    else:
        axes[pos[0], pos[1]].set_xlabel("Samples (million)")

plt.show()

# ======================================================================================================================
# Load the data
# ======================================================================================================================
repo_path = Path(r"C:\Users\Joe\work\git-repos\swc-timeseries-analysis-course-2023")

pandas_dataframe = pd.read_csv(repo_path / "sub-001_drug-cch_rawdata.csv")

raw_signal = pandas_dataframe["current_pa"].to_numpy()
demean_signal = raw_signal - np.mean(raw_signal)
time_s = pandas_dataframe["time_s"].to_numpy()

# Plot the data
plt.plot(np.arange(demean_signal.size), demean_signal)
plt.ylabel("Current (pA)")
plt.xlabel("Samples (million)")
plt.title(" Raw Signal")
plt.show()

# ======================================================================================================================
# Subset data plot (Major DRY)
# ======================================================================================================================
repo_path = Path(r"C:\Users\Joe\work\git-repos\swc-timeseries-analysis-course-2023")

pandas_dataframe = pd.read_csv(repo_path / "sub-001_drug-cch_rawdata.csv")

raw_signal = pandas_dataframe["current_pa"].to_numpy()
demean_signal = raw_signal - np.mean(raw_signal)
time_s = pandas_dataframe["time_s"].to_numpy()

# Plot the data
sub_signal = demean_signal[700000:710000]
plt.plot(np.arange(sub_signal.size) / sub_signal.size, sub_signal)
plt.ylabel("Current (pA)")
plt.xlabel("Time (s)")
plt.title(" Raw Signal")
plt.show()


# ======================================================================================================================
# Take the DFT
# =====================================================================================================================

for in_hz in [True, False]:

    freqs = np.arange(N)

    if in_hz:
        sampling_rate = 10000
        freqs = freqs * (10000/N)
        xlabel = "Frequency (Hz)"
    else:
        xlabel = "Frequency (Cycles)"

    signal_demean = raw_signal - np.mean(raw_signal)

    signal_fft = np.fft.fft(signal_demean)

    magnitude_fft = np.abs(signal_fft)

    # Scaling: note this scales 0 Hz freq by 2x what it should be.
    # In our case, we don't need to worry as it is zero'd out.
    # Don’t scale if planning on taking the inverse DFT to get back your original signal.
    scale_magnitude_fft = magnitude_fft * 2 / N


    plt.stem(freqs[:1000],
             scale_magnitude_fft[:1000],
             markerfmt=" ",
             basefmt="k")
    plt.ylabel("Scaled Magnitude")
    plt.xlabel(xlabel)
    plt.title("Demeaned Signal Frequency Spectrum")
    plt.show()

    scale_magnitude_fft = np.fft.fftshift(scale_magnitude_fft)
    plt.stem(freqs[-1000:],
             scale_magnitude_fft[-1000:],
             markerfmt=" ",
             basefmt="k",
    )
    frame1 = plt.gca()
    frame1.axes.ticklabel_format(style='plain', useOffset=False)  # stop scientific notation, stop x-axis as negative indicies

    plt.ylabel("Scaled Magnitude")
    plt.xlabel(xlabel)
    plt.title("Demeaned Signal Frequency Spectrum")
    plt.show()

