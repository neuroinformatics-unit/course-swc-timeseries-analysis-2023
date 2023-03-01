"""
Generate a 4x2 subplot of sine waves at 1-6, 5000 and 10000 Hz
"""

import numpy as np
import matplotlib.pyplot as plt

time_s = 120
N = 100000
sampling_rate = N / time_s
sample_spacing = 1 / sampling_rate
n = np.linspace(0, time_s-sample_spacing, N)


fix, axes = plt.subplots(4, 2)

for pos, freq in zip([[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1], [3, 0], [3, 1]],
                     [1,      2,      3,      4,      5,     6,      5000, 10000]):

    axes[pos[0], pos[1]].plot(n, np.sin(2 * np.pi * (n*sampling_rate/N) * freq))
    axes[pos[0], pos[1]].set_title(f"{freq} Hz")
    axes[pos[0], pos[1]].set_xlim([0, time_s])

    if pos[0] != 3:
        axes[pos[0], pos[1]].get_xaxis().set_visible(False)
    else:
        axes[pos[0], pos[1]].set_xlabel("Time (s)")

plt.show()
