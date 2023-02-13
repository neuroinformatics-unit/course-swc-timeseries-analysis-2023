"""
make a 3 x 2 subplot of sine and cosine at different frequencies (1, 2, 10 Hz)
and a plot of a sine wave at different amplitudes (1, 2, 4) as well
as the magnitude vs. amplitude of a sine wave.
"""
import numpy as np
import matplotlib.pyplot as plt


# Frequencies
fig, axes = plt.subplots(3, 2)
fig.set_figheight(8)
fig.set_figwidth(7)

N = 10000
n = np.linspace(0, 1, N)

for row, freq in enumerate([1, 2, 10]):

    axes[row, 0].plot(n, np.sin(2 * np.pi * n * freq))
    axes[row, 0].set_title(f"Sine ({freq} Hz)")

    axes[row, 1].plot(n, np.cos(2 * np.pi * n * freq), c="tab:red")
    axes[row, 1].set_title(f"Cosine ({freq} Hz)")

axes[2, 0].set_xlabel("Time (s)")
axes[2, 1].set_xlabel("Time (s)")

plt.tight_layout()
plt.show()

# Amplitude and Magnitujde

fig, axes = plt.subplots(2, 1)
fig.set_figwidth(4.5)
fig.set_figheight(6)

for amplitude in [1, 2, 4]:
    axes[0].plot(n, amplitude * np.sin(2 * np.pi * n))

axes[0].legend(["Amplitude = 1", "Amplitude = 2", "Amplitude = 4"])
axes[0].set_title("Sine curves of different amplitude")
axes[1].set_xlabel("Time (s)")


axes[1].plot(n, 1 * np.sin(2 * np.pi * n))
axes[1].plot(n, np.abs(1 * np.sin(2 * np.pi * n)))

axes[0].set_title("Magnitude = np.abs(Amplitude) =      ")
plt.legend(["Amplitude", "Magnitude"])
plt.tight_layout()

plt.show()

