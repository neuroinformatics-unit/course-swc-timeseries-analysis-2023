import numpy as np
import matplotlib.pyplot as plt

N = 1000
n = np.linspace(0, 1, N)

amplitude_cos = 4
freq_cos_hz = 10

amplitude_sin = 8
freq_sin_hz = 6
signal = (
         amplitude_cos * np.cos(2 * np.pi * n *freq_cos_hz) +
         amplitude_sin * np.sin(2 * np.pi * n * freq_sin_hz)
         )

plt.plot(n, signal)
plt.xlabel("Time (s)")
plt.show()