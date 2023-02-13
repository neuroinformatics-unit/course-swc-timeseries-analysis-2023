"""
Plot an example of sampling a signal at 4 Hz
"""
import matplotlib.pyplot as plt
import numpy as np

n = np.linspace(0, 1, 1000)
x = 2 * np.sin(2 * np.pi * n * 1) + 4 * np.cos(2 * np.pi * n * 5) + 5 * np.sin(2 * np.pi * n * 3)
x += np.max(x) + 2

plt.plot(n, x)
plt.stem([0, 0.25, 0.5, 0.75], [x[0], x[250], x[500], x[750]], linefmt="k", markerfmt="k", basefmt="k")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (Vm)")
plt.show()
