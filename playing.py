import matplotlib.pyplot as plt
import numpy as np

x = [0, 1, 2, 3, 4]
hp_filt = [0.2, 0.2, 0.2, 0.2, 0.2]

plt.stem(x, hp_filt, basefmt="k")
plt.xlabel("Index")
plt.ylabel("Filter Coefficients")
plt.title("Filter to convolve with our signal")
plt.xticks(range(5))
plt.show()


hp_filt = [-0.2, 0.2, -0.2, 0.2, -0.2]

plt.stem(x, hp_filt, basefmt="k")
plt.xlabel("Index")
plt.ylabel("Filter Coefficients")
plt.title("Filter to convolve with our signal")
plt.xticks(range(5))
plt.show()

signal = [1, 2, 0.5, 0.25, 4]

plt.plot(x, signal, "-o")
plt.xlabel("Index")
plt.ylabel("Signal")
plt.xticks(range(5))
plt.show()

filt_signal = np.array(hp_filt) * np.array(signal)

plt.plot(x, filt_signal, "-o")
plt.xlabel("Index")
plt.ylabel("Signal")
plt.xticks(range(5))
plt.show()
