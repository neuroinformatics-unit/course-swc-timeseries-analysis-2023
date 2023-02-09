import matplotlib.pyplot as plt
import numpy as np

N = 4
plt.plot(np.arange(4)/4, [0, 1, 0, -1])
plt.plot(np.arange(4)/4, [1, 0, -1, 0])
plt.legend(["Sine", "Cosine"])
plt.xlabel("Time (s)")
plt.show()