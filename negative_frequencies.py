import matplotlib.pyplot as plt
import numpy as np

N=10000
n = np.linspace(-1, 1, N)
sin = np.sin(2 * np.pi * n)
cos = np.cos(2 * np.pi * n)

plt.plot(n, sin)
plt.plot(n, cos)
plt.legend(["Sin", "Cos"])
plt.axvline(color="k")
plt.axhline(color="k")
plt.show()


N=10000
n = np.linspace(-1, 1, N)
pos_sin = np.sin(2 * np.pi * n)
neg_sin = np.sin(2 * np.pi * -n)

plt.plot(n, pos_sin)
plt.plot(n, neg_sin)
plt.legend(["Positive Sin", "Negative Sin"], loc="lower left")
plt.axvline(color="k")
plt.axhline(color="k")
plt.show()

X = np.fft.fft(pos_sin)
freqs = np.fft.fftfreq(N, d=1/(N/2))
plt.stem(freqs, np.abs(X) * 2 / N)
plt.xlim(-2, 2)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Scaled Magnitude")
plt.show()


n = np.linspace(0, 1, N)
sin = np.sin(2 * np.pi * n)

X = np.fft.fft(sin)
X[np.abs(X) * 2 / N < 0.01] = 0 + 1j*0  # kill  error
freqs = np.fft.fftfreq(N, d=1/N)
plt.stem(freqs, np.angle(X) / np.pi)
plt.ylim(-0.6, 0.6)
plt.xlim(-2, 2)
plt.title("Sin Phase")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Phase ($\pi$)")
plt.show()

cos = np.cos(2 * np.pi * n)

X = np.fft.fft(cos)
X[np.abs(X) * 2 / N < 1] = 0 + 1j*0  # kill  error
freqs = np.fft.fftfreq(N, d=1/N)
plt.stem(freqs, np.angle(X) / np.pi)
plt.ylim(-0.6, 0.6)
plt.xlim(-2, 2)
plt.title("Cos Phase")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Phase ($\pi$)")
plt.show()
