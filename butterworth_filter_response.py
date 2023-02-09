from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

b, a = signal.butter(2,
                     Wn=1,
                     btype='low',
                     analog=True,
                     output='ba')

w, h = signal.freqs(b, a, worN=np.linspace(0, 1000, 10000))

# copied from scipy freqz
fig, axes = plt.subplots(1, 2)
axes[0].set_title('Analog Butter. freq response')
axes[0].semilogx(w, 20 * np.log10(abs(h)), 'b')  # 20 * np.log10(abs(h))
axes[0].set_ylabel('Amplitude [dB]', color='k')
axes[0].set_xlabel('Frequency [rad /s]')


b, a = signal.butter(2,
                     Wn=1,
                     btype='low',
                     analog=False,
                     output='ba',
                     fs=20000)

w, h = signal.freqz(b, a, worN=np.linspace(0, 1000, 10000), fs=20000)

# copied from scipy freqz
axes[1].set_title("Digital Butter. freq response")
axes[1].semilogx(w, 20 * np.log10(abs(h)), 'b')  # 20 * np.log10(abs(h))
axes[1].set_xlabel('Frequency [rad /s]')

plt.show()