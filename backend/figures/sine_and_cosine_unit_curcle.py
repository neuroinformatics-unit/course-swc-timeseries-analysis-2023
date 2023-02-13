"""
Show the evolution of the sine and cosine function. Meant to be
companion to unit circle example.
"""
import numpy as np
import matplotlib.pyplot as plt

N = 10000
n = np.linspace(0, 2 * np.pi, N)


for angles in [[np.pi/16, np.pi / 4, 5*np.pi/12], [3*np.pi/4, 5*np.pi/4, 7 * np.pi/4]]:
    fig, axes = plt.subplots(3, 1)
    fig.set_figheight(10)
    fig.set_figwidth(8)
    for row, theta in enumerate(angles):

        cos = np.cos(n)
        sin = np.sin(n)


        cos[n > theta] = np.nan
        sin[n > theta] = np.nan

        axes[row].plot(n, sin, color=np.array([0, 176, 80])/255, lw=2)
        axes[row].plot(n, cos, color=np.array([237, 125, 49])/255, lw=2)

        axes[row].set_ylim(-1.1, 1.1)
        axes[row].set_yticks([-1, 0, 1])
        axes[row].set_xlim(0, np.pi * 2)

        if row != 2:
            axes[row].get_xaxis().set_visible(False)

        axes[row].legend(["Sin", "Cos"], loc="upper right")

    x_label = ["0",  r"$\frac{\pi}{4}$", r"$\frac{2\pi}{4}$", r"$\frac{3\pi}{4}$", r"$\pi$", r"$\frac{5\pi}{4}$", r"$\frac{6\pi}{4}$",   r"$\frac{7\pi}{4}$",r"$2\pi$"]
    axes[-1].set_xticks([np.pi/4 * i for i in range(8+1)])
    axes[-1].set_xticklabels(x_label, fontsize=20)

    plt.xlim(0, np.pi * 2)
    plt.show()

num_samples = 10000
frequency = 1

x_axis = np.linspace(0, 1, num_samples)
sine_wave = np.sin(2 * np.pi * n  * frequency)