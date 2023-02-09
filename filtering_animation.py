import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os

filter_type = "high_pass"  # high_pass or low_pass
plot_ffts = True
save_animation = True
save_fps = 180
gif_or_mp4 = "gif"

fig, axes = plt.subplots(3, 1)
ax1, ax2, ax3 = axes

def setlabels():
    ax1.set_title("Raw Signal")
    ax2.set_title("Filter")
    ax3.set_title("Filtered Signal")
    ax3.set_xlabel("Index")

def setlimits(lims, N, filt_N):
    lower = -filt_N + 1
    upper = N + filt_N + 1
    ax1.set_xlim(lower, upper)
    ax2.set_xlim(lower, upper)
    ax3.set_xlim(lower, upper)
    ax1.set_ylim(-lims, lims)
    ax3.set_ylim(-lims, lims)

def init():
    pass

if filter_type == "high_pass":
    filter_coefs = np.array([-1, 1, -1, 1, -1]) / 5
elif filter_type == "low_pass":
    filter_coefs = np.ones(5) / 5
else:
    raise BaseException("Filter type must be high_pass or low_pass")

N = 50
filt_N = len(filter_coefs)
MAIN_N = N + 2 * filt_N - 2
MAIN_n = np.arange(-filt_N + 1, N + filt_N - 1)
filter = np.zeros(MAIN_N)
filter[:filt_N] = filter_coefs

n = np.linspace(0, 1, N)
signal = (50 * np.sin(2*np.pi*n*100) +
          3 * np.sin(2*np.pi*n*72) +
          25 * np.cos(2*np.pi*n*3) +
          15 * np.cos(2*np.pi*n*12))

signal = np.r_[(filt_N - 1) * [np.NaN], signal, (filt_N - 1) * [np.NaN]]

ax1_lines = ax1.plot(MAIN_n, signal, '-o', c="k", markevery=[filt_N - 1])
a2_stem = ax2.stem(MAIN_n, filter, basefmt="k", linefmt="k")
plt.setp(a2_stem[0], "markerfacecolor", "#1f77b4", "markeredgewidth", 0.5)

ax3_lines = ax3.plot([], [], c="#1f77b4")

lims = np.nanmax(signal) * np.nanmax(filter) * filt_N + 10
setlabels()
setlimits(lims, N, filt_N)
fig.tight_layout()

output = np.empty(MAIN_N) 
output.fill(np.NaN)

def animate(i):

    filter = np.zeros(MAIN_N)
    filter[i:i+filt_N] = filter_coefs

    ax2.clear()
    a2_stem = ax2.stem(MAIN_n, filter, basefmt="k", linefmt="k")
    plt.setp(a2_stem[0], "markerfacecolor", "#1f77b4", "markeredgewidth", 0.5)

    ax1.clear()
    ax1.plot(MAIN_n, signal, '-o', c="k", markevery=range(i, i+filt_N))  # hard to update this with set_data()

    output[i + filt_N - 1] = np.nansum(filter * signal)

    ax3_lines[0].set_data(MAIN_n, output)  

    setlabels()
    setlimits(lims, N, filt_N)

    if i == MAIN_N - filt_N:
        ani.event_source.stop()

ani = animation.FuncAnimation(fig, animate, frames=range(0, MAIN_N - filt_N + 1), interval=5, init_func=init)

if save_animation:
    if gif_or_mp4 == "gif":
        ani.save(f"FIR_{filter_type}_{save_fps}fps.gif", writer='imagemagick', fps=save_fps)
    else:
        FFwriter = animation.FFMpegWriter(fps=save_fps, extra_args=['-vcodec', 'libx264'])
        ani.save(f"FIR_{filter_type}_{save_fps}fps.mp4", writer=FFwriter)
else:
    plt.show()

assert np.sum(~np.isnan(output)) == N + filt_N - 1  # sanity check on 'convolution' (not flipped) output size N + K - 1

if plot_ffts:

    fig, axes = plt.subplots(1, 2)
    freqs = np.fft.rfftfreq(N, 1 / N)
    mag_orig = np.abs(np.fft.rfft(signal[~np.isnan(signal)])) * 2 / N
    max_lim = np.max(mag_orig) + 5

    axes[0].stem(freqs, mag_orig, markerfmt="", basefmt="k")
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Scaled Magnitude")
    axes[0].set_ylim(0, max_lim)

    output_no_nan = output[~np.isnan(output)]
    if filt_N % 2 == 0:
        centered_output = output_no_nan[int(filt_N/2):-int(filt_N/2)]
    else:
        centered_output = output_no_nan[int((filt_N - 1) / 2):-int((filt_N - 1) / 2)]

    assert centered_output.size == N

    mag_filt = np.abs(np.fft.rfft(centered_output)) * 2 / N
    axes[1].stem(freqs, mag_filt, markerfmt="", basefmt="k")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Scaled Magnitude")
    axes[1].set_ylim(0, max_lim)

    fig.tight_layout()
    plt.show()

