"""
Create an animation of a FIR filter implementation. A caveat, the 
input signal is not reversed, but apart from this it is convolution
(and, if you imagine the signal has already been reversed, it is a convolution).

Show with high pass or low pass FIR coefficients, Option to plot
the DFT of the signal, save as GIF or MP4.

NOTES:
as per DSP convention, n is the full signal index array (e.g. x(n))
and N is the num_samples.

Ideally, the stem plot should be updated at data
level with set_ydata. This works well for the markers but the
line need to be redrawn, in a convoluted way, so for convenience.
just re-plot. This might cause speed issues but does not matter
as saving to mp4 / gif.
"""
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

filter_type = "low_pass"  # high_pass or low_pass
plot_dft = True
save_animation = False
save_fps = 180
gif_or_mp4 = "gif"

# Setup PLots

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
    """ Required to stop animate adding +1 to iteration.
    Should move plot setup here, but lead to weird flickering"""
    pass

if filter_type == "high_pass":
    filter_coefs = np.array([-1, 1, -1, 1, -1]) / 5
elif filter_type == "low_pass":
    filter_coefs = np.ones(5) / 5
else:
    raise BaseException("Filter type must be high_pass or low_pass")

# Setup filter coefficients and n
N = 50
filt_N = len(filter_coefs)
main_N = N + 2 * filt_N - 2
main_n = np.arange(-filt_N + 1, N + filt_N - 1)
filter = np.zeros(main_N)
filter[:filt_N] = filter_coefs

# Setup signal
n = np.linspace(0, 1, N)
signal = (50 * np.sin(2*np.pi*n*100) +
          3 * np.sin(2*np.pi*n*72) +
          25 * np.cos(2*np.pi*n*3) +
          15 * np.cos(2*np.pi*n*12))

signal = np.r_[(filt_N - 1) * [np.NaN], signal, (filt_N - 1) * [np.NaN]]

# Setup plots

ax1_lines = ax1.plot(main_n, signal, '-o', c="k", markevery=[filt_N - 1])
a2_stem = ax2.stem(main_n, filter, basefmt="k", linefmt="k")
plt.setp(a2_stem[0], "markerfacecolor", "#1f77b4", "markeredgewidth", 0.5)

ax3_lines = ax3.plot([], [], c="#1f77b4")

lims = np.nanmax(signal) * np.nanmax(filter) * filt_N + 10
setlabels()
setlimits(lims, N, filt_N)
fig.tight_layout()

output = np.empty(main_N) 
output.fill(np.NaN)

def animate(i):
    """
    Update the first plot (raw signal, position of the pointwise multiplication)
    second plot (filter coefficient position) and third plot (filtered signal).
    """
    filter = np.zeros(main_N)
    filter[i:i+filt_N] = filter_coefs

    ax1.clear()
    ax2.clear()

    ax1.plot(main_n, signal, '-o', c="k", markevery=range(i, i+filt_N))

    a2_stem = ax2.stem(main_n, filter, basefmt="k", linefmt="k")
    plt.setp(a2_stem[0], "markerfacecolor", "#1f77b4", "markeredgewidth", 0.5)

    output[i + filt_N - 1] = np.nansum(filter * signal)
    ax3_lines[0].set_data(main_n, output)  

    setlabels()
    setlimits(lims, N, filt_N)

    if i == main_N - filt_N:
        ani.event_source.stop()

ani = animation.FuncAnimation(fig, animate, frames=range(0, main_N - filt_N + 1), interval=5, init_func=init)

if save_animation:
    if gif_or_mp4 == "gif":
        ani.save(f"FIR_{filter_type}_{save_fps}fps.gif", writer='imagemagick', fps=save_fps)
    else:
        FFwriter = animation.FFMpegWriter(fps=save_fps, extra_args=['-vcodec', 'libx264'])
        ani.save(f"FIR_{filter_type}_{save_fps}fps.mp4", writer=FFwriter)
else:
    plt.show()

assert np.sum(~np.isnan(output)) == N + filt_N - 1  # sanity check on 'convolution' (not flipped) output size N + K - 1

if plot_dft:

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

