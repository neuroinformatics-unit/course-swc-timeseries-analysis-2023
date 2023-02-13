"""
Make an animation of an IIR filter, similar to the FIR
filter animation but with an extra row for feedback coefficients.

see filtering_animation_FIR.py for more details. This script is very
similar and should be refactored to reduce code reuse.
"""
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

plot_dft = True
save_animation = False
save_fps = 5
gif_or_mp4 = "mp4"

# Setup plots

fig, axes = plt.subplots(4, 1)
ax1, ax2, ax3, ax4 = axes

def init():
    """ Required to stop animate adding +1 to iteration.
    Should move plot setup here, but lead to weird flickering"""
    pass

def setlabels():
    ax1.set_title("Raw Signal")
    ax2.set_title("Filter Coefficients (Feedforward)")
    ax3.set_title("Filtered Signal")
    ax4.set_xlabel("Time (s)")
    ax4.set_title("Filter Coefficients (Feedback)")

def setlimits(lims, b_N):
    lower = -b_N + 1
    upper = N + b_N + 1
    ax1.set_xlim(lower, upper)
    ax2.set_xlim(lower, upper)
    ax3.set_xlim(lower, upper)
    ax4.set_xlim(lower, upper)
    ax1.set_ylim(-lims, lims)
    ax3.set_ylim(-lims, lims)

# Setup feedforward coefficients
N = 50
b = np.ones(5) / 5
b_N = len(b)
main_N = N + 2 * b_N - 2
main_n = np.arange(-b_N + 1, N + b_N - 1)
filter = np.zeros(main_N)
filter[:b_N] = b

# Setup signal
n = np.linspace(0, 1, N)
signal = (50 * np.sin(2*np.pi*n*100) +
          3 * np.sin(2*np.pi*n*72) +
          25 * np.cos(2*np.pi*n*3) +
          15 * np.cos(2*np.pi*n*12))

signal = np.r_[(b_N - 1) * [np.NaN], signal, (b_N - 1) * [np.NaN]]

# Setup feedback coefficients
fb_filter = np.zeros(main_N)
a = np.array([1, 0.35, 0.15])  # note as structured, len(a) must < len(b)
a_N = len(a)
fb_filter[b_N - a_N: b_N] = np.flip(a)

# Initialise Plots
ax1_lines = ax1.plot(main_n, signal, c="k", markevery=[b_N - 1])

a2_stem = ax2.stem(main_n, filter, basefmt="k", linefmt="k")
plt.setp(a2_stem[0], "markerfacecolor", "#1f77b4", "markeredgewidth", 0.5)

ax3_lines = ax3.plot([], [], c="tab:green")

a4_stem = ax4.stem(main_n, fb_filter, basefmt="k", linefmt="k")
plt.setp(a4_stem[0], "markerfacecolor", "yellow")
ax4.set_ylim(-0.1, 1.1)

lims = 130
setlimits(lims, b_N)
setlabels()
fig.set_figheight(6)
fig.tight_layout()

# Init and run animation

output = np.empty(main_N)
output.fill(np.NaN)

def animate(i):
    """
    Update the first plot (coefficient positions), second plot (feedforward coefficient positions)
    third plot: IIR output, fourth plot (feedback coefficient positions).
    """
    filter = np.zeros(main_N)
    filter[i:i+b_N] = b

    ax1.clear()
    ax2.clear()
    ax4.clear()

    # Update coefficinet positions
    ax1.plot(main_n, signal, '-o', c="k", markevery=range(i, i + b_N))

    a2_stem = ax2.stem(main_n, filter, basefmt="k", linefmt="k")
    plt.setp(a2_stem[0], "markerfacecolor", "#1f77b4", "markeredgewidth", 0.5)


    fb_filter = np.zeros(main_N)
    fb_filter[i + b_N - a_N : i + b_N] = np.flip(a)

    a4_stem = ax4.stem(main_n, fb_filter, basefmt="k", linefmt="k")
    plt.setp(a4_stem[0], "markerfacecolor", "yellow")
    ax4.set_ylim(-0.1, 1.1)  # TODO:

    # Calcuate IIR filter output and plot
    current_idx = i + b_N - 1
    feedforward_sum = np.nansum(filter * signal)
    output[current_idx] = feedforward_sum * a[0] + np.nansum([output[current_idx - j] * a[j] for j in range(1, a_N)])

    ax3_lines[0].set_data(main_n, output)

    setlimits(lims, b_N)
    setlabels()

    if i == main_N - b_N:
        ani.event_source.stop()

ani = animation.FuncAnimation(fig, animate, frames=range(main_N - b_N + 1), interval=5, init_func=init)
if save_animation:
    if gif_or_mp4 == "gif":
        ani.save(f"IIR_{save_fps}fps.gif", writer='imagemagick', fps=save_fps)
    else:
        FFwriter = animation.FFMpegWriter(fps=save_fps, extra_args=['-vcodec', 'libx264'])
        ani.save(f"IIR_{save_fps}fps.mp4", writer=FFwriter)
else:
    plt.show()
    
assert np.sum(~np.isnan(output)) == N + b_N - 1  # sanity check on 'convolution' (not flipped) output size N + K - 1

if plot_dft:

    fig, axes = plt.subplots(1, 2)
    freqs = np.fft.rfftfreq(N, 1 / N)
    mag_orig = np.abs(np.fft.rfft(signal[~np.isnan(signal)])) * 2 / N

    output_no_nan = output[~np.isnan(output)]  # TODO: direct copy!!
    if b_N % 2 == 0:
        centered_output = output_no_nan[int(b_N/2):-int(b_N/2)]
    else:
        centered_output = output_no_nan[int((b_N - 1) / 2):-int((b_N - 1) / 2)]

    assert centered_output.size == N

    mag_filt = np.abs(np.fft.rfft(centered_output)) * 2 / N

    mag_filt *= np.max(mag_orig) / np.max(mag_filt)
    max_lim = np.max([np.max(mag_orig), np.max(mag_filt)]) + 5

    axes[0].stem(freqs, mag_orig, markerfmt="", basefmt="k")
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Scaled Magnitude")
    axes[0].set_ylim(0, max_lim)

    axes[1].stem(freqs, mag_filt, markerfmt="", basefmt="k")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Scaled Magnitude")
    axes[1].set_ylim(0, max_lim)

    plt.tight_layout()
    plt.show()

