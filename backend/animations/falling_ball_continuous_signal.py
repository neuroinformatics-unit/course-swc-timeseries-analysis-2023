"""
Make a animation of a ball falling with the position indicated by a "continous"
function. Save as mp4.

Saving requires ffmpeg to be installed.

NOTES:
https://stackoverflow.com/questions/44938231/animating-a-stem-plot-in-matplotlib
"""
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

save_animation = False
save_fps = 30


# Initialise the plots. Don't use the func_init
# on animation as it created flicker at the start.

fig, axes = plt.subplots(1, 2)
ax1, ax2 = axes
h_stem = ax1.stem(0, 1, linefmt="w", markerfmt="#1f77b4")
h_stem[0].set_markersize(10)
line, = ax2.plot([], [], lw=2, c="k")

ax1.set_ylim(0, 1)
ax1.set_xlim(-0.01, 0.01)
ax1.get_xaxis().set_visible(False)
ax1.set_ylabel("Height (m)")
ax2.set_xlabel("Time (s)")
ax2.set_ylim(0, 1)
ax2.set_xlim(0, 1)

iter = np.linspace(0, 1, 48)

# Setup and run animation

def run(i):
    h_stem[0].set_ydata(1 - iter[i])
    line.set_data(iter[:i+1], 1 - iter[:i+1])
    if 1 - iter[i] == 0:
        ani.event_source.stop()

def data_gen():
    for i in reversed(np.linspace(0, 1, 48)):
        yield i

ani = animation.FuncAnimation(fig, run, frames=range(iter.size), interval=10, repeat=False)

if save_animation:
    FFwriter = animation.FFMpegWriter(fps=save_fps, extra_args=['-vcodec', 'libx264'])
    ani.save(f"continuous_ball_{save_fps}_norepeat.mp4", writer=FFwriter)
else:
    plt.show()
