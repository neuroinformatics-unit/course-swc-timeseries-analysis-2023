import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
# https://stackoverflow.com/questions/44938231/animating-a-stem-plot-in-matplotlib
fig, axes = plt.subplots(1, 2)
show_line = False

save_animation = True
save_fps = 5

ax1, ax2 = axes
h_stem = ax1.stem(0, 1, linefmt="w", markerfmt="#1f77b4")
h_stem[0].set_markersize(10)

h2_stem = ax2.stem(0, 1, linefmt="k", markerfmt="k", basefmt="k")
h2_stem[0].set_markerfacecolor('none')

ax1.set_ylim(0, 1)
ax1.set_xlim(-0.01, 0.01)
ax1.get_xaxis().set_visible(False)
ax1.set_ylabel("Height (m)")
ax2.set_xlabel("Time (s)")
ax2.set_ylim(0, 1)
ax2.set_xlim(0, 1)

iter = np.linspace(0, 1, 48)
running_stem_time = []
running_stem_data = []
import time

line, = ax2.plot([], [], lw=2, c="k")

def run(i):
    h_stem[0].set_ydata(1 - iter[i])

    if show_line:
        line.set_data(iter[:i + 1], 1 - iter[:i + 1])

    if i in np.arange(0, 48, 12):
        running_stem_time.append(iter[i])
        running_stem_data.append(1 - iter[i])
        h2_stem = ax2.stem(running_stem_time, running_stem_data, linefmt="k", markerfmt="k", basefmt="k")
        h2_stem[0].set_markerfacecolor('none')

    if 1 - iter[i] == 0:
        ani.event_source.stop()



def data_gen():
    for i in reversed(np.linspace(0, 1, 48)):
        yield i

ani = animation.FuncAnimation(fig, run, frames=range(iter.size), interval=50)
if save_animation:
    FFwriter = animation.FFMpegWriter(fps=save_fps, extra_args=['-vcodec', 'libx264'])
    ani.save(f"discrete_ball_line_{show_line}_{save_fps}fps.mp4", writer=FFwriter)
else:
    plt.show()

plt.show()