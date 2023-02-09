import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
# https://stackoverflow.com/questions/44938231/animating-a-stem-plot-in-matplotlib

save_animation = True
save_fps = 30

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

def run(i):
    h_stem[0].set_ydata(1 - iter[i])
    line.set_data(iter[:i+1], 1 - iter[:i+1])
    if 1 - iter[i] == 0:
        ani.event_source.stop()

def data_gen():
    for i in reversed(np.linspace(0, 1, 48)):
        yield i

def init():
    ax1.set_ylim(0, 1)
    ax1.set_xlim(-0.01, 0.01)
    ax1.get_xaxis().set_visible(False)
    ax1.set_ylabel("Height (m)")
    ax1.set_ylabel("Height (m)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylim(0, 1)
    ax2.set_xlim(0, 1)

    return h_stem,

ani = animation.FuncAnimation(fig, run, frames=range(iter.size), interval=10, repeat=False)
if save_animation:
    # FFwriter = animation.FFMpegWriter(fps=save_fps, extra_args=['-vcodec', 'libx264'])
    ani.save(f"continuous_ball_{save_fps}_norepeat.gif", writer=animation.PillowWriter(fps=save_fps))# , fps=save_fps)
else:
    plt.show()

plt.show()