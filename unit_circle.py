import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

circle1 = plt.Circle((0, 0), 1
                     , fill=False)

V = np.array([[1,1], [-2,2], [4,-7]])
origin = np.array([[0, 0, 0],[0, 0, 0]])

plt.quiver(*origin, V[:,0], V[:,1], color=['r','b','g'], scale=21 )

ax.add_patch(circle1)

plt.show()