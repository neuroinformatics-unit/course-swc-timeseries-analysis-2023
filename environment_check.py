"""
The purpose of this script is to check the environment is working.
You can use "pip install <package_name>" to install the below dependencies
in your environment.

The dependencies of the course are:
    numpy
    scipy
    matplotlib
    pandas
    seaborn

Running the script should display a plot. If you are having trouble
displaying the plot or setting up an environment, please contact
Joe Ziminski (j.ziminski@ucl.ac.uk) or Adam Tyson (adam.tyson@ucl.ac.uk)
and we'd be happy to help.
"""
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

frequency_hz = 4

num_samples = 10000
n = np.linspace(0, 1, num_samples)

plt.plot(n, np.sin(2 * np.pi * n * frequency_hz))
plt.plot(n, np.cos(2 * np.pi * n * frequency_hz), c="tab:red")

plt.xlabel("Time (s)")
plt.legend(["Sine", "Cosine"], loc="upper right")
plt.title("The environment is working. Feel free to close this plot.")
plt.show()

