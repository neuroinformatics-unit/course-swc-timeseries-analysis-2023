import matplotlib.pyplot as plt

def show_line_plot(x, y, xlabel="", ylabel="", title=""):
   """ Make and show a matplotlib lineplot """
   plt.plot(x, y)
   plt.ylabel(ylabel)
   plt.xlabel(xlabel)
   plt.title(title)
   plt.show()



















