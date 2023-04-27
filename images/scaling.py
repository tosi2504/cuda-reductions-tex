import matplotlib.pyplot as plt
import numpy as np

from pylab import rcParams
rcParams['figure.figsize'] = (5.9, 4)
rcParams['font.size'] = 14
rcParams['lines.linewidth'] = 2.5
rcParams['axes.linewidth'] = 1.3

def load_data(filename):
    with open(filename, "r") as file:
        lines = [line.strip().split(" ") for line in file.readlines()]
    X, Y = list(), list()
    for size, duration in lines:
        X.append(int(size))
        Y.append(float(duration))
    return X, Y

X, Y = load_data("scaling_rtx3070.txt")
Xsuper, Ysuper = load_data("scaling_a100.txt")

print(X, Y)
print(Xsuper, Ysuper)
fig, ax = plt.subplots(1, 1)
ax.scatter(X, Y, marker = "x", label="RTX 3070", color="black")
ax.plot(X, Y, linestyle="dashed", color="black", linewidth=2)
ax.scatter(Xsuper, Ysuper, marker = "x", label="A100", color="red")
ax.plot(Xsuper, Ysuper, linestyle="dashed", color="red", linewidth=2)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Array size")
ax.set_ylabel("Mean execution time in $\mu s$")
ax.set_xticks(X[::2])
ax.set_xticklabels(["$2^{"+str(e)+"}$" for e in range(10,len(X)+10,2)])
ax.legend()
ax.grid(True)



fig.tight_layout()
plt.show()
