import matplotlib.pyplot as plt
import numpy as np

from pylab import rcParams
rcParams['figure.figsize'] = (5.9, 4)
rcParams['font.size'] = 14
rcParams['lines.linewidth'] = 2.5
rcParams['axes.linewidth'] = 1.3


X = [32, 64, 128, 256, 512, 1024]
Y = [6465.12, 2742.38, 2705.88, 2684.25, 2934.62, 4498.38]
Xsuper = [32, 64, 128, 256, 512, 1024]
Ysuper = [3267.5, 1605.5, 836.25, 845.25, 948.5, 1218.62]

fig, ax = plt.subplots(1, 1)
ax.scatter(X, Y, marker = "x", label="RTX 3070", color="black")
ax.plot(X, Y, linestyle="dashed", color="black", linewidth=2)
ax.scatter(Xsuper, Ysuper, marker = "x", label="A100", color="red")
ax.plot(Xsuper, Ysuper, linestyle="dashed", color="red", linewidth=2)
ax.set_xscale("log")
ax.set_xlabel("Number of threads per block")
ax.set_ylabel("Mean execution in $\mu s$")
ax.set_xticks(X)
ax.set_xticklabels([str(x) for x in X])
ax.set_yticks([y*1000 for y in  [1, 2, 3, 4, 5, 6]])
ax.legend()
ax.grid(True)



fig.tight_layout()
plt.show()
