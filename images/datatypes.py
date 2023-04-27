import matplotlib.pyplot as plt
import numpy as np

from pylab import rcParams
rcParams['figure.figsize'] = (5.9, 3)
rcParams['font.size'] = 12
rcParams['lines.linewidth'] = 2.5
rcParams['axes.linewidth'] = 1.3

rtx3070 = dict()
rtx3070["int32"] = 2597.7
rtx3070["int64"] = 5170.51
rtx3070["float"] = 2594.71
rtx3070["double"] = 5172.44

X = ["int32", "int64", "float", "double"]
Y = [2597.7, 5170.51, 2594.71, 5172.44]
Ysuper = [882.512, 1508.18, 969.211, 1509.76]

a100 = dict()
a100["int32"] = 882.512
a100["int64"] = 1508.18
a100["float"] = 969.211
a100["double"] = 1509.76


fig, (axr, axa) = plt.subplots(1, 2)
axr.bar(X, Y, color="black")
axr.set_ylabel("Mean execution time in $\mu s$")
axa.bar(X, Ysuper, color="red")



fig.tight_layout()
plt.show()
