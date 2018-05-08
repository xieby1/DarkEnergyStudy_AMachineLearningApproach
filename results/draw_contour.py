# -*- coding: utf-8 -*-

import numpy
import matplotlib.pyplot as plt
import seaborn

seaborn.set(color_codes=True)
fig,ax=plt.subplots()

name = "20180507_221420sigma"

distri = numpy.loadtxt(name)
ax=seaborn.kdeplot(distri[:,0], distri[:,1], cmap="Blues",shade="True", shade_lowest=False)
#plt.scatter(distri[:,0], distri[:,1], marker='.', s=7, alpha=0.5)
#plt.plot([-1, -1],[distri[:,1].min(), distri[:,1].max()],"--", color='gray', linewidth=1.0, alpha=0.5)
#plt.plot([distri[:,0].min(), distri[:,0].max()],[0, 0],"--", color='gray', linewidth=1.0, alpha=0.5)

plt.savefig(name+"contour.png", format="png")

plt.show()

