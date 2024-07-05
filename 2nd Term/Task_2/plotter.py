import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

def plotAll(x, y, z, fname = 't.png', title = 'In-plane'):
    levels = MaxNLocator(nbins=50).tick_values(z.min(), z.max())
    cmap = plt.colormaps['magma']

    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(1,2,1)

    cf = ax1.contourf(x, y, z, levels=levels, cmap=cmap)
    fig.colorbar(cf, ax=ax1)
    ax1.set_title(title)

    ax2 = fig.add_subplot(1,2,2, projection='3d')
    ax2.plot_wireframe(x, y, z, color='purple')
    ax2.set_title(r'3D Plot')
    ax2.contour(x, y, z, zdir='z', offset=-10, levels=levels, cmap=cmap)

    plt.savefig(fname)
    plt.show()
