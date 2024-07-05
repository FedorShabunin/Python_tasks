import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

def plotAll(x, y, z, fname = 't.png', title = 'Plot'):
    levels = MaxNLocator(nbins=50).tick_values(z.min(), z.max())
    cmap = plt.colormaps['RdYlBu']

    fig = plt.figure(figsize=(12, 8))
    #ax1 = fig.add_subplot(1,2,1)

    #cf = ax1.contour(x, y, z, levels=levels, linewidth=1)
    #fig.colorbar(cf, ax=ax1)
    #ax1.set_title(title)
    #ex, ey = z.copy(), z.copy()
    #ex[1:-1, 1:-1] = (-z[0:-2,1:-1] + z[2:,1:-1])/2
    #ey[1:-1, 1:-1] = (-z[1:-1,0:-2] + z[1:-1,2:])/2
    #id = 10
    #idy = 5
    #ex[id,idy:-1-idy] = 0
    #ex[-1-id,idy:-1-idy] = 0
    #ey[id,idy:-1-idy] = 0
    #ey[-1-id,idy:-1-idy] = 0
    #ax1.streamplot(x[:, 0],y[0, :],ex,ey, cmap = plt.colormaps['inferno'])

    ax2 = fig.add_subplot(1,1,1, projection='3d')
    ax2.plot_wireframe(x, y, z, color='blue')
    ax2.set_title(r'z = $U(x,y)$')
    ax2.contourf(x, y, z, zdir='z', offset=-10, levels=levels, cmap=cmap)

    plt.savefig(fname)
    plt.show()