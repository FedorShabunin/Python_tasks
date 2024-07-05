import numpy as np

def iterate(Z):
    N = (Z[0:-2,0:-2] + Z[0:-2,1:-1] + Z[0:-2,2:] +
         Z[1:-1,0:-2]                + Z[1:-1,2:] +
         Z[2:  ,0:-2] + Z[2:  ,1:-1] + Z[2:  ,2:])

    birth = (N == 3) & (Z[1:-1,1:-1]==0)
    survive = ((N == 2) | (N == 3)) & (Z[1:-1,1:-1] == 1)
    Z[...] = 0
    Z[1:-1,1:-1][birth | survive] = 1
    return Z
    
    
import time
from IPython import display
import matplotlib.pyplot as plt
import matplotlib.animation as animation

Z = np.random.randint(0,2,(50,50))

def animate(frame):
    im.set_data(iterate(Z))
    return im,

fig = plt.gcf()

im = plt.imshow(Z)
plt.show()

anim = animation.FuncAnimation(fig, animate, frames=500,
                               interval=50, blit = True)

anim.save('Life.mp4')
