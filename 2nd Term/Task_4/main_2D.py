import sys
import os
sys.path.append(os.path.join(sys.path[0], 'solver'))

from two import Domain2d

from matplotlib import pyplot as plt
import numpy as np
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor as TPE
from time import time

def proc(t):
    return Domain2d(20, 1, B=0.1, T = t, J = -1).run_long_metro(15000, ensemble_rate = 0.5)

Ts = np.linspace(1, 5, 150)

start = time()
with TPE(50) as p:
    rr = p.map(proc, Ts)
rr = list(rr)
endd = time()
print(f'{endd - start:.3e} sec')

cc_list = [rri[3] for rri in rr]
uu_list = [rri[4] for rri in rr]
mm_list = [rri[5] for rri in rr]

fig = plt.figure(figsize=(12, 12))
ax1 = fig.add_subplot(3,1,2)
ax1.set_ylabel('C(T)')
ax1.plot(Ts, cc_list, color = 'Purple')
ax2 = fig.add_subplot(3,1,1)
ax2.set_ylabel('U(T)')
ax2.plot(Ts, uu_list, color = 'Purple')
ax3 = fig.add_subplot(3,1,3)
ax3.set_ylabel('M(T)')
ax3.plot(Ts, mm_list, color = 'Purple')
plt.show()

d2 = Domain2d(20, 1, B=0.1, T = 0.5, J =-1)
sst = d2.get_spins()

fig = plt.figure(figsize=(12, 12))
ax1 = fig.add_subplot(1,1,1)

cf = ax1.imshow(sst, interpolation='nearest', cmap = 'magma')
plt.colorbar(cf)
ax1.set_title('Initial state')

plt.show()

d2.run_long_metro(10000, ensemble_rate = 0.5)
sst = d2.get_spins()

fig = plt.figure(figsize=(12, 12))
ax1 = fig.add_subplot(1,1,1)

cf = ax1.imshow(sst, cmap = 'magma')
plt.colorbar(cf)
ax1.set_title('Final state')

plt.show()