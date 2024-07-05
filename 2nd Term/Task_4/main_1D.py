import sys
import os
sys.path.append(os.path.join(sys.path[0], 'solver'))

from one import Domain

from matplotlib import pyplot as plt
import numpy as np
from concurrent.futures import ThreadPoolExecutor as TPE
from time import time

d1 = Domain(500, 1, B=0.1, T = 1.0, J = -1)
st, se, sm, cc,uu,mm = d1.run_long_metro(10000, full_log = 1)

fig = plt.figure(figsize=(12, 12))
ax1 = fig.add_subplot(2,1,1)

cf = ax1.imshow(st, aspect=10, cmap='magma')
fig.colorbar(cf, ax=ax1, orientation='vertical')
ax1.set_title('Spin evolution')
ax1.set_xlabel('Time')
ax1.set_ylabel('Position')

ax2 = fig.add_subplot(2,1,2, aspect = 1)
ax2.plot(se, color = 'Purple')
ax2.set_title('Energy evolution')
plt.show()

async def process(d: Domain, steps:int):
    #st, se, sm, cc,uu,mm = d1.run_long_metro(10000, full_log = 1)
    return d.run_long_metro(10000, full_log = 1)

def proc(t):
    return Domain(500, 1, B=0.1, T = t, J = -1).run_long_metro(10000, ensemble_rate = 0.5)

Ts = np.linspace(1.0, 5, 50)

with TPE(50) as p:
    rr = p.map(proc, Ts)
rr = list(rr)

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