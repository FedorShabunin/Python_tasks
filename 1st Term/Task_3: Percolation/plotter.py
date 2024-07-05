from calendar import day_abbr
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
from data_runner import Axis
import h5py as h


_default_axes = Axis(start = 0., end = 1., step = 0.2)

def plot(nvalues: Optional[tuple] = (10, 15, 20), axes: Axis = _default_axes):
   
    fig, ax = plt.subplots(1,1, figsize=(10,6))
    for i in nvalues:
        with h.File(f'data{i}.hdf5', 'r') as f:
            data_set = f['default']
            p, cond = data_set[:, 0], data_set[:, 1]
       
        ax.plot(p, cond, 'o-', linewidth=2.0, label=f'$n={i}$')

    ax.set(xlim=(0, 1), xticks=np.arange(*axes),
        ylim=(0, 1), yticks=np.arange(*axes))
    plt.title('Percolation for 2D')
    plt.xlabel('p')
    plt.ylabel('$P_{cond}$')
    plt.legend(loc='upper left')
    plt.savefig('plot.svg', dpi = 120)
