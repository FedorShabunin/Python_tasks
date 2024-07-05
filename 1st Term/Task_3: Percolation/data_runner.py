import numpy as np
from matrix import move_down, gen
from tqdm import tqdm
from typing import Optional
from typing import NamedTuple
import h5py as h

class Axis(NamedTuple):
    start: float
    end: float
    step: float

default_nvalues = (10, 15, 20)

def pcond(n: int, p: float, k: Optional[int] = 1000) -> float:

    return sum([move_down(gen(n, p)) for i in tqdm(range(k), desc=f'evaluating for p = {p:.3f}')])/k

def run(nvalues: Optional[tuple] = default_nvalues, density: Optional[int] = 25, k: Optional[int] = 1000) -> None:
   
    ax = Axis(start = 0, end = 1, step = density)
    for i in nvalues:
        values = []
        for p in np.linspace(*ax):
            values += [(p, pcond(i, p, k))]
        with h.File(f'data{i}.hdf5', 'w') as f:
            f.create_dataset("default", (density, 2), 'f', data=values)
            
