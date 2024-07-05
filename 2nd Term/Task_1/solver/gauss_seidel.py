import sys
import os
sys.path.append(os.path.join(sys.path[0], 'solver'))
print(sys.path)
from .solver import ISolver, np
from relaxation import relaxation

class Solver(ISolver):
    def run(self):
        z, rel, iter = relaxation(self.bound_values,self.bound_map, 1e-2, self.dh, 1)
        return np.asarray(z), rel, iter