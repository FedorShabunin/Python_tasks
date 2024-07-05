import sys
import os
sys.path.append(os.path.join(sys.path[0], 'solver'))
from .solver import ISolver, np
from relaxation import relaxation

class Solver(ISolver):
    def run(self, omega):
        z, rel, iter = relaxation(self.bound_values,self.bound_map, 1e-2, self.dh, omega)
        return np.asarray(z), rel, iter