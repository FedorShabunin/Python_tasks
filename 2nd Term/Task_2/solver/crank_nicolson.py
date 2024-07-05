import sys
import os
# sys.path.append('./solver/lib/')
#sys.path.append('./solver/pyx/')
sys.path.append(os.path.join(sys.path[0], 'solver'))
from .solver import ISolver, np
from crank import nicolson

class Solver(ISolver):
    def run(self):
        al = self.dt/(self.dh*self.dh)
        U = nicolson(self.map, al)
        return np.asarray(U)