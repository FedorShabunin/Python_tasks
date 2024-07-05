from .solver import ISolver, np
class Solver(ISolver):
    def run(self):
        it = 0
        rel_res = 1.0

        z = self.map
        h = self.dh
        z1 = 1000
        itmax = 500
        rtol = 1e-5
        while (rel_res>rtol) and (it<=itmax):
            #z0 = np.sum(np.square(z))
            dz = (z[1:-1,0:-2] + z[0:-2,1:-1] + z[1:-1,2:]+ z[2:,1:-1])/4 - z[1:-1, 1:-1]
            z0 = z1
            z1 = np.sqrt(np.sum(np.square(dz)))
            z[1:-1, 1:-1] += dz
            z = z*(~self.bound_map) + self.bound_values
            it += 1    
        return z
        
