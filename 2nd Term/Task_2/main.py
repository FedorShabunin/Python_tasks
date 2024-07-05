#!/usr/bin/env pytdhon3

from solver import *
import numpy as np
import time

T0 = 100.0
u_e = 10.0
dh = 0.5
dt = 0.1
space = solver.Space(
        x = solver.Interval(0, 10), t = solver.Interval(0, 45)
    )
nsx = int((space.x.end - space.x.start )/dh)
nst = int((space.t.end - space.t.start )/dt)

bc = np.zeros((nsx, nst))
bc[0, :] = 0
bc[-1, :] = 0
bc[:, 0] = T0
#bc[:, 0] = np.sin(np.pi*(np.mgrid[space.x.start:space.x.end:dh])/space.x.end) # eigenfunction initial condition
#bc[1:nsx//2, 0] = T0/2
#bc[nsx//2:, 0] = T0

js = eftcs.Solver(
   space=space,
   dh = dh,
   dt = dt,
   rtol = 1e-1
)

#js = crank_nicolson.Solver(
#    space=space,
#    dh = dh,
#    dt = dt,
#    rtol = 1e-1
#)

js.set_bounds(bc)
#print(js.axis_x)
#print(js.map)

start = time.time()
z = js.run()
end = time.time()
print(f'{end - start:.3e} sec')


from plotter import plotAll
plotAll(js.axis_x, js.axis_t, z, fname='t01j.png')
  