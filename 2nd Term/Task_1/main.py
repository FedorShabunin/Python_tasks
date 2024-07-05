#!/usr/bin/env python3

from solver import solver, jacobi, gauss_seidel, super_relaxation
import numpy as np

U0 = 100.0
h = 1
sizes = solver.Sizes(
        x = solver.Interval(0, 30), y = solver.Interval(0, 30)
    )
nsx = int((sizes.x.end - sizes.x.start )/h)
nsy = int((sizes.y.end - sizes.y.start )/h)

bm = np.zeros((nsx, nsy)).astype(bool)
id = 5
idy = 5
ix = 15
w = 1
#bm[0,:] = True
bm[ix:ix+1+w,:] = True
bm[-ix:-ix-1-w,:] = True
#bm[id,idy:-1-idy] = True
#bm[-1-id,idy:-1-idy] = True
bc = np.zeros((nsx, nsy))
#bc[0,:] = U0
bc[ix:ix+1+w,:] = U0
bc[-ix:-ix-1-w,:] = -U0
#bc[id, idy:-1-idy] = U0
#bc[-1-id, idy:-1-idy] = -U0

#js = jacobi.Solver(
#    sizes=sizes,
#    dh = h,
#    rtol = 1e-1
#)
js = gauss_seidel.Solver(
    sizes=sizes,
    dh = h,
    rtol = 1e-1
)
#js = super_relaxation.Solver(
#    sizes=sizes,
#    dh = h,
#    rtol = 1e-1
#)
js.set_bounds(bm, bc)
#print(js.axis_x)
#print(js.map)

z, gauss_rtol, iter = js.run()

print("Number of iterations:", iter)

from plotter import plotAll
plotAll(js.axis_x, js.axis_y, z, fname='t1new.png')