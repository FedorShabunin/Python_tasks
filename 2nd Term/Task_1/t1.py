import numpy as np
from plotter import plotAll

# variables
a = 10
U0 = 100.0

# domain
dx, dy = a*1e-2, a*1e-2
x, y = np.mgrid[slice(0, a + dx, dx), slice(0, a + dy, dy)]

# analitical solution
U = 4*U0/np.pi * np.sum([(np.sin(np.pi*(2*k+1)/a*y)*1/(2*k+1)*(
    np.cosh(np.pi*(2*k+1)/a*x) - np.sinh(np.pi*(2*k+1)/a*x) / np.tanh(np.pi*(2*k+1)))) for k in range(60)], axis=0)

# applying boundary conditions
U[0, :] = U0
U[-1, :] = 0.

plotAll(x, y, U, fname='t01.png', title='Analytical solution map')
