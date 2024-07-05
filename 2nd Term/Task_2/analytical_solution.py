import numpy as np
from plotter import plotAll

#variables
T0 = 100 # Celcius deg
L = 10 # cm
Tau = 45 # sec
a = 1

# domain
dx, dt = L*1e-2, Tau*1e-2
x, t = np.mgrid[slice(0, L + dx, dx), slice(0, Tau + dt, dt)]

# analitical solution
U = 4*T0/np.pi * np.sum([(np.sin(np.pi*(2*k+1)/L*x)*1/(2*k+1)*(
       np.exp(-(a*np.pi*(2*k+1)/L)*(a*np.pi*(2*k+1)/L)*t)
    )) for k in range(60)], axis=0)

# eigenfunction initial condition 
U = np.sin(np.pi*x/L) * np.exp(-t*(a*np.pi/L)**2)

# applying boundary conditions
U[0, :] = 0.
U[-1, :] = 0.
#U[:, 0] = T0
# eigenfunction initial condition 
#U[:, 0] = np.sin(np.pi*x[0]/L)

plotAll(x, t, U, fname='t01.png', title='Analytical solution map')

