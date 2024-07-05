# -*- coding: utf-8
#!/usr/bin/env python

import pylab
import numpy as np
from solver import *
from initials import *
from matplotlib import pyplot as plt, animation


dx = 0.1
dt = 0.05
x_s = 50.0
t = 50.0
p0 = 0.0

space = solver.Space(
        x = solver.Interval(-x_s, x_s, dx), t = solver.Interval(0, t, dt)
    )

X = space.x.get_axis()
T = space.t.get_axis()

b = 3.5
a = 0.5

c = a+b/2.0
m = b - a 

init = mapped(X, InitialWave.gauss_package(x0=-10, sigma=m, p0=1.0))
# init = mapped(X, InitialWave.gauss_package(x0=c, sigma=m*0.9, p0=0.0))
init /= np.sqrt(np.abs(np.dot(init, np.conj(init))))


# /V = mapped(X, Potential.double_rect_well(in_bound = a,out_bound = b, magnitude=-1.5))
V = mapped(X, Potential.finite_step(start=10.0, height=-0.5))

slv = rhombus_method.Solver(
   space=space,
   V=V
).set_initials(init)
u = slv.run()

fig = plt.figure()
ax = fig.add_subplot(111)
nss = np.sum(np.abs(u[:,:])**2, axis=0)
ax.plot(T[:-20], nss[20:]/nss[-1], color='Purple')
ax.set_title("Normalization")
plt.show()
exit()


skip = 10

fig = plt.figure()
ax = fig.add_subplot(111)
line, = ax.plot(X, 10*np.abs(init)**2, color='Purple')
ax.plot(X,V, color='Blue')
ax.set_title("Absolute value of wave function squared")
ax.set_xlabel('x')
ax.set_ylabel('$|\psi(x,t)|^2$')
# ax.pylab.plot(X,V,'x:', '$|\psi(x,t)|^2$')

def animate(i):
    line.set_ydata( 10*np.abs(u[:,int(skip*i)])**2 + p0*p0)
    return line


ani = animation.FuncAnimation(fig, func=animate, frames=len(T)//skip-1, interval=100)
plt.show()
ani.save('animation.gif', fps=30)

