#!/usr/bin/env python
# coding: utf-8

import numpy as np

from scipy import constants
from scipy import integrate

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

h = constants.hbar
c = constants.c
m = constants.m_e
e = constants.e

l_c = h / (m * c)
t_c = l_c / c
p_c = m * c
E_c = m ** 2 * c ** 3 / (e * h)

E = 7

x_0 = 1
y_0 = 2
z_0 = 3
p_x_0 = 7
p_y_0 = 8
p_z_0 = 0


def particle(t, r):
    X, Y, Z, Px, Py, Pz = r

    fx = Px / np.sqrt(1 + Px ** 2 + Py ** 2 + Pz ** 2)
    fy = Py / np.sqrt(1 + Px ** 2 + Py ** 2 + Pz ** 2)
    fz = Pz / np.sqrt(1 + Px ** 2 + Py ** 2 + Pz ** 2)
    fpx = E * (1 - Pz / np.sqrt(1 + Px ** 2 + Py ** 2 + Pz ** 2))
    fpy = 0
    fpz = Px / np.sqrt(1 + Px ** 2 + Py ** 2 + Pz ** 2) * E
    
    return fx, fy, fz, fpx, fpy, fpz

sol = integrate.solve_ivp(particle, t_span=(0, 50), y0 = (x_0, y_0, z_0, p_x_0, p_y_0, p_z_0), t_eval=np.linspace(0, 10, 1000))
x, y, z, px, py, pz = sol.y

t = sol.t

A = np.sqrt(1 + px ** 2 + py ** 2 + pz ** 2) - pz
e_cons = np.sqrt(1 + p_y_0 ** 2)

x_teor = 1 / (2 * A * E) * px ** 2
x_teor = x_teor - x_teor[0]
y_teor = p_y_0 / (E * A) * px
y_teor = y_teor - y_teor[0]
z_teor = (-1 + (e_cons / A) ** 2) / (2 * E) * px + 1 / (6 * A ** 2 * E) * px ** 3
z_teor = z_teor - z_teor[0]

    
fig, ax1 = plt.subplots(figsize = (10, 8))
ax1.plot(x, y, label = r'$Численно$', color = 'blue')
ax1.plot(x_0 + x_teor, y_0 + y_teor, label = r'$Теоретически$', color = 'green')
ax1.set_xlabel(r'$X\;[l_c\;=\;\frac{\hbar}{m \cdot c}]$', size  = 12)
ax1.set_ylabel(r'$Y\;[l_c\;=\;\frac{\hbar}{m \cdot c}]$', size  = 12)
ax1.set_title(r'$0XY$', size  = 20)
ax1.legend(prop={'size': 15}, title = r'$Метод\;решения$')
ax1.grid()

fig, ax2 = plt.subplots(figsize = (10, 8))
ax2.plot(y, z, label = r'$Численно$', color = 'blue')
ax2.plot(y_0 + y_teor, z_0 + z_teor, label = r'$Теоретически$', color = 'green')
ax2.set_xlabel(r'$y\;[l_c\;=\;\frac{\hbar}{m \cdot c}]$', size  = 12)
ax2.set_ylabel(r'$z\;[l_c\;=\;\frac{\hbar}{m \cdot c}]$', size  = 12)
ax2.set_title(r'$0YZ$', size  = 20)
ax2.legend(prop={'size': 15}, title = r'$Метод\;решения$')
ax2.grid()

fig, ax3 = plt.subplots(figsize = (10, 8))
ax3.plot(x, z, label = r'$Численно$', color = 'blue')
ax3.plot(x_0 + x_teor, z_0 + z_teor, label = r'$Теоретически$', color = 'green')
ax3.set_xlabel(r'$x\;[l_c\;=\;\frac{\hbar}{m \cdot c}]$', size  = 12)
ax3.set_ylabel(r'$z\;[l_c\;=\;\frac{\hbar}{m \cdot c}]$', size  = 12)
ax3.set_title(r'$0XZ$', size = 20)
ax3.legend(prop={'size': 15}, title = r'$Метод\;решения$')
ax3.grid()

fig = plt.figure(figsize = (10, 8))
ax = plt.axes(projection ='3d')
ax.plot3D(x, y, z, label = r'$Численно$', color = 'blue')
ax.plot3D(x_0 + x_teor, y_0 + y_teor, z_0 + z_teor, label = r'$Теоретически$', color = 'green')
ax.set_title(r'$3D$')
ax.set_xlabel(r'$X$')
ax.set_ylabel(r'$Y$')
ax.set_zlabel(r'$Z$')
ax.legend(prop={'size': 15}, title = r'$Метод\;решения$')

plt.show()
