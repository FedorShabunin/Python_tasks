# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 14:14:17 2021

@author: Артём
"""

import numpy as np

import matplotlib.pyplot as plt

t = np.linspace(0, 7, 100)

B = np.array([0, 0, -1])  # магнитное поле по оси z

E = np.array([0, 0, 0])

v0 = np.array([1, 0, 0])

v = np.zeros((t.size, 3))
v[0, :] = v0  # задали начальную скорость

dt = t[1] - t[0]

for i in range(1, t.size, 1):
    v_minus = v[i-1, :] + E * dt/2
    t1 = B * dt/2
    v_prime = v_minus + np.cross(v_minus, t1)
    s = 2*t1/(1+np.dot(t1, t1))
    v_plus = v_minus + np.cross(v_prime, s)
    v[i, :] = v_plus + E * dt/2
    
x = np.zeros((t.size, 3))

x0 = np.array([0, 0, 0])

x[0, :] = x0

for i in range(1, t.size, 1):
    x[i, :] = x[i-1, :] + v[i, :] * dt
    
plt.plot(x[:, 0], x[:, 1])
plt.show()