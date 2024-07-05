#!/usr/bin/env python
# coding: utf-8
import numpy as np

from scipy import constants as con
from scipy import integrate
from scipy.constants import e, m_e

from sympy.abc import x, y, z, t
from sympy import sympify
from sympy import lambdify

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

import math

class ElectromagneticField():
    def __init__(self, E_x = 0, E_y = 0, E_z = 0, H_x = 0, H_y = 0, H_z = 0):
        self.E = np.array([E_x, E_y, E_z])
        self.H = np.array([H_x, H_y, H_z])
    '''
    def __init__(self, E_x = 0, E_y = 0, E_z = 0, H_x = 0, H_y = 0, H_z = 0):
        self.E_x = E_x
        self.E_y = E_y
        self.E_z = E_z
        self.H_x = H_x
        self.H_y = H_y
        self.H_z = H_z
        self.Fields = {
            'E_x': self.E_x, 
            'E_y': self.E_y,
            'E_z': self.E_z,
            'H_x': self.H_x,
            'H_y': self.H_y,
            'E_z': self.E_z,
        }
    '''
    def components_E_field(self, coordinate = np.zeros(4)):
        
        E_components_field = np.array([lambdify([x, y, z, t], self.E[i], "numpy")(coordinate[0], coordinate[1], coordinate[2], coordinate[3]) 
                              for i in range(0, 3)], dtype=object)
        return E_components_field
            
    def components_H_field(self, coordinate = np.zeros(4)):
        
        H_components_field = np.array([lambdify([x, y, z, t], self.H[i], "numpy")(coordinate[0], coordinate[1], coordinate[2], coordinate[3])
                              for i in range(0, 3)], dtype=object) 
        return H_components_field
    '''    
    def components_field(self, x_1, y_1, z_1, t_1, E_only=True, H_only=True):
        
        self.E_components_field = np.zeros(3)
        self.E_components_field[0] = sympify(self.E_x).evalf(subs={x: x_1, y: y_1, z: z_1, t: t_1})
        self.E_components_field[1] = sympify(self.E_y).evalf(subs={x: x_1, y: y_1, z: z_1, t: t_1})
        self.E_components_field[2] = sympify(self.E_z).evalf(subs={x: x_1, y: y_1, z: z_1, t: t_1})
        
        self.H_components_field = np.zeros(3)
        self.H_components_field[0] = sympify(self.H_x).evalf(subs={x: x_1, y: y_1, z: z_1, t: t_1})
        self.H_components_field[1] = sympify(self.H_y).evalf(subs={x: x_1, y: y_1, z: z_1, t: t_1})
        self.H_components_field[2] = sympify(self.H_z).evalf(subs={x: x_1, y: y_1, z: z_1, t: t_1})
        
        if E_only and H_only: return self.E_components_field, self.H_components_field
        if E_only: return self.E_components_field
        if H_only: return self.H_components_field
    '''
    
    #def sum_of_squares(self, E_only=True, H_only=True):
        #'''Сумма квадратов компонет поля'''
        #self.E_field_components_squared = np.sum((self.E_components_field)**2)
        #self.H_field_components_squared = np.sum((self.H_components_field)**2)
        
        #if E_only and H_only: return self.E_field_components_squared, self.H_field_components_squared
        #if E_only: return self.E_field_components_squared
        #if H_only: return self.H_field_components_squared

    def module_E(self, coordinate = np.zeros(4)):
        cmp_E = self.components_E_field(coordinate)
        E_module = np.sqrt(np.sum(np.dot(cmp_E, cmp_E)))
        return E_module
    
    def module_H(self, coordinate = np.zeros(4)):
        cmp_H = self.components_H_field(coordinate)
        H_module = np.sqrt(np.sum(np.dot(cmp_H, cmp_H)))
        return H_module

    def plot_field_components_in_plane(self):
        '''Компоненты поля, отбираем оси и координаты'''
        axes = ['x', 'y', 'z', 't']
        axes_1 = [1, 2, 3, 4]
        axes_2 = []
        other_var_value = np.zeros(2)
        other_var = []
        
        print("Какую компоненту поля необходимо построить?")
        cmp = input()        
        print('Введите абсциссу для графика: ', end = '')
        abscissa = input()
        print('Введите ординату для графика: ', end = '')
        ordinate = input()

        for i in axes_1:
            if axes[i-1] == abscissa:
                axes_2 = axes_2 + [i]
            if axes[i-1] == ordinate:
                axes_2 = axes_2 + [i]

        print('Зафиксируйте остальные переменные:')
        k = 0 
        for i in axes_1:
            if i not in axes_2:
                print(axes[i-1], '= ', end = '')
                other_var_value[k] = float(input())
                other_var = other_var + list(axes[i-1])
                k = k + 1 
        
        def function(coordinate, cmp):
            '''Построение'''
            #expr = sympify(self.Fields[cmp])
            
            #return lambdify([x, y, z, t], expr, "numpy")(*coordinate) 
            if cmp == 'E_x':
                return self.components_E_field(coordinate)[0]
            elif cmp == 'E_y':
                return self.components_E_field(coordinate)[1]
            elif cmp == 'E_z':
                return self.components_E_field(coordinate)[2]
            elif cmp == 'H_x':
                return self.components_H_field(coordinate)[0]
            elif cmp == 'H_y':
                return self.components_H_field(coordinate)[1]
            elif cmp == 'H_z':
                return self.components_H_field(coordinate)[2]

        absc, ordn = np.meshgrid(np.arange(-10, 10, 0.05), np.arange(-10, 10, 0.05))
        coordinate = np.array([0, 0, 0, 0], dtype=object)
        j = 0
        for i in range(0, 4):
            if(axes[i] == abscissa):
                coordinate[i] = absc
            elif(axes[i] == ordinate):
                coordinate[i] = ordn
            else:
                coordinate[i] = other_var_value[j] 
                j = j + 1
            
        z_grid = function(coordinate, cmp)
        #z_grid = z_grid.astype(np.float64)
        #print(z_grid.shape)
        
        plt.figure(figsize=(9, 6))
        plt.imshow(z_grid, extent=[-4, 4, -4, 4], origin="lower", cmap = 'inferno')
        plt.colorbar()
        plt.title(f"{cmp[0]}: {cmp[2]} компонента поля с {other_var[0]} = {other_var_value[0]}"
                  f" и {other_var[1]} = {other_var_value[1]}")
        plt.xlabel(f"{abscissa}")
        plt.ylabel(f"{ordinate}")
        plt.show()

    
    def plot_density_of_energy(self):
        '''Плотность энергии, отбираем оси и фиксируем координаты'''
        axes = ['x', 'y', 'z', 't']
        axes_1 = [1, 2, 3, 4]
        axes_2 = []
        other_var_value = np.zeros(2)
        other_var = []
      
        print('Введите абсциссу для графика: ', end = '')
        abscissa = input()
        print('Введите ординату для графика: ', end = '')
        ordinate = input()

        for i in axes_1:
            if axes[i-1] == abscissa:
                axes_2 = axes_2 + [i]
            if axes[i-1] == ordinate:
                axes_2 = axes_2 + [i]

        print('Зафиксируйте остальные переменные:')
        k = 0 
        for i in axes_1:
            if i not in axes_2:
                print(axes[i-1], '= ', end = '')
                other_var_value[k] = float(input())
                other_var = other_var + list(axes[i-1])
                k = k + 1 
        
        def function(coordinate):
            '''Построение'''
            #x_1 = coordinate[0]
            #y_1 = coordinate[1]
            #z_1 = coordinate[2]
            #t_1 = coordinate[3]
            #E_sq = sympify(self.E_x)**2 + sympify(self.E_y)**2 + sympify(self.E_z)**2
            #H_sq = sympify(self.H_x)**2 + sympify(self.H_y)**2 + sympify(self.H_z)**2
            #expr = (E_sq + H_sq) / (4 * np.pi)
            #f = lambdify([x, y, z, t], expr, "numpy") 
            #return f(x_1, y_1, z_1, t_1)
            cmp_E = self.components_E_field(coordinate)
            cmp_H = self.components_H_field(coordinate)
            dens_nrj = (np.sum(cmp_E**2) + np.sum(cmp_H**2))  / (4 * np.pi)
            return dens_nrj

        absc, ordn = np.meshgrid(np.arange(-1, 1, 0.01), np.arange(-1, 1 , 0.01))
        coordinate = [0, 0, 0, 0]
        j = 0
        for i in range(0, 4):
            if(axes[i] == abscissa):
                coordinate[i] = absc
            elif(axes[i] == ordinate):
                coordinate[i] = ordn
            else:
                coordinate[i] = other_var_value[j] 
                j = j + 1
            
        #print(coord)
        z_grid = function(coordinate)
        z_grid = z_grid.astype(np.float64)

        plt.figure(figsize=(10,5))
        plt.imshow(z_grid, extent=[-1, 1, -1, 1], origin="lower", cmap = 'inferno')
        plt.title(f"Плотность энергии в плоскости ({abscissa},{ordinate}) c "
                  f"{other_var[0]} = {other_var_value[0]} и {other_var[1]} = {other_var_value[1]}")
        plt.xlabel(f"{abscissa}")
        plt.ylabel(f"{ordinate}")
        plt.colorbar()
        plt.show()

class Particle():
    def __init__(self, mass=0, charge=0, coordinate_initial = np.zeros(3), momentum_initial = np.zeros(3), field=None):
        self.mass = mass
        self.charge = charge
        self.coordinate_initial = coordinate_initial
        self.momentum_initial = momentum_initial
        self.field = field
    
    def sol_motion_eq(self,):
        '''
        def f(t_1, r):
            from sympy.abc import x, y, z, t
            X, Y, Z, P_x, P_y, P_z = r
            V_x = P_x / np.sqrt(1 + P_x**2 + P_y**2 + P_z**2)
            V_y = P_y / np.sqrt(1 + P_x**2 + P_y**2 + P_z**2)
            V_z = P_z / np.sqrt(1 + P_x**2 + P_y**2 + P_z**2)
            
            E_x = E_H_field.components_field(r[0], r[1], r[2], t_1, True, False)[0]
            E_y = E_H_field.components_field(r[0], r[1], r[2], t_1, True, False)[1]
            E_z = E_H_field.components_field(r[0], r[1], r[2], t_1, True, False)[2]
            H_x = E_H_field.components_field(r[0], r[1], r[2], t_1, False, True)[0]
            H_y = E_H_field.components_field(r[0], r[1], r[2], t_1, False, True)[1]
            H_z = E_H_field.components_field(r[0], r[1], r[2], t_1, False, True)[2]
            
            f_X = V_x
            f_Y = V_y
            f_Z = V_z
            if t_1 < T_off: 
                f_P_x = (E_x + V_y*H_z - V_z*H_y)
                f_P_y = (E_y + V_z*H_x - V_x*H_z) 
                f_P_z = (E_z + V_x*H_y - V_y*H_x)
            else: 
                f_P_x = (E_x + V_y*H_z - V_z*H_y) * np.exp(-t_1**2)
                f_P_y = (E_y + V_z*H_x - V_x*H_z) * np.exp(-t_1**2)
                f_P_z = (E_z + V_x*H_y - V_y*H_x) * np.exp(-t_1**2)
            return f_X, f_Y, f_Z, f_P_x, f_P_y, f_P_z
            '''
        def f(t_1, r):
            #from sympy.abc import x, y, z, t
            
            # a, b, c, p_x, p_y, p_z = r
            # print(len(r))
            #X, Y, Z, P_x, P_y, P_z = r
            #p = np.array([P_x, P_y, P_z])
            p = r[3:6]
            v = p / np.sqrt(1 + np.sum(np.dot(p, p)))
            ee = self.field.components_E_field(np.append(r[:3], t_1)).astype(np.float64)
            hh = self.field.components_H_field(np.append(r[:3], t_1)).astype(np.float64)
            
            f_r = v
            f_p = self.charge*(ee + np.cross(v, hh))
            f = np.hstack((f_r, f_p))
            return f

        sol = integrate.solve_ivp(f, t_span=(0, 10), t_eval=np.linspace(0, 10, 2000),
                                 y0=(self.coordinate_initial[0], self.coordinate_initial[1], self.coordinate_initial[2], 
                                     self.momentum_initial[0], self.momentum_initial[1], self.momentum_initial[2]))
        
        x, y, z, p_x, p_y, p_z = sol.y
        t = sol.t
        self.x = x
        self.y = y
        self.z = z
        self.t = t
        self.p_x = p_x
        self.p_y = p_y
        self.p_z = p_z
        
        return x, y, z, p_x, p_y, p_z, t
        
    def plot_trj_in_plane(self):
        #отбор осей, в которых будем строить
        print('Мы хотим построить траекторию движения заряженной частицы в некоторой плоскости.')
        check_list = ['x', 'y', 'z', 't']
        check_list_int_1 = [1, 2, 3, 4]
        check_list_int_2 = []
        chosen_var = []
      
        print('Введите абсциссу для графика: ', end = '')
        abscissa = input()
        print('Введите ординату для графика: ', end = '')
        ordinate = input()

        for i in check_list_int_1:
            if check_list[i-1] == abscissa:
                check_list_int_2 = check_list_int_2 + [i]
        
        for i in check_list_int_1:
            if check_list[i-1] == ordinate:
                check_list_int_2 = check_list_int_2 + [i]
        
        #Построение 
        to_chose = np.zeros((4, len(self.x)))
        to_chose[0] = self.x
        to_chose[1] = self.y
        to_chose[2] = self.z
        to_chose[3] = self.t
        
        i1 = int(check_list_int_2[0]-1)
        i2 = int(check_list_int_2[1]-1)
        absc = to_chose[i1]
        ordn = to_chose[i2]
        
        fig, ax = plt.subplots(figsize = (10, 8))
        ax.plot(absc, ordn, label = r'$Численное\;интегрирование$')
        if abscissa == 't':
            ax.set_xlabel(r"$t\;\;\;\left(в\;единицах\;\frac{\hbar}{m \cdot c^2}\right)$", size = 16)
            ax.set_ylabel(f"{ordinate}"+r'$\;\;\;\left(в\;единицах\;\frac{\hbar}{m \cdot c}\right)$', size = 16)
        elif ordinate == 't':
            ax.set_xlabel(f"{abscissa}"+r'$\;\;\;\left(в\;единицах\;\frac{\hbar}{m \cdot c}\right)$', size = 16)
            ax.set_ylabel(r"$t\;\;\;\left(в\;единицах\;\frac{\hbar}{m \cdot c^2}\right)$", size = 16)
        else:
            ax.set_xlabel(f"{abscissa}"+r'$\;\;\;\left(в\;единицах\;\frac{\hbar}{m \cdot c}\right)$', size = 16)
            ax.set_ylabel(f"{ordinate}"+r'$\;\;\;\;\left(в\;единицах\;\frac{\hbar}{m \cdot c}\right)$', size = 16)
        ax.set_title(fr'$Движение\;в\;({abscissa}, {ordinate})\;плоскости$', size  = 16)
        ax.legend(prop={'size': 15}, title = r'$Вид\;кривых$')
        ax.grid()
        plt.show()
        
    def plot_trj_in_3D(self):
        fig = plt.figure(figsize=(10, 8))
        ax = plt.axes(projection ='3d')
        ax.plot3D(self.x, self.y, self.z, color='purple')
        ax.set_title(r'$Движение\;частицы\;в\;3D$')
        ax.set_xlabel(r"$x\;\;\;\left(в\;единицах\;\frac{\hbar}{m \cdot c}\right)$", size  = 16)
        ax.set_ylabel(r"$y\;\;\;\left(в\;единицах\;\frac{\hbar}{m \cdot c}\right)$", size  = 16)
        ax.set_zlabel(r"$z\;\;\;\left(в\;единицах\;\frac{\hbar}{m \cdot c}\right)$", size  = 16)
        plt.show()
        
    def plot_mnt_and_nrj(self):
        fig, ax1 = plt.subplots(figsize = (10, 8))
        ax1.plot(self.t, self.p_x, label = r'$p_x(t)\;\;\;\left(в\;единицах\; m \cdot c\right)$', color = 'teal')
        ax1.plot(self.t, self.p_y, label = r'$p_y(t)\;\;\;\left(в\;единицах\; m \cdot c\right)$', color = 'purple')
        ax1.plot(self.t, self.p_z, label = r'$p_z(t)\;\;\;\left(в\;единицах\; m \cdot c\right)$', color = 'orange')
        ax1.set_xlabel(r"$t\;\;\;\left(в\;единицах\;\frac{\hbar}{m \cdot c^2}\right)$", size  = 16)
        ax1.set_ylabel(r'$p_{x,\:y,\:z}\;\;\;\left(в\;единицах\; m \cdot c\right)$', size  = 16)
        ax1.set_title(r'$p_{x,\:y,\:z}(t)\;зависимость$', size  = 16)
        ax1.legend(prop={'size': 15}, title = r'$Вид\;кривых$')
        ax1.grid()
        
        self.K = (self.p_x**2 + self.p_y**2 + self.p_z**2) / 2
        
        fig, ax2 = plt.subplots(figsize = (10, 8))
        ax2.plot(self.t, self.K, label = r'$E_{kin}(t)$', color = 'teal')
        ax2.set_xlabel(r"$t\;\;\;\left(в\;единицах\;\frac{\hbar}{m \cdot c^2}\right)$", size  = 16)
        ax2.set_ylabel(r"$E_{kin}\;\;\;\left(в\;единицах\;m \cdot c^2 \right)$", size  = 16)
        ax2.set_title(r'$E_{kin}(t) = \frac{p(t)^2}{2m}\;\,зависимость$', size  = 16)
        ax2.legend(prop={'size': 15}, title = r'$Вид\;кривых$')
        ax2.grid()
        
        plt.show()

#Скрещенное:
#E_H_field = ElectromagneticField(E_x='0.0001', E_y=0, E_z=0, H_x=0, H_y='10', H_z=0)
#1) круговая плоская поляризация
#E_H_field = ElectromagneticField(E_x=0, E_y='5*cos(2*(t-x))', E_z='5*sin(2*(t-x))', H_x=0, H_y='-10*sin(2*(t-x))', H_z='5*cos(2*(t-x))')
#2) линейная поляризация
E_H_field = ElectromagneticField(E_x=0, E_y='5*cos(2*(t-x))', E_z=0, H_x=0, H_y=0, H_z='10*cos(2*(t-x))')
#3) стоячая волна
#E_H_field = ElectromagneticField(E_x=0, E_y='10*cos(x)*cos(t)', E_z=0, H_x=0, H_y=0, H_z='10*sin(x)*sin(t)')
#Лазер:
#E_H_field = ElectromagneticField(E_x=0, E_y=0, E_z=f'5/(1+(2*z)**2)*exp(-(x**2+y**2)/(1+(2*z)**2))*cos(t)*'
#                                 f'cos((z + (x**2+y**2)/(2*z*(1+(1/(2*z))**2))) - (arctan(1/(2*z))))', H_x=0, H_y=0, H_z=0)
#E_H_field.plot_field_components_in_plane()
#E_H_field.plot_density_of_energy()


electron = Particle(mass=1, charge=1, coordinate_initial=np.array([0, 0, 0]), momentum_initial=np.array([0.1, 0.1, 0.1]), field = E_H_field)
print(electron.__dict__)
electron.sol_motion_eq()
electron.plot_trj_in_plane()
electron.plot_trj_in_3D()
electron.plot_mnt_and_nrj()