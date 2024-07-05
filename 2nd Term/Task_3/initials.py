import numpy as np
from typing import Iterable, Callable, Union

class Potential:
    @staticmethod
    def rect_well(in_bound:float, out_bound:float, magnitude:float = -1.0):
        def v(x):
            return magnitude if x > in_bound and x < out_bound else 0
        return v

    @staticmethod
    def double_rect_well(in_bound:float, out_bound:float, magnitude:float = -1.0):
        def v(x):
            return magnitude if abs(x) > in_bound and abs(x) < out_bound else 0 if abs(x) < in_bound else 0.0
        return v
    
    @staticmethod
    def cosh_sq_well(a, b, magnitude:float = -1.0):
        def v(x):
            return magnitude/(np.cosh(((x**2-a**2)/(b*2)))**2)
        return v
    
    @staticmethod
    def cosh_sq_well1(a, b, magnitude:float = -1.0):
        def v(x):
            return magnitude/np.cosh((x-(b+a/2.0))/b)**2 + magnitude/np.cosh((x+(b+a/2.0))/b)**2 
        return v
    
    @staticmethod
    def finite_step(start, height):
        def v(x):
            return 0 if x < start else height
        return v
    
class InitialWave:
    @staticmethod
    def gauss_package(x0:float, sigma:float, p0:float):
        def Psi(x):
            return np.exp(-0.5*((x - x0)/(sigma/3.0))**2 + 1j*p0*x) * 1/np.sqrt(2.0*np.pi*sigma/3.0)
        return Psi
    
def mapped(x: Iterable[Union[float, np.float64, np.complex128]], f: Callable[[float], np.ndarray], dt=np.complex128):
    return np.array(list(map(f, x)), dtype=dt)