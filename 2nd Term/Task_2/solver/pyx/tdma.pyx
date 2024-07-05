import cython
cimport cython

import numpy as np
cimport numpy as np
from libc cimport abs

'''
[
    [b0 c0  0 0  0  0 ...],
    [a1 b1 c1 0  0  0 ...],
    [0  a2 b2 c2 0  0 ...],
    [0  0  a3 b3 c3 0 ...],
    ...
] * [x_i] = [d_i]
'''
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef double[:] solve(double[:] a,
    double[:] b,
    double[:] c,
    double[:] d):
    cdef:
        double[:] a_mv = a.copy()
        double[:] b_mv = b.copy()
        double[:] c_mv = c.copy()
        double[:] d_mv = d.copy()
        unsigned int n = len(a) - 1
        unsigned int i
    
    c_mv[0] /= b_mv[0]
    d_mv[0] /= b_mv[0]

    for i in range(1, n):
        c_mv[i] /= b_mv[i] - a_mv[i] * c_mv[i-1]
        d_mv[i] = (d_mv[i] - a_mv[i] * d_mv[i-1])/(b_mv[i] - a_mv[i]*c_mv[i-1])
    d_mv[n] = (d_mv[n] - a_mv[n]*d_mv[n-1]) / (b_mv[n] - a_mv[n]*c_mv[n-1])
    for i in range(n-1, -1, -1):
        d_mv[i] -= c_mv[i]*d_mv[i+1]
    return d_mv

'''
[
    [b0 c0  0 0  0  0 ...],
    [a1 b1 c1 0  0  0 ...],
    [0  a2 b2 c2 0  0 ...],
    [0  0  a3 b3 c3 0 ...],
    ...
] * [x_i] = [d_i]
'''
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef double[:] solve_mv(double[:] a_mv,
    double[:] b_mv,
    double[:] c_mv,
    double[:] d_mv,
    unsigned int n):
    cdef unsigned int i
    
    c_mv[0] /= b_mv[0]
    d_mv[0] /= b_mv[0]

    for i in range(1, n):
        c_mv[i] /= b_mv[i] - a_mv[i] * c_mv[i-1]
        d_mv[i] = (d_mv[i] - a_mv[i] * d_mv[i-1])/(b_mv[i] - a_mv[i]*c_mv[i-1])
    d_mv[n] = (d_mv[n] - a_mv[n]*d_mv[n-1]) / (b_mv[n] - a_mv[n]*c_mv[n-1])
    for i in range(n-1, -1, -1):
        d_mv[i] -= c_mv[i]*d_mv[i+1]
    return d_mv