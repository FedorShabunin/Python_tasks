import cython
cimport cython

import numpy as np
cimport numpy as np
from libc cimport abs
from tdma import solve

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef double[:,:] nicolson(np.ndarray[np.float64_t, ndim=2] U0,
        double al):
        cdef:
            unsigned int i, j
            np.ndarray[np.float64_t, ndim=2] U = U0.copy()
            double[:,:] U_mv = U
            
            unsigned int n = U0.shape[0]
            unsigned int m = U0.shape[1]
            np.ndarray[np.float64_t, ndim=1] a0 = np.array([0.,] + [-1.,]*(n-3))
            np.ndarray[np.float64_t, ndim=1] b0 = np.array([2.0 + 2.0/al ,]*(n-2))
            np.ndarray[np.float64_t, ndim=1] c0 = np.array([-1.,]*(n-3) + [0.,])
            np.ndarray[np.float64_t, ndim=1] a
            np.ndarray[np.float64_t, ndim=1] b
            np.ndarray[np.float64_t, ndim=1] c
            np.ndarray[np.float64_t, ndim=1] d
            np.ndarray[np.float64_t, ndim=1] stp = np.zeros((n-2,))
            double[:] stp_mv = stp

        for i in range(1,m-1):
            a = a0.copy()
            b = b0.copy()
            c = c0.copy()
            stp = U[:, i-1]
            d = np.array([stp[j-1] + (2./al - 2.)* stp[j] + stp[j+1] for j in range(1, n-1)])
            d[0] += U[0, i]
            d[-1] += U[-1, i]
            stp = np.asarray(solve(a,b,c,d))
            U[1:-1, i] = stp[:]
        
        return U_mv