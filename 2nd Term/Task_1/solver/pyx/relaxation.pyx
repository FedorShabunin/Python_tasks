import cython
cimport cython

import numpy as np
cimport numpy as np
from libc cimport abs

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef max(double a, double b):
    return a if a > b else b

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)

cpdef relaxation(np.ndarray[np.float64_t, ndim=2] U1,
                 np.ndarray[np.uint8_t, ndim = 2, cast=True] BM,
                 double rtol, double h, double omega):
    cdef int i, j, it
    cdef double rel_res, dU_max, dU, R
    cpdef np.ndarray U0 = U1.copy()

    cdef unsigned int rows = U1.shape[0]
    cdef unsigned int cols = U1.shape[1]
    cdef double [:,:] U = U1
    cdef char [:,:] BM_mv = BM
    cdef double [:,:] U0_mv = U0

    
    it=0
    rel_res=1.0
        
    itmax=10000
    while ((rel_res>rtol) and (it<=itmax)):
        dU_max=0.0
        for j in range(1,rows-2):
                for i in range(1,cols-2):
                    R = 0.25*(U[j,i-1] + U[j-1,i] + U[j,i+1] + U[j+1,i]) - U[j,i] 
                    dU =  omega*R
                    U[j,i] += dU
                    if BM_mv[j,i]:
                      U[j,i] = U0_mv[j,i]
                    dU_max=max(abs(dU),dU_max)
                   
        rel_res=dU_max/np.max(np.abs(U[:,:]))
#         print 'rel_res', rel_res
        it+=1 
    return U, rel_res, it