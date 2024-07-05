import cython
cimport cython
import numpy as np
cimport numpy as np
from cython.parallel cimport prange
from libc.string cimport memset
from libc.stdlib cimport malloc, free
from libc.math cimport exp
from libc.stdlib cimport rand, RAND_MAX

cpdef double gen_pure_random():
    return (<double>rand())/(<double>RAND_MAX)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef class Domain:
  cdef:
    char [:] spins_mv
    long N
    double J
    double B
    double T
  def __init__(self, long N, char cold_start = 0, double T = 1.0, double J = 1.0, double B = 0.0):
    cdef np.ndarray[np.int8_t, ndim=1, negative_indices=False, mode='c', cast=True] spins
    self.N = N
    self.J = J
    self.B = B
    self.T = T
    if cold_start == -1:
      spins = np.ones((N,), dtype=np.byte)*(-1)
    elif (cold_start):
      spins = np.ones((N,), dtype=np.byte)
    else:
      spins = np.random.randint(0, 2, (N,), dtype=np.byte) * 2 - np.ones((N,), dtype=np.byte)
    self.spins_mv = spins

  def get_spins(self):
    return np.asarray(self.spins_mv)

  cpdef int run_metro(self, double r_ = -1.0, int rx_ = -1):
    cdef double E0
    cdef double E1
    cdef double R
    cdef double r
    E0 = self.E()
    if rx_ == -1:
      tr_idx = int(gen_pure_random() * self.N)
    else:
      tr_idx = rx_
#     tr_idx = np.random.randint(0, self.N)
    self.spins_mv[tr_idx] *= -1
    E1 = self.E()
    if E1 < E0:
      return E1<E0
    else:
      R = exp(-(E1 - E0)/self.T)
      if r_ == -1.0:
        r = gen_pure_random()
      else:
        r = r_
      if R < r:
        self.spins_mv[tr_idx] *= -1
      return R>r

  cpdef run_long_metro(self, long steps, double ensemble_rate = 0.9, char full_log = 0):
    cdef np.ndarray[np.int8_t, ndim=2, negative_indices=False, mode='c'] spin_log
    cdef np.ndarray[np.float64_t, ndim=1, negative_indices=False, mode='c'] elog
    cdef np.ndarray[np.float64_t, ndim=1, negative_indices=False, mode='c'] mlog
    cdef np.ndarray[np.float64_t, ndim=1, negative_indices=False, mode='c'] randoms_doubles
    cdef np.ndarray[np.int64_t, ndim=1, negative_indices=False, mode='c'] randoms_ints
    cdef double[:] randoms_d_mv
    cdef long[:] randoms_i_mv
    cdef char[:,:] spin_log_mv
    cdef double[:] elog_mv
    cdef double[:] mlog_mv
    cdef long i
    cdef pre_step
    cdef long j
    cdef double E2_ = 0
    cdef double E_ = 0
    cdef double M_ = 0
    cdef double C
    cdef double U
    cdef double M
    
    pre_steps = long((<double>steps) * ensemble_rate)
    randoms_doubles = np.random.random((steps,))
    randoms_ints = np.random.randint(0, self.N, (steps,))
    randoms_d_mv = randoms_doubles
    randoms_i_mv = randoms_ints

    spin_log = np.zeros((self.N, steps), dtype=np.byte)
    elog = np.zeros((steps,))
    mlog = np.zeros((steps,))
    spin_log_mv = spin_log
    elog_mv = elog
    mlog_mv = mlog
      
    for i in range(steps):
      spin_log_mv[:, i] = self.spins_mv[:]
      elog_mv[i] = self.E()
      mlog_mv[i] = self.M()
      self.run_metro(randoms_d_mv[i], randoms_i_mv[i])
    
    for i in range(pre_steps, steps):
      E2_ += elog_mv[i]*elog_mv[i]
      E_ += elog_mv[i]
      M_ += mlog_mv[i]
    E2_ /= (<double>(steps - pre_steps))
    E_ /= (<double>(steps - pre_steps))
    M_ /= (<double>(steps - pre_steps))
    
    M = M_
    U = E_
    C = 1.0/(<double>self.N) * (E2_ - E_*E_)/(self.T*self.T)
    
    if full_log:
        return spin_log, elog, mlog, C,U,M
    else:
        return spin_log[:, pre_steps:steps], elog[pre_steps:steps], mlog[pre_steps:steps], C,U,M

    
  
  cpdef double E(self):
    cdef double energy = 0
    for i in range(len(self.spins_mv)-1):
      energy += self.spins_mv[i] * self.spins_mv[i+1]
    energy *= self.J
    energy += -self.B*np.sum(self.spins_mv.base)
    return energy

  cpdef double M(self):
    return np.sum(self.spins_mv.base)/self.N

  def get_E(self):
    return self.E()