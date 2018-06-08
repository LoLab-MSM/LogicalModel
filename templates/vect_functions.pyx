# distutils: language = c++
from __future__ import division
import numpy as np
cimport numpy as np
cimport cython
from libcpp.vector cimport vector

DTYPE = np.int
ctypedef np.int_t DTYPE_t
@cython.boundscheck(False)
@cython.wraparound(False)
def function(object[DTYPE_t, ndim=1, mode="c"] x):
    cdef int vectorsize = {{vectorsize}}
    cdef vector[int] vect
    cdef int i
{{vector}}
    for i in range(vectorsize):
        vect[i] = vect[i]%3
    return vect
