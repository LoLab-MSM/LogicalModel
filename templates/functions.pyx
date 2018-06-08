
from __future__ import division
import numpy as np
cimport numpy as np
cimport cython

DTYPE = np.int
ctypedef np.int_t DTYPE_t
@cython.boundscheck(False)
def changebase(number, numNodes, numStates):
    cdef int counter = -1
    cdef long Number = number
    cdef int NumNodes = numNodes
    cdef int NumStates = numStates
    cdef np.ndarray[DTYPE_t, ndim=1] state = np.zeros([numNodes],dtype=int)
    cdef long quotient = np.int(Number/NumStates)
    cdef int remainder = Number % NumStates
    Number = quotient
    state[counter] = remainder
    counter -=1
    while quotient !=0:
        quotient = np.int(Number/NumStates)
        remainder = Number % NumStates
        state[counter] = remainder
        counter -=1
        Number = quotient
    return state


@cython.boundscheck(False)
@cython.wraparound(False)
def function(object[DTYPE_t, ndim=1, mode="c"] x not None):
    cdef int vectorsize = {{vectorsize}}
    cdef int i
    cdef np.ndarray[DTYPE_t, ndim=1] vect = np.zeros([vectorsize], dtype=int)
{{vector}}
    for i in range(vectorsize):
        vect[i] = vect[i]%3
    return vect, tuple(vect)
