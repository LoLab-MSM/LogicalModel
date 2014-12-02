from __future__ import division
import numpy as np
cimport numpy as np
cimport cython

DTYPE = np.int
ctypedef np.int_t DTYPE_t
@cython.boundscheck(False)
def function(np.ndarray[DTYPE_t, ndim=1] x):
	cdef int vectorsize = x.shape[0]
	cdef np.ndarray[DTYPE_t, ndim=1]	vector = np.zeros([vectorsize],dtype=int)
	vector[0] = x[1]
	vector[1] = x[0]
	vector[2] = 2+2*x[0]+x[0]*x[0]*x[1]+x[0]*x[1]*x[1]+x[0]*x[0]*x[1]*x[1]
	for i in xrange(0,len(vector)):
		while vector[i] > 3:
			vector[i] = vector[i]%3
	return vector
