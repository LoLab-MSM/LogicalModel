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
	vector[0] = x[1]+2*x[1]*x[1]*x[2]+2*x[1]*x[2]*x[2]+2*x[1]*x[1]*x[2]*x[2]+2*x[1]*x[1]*x[3]+2*x[1]*x[2]*x[3]+x[1]*x[1]*x[2]*x[3]+2*x[1]*x[1]*x[2]*x[2]*x[3]+x[1]*x[2]*x[2]*x[2]*x[3]+2*x[1]*x[3]*x[3]+2*x[1]*x[1]*x[3]*x[3]+2*x[1]*x[1]*x[2]*x[3]*x[3]+x[1]*x[2]*x[2]*x[3]*x[3]
	vector[1] = x[1]+2*x[1]*x[1]+x[1]*x[1]*x[4]+x[4]*x[4]+2*x[1]*x[4]*x[4]+x[1]*x[5]+2*x[1]*x[1]*x[5]+2*x[1]*x[1]*x[4]*x[5]+2*x[1]*x[4]*x[4]*x[5]+x[5]*x[5]+x[1]*x[5]*x[5]+2*x[1]*x[1]*x[5]*x[5]+x[1]*x[1]*x[4]*x[5]*x[5]+2*x[4]*x[4]*x[5]*x[5]+2*x[1]*x[4]*x[4]*x[5]*x[5]+2*x[1]*x[1]*x[4]*x[4]*x[5]*x[5]
	vector[2] = 1+x[2]*x[2]+2*x[4]+x[2]*x[4]+2*x[2]*x[2]*x[4]+x[4]*x[4]+2*x[2]*x[4]*x[4]+2*x[5]+x[2]*x[5]+2*x[2]*x[2]*x[5]+x[4]*x[5]+2*x[2]*x[4]*x[5]+x[2]*x[2]*x[4]*x[5]+2*x[4]*x[4]*x[5]+x[2]*x[4]*x[4]*x[5]+2*x[2]*x[2]*x[4]*x[4]*x[5]+x[5]*x[5]+2*x[2]*x[5]*x[5]+2*x[4]*x[5]*x[5]+x[2]*x[4]*x[5]*x[5]+2*x[2]*x[2]*x[4]*x[5]*x[5]+x[4]*x[4]*x[5]*x[5]+2*x[2]*x[4]*x[4]*x[5]*x[5]+2*x[2]*x[2]*x[4]*x[4]*x[5]*x[5]
	vector[3] = 1+x[3]*x[3]+2*x[4]+x[3]*x[4]+2*x[3]*x[3]*x[4]+x[4]*x[4]+2*x[3]*x[4]*x[4]+2*x[5]+x[3]*x[5]+2*x[3]*x[3]*x[5]+x[4]*x[5]+2*x[3]*x[4]*x[5]+x[3]*x[3]*x[4]*x[5]+2*x[4]*x[4]*x[5]+x[3]*x[4]*x[4]*x[5]+2*x[3]*x[3]*x[4]*x[4]*x[5]+x[5]*x[5]+2*x[3]*x[5]*x[5]+2*x[4]*x[5]*x[5]+x[3]*x[4]*x[5]*x[5]+2*x[3]*x[3]*x[4]*x[5]*x[5]+x[4]*x[4]*x[5]*x[5]+2*x[3]*x[4]*x[4]*x[5]*x[5]+2*x[3]*x[3]*x[4]*x[4]*x[5]*x[5]
	vector[4] = 1+2*x[0]+x[0]*x[0]+x[0]*x[4]+2*x[0]*x[0]*x[4]+x[4]*x[4]+2*x[0]*x[4]*x[4]
	vector[5] = 1+2*x[0]+x[0]*x[0]+x[0]*x[5]+2*x[0]*x[0]*x[5]+x[5]*x[5]+2*x[0]*x[5]*x[5]
	for i in xrange(0,len(vector)):
		while vector[i] > 3:
			vector[i] = vector[i]%3
	return vector
