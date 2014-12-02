# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 13:52:14 2014

@author: James C Pino ,  Leonard Harris, Carlos F. Lopez
"""
import pylab as plt
import numpy as np
import re
import sys

f = open(sys.argv[1],'r')
f = t = map(lambda s: s.strip(), f)
f_new = open(sys.argv[2],'w')
modules = 'from __future__ import division\nimport numpy as np\ncimport numpy as np\ncimport cython'

print>>f_new,modules
header = '\nDTYPE = np.int\nctypedef np.int_t DTYPE_t\n@cython.boundscheck(False)\ndef function(np.ndarray[DTYPE_t, ndim=1] x):\n\tcdef int vectorsize = x.shape[0]\n\tcdef np.ndarray[DTYPE_t, ndim=1]\tvector = np.zeros([vectorsize],dtype=int)'
print>>f_new,header
count=0
for line in f:
    count+=1
    #print line
    #line = re.sub('f','vector',line)
    for i in reversed(range(0,22)):
        exp = 'f'+str(i)
        repl = 'vector[%d]'%(int(i)-1)
        line = re.sub(exp,repl,line)
        exp1 = 'x'+str(i)
        repl1 = 'x[%d]'%(int(i)-1)
        #print exp1,repl1
        line = re.sub(exp1,repl1,line)
        exp2 = 'x\['+str(i)+'\]\^2'
        X = 'x[%d]*x[%d]' % (int(i) ,int(i))
        line = re.sub(exp2,X,line)
        exp3 = 'x\['+str(i)+'\]\^3'
        X = 'x[%d]*x[%d]*x[%d]' % (int(i) ,int(i),int(i))
        line = re.sub(exp3,X,line)
    print>>f_new,'\t',line


    #print line[:-1]
print>>f_new,'\tfor i in xrange(0,len(vector)):\n\t\twhile vector[i] > 3:\n\t\t\tvector[i] = vector[i]%3\n\treturn vector'
f_new.close()
print " Ready to compile with cython.\
\n Use command cython filename.pyx to convery to c code\
\n Then compile with c code with gcc, example if filename = ironFunct2 \
\n gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I/usr/include/python2.7 -o ironFunct2.so ironFunct2.c"




