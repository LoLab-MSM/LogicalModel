# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 16:55:23 2014

@author: James C Pino, Leonard A Harris

"""
import numpy as np
import time
import argparse
from importlib import import_module
import sys
import os.path
import re
import os
import warnings
import pycuda.gpuarray as gpuarray
import pycuda.autoinit

#p = argparse.ArgumentParser()
#p.add_argument("-n","--nstates",  type=str, help='provide a number of states')
#p.add_argument("-s","--start",    type=str, help='starting string to convert to base Nstates')
#p.add_argument("-e","--end",      type=str, help='ending string to convert to based Nstates')
#p.add_argument("-m","--model",    type=str, help='model to run simulation, assumes file to end in .txt')
#p.add_argument("-v","--verbose",  type=str, help='if you want verbose updates (use with single processor)')
#p.add_argument("-p","--parallel", type=str, help='run in parallel, use 0 or 1')
#args = p.parse_args()

#numStates=int(args.nstates)
numStates = 3
start = 0
end = 3**3
numNodes = 3

def get_num_nodes(model_file):

    global numNodes

    f = open(model_file,'r')
    functions = map(lambda s: s.strip(), f)
    numNodes = len(functions)
    f.close()
def compile_cython_code(model_file, overwrite=False):

    global function
    global numNodes

    f = open(model_file,'r')
    functions = map(lambda s: s.strip(), f)
    numNodes = len(functions)
    dir,file = os.path.split(model_file)
    prefix = file.split('.')[0]

    outstring = \
    'from __future__ import division\
    \nimport numpy as np\
    \ncimport numpy as np\
    \ncimport cython\
    \n\nDTYPE = np.int\
    \nctypedef np.int_t DTYPE_t\
    \n@cython.boundscheck(False)\
    \ndef function(np.ndarray[DTYPE_t, ndim=1] x):\
    \n\tcdef int vectorsize = x.shape[0]\
    \n\tcdef np.ndarray[DTYPE_t, ndim=1] vector = np.zeros([vectorsize],dtype=int)\n'
    for line in functions:
        # Replace function name
        line = re.sub('f\d+', 'v%d' % (int(re.match('f(\d+)', line).group(1))-1), line)
        # Convert powers (^) to simple multiplications
        matches = re.findall('(x\d+)\^(\d+)', line)
        for m in matches:
            repl = m[0]
            for i in range(1,int(m[1])):
                repl += "*%s" % m[0]
            line = re.sub('x\d+\^\d+', repl, line, count=1)
        # Replace node names
        matches = np.array(re.findall('x(\d+)', line), dtype=int)
        for m in matches:
            line = re.sub('x\d+', 'x%d' % (m-1), line, count=1)
        outstring += '\t%s\n' % line
        print line
    for line in functions:
        print 'gpuarray.to_gpu(x%s)' % str(int(re.match('f(\d+)', line).group(1))-1)
    outstring += '\tfor i in xrange(0,len(vector)):\n\t\twhile vector[i] > %d:\n\t\t\tvector[i] = vector[i]%%%d\n\treturn vector\n' % (numStates, numStates)

compile_cython_code("/home/pinojc/Projects/LogicalModel/Models/func-example.txt")
Model = "/home/pinojc/Projects/LogicalModel/Models/func_example.txt"
def compile_cython_code2(model_file, overwrite=False):

    global function
    global numNodes

    f = open(model_file,'r')
    functions = map(lambda s: s.strip(), f)
    numNodes = len(functions)
    dir,file = os.path.split(model_file)
    prefix = file.split('.')[0]

    # Warn the user if .so file already exists
    if os.path.exists(prefix+'.so'):
        warning_string = 'Shared object file \'%s\' already exists: ' % (prefix+'.so')
        if overwrite:
            warnings.warn(warning_string + 'overwriting file.')
            os.remove(prefix+'.so')
        else:
            warnings.warn(warning_string + 'moving on.')
            function = import_module(prefix).function
            return

    pyxfile = open(prefix+'.pyx','w')

    outstring = \
    'from __future__ import division\
    \nimport numpy as np\
    \ncimport numpy as np\
    \ncimport cython\
    \n\nDTYPE = np.int\
    \nctypedef np.int_t DTYPE_t\
    \n@cython.boundscheck(False)\
    \ndef function(np.ndarray[DTYPE_t, ndim=1] x):\
    \n\tcdef int vectorsize = x.shape[0]\
    \n\tcdef np.ndarray[DTYPE_t, ndim=1] vector = np.zeros([vectorsize],dtype=int)\n'
    for line in functions:
        # Replace function name
        line = re.sub('f\d+', 'vector[%d]' % (int(re.match('f(\d+)', line).group(1))-1), line)
        # Convert powers (^) to simple multiplications
        matches = re.findall('(x\d+)\^(\d+)', line)
        for m in matches:
            repl = m[0]
            for i in range(1,int(m[1])):
                repl += "*%s" % m[0]
            line = re.sub('x\d+\^\d+', repl, line, count=1)
        # Replace node names
        matches = np.array(re.findall('x(\d+)', line), dtype=int)
        for m in matches:
            line = re.sub('x\d+', 'x[%d]' % (m-1), line, count=1)
        outstring += '\t%s\n' % line
    outstring += '\tfor i in xrange(0,len(vector)):\n\t\twhile vector[i] > %d:\n\t\t\tvector[i] = vector[i]%%%d\n\treturn vector\n' % (numStates, numStates)
    pyxfile.write(outstring)
    pyxfile.close()
    #os.system('sleep 2s')
    setup = open('setup.py','w')
    outstring = \
    "from distutils.core import setup\
    \nimport numpy\
    \nfrom Cython.Build import cythonize\
    \nsetup(\
    \n    ext_modules = cythonize('"+prefix+".pyx',),\
          include_dirs = [numpy.get_include()])"
    setup.write(outstring)
    setup.close()
    os.system('python setup.py build_ext --inplace')
    sys.path.append(dir)
    function = import_module(prefix).function
def changebase(number):
    counter = -1
    state = np.zeros(numNodes,dtype=int)
    quotient = number/numStates
    remainder = int(number) % int(numStates)
    number = quotient
    state[counter] = remainder
    counter -=1
    while quotient !=0:
        quotient = int(number)/int(numStates)
        remainder = int(number) % int(numStates)
        state[counter] = remainder
        counter -=1
        number = quotient
    return state

def getNextState(state):
    nextstate = function(state)
    return nextstate

def checkStates(x):
    if ((x[:-1,:] == x[-1,:]).sum(axis=1)==numNodes).any() ==1:
        i = np.where((x[:-1,:] != x[-1,:]).sum(axis=1) == 0)
        return i[0][0]
    else:
        return -1
def function(x0,x1,x2):
    x0_gpu = gpuarray.to_gpu(x0.astype(np.float32))
    x1_gpu = gpuarray.to_gpu(x1.astype(np.float32))
    x2_gpu = gpuarray.to_gpu(x2.astype(np.float32))
    v0 = x1_gpu.get()
    v1 = x0_gpu.get()
    v2 = (2+2*x1_gpu+x1_gpu*x1_gpu*x2_gpu+x1_gpu*x2_gpu*x2_gpu+x1_gpu*x1_gpu*x2_gpu*x2_gpu).get()
    return v0,v1,v2


import itertools
x = np.empty((3**3,3), dtype=int)
counter = 0
print list(itertools.product([0,1,2], repeat=3))
for comb in list(itertools.product([0,1,2], repeat=3)):
    x[counter,0] = comb[0]
    x[counter,1] = comb[1]
    x[counter,2] = comb[2]
    counter+=1
#print x

for i in range(10):
    x1,x2,x3= x[:,0],x[:,1],x[:,2]
    x1t,x2t,x3t= function(x1,x2,x3)
    x1t=x1t%3
    x2t=x2t%3
    x3t=x3t%3
    xt = np.column_stack((x1t,x2t,x3t))
    if ((xt[:,:] == x[:,:]).sum(axis=1)==3).any() ==1:
        j = np.where((xt[:,:] == x[:,:]).sum(axis=1) == 0)
        for i in j:
            print xt[j]
    x = xt
    #print x1,x2,x3
print np.shape(x1)
def run(x):
    counter = 1
    blank[0:2,:] = x
    while checkStates(blank[0:counter,:])==-1:
        blank[counter+1,:]=getNextState(blank[counter,:])
        counter += 1
    xx = blank[checkStates(blank[:counter,:]):counter]
    ncols = xx.shape[1]
    dtype = xx.dtype.descr * ncols
    struct = xx.view(dtype)
    uniq = np.unique(struct)
    uniq = uniq.view(xx.dtype).reshape(-1, ncols)
    uniq[uniq[:,1].argsort()]
    uniq = uniq[0]
    data = ''
    for u in uniq:
        data+=str(u)
    return data

def main():
    print 'Started '
    start_time = time.time()
    data = dict()
    compile_cython_code2(Model, overwrite=False)
    x = np.zeros((2,numNodes),dtype=int)
    for i in xrange(start,end):
        x[0,:]=changebase(i)
        x[1,:]=getNextState(x[0,:])
        tmp=run(x)
        if tmp in data:
            data[tmp]+=1
        else:
            data[tmp]=1
    print 'Computed %s samples %.4f minutes' %(str(samplesize),(time.time() - start_time)/60)
    print 'Attractors ',data.keys()
    print 'Frequencies ',data.values()
    print 'Total ',np.sum(data.values())
blank = np.empty((numStates**9,numNodes), dtype=int)
samplesize = end - start
main()    

#Model = str(args.model)
#get_num_nodes(Model)
#blank = np.empty((numStates**9,numNodes), dtype=int)
#x_gpu1 = np.zeros()



"""
z = np.linspace(0,1000,10000000)
x2 = np.asarray(z.astype(np.float32))
x1 = np.asarray(z.astype(np.float32))

x1_gpu = gpuarray.to_gpu(x1)
x2_gpu = gpuarray.to_gpu(x2)
two = gpuarray.to_gpu(np.array([2]*len(z)).reshape(np.shape(x2)).astype(np.float32))
tw = np.array([2]*len(z))
print tw
import time
startT = time.time()
for i in range(10):
    f1 = x1_gpu.get()
    f2 = x2_gpu.get()
    f3 = (two+two*x1_gpu + x1_gpu*x2_gpu +x1_gpu*x2_gpu+x1_gpu*x1_gpu+x2_gpu+x2_gpu).get()
    x1_gpu = gpuarray.to_gpu(f1)
    x2_gpu = gpuarray.to_gpu(f2)
print 'time',time.time() - startT
#print f3
startT=time.time()
for i in range(10):
    f1 = x1
    f2 = x2
    f3 = tw+tw*x1 + x1*x2 +x1*x2+x1*x1+x2+x2

print 'time', time.time() - startT

"""

