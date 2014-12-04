# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 16:55:23 2014

@author: James C Pino ,  Leonard Harris, Carlos F. Lopez

"""


import numpy as np
import time
import argparse
from importlib import import_module
import sys
import os.path
import re
import os
parser = argparse.ArgumentParser()
parser.add_argument("-n","--numberstates", type=str, help="provide a number of states")
parser.add_argument("-s","--start",type=str,help='starting string to convert to base Nstates')
parser.add_argument("-e","--end",type=str,help='ending string to convert to based Nstates')
parser.add_argument("-m","--model",type=str,help='model to run simulation, assumes file to end in .txt')
parser.add_argument("-v","--verbose",type=str,help='if you want verbose updates (use with single processor)')
parser.add_argument("-p","--parallel",type=str,help='run in parallel, use 0 or 1')
args = parser.parse_args()



def compile_cython_code(Model):
    f = open(Model,'r')
    f = map(lambda s: s.strip(), f)
    global  numNodes
    numNodes = len(f)
    if os.path.exists('model_code.pyx'):
        print 'Cython code already exists. Ensure that you are running the correct model.'
        return
    f_new = open('model_code.pyx','w')
    modules = 'from __future__ import division\
    \nimport numpy as np\
    \ncimport numpy as np\
    \ncimport cython'
    print>>f_new,modules
    header = '\nDTYPE = np.int\
    \nctypedef np.int_t DTYPE_t\
    \n@cython.boundscheck(False)\
    \ndef function(np.ndarray[DTYPE_t, ndim=1] x):\
    \n\tcdef int vectorsize = x.shape[0]\
    \n\tcdef np.ndarray[DTYPE_t, ndim=1]\
    \tvector = np.zeros([vectorsize],dtype=int)'
    print>>f_new,header
    count=0
    for line in f:
        count+=1
        for i in reversed(range(0,22)):
            exp = 'f'+str(i)
            repl = 'vector[%d]'%(int(i)-1)
            line = re.sub(exp,repl,line)
            exp1 = 'x'+str(i)
            repl1 = 'x[%d]'%(int(i)-1)
            line = re.sub(exp1,repl1,line)
            exp2 = 'x\['+str(i)+'\]\^2'
            X = 'x[%d]*x[%d]' % (int(i) ,int(i))
            line = re.sub(exp2,X,line)
            exp3 = 'x\['+str(i)+'\]\^3'
            X = 'x[%d]*x[%d]*x[%d]' % (int(i) ,int(i),int(i))
            line = re.sub(exp3,X,line)
        print>>f_new,'\t',line
    print>>f_new,'\tfor i in xrange(0,len(vector)):\n\t\twhile vector[i] > 3:\n\t\t\tvector[i] = vector[i]%3\n\treturn vector'
    f_new.close()
    os.system('sleep 2s')
    setup = open('setup.py','w')
    print>>setup,"from distutils.core import setup\
    \nimport numpy\
    \nfrom Cython.Build import cythonize\
    \nsetup(\
    \n    ext_modules = cythonize('model_code.pyx',),\
          include_dirs = [numpy.get_include()])"
    setup.close()
    os.system('python setup.py build_ext --inplace')



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
    x = np.zeros((2,numNodes),dtype=int)
    #for i in xrange(samplesize):
    for i in xrange(start,end):
        if v == 1:
            print str(i+1),'/',samplesize
        x[0,:]=changebase(i)
        x[1,:]=getNextState(x[0,:])
        tmp=run(x)
        if tmp in data:
            data[tmp]+=1
        else:
            data[tmp]=1
    print 'time of '+str(samplesize)+' calculations '+ str((time.time() - start_time)/60)+' minutes'
    print 'Attractors ',data.keys()
    print 'Frequencies ',data.values()
    print np.sum(data.values())

Model = str(args.model)
compile_cython_code(Model)
numStates=int(args.numberstates)
start = int(args.start)
if args.end != None:
    end = int(args.end)
else:
    end = numStates**numNodes
if args.start != None:
    start = int(args.start)
else:
    start = 0
parallel = int(args.parallel)
v = int(args.verbose)
directory, file = os.path.split('model_code')
sys.path.append(directory)
Function = import_module(file)
global function
function = Function.function
global blank
blank = np.empty((numStates**9,numNodes), dtype=int)
samplesize = end - start




if parallel == False:
    print "Running on single CPU"
    main()
if parallel == True:
    import pypar
    # Must have pypar installed, uses a "stepping" of 100, which means splits up
    # the job in batches of 100 over the processors
    directory, file = os.path.split('model_code')
    sys.path.append(directory)
    Function = import_module(file)
    function = Function.function
    blank = np.empty((numStates**9,numNodes), dtype=int)
    #Initialise
    t = pypar.time()
    P = pypar.size()
    p = pypar.rank()
    processor_name = pypar.get_processor_name()
    # Block stepping
    stepping = 10

    B = samplesize/stepping # Number of blocks
    print 'B=',B
    print 'Processor %d initialised on node %s' % (p, processor_name)
    assert P > 1, 'Must have at least one slave'
    assert B > P - 1, 'Must have more work packets than slaves'



    if p == 0:

        print 'samplesize = ',samplesize
        print 'split up into %s segements' % str(1.*B)
        #Create array for storage
        Results = dict()
        # Create work pool (B blocks)
        workpool = []
        for i in range(start, end, stepping):
            workpool.append(i)
        # Distribute initial work to slaves
        w = 0
        for d in range(1, P):
            pypar.send(workpool[w], destination=d)
            w += 1
        # Receive computed work and distribute more
        terminated = 0
        while(terminated < P - 1):
            data,status= pypar.receive(pypar.any_source,return_status=True)
            for tmp in data[1]:     # check to see if new states are already present
                if tmp in Results:
                    Results[tmp]+=1
                else:
                    Results[tmp]=1
            d = status.source  # Id of slave that just finished
            if w < len(workpool):
                # Send new work to slave d
                pypar.send(workpool[w], destination=d)
                w += 1
            else:
                # Tell slave d to terminate
                pypar.send(None, destination=d)
                terminated += 1
        print 'Computed '+str(samplesize)+' samples in %.2f seconds' % (pypar.time() - t)

    else:
        while(True):
            # Receive work (or None)
            W = pypar.receive(source=0)
            #print W
            if W is None:
                #print 'Slave p%d finished: time = %.2f ' % (p, pypar.time() - t)
                break
            # Compute allocated work
            data = []
            for i in xrange(0,stepping):
                if v == 1:
                    print str(W+i),'/',end
                if  W+i > end:
                    #print 'Finished with p%d'% (p)
                    break
                else:
                    x = changebase(W+i)
                    x = np.vstack((x,getNextState(x)))
                    tmp = run(x)
                    data.append(tmp)

            # Return result
            pypar.send((W,data), destination=0)
    pypar.finalize()
    if p == 0:
        print Results.keys()
        print Results.values()
        print np.sum(Results.values())


