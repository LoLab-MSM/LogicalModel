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
from changeBase import changebase
from changeBase import run
import multiprocessing as mp
p = argparse.ArgumentParser()
p.add_argument("-n","--nstates",  type=str, help='provide a number of states')
p.add_argument("-s","--start",    type=str, help='starting string to convert to base Nstates')
p.add_argument("-e","--end",      type=str, help='ending string to convert to based Nstates')
p.add_argument("-m","--model",    type=str, help='model to run simulation, assumes file to end in .txt')
p.add_argument("-v","--verbose",  type=str, help='if you want verbose updates (use with single processor)')
p.add_argument("-p","--parallel", type=str, help='run in parallel, use 0 or 1')
args = p.parse_args()

#numStates=int(args.nstates)
numStates=3
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

    # Warn the user if .so file already exists
    if os.path.exists(prefix+'.so'):
        warning_string = 'Shared object file \'%s\' already exists: ' % (prefix+'.so')
        if overwrite:
            #warnings.warn(warning_string + 'overwriting file.')
            os.remove(prefix+'.so')
        else:
            #warnings.warn(warning_string + 'moving on.')
            function = import_module(prefix).function
            return

    pyxfile = open(prefix+'.pyx','w')

    outstring = \
    '# cython: profile=True\
    \nfrom __future__ import division\
    \nimport numpy as np\
    \ncimport numpy as np\
    \ncimport cython\
    \n\nDTYPE = np.int\
    \nctypedef np.int_t DTYPE_t\
    \n@cython.boundscheck(False)\
    \n@cython.wraparound(False)\
    \ndef function(object[DTYPE_t, ndim=1, mode="c"] x not None):\
    \n\tcdef unsigned int vectorsize = %d\
    \n\tcdef np.ndarray[DTYPE_t, ndim=1] vector = np.zeros([vectorsize],dtype=int)\n'%numNodes
    #for i in range(numNodes):
    #    outstring+='\tcdef unsigned int  X%s = x[%s]\n'%(str(i),str(i))
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
            line = re.sub('x\d+', 'x[<unsigned int>%d]' % (m-1), line, count=1)
        outstring += '\t%s\n' % line
    #outstring += '\tvector = np.fmod(vector,%d)\n\treturn vector\n' %  numStates
    outstring += '\tfor i in xrange(%d):\n\t\tvector[i] = vector[i]%%%d\n\treturn vector,tuple(vector)\n' %  (numNodes,numStates)
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

import pygraphviz as pyg

def return_attractor(point,G):
    
    path = set()
    tmp,tmp2 = function(np.array(point))
    x = str(tmp2).replace(',','').replace('(','').replace(')','').replace(' ','')
    print x,
    while not tmp2 in path:
        path.add(tmp2)
        tmp,tmp2 = function(tmp)
        y = str(tmp2).replace(',','').replace('(','').replace(')','').replace(' ','')
        #G.add_edge(x,y)
        x = y
        print '->',y,
    
    print 
    return path.pop()
#@profile
def run(i):
    
    x = changebase(i,numNodes,numStates)
    loc = tuple(x)
    path = set()
    path.add(loc)
    tmp,tmp2 = function(x)
    while not tmp2 in path:
        path.add(tmp2)
        tmp,tmp2 = function(tmp)

    path = set()    
    path.add(tmp2)
    tmp,tmp2 = function(tmp)
    while not tmp2 in path:
        path.add(tmp2)
        tmp,tmp2 = function(tmp)
    return path.pop()

import multiprocessing as mp
#@profile
def main():
    #print 'Started '
    start_time = time.time()
    data = dict()
    compile_cython_code(Model, overwrite=True)
    pool = mp.Pool()
    a = xrange(start,end)
    b = pool.map(run,a,100)
    pool.close()
    pool.join()
    
    for i in b:
        try:
            data[i] += 1
        except:
            data[i] = 1
    endT = time.time()
    print 'start = %s, end = %s'%(start,end)
    print 'Computed %s samples %.4f minutes' %(str(samplesize),(endT - start_time)/60)
    #print 'Attractors ',data.keys()
    print 'Frequencies ',data.values()
    print 'Total ',np.sum(data.values())
#@profile    
def main1():
    #print 'Started '
    start_time = time.time()
    data = dict()
    compile_cython_code(Model, overwrite=False)
    for i in xrange(start,end):
        tmp = run(i)
        try:
            data[tmp] += 1
        except:
            data[tmp] = 1
    endT = time.time()
    print 'start = %s, end = %s'%(start,end)
    print 'Computed %s samples %.4f minutes' %(str(samplesize),(endT - start_time)/60)
    G = pyg.AGraph(directed=True)
    for i in data.keys():
        return_attractor(i,G)
    #G.draw('%s.pdf'%str('attractors3'),prog='dot')
    print 'Attractors ',data.keys()
    print 'Frequencies ',data.values()
    print 'Total ',np.sum(data.values())    
if args.model == None:
    #Model = 'Models/func_example.txt'
    #Model = 'Models/core_iron_6variables_3states.txt'
    Model ='Models/final_continuous_model_21_nodes.txt'
else:
    Model = str(args.model)
get_num_nodes(Model)

if args.start != None:
    start = int(args.start)
else:
    start = 0

if args.end != None:
    end = int(args.end)
else:
    end = numStates**numNodes
    #end =10000
if args.parallel != None:
    parallel = int(args.parallel)
else:
    parallel = 0
    
if args.verbose != None:
    v = int(args.verbose)
else:
    v = 0
samplesize = end - start

if parallel == False:
    #print "Running on single CPU"
    main1()
    #import profile
    #profile.run('main1()',sort=2)
if parallel == True:
    import pypar
    # Must have pypar installed, uses a "stepping" of 100, which means splits up
    # the job in batches of 10 over the processors

    #Initialise
    t = pypar.time()
    P = pypar.size()
    p = pypar.rank()
    processor_name = pypar.get_processor_name()
    
    # Block stepping
    
    stepping = 100
    dir,file = os.path.split(Model)
    prefix = file.split('.')[0]
    if p == 0:
        compile_cython_code(Model, overwrite=False)
    else:
        while os.path.exists(prefix+'.so') == False:
            time.sleep(.2)
        Function = import_module(prefix)
        function = Function.function

    B = samplesize/stepping # Number of blocks
    print 'Processor %d initialised on node %s' % (p, processor_name)
    assert P > 1, 'Must have at least one slave'
    assert B > P - 1, 'Must have more work packets than slaves'

    if p == 0:

        print 'samplesize = ',samplesize
        print 'split up into %s segments' % str(B)
        #Create array for storage
        Results = dict()
        # Create work pool (B blocks)
        workpool = []
        for i in xrange(start, end, stepping):
            workpool.append(i)
        # Distribute initial work to slaves
        w = 0
        for d in xrange(1, P):
            pypar.send(workpool[w], destination=d)
            w += 1
        # Receive computed work and distribute more
        terminated = 0
        while(terminated < P - 1):
            data,status= pypar.receive(pypar.any_source,return_status=True)
            for tmp in data[1]:     # check to see if new states are already present
                if tmp in Results:
                    Results[tmp]+=data[1][tmp]
                else:
                    Results[tmp]=data[1][tmp]
            d = status.source  # Id of slave that just finished
            if w < len(workpool):
                # Send new work to slave d
                pypar.send(workpool[w], destination=d)
                w += 1
            else:
                # Tell slave d to terminate
                pypar.send(None, destination=d)
                terminated += 1
        print 'Computed '+str(samplesize)+' samples in %.4f minutes' % ((pypar.time() - t)/60)

    else:
        while(True):
            # Receive work (or None)
            W = pypar.receive(source=0)
            if W is None:
                break
            # Compute allocated work
            data = {}
            for i in xrange(0,stepping):
                if  W+i >= end:
                    break
                else:
                    if v == 1:
                        print str(W+i+1),'/',end
                    tmp = run(W+i)
                    try:
                        data[tmp]+=1
                    except:
                        data[tmp]=1
                    #data.append(tmp)
            # Return result
            pypar.send((W,data), destination=0)
    pypar.finalize()
    if p == 0:
        print 'Attractors ',Results.keys()
        print 'Frequencies ',Results.values()
        print 'Total ',np.sum(Results.values())



