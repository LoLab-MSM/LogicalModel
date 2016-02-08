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
import re
import os
import warnings
from changeBase import changebase

p = argparse.ArgumentParser()
p.add_argument("-n", "--nstates", type=str, help='provide a number of states')
p.add_argument("-s", "--start", type=str, help='starting string to convert to base Nstates')
p.add_argument("-e", "--end", type=str, help='ending string to convert to based Nstates')
p.add_argument("-m", "--model", type=str, help='model to run simulation, assumes file to end in .txt')
p.add_argument("-v", "--verbose", type=str, help='if you want verbose updates (use with single processor)')
args = p.parse_args()

# numStates=int(args.nstates)
numStates = 3


def get_num_nodes(model_file):
    global numNodes
    f = open(model_file, 'r')
    functions = map(lambda s: s.strip(), f)
    numNodes = len(functions)
    f.close()


def compile_cython_code(model_file, overwrite=False):
    global function
    global numNodes

    f = open(model_file, 'r')
    functions = map(lambda s: s.strip(), f)
    numNodes = len(functions)
    directory, filename = os.path.split(model_file)
    prefix = filename.split('.')[0]

    # Warn the user if .so file already exists
    if os.path.exists(prefix + '.so'):
        warning_string = 'Shared object file \'%s\' already exists: ' % (prefix + '.so')
        if overwrite:
            warnings.warn(warning_string + 'overwriting file.')
            os.remove(prefix + '.so')
        else:
            warnings.warn(warning_string + 'moving on.')
            function = import_module(prefix).function
            return

    pyxfile = open(prefix + '.pyx', 'w')

    outstring = '\
    \nfrom __future__ import division\
    \nimport numpy as np\
    \ncimport numpy as np\
    \ncimport cython\
    \n\nDTYPE = np.int\
    \nctypedef np.int_t DTYPE_t\
    \n@cython.boundscheck(False)\
    \n@cython.wraparound(False)\
    \ndef function(object[DTYPE_t, ndim=1, mode="c"] x not None):\
    \n\tcdef int vectorsize = %d\
    \n\tcdef int i\
    \n\tcdef np.ndarray[DTYPE_t, ndim=1] vector = np.zeros([vectorsize],dtype=int)\n' % numNodes

    function_output = ''

    for line in functions:
        # Replace function name
        line = re.sub('f\d+', 'vector[%d]' % (int(re.match('f(\d+)', line).group(1)) - 1), line)
        # Convert powers (^) to simple multiplications
        matches = re.findall('(x\d+)\^(\d+)', line)
        for m in matches:
            repl = m[0]
            for i in range(1, int(m[1])):
                repl += "*%s" % m[0]
            line = re.sub('x\d+\^\d+', repl, line, count=1)
        # Replace node names
        matches = np.array(re.findall('x(\d+)', line), dtype=int)
        for m in matches:
            line = re.sub('x\d+', 'x[<int>%d]' % (m - 1), line, count=1)
        function_output += '\t%s\n' % line

    outstring+=function_output
    #print function_output
    outstring += '\tfor i in xrange(%d):\n\t\tvector[i] = vector[i]%%%d\n\treturn vector,tuple(vector)\n' % (
        numNodes, numStates)
    pyxfile.write(outstring)
    pyxfile.close()
    # os.system('sleep 2s')
    setup = open('temp_setup.py', 'w')
    outstring = \
        "from distutils.core import setup\
    \nimport numpy\
    \nfrom Cython.Build import cythonize\
    \nsetup(\
    \n    ext_modules = cythonize('" + prefix + ".pyx',),\n\
          include_dirs = [numpy.get_include()])"
    setup.write(outstring)
    setup.close()
    os.system('/Users/jamespino/git/RTA2/rta-env/bin/python temp_setup.py build_ext --inplace')
    sys.path.append(directory)
    function = import_module(prefix).function


def return_attractor(point, G):
    path = set()
    tmp, tmp2 = function(np.array(point))
    x = str(tmp2).replace(',', '').replace('(', '').replace(')', '').replace(' ', '')
    print point,
    while tmp2 not in path:
        path.add(tmp2)
        tmp, tmp2 = function(tmp)
        y = str(tmp2).replace(',', '').replace('(', '').replace(')', '').replace(' ', '')
        G.add_edge(x, y)
        x = y
        print '->', tmp2,

    print
    return path.pop()


def run(i):
    x = changebase(i, numNodes, numStates)
    loc = tuple(x)
    path = set()
    path.add(loc)
    tmp, tmp2 = function(x)
    if tmp2 in all_dict:
        all_dict[loc] = all_dict[tmp2]
        return all_dict[tmp2]
    while tmp2 not in path:
        path.add(tmp2)
        tmp, tmp2 = function(tmp)
        if tmp2 in data:
            for j in path:
                all_dict[j] = tmp2
            return tmp2
    path2 = set()
    path2.add(tmp2)
    tmp, tmp2 = function(tmp)
    while not tmp2 in path:
        path2.add(tmp2)
        tmp, tmp2 = function(tmp)
    attractor = path2.pop()
    for i in path:
        all_dict[i] = attractor
    # print 'Found unique attractor = %s' % str(attractor)
    return attractor


all_dict = {}
data = dict()

def main():
    print 'Started '
    compile_cython_code(Model, overwrite=False)
    start_time = time.time()
    for i in xrange(start, end):
        tmp = run(i)
        try:
            data[tmp] += 1
        except:
            data[tmp] = 1
    endT = time.time()
    print 'Computed %s samples %.4f minutes' % (str(samplesize), (endT - start_time) / 60.)
    print numStates ** numNodes/100000*(endT - start_time) / 60.
    draw = False
    if draw:
        import pygraphviz as pyg
        G = pyg.AGraph(directed=True)
        for i in data.keys():
            return_attractor(i, G)
        G.draw('%s.pdf' % str('attractors3'), prog='dot')
    print 'Attractors ', data.keys()
    print 'Frequencies ', data.values()
    print 'Total ', np.sum(data.values())


if args.model is None:
    # Model = 'Models/func_example.txt'
    # Model = 'Models/core_iron_6variables_3states.txt'
    #Model = '/Users/jamespino/git/LogicalModel/NewModels_2015_8_17/Ftmt_oe_24.txt'
    Model = 'Models/final_continuous_model_21_nodes.txt'
else:
    Model = str(args.model)
get_num_nodes(Model)

if args.start is not None:
    start = int(args.start)
else:
    start = 0

if args.end is not None:
    end = int(args.end)
else:
    end = numStates ** numNodes
    end = 100000

if args.verbose is not None:
    v = int(args.verbose)
else:
    v = 0

samplesize = end - start

if __name__ == "__main__":
    main()
    print "Len of dict = %s " % len(all_dict)
    print "Size of set = %s gb " % str(np.float(sys.getsizeof(all_dict)) / (10. ** 9))
