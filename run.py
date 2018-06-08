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
import subprocess
import jinja2
import pydotplus as pydot


env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(
        searchpath=os.path.join(os.path.dirname(__file__), 'templates')
    )
)

template = env.get_template('functions.pyx')
setup_template = env.get_template('temp_setup.py')


p = argparse.ArgumentParser()
p.add_argument("-n", "--nstates", type=str, help='provide a number of states')
p.add_argument("-s", "--start", type=str, help='starting string to convert to base Nstates')
p.add_argument("-e", "--end", type=str, help='ending string to convert to based Nstates')
p.add_argument("-m", "--model", type=str, help='model to run simulation, assumes file to end in .txt')
p.add_argument("-v", "--verbose", type=str, help='if you want verbose updates (use with single processor)')
args = p.parse_args()

# n_states=int(args.nstates)
n_states = 3


def compile_cython_code(model_file):
    with open(model_file, 'r') as f:
        functions = map(lambda s: s.strip(), f)
    num_nodes = len(functions)
    directory, filename = os.path.split(model_file)
    prefix = filename.split('.')[0]
    try:
        return import_module(prefix).function,\
               import_module(prefix).changebase, num_nodes
    except ImportError:
        pass

    function_output = ''

    for line in functions:
        # Replace function name
        index = 'vect[%d]' % (int(re.match('f(\d+)', line).group(1)) - 1)
        line = re.sub('f\d+', index, line)

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
            line = re.sub('x\d+', 'x[%d]' % (m - 1), line, count=1)
        function_output += '    %s\n' % line

    with open(prefix + '.pyx', 'w') as pyxfile:
        pyxfile.write(template.render(
            {'vector': function_output,
             'vectorsize': num_nodes}
        ))

    with open('temp_setup.py', 'w') as setup:
        setup.write(setup_template.render({'prefix': prefix}))

    python = r'C:\Users\James Pino\Miniconda2\envs\pysb_env\python.exe'

    cython_feedback = subprocess.call(
        [python, 'temp_setup.py', 'build_ext', '--inplace']
    )

    if cython_feedback:
        print(cython_feedback)
    return import_module(prefix).function, \
           import_module(prefix).changebase, num_nodes


def test(num, n_nodes, n_states):
    state = np.zeros(n_nodes, dtype=np.int)
    quotient = np.int(num / n_states)
    remainder = num % n_states
    num = quotient
    counter = -1
    state[counter] = remainder
    counter -= 1
    while quotient != 0:
        quotient = np.int(num / n_states)
        remainder = num % n_states
        state[counter] = remainder
        counter -= 1
        num = quotient
    return state


# @profile
def run(state):
    x = changebase(state, n_nodes, n_states)
    loc = tuple(x)
    path = {loc}

    tmp, tmp2 = update_func(x)

    if tmp2 in all_dict:
        all_dict[loc] = all_dict[tmp2]
        return all_dict[tmp2]

    while tmp2 not in path:
        path.add(tmp2)
        tmp, tmp2 = update_func(tmp)
        if tmp2 in data:
            for j in path:
                all_dict[j] = tmp2
            return tmp2

    path2 = {tmp2}
    tmp, tmp2 = update_func(tmp)
    while tmp2 not in path:
        path2.add(tmp2)
        tmp, tmp2 = update_func(tmp)
    attractor = path2.pop()
    print(attractor)
    for i in path:
        all_dict[i] = attractor

    return attractor


all_dict = {}
data = dict()


def main():
    print('Started ')

    start_time = time.time()

    for i in range(start, end):
        tmp = run(i)
        try:
            data[tmp] += 1
        except:
            data[tmp] = 1

    end_t = time.time()
    time_mins = (end_t - start_time) #/ 60.

    print('Computed {0} samples {1:.4f} seconds'.format(samplesize, time_mins))
    print(time_mins*total_possible/end)

    draw = True
    if draw:
        draw_attractors(data_dict=data)

    print('Attractors ', data.keys())
    print('Frequencies ', data.values())
    print('Total ', np.sum(data.values()))


def draw_attractors(data_dict):
    graph = pydot.Dot(directed=True)
    nodes = set()
    for i in data_dict.keys():
        path = return_attractor(i, graph)
        nodes.update(path)
    graph.write('%s.pdf' % str('attractors3'), prog='dot', format='pdf')


def return_attractor(point, graph):
    path = set()
    tmp, tmp2 = update_func(np.array(point))
    x = ''.join([str(i) for i in tmp2])
    string_path = x
    while tmp2 not in path:
        path.add(tmp2)
        tmp, tmp2 = update_func(tmp)
        y = ''.join([str(i) for i in tmp2])
        graph.add_edge(pydot.Edge(x, y))
        x = y
        string_path += ' -> ' + y
    print(string_path)
    return path


if args.model is None:
    # Model = 'Models/func_example.txt'
    # Model = 'Models/core_iron_6variables_3states.txt'
    # Model = '/Users/jamespino/git/LogicalModel/NewModels_2015_8_17/Ftmt_oe_24.txt'
    Model = 'Models/final_continuous_model_21_nodes.txt'
else:
    Model = str(args.model)

update_func, changebase, n_nodes = compile_cython_code(Model)

if args.start is not None:
    start = int(args.start)
else:
    start = 0

if args.end is not None:
    end = int(args.end)
else:
    total_possible = n_states ** n_nodes
    end = 100000
    if end > total_possible:
        end = total_possible

if args.verbose is not None:
    v = int(args.verbose)
else:
    v = 0

samplesize = end - start

if __name__ == "__main__":
    main()

    print("Len of dict = %s " % len(all_dict))
    print("Size of set = %s gb " % str(np.float(sys.getsizeof(all_dict)) / (10. ** 9)))
