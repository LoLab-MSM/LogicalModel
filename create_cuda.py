# -*- coding: utf-8 -*-
"""
@author: James C Pino

"""
import numpy as np
import re
import os
import subprocess
import jinja2
import pycuda as cuda
from pycuda.driver import init as pycuda_init
import pycuda.compiler
import pycuda.tools as tools
import pycuda.driver as driver
import pycuda.gpuarray as gpuarray

env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(
        searchpath=os.path.join(os.path.dirname(__file__), 'templates')
    )
)

template = env.get_template('cuda_attempt.cu')

n_states = 3


def compile_cython_code(model_file):
    with open(model_file, 'r') as f:
        functions = map(lambda s: s.strip(), f)
    num_nodes = len(functions)

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
        function_output += '    %s;\n' % line
    code = template.render(
            {'functions': function_output,
             'num_states': n_states,
             'n_nodes': num_nodes},
    )
    with open('test.cpp', 'w') as pyxfile:
        pyxfile.write(code)
    return code

Model = 'Models/func_example.txt'
# Model = 'Models/core_iron_6variables_3states.txt'
# Model = 'Models/final_continuous_model_21_nodes.txt'


code = compile_cython_code(Model)


kernel = pycuda.compiler.SourceModule(code, no_extern_c=True)


runner = kernel.get_function("AttractorFinder")
quit()
_total_threads = 10000


species_matrix = np.zeros((10000,))

species_matrix_gpu = gpuarray.to_gpu(species_matrix)

# allocate space on GPU for results
result = driver.managed_zeros(
    shape=(_total_threads, 1, 1),
    dtype=np.int32, mem_flags=driver.mem_attach_flags.GLOBAL
)


# perform simulation
self._ssa_all(species_matrix_gpu, result, time_points_gpu, n_results,
              param_array,
              block=(self._threads, 1, 1), grid=(self._blocks, 1)
              )

# Wait for kernel completion before host access
pycuda.autoinit.context.synchronize()
