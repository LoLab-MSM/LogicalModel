from distutils.core import setup
 import numpy
from Cython.Build import cythonize
setup(
    ext_modules = cythonize('func_example.pyx',include_path = [numpy.get_include()])
    )
