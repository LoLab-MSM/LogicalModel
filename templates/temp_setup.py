from distutils.core import setup
import numpy
from distutils.core import setup, Extension
from Cython.Build import cythonize

setup(
    ext_modules=cythonize('{{prefix}}.pyx',
                          # language="c++",
                          ),
    include_dirs=[numpy.get_include()]
)
