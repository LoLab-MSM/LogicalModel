from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
        name = "changeBase",
        ext_modules = cythonize('changeBase.pyx'),  # accepts a glob pattern
        include_dirs=[numpy.get_include()]
)

