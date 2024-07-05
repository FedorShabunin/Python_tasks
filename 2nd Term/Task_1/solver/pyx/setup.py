#!/usr/bin/env python3

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

compilation_args = ['-fopenmp', '-O3']

ext_modules = [
    Extension(
        "relaxation",
        ["relaxation.pyx"],
        extra_compile_args=compilation_args,
        extra_link_args =compilation_args,
        include_dirs=[numpy.get_include()],
    ),
]


setup(
    ext_modules=cythonize(ext_modules)
)