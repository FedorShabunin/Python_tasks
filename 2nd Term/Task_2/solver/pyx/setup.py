#!/usr/bin/env python3

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

compilation_args = ['-fopenmp', '-O3']

source_files = ['tdma', 'crank']

ext_modules = [
    Extension(
        filename,
        [f"{filename}.pyx"],
        extra_compile_args = compilation_args,
        extra_link_args = compilation_args,
        include_dirs=[numpy.get_include()],
        compiler_directives={'language_level' : "3"}
    )
    for filename in source_files
]

setup(
    ext_modules=cythonize(ext_modules)
)