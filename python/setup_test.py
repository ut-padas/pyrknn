from distutils.core import setup
from Cython.Build import cythonize
from setuptools.extension import Extension

extensions = [
    Extension("combinatorics",
        sources=["comb.pyx"],
        language="c++",
        extra_compile_args=["-std=c++11","-O3","-fopenmp"],
        extra_link_args=["-fopenmp"])]
setup(ext_modules = extensions,zip_safe=False)