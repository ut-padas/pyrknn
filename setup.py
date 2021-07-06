import sys

from skbuild import setup
from setuptools import find_namespace_packages 
setup(
    name="pyrknn",
    version="0.0.1",
    description="a minimal example package (cpu dense only)",
    author='Chao Chen, William Ruys',
    license="MIT",
    packages=find_namespace_packages(where='src'),
    package_dir = {"":"src"},
    install_requires=['cython', 'numpy', 'scipy', 'numba', 'mpi4py', 'scikit-learn']
)
