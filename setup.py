import sys

from skbuild import setup

setup(
    name="pyrknn",
    version="0.0.1",
    description="a minimal example package (cpu dense only)",
    author='Chao Chen, William Ruys',
    license="MIT",
    packages=['pyrknn'],
    install_requires=['cython', 'numpy', 'scipy', 'numba', 'mpi4py', 'scikit-learn']
    #tests_require=['pytest'],
    #setup_requires=setup_requires
)
