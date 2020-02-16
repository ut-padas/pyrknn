


Will's Note (Updated Feb 15):
Adding this to your PYTHON_PATH env variable i.e. `export PYTHON_PATH=$PYTHON_PATH:$(pwd)/knn` should work.
Then you can run the example script: example/test_accuracy.py

TODO(p1)[Will] Fix build system.

Currently missing in the current src (need to merge in):
- Hongru's gpu kselect and scan
- Will's CUDA median function


Outline of File Structure

examples/

    - For example scripts showing functionality. Eventually include scripts to reproduce results. 

test/

    - For now, just for debugging scripts. Ideally, clean this up and use a unittest module. (Save this for after PARLA implementation.)
    - TODO(p1)[ALL] We need test and timing scripts for kernels
    - 
src/kernels/cpu

- cython wrapper for functions in impl (pxd, pyx, setup.py)

src/kernels/cpu/impl/

- source files for cpp impl of kernels
-
src/kernels/gpu/

- cython wrapper for functions in impl (pxd, pyx, setup.py)
- pure python gpu functions (NUMBA, CuPy, etc)
-
src/kernels/gpu/impl

- source files for cpp/cuda impl of kernels
-
src/kdforest/reference/

- Pure Python Implementation of KNN Scripts
-
- The important file here is: util.py, it contains kernels to be replaced with specific variants
-
- TODO(p1)[Will] Test and switch all kernels in util.py to specific variants
src/kdforest/parla/

- Parla Tasking Implementation

- TODO(p2)[Will] Do this. (Side task: parallelize python parts within tasks using numba, avoid using dictionary to gather query ids)

A quick note on TODO format:
    Very informally, mostly so I can set reminders for myself to without going through the git issue system. 
    TODO(p#)[user] where # marks the priority of the work (vague 1 - 5) to be completed and user is assigned the work. User is optional.
    Unmarked is considered to have low priority


The following has been restructured/renamed/etc and will no longer work to build:

Hongru's Note (Feb 14):

cpu/combinatorics_cpu.hpp is originally located at HMLP/primitives/. 

To create shared library object, run 
'''python setup_test.py build_ext --inplace'''
 
