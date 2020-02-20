



Will's Note (Updated Feb 15):
Set the appropriate enviornment variables in `set_env.sh`. Run make to build. 
Adding this to your PYTHON_PATH env variable i.e. `export PYTHON_PATH=$PYTHON_PATH:$(pwd)/prknn` should work.
Then you can run the example script: example/test_accuracy.py

Numba works on Frontera with CUDA/10.0
Set: CUDA_HOME=$TACC_CUDA_DIR
You might need to set: NUMBA_CUDA_DRIVER=/usr/lib64/libcuda.so.1

TODO(p1)[Will] Fix build system.

Currently missing in the current src (need to merge in):
- Hongru's gpu kselect and scan
- Will's CUDA median function


##Outline of File Structure

examples/

    - For example scripts showing functionality. Eventually include scripts to reproduce results. 

test/

    - For now, just for debugging scripts. Ideally, clean this up and use a unittest module. (Save this for after PARLA implementation.)
    - TODO(p1)[ALL] We need test and timing scripts for kernels
 
src/prknn/kernels/cpu

    - cython wrapper for functions in impl (pxd, pyx, setup.py)

src/prknn/kernels/cpu/impl/

    - source files for cpp impl of kernels

src/prknn/kernels/gpu/

    - cython wrapper for functions in impl (pxd, pyx, setup.py)
    - pure python gpu functions (NUMBA, CuPy, etc)

src/prknn/kernels/gpu/impl

    - source files for cpp/cuda impl of kernels

src/prknn/kdforest/reference/

    - Pure Python Implementation of KNN Scripts
    - The important file here is: util.py, it contains kernels to be replaced with specific variants
    - TODO(p1)[Will] Test and switch all kernels in util.py to specific variants

src/prknn/kdforest/parla/

    - Parla Tasking Implementation
    - TODO(p2)[Will] Do this. (Side task: parallelize python parts within tasks using numba, avoid using dictionary to gather query ids)

##A quick note on TODO format:

Very informally, mostly so I can set reminders for myself to without going through the git issue system. 
TODO(p#)[user] where # marks the priority of the work (vague 1 - 5) to be completed and user is assigned the work. User is optional.
Unmarked is considered to have low priority
