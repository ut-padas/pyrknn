import sys
import os
import skbuild
from setuptools import find_namespace_packages

from string import Template

def main():
    #Load README.md as long description for PyPI #TODO: Change to README_PYPI)
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()

    #Download and Update GIT Submodules
    #if os.path.exists(".git"):
    #    import pip._internal.vcs.git as git
    #
    #    g = git.Git()  # NOTE: pip API's are internal, this has to be refactored
    #
    #    g.run_command(["submodule", "sync"])
    #    g.run_command(
    #        ["submodule", "update", "--init", "--recursive"]
    #    )

    #Set default values of GPU Support and Numba_Threads
    use_cuda = False
    use_gsknn = False
    use_mkl = False
    build_sparse = False
    numba_threads = 8


    if env_build_sparse := os.getenv("PYRKNN_BUILD_SPARSE"):
        build_sparse = env_build_sparse

    if env_use_mkl := os.getenv("PYRKNN_USE_MKL"):
        use_mkl = env_use_mkl

    if env_use_cuda := os.getenv("PYRKNN_USE_CUDA"):
        use_cuda = env_use_cuda

    if env_numba_threads := os.getenv("PYRKNN_NUMBA_THREADS"):
        numba_threads = env_numba_threads

    if env_use_gsknn := os.getenv("PYRKNN_USE_GSKNN"):
        use_gsknn = env_use_gsknn


    #Configure GPU Support

    config_dict = {
    'pyrknn_cuda' : use_cuda,
    'pyrknn_numba_threads' : numba_threads,
    }

    with open('src/pyrknn/kdforest/config.in', 'r') as f_in:
        src = Template(f_in.read())
        configured = src.substitute(config_dict)

        with open('src/pyrknn/kdforest/config.py', 'w') as f_out:
            f_out.write(configured)


    #Set CMake Arguments
    cmake_args = []
    cmake_args.append("-DPROD=1")
    cmake_args.append("-DFRONTERA=1")

    if(use_gsknn):
        cmake_args.append("-DUSE_GSKNN=1")
    else:
        cmake_args.append("-DUSE_GSKNN=0")

    if(use_cuda):
        cmake_args.append("-DPYRKNN_USE_CUDA=1")
    else:
        cmake_args.append("-DPYRKNN_USE_CUDA=0")

    if(use_mkl):
        cmake_args.append("-DUSE_MKL=1")
    else:
        cmake_args.append("-DUSE_MKL=0")

    if(build_sparse):
        cmake_args.append("-DBUILD_SPARSE=1")
    else:
        cmake_args.append("-DBUILD_SPARSE=0")

    #Fix for MKL in Conda install
    if mkl_prefix := os.getenv("CONDA_PREFIX"):

        mkl_preload = []
        #mkl_preload.append(mkl_prefix+r"/lib/libmkl_core.so")
        #mkl_preload.append(mkl_prefix+r"/lib/libmkl_sequential.so")
        #mkl_preload.append(mkl_prefix+r"/lib/libmkl_intel_lp64.so")
        #mkl_preload.append(mkl_prefix+r"/lib/libmkl_avx512.so")

        #if os.getenv("LD_PRELOAD"):
        #    os.environ["LD_PRELOAD"] += os.pathsep + os.pathsep.join(mkl_preload)
        #else:
        #    os.environ["LD_PRELOAD"] = os.pathsep + os.pathsep.join(mkl_preload)


    os.environ["GSKNN_ARCH_MAJOR"] = "x86_64"
    os.environ["GSKNN_ARCH_MINOR"] = "sandybridge"

    skbuild.setup(
        name="pyrknn",
        version="0.0.6",
        description="a minimal example package (cpu dense only)",
        author='Chao Chen, William Ruys',
        author_email="will@oden.utexas.edu",
        license="GNU GPL3",
        packages=find_namespace_packages(where='src'),
        package_dir = {"":"src"},
        python_requires=">=3.8",
        cmake_args=cmake_args
    )

if __name__ == "__main__":
    main()
