from setuptools import setup, find_packages
from setuptools.extension import Extension

#import numpy.distutils.misc.util
import argparse
import sys, os

import numpy as np

#Check is Cython is installed
try:
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext
except:
    raise Exception('Cython is not installed or could not be found by your PYTHON_PATH')

os.environ["CC"] = "mpicc"
os.environ["CXX"] = "mpicxx"

#Compile GPU Kernels?
use_cuda = False
try:
    use_cuda = int(os.environ["PYRKNN_USE_CUDA"])
    if use_cuda == 0:
        use_cuda = False
    else:
        use_cuda = True
    print("PYRKNN_USE_CUDA is set to ", use_cuda)
    print("LKJLJLKJLJKJLJ")
except:
    print("Environment variable PYRKNN_USE_CUDA is not set. Assuming CUDA is not being used.")

#The location of the CUDA_LIB should be set. This can be important for cupy.
CUDA_LIB = ""
try:
    CUDA_LIB = os.environ["CUDA_LIB"]
except:
    raise Exception("Enviornment variable CUDA_LIB is not set. Please set this to your CUDA library path.")

#We currently require GSKNN for CPU Dense Distance Kernel evaluation.   TODO: Make this optional
GSKNN_DIR = ""
try:
    GSKNN_DIR = os.environ["GSKNN_DIR"]
except:
    raise Exception("Environment variable GSKNN_DIR is not set. Please set this to link against the GSKNN Library")

#We require Eigen for the computation of Sparse exact knn solution of the GPU
EIGEN_DIR = ""
try:
    EIGEN_DIR = os.environ["EIGEN_ROOT"]
except:
    raise Exception("Environment variable EIGEN_ROOT is not set. Please set this to link against the Eigen header library")

GPU_IMPL_DIR = "pyrknn/kernels/gpu/impl"
CPU_IMPL_DIR = "pyrknn/kernels/cpu/impl"

#Parse command line arguments to get required files
#TODO: Remove this, I don't believe its currently necessary
#parser = argparse.ArgumentParser(description='Cython setup script to link against compiled object files')
#parser.add_argument('--gpu_obj', '--cuda_obj', nargs="+", help='Pass a list of file names to link as cuda objects to the cython script', required=False, dest="gpu_obj", default=[])
#parser.add_argument('--cpu_obj', nargs="+", help='Pass a list of file names to link as cuda objects to the cython script', required=False, dest="cpu_obj", default=[])
#parser.add_argument("--setup", nargs="+", help="Pass setuptools arguments into setup.py (Required)", required=True, dest="setup")
#args, unknown = parser.parse_known_args()

#sys.argv = ['setup.py'] + args.setup + unknown

#setup the shared include directories
inc_dirs = []
inc_dirs = inc_dirs + [np.get_include()]

#setup the cpu include directories
cpu_inc_dirs = inc_dirs
cpu_inc_dirs = cpu_inc_dirs + [GSKNN_DIR+'build/include']
cpu_inc_dirs = cpu_inc_dirs + [EIGEN_DIR]
cpu_inc_dirs = cpu_inc_dirs + [CPU_IMPL_DIR+'/sparse']
cpu_inc_dirs = cpu_inc_dirs + [CPU_IMPL_DIR+'/exact']
cpu_inc_dirs = cpu_inc_dirs + [GPU_IMPL_DIR+'/readSVM']
cpu_inc_dirs = cpu_inc_dirs + [GPU_IMPL_DIR+'/util']
cpu_inc_dirs = cpu_inc_dirs + [CUDA_LIB]
cpu_inc_dirs = cpu_inc_dirs + [CUDA_LIB+'/stubs']
#setup the gpu include directories
gpu_inc_dirs = inc_dirs + [CUDA_LIB]
gpu_inc_dirs = gpu_inc_dirs + [CUDA_LIB+'/stubs/']

gpu_inc_dirs = gpu_inc_dirs + [GPU_IMPL_DIR+'/gemm']
gpu_inc_dirs = gpu_inc_dirs + [GPU_IMPL_DIR+'/merge']
gpu_inc_dirs = gpu_inc_dirs + [GPU_IMPL_DIR+'/transpose']
gpu_inc_dirs = gpu_inc_dirs + [GPU_IMPL_DIR+'/reorder']
gpu_inc_dirs = gpu_inc_dirs + [GPU_IMPL_DIR+'/sort']
gpu_inc_dirs = gpu_inc_dirs + [GPU_IMPL_DIR+'/sparse']
gpu_inc_dirs = gpu_inc_dirs + [GPU_IMPL_DIR+'/singleton']
gpu_inc_dirs = gpu_inc_dirs + [GPU_IMPL_DIR+'/dense']
gpu_inc_dirs = gpu_inc_dirs + [GPU_IMPL_DIR+'/orthogonal']
gpu_inc_dirs = gpu_inc_dirs + [GPU_IMPL_DIR+'/util']
gpu_inc_dirs = gpu_inc_dirs + [GPU_IMPL_DIR+'/readSVM']
gpu_inc_dirs = gpu_inc_dirs + [GPU_IMPL_DIR+'/']

#setup the shared library directories
lib_dirs = []

#setup the cpu library directories
cpu_lib_dirs = lib_dirs
cpu_lib_dirs = cpu_lib_dirs + [GSKNN_DIR+'build/lib']
cpu_lib_dirs = cpu_lib_dirs + [EIGEN_DIR]
cpu_lib_dirs = cpu_lib_dirs + [CPU_IMPL_DIR+'/sparse']
cpu_lib_dirs = cpu_lib_dirs + [CPU_IMPL_DIR+'/exact']
cpu_lib_dirs = cpu_lib_dirs + [GPU_IMPL_DIR+'/readSVM']
cpu_lib_dirs = cpu_lib_dirs + [GPU_IMPL_DIR+'/util']
cpu_lib_dirs = cpu_lib_dirs + [CUDA_LIB]
cpu_lib_dirs = cpu_lib_dirs + [CUDA_LIB+'/stubs']

#setup the gpu include directories
#setup the gpu library directories
gpu_lib_dirs = lib_dirs + [CUDA_LIB]
gpu_lib_dirs = gpu_lib_dirs + [CUDA_LIB+'/stubs/']

gpu_lib_dirs = gpu_lib_dirs + [GPU_IMPL_DIR+'/gemm']
gpu_lib_dirs = gpu_lib_dirs + [GPU_IMPL_DIR+'/merge']
gpu_lib_dirs = gpu_lib_dirs + [GPU_IMPL_DIR+'/transpose']
gpu_lib_dirs = gpu_lib_dirs + [GPU_IMPL_DIR+'/reorder']
gpu_lib_dirs = gpu_lib_dirs + [GPU_IMPL_DIR+'/sort']
gpu_lib_dirs = gpu_lib_dirs + [GPU_IMPL_DIR+'/sparse']
gpu_lib_dirs = gpu_lib_dirs + [GPU_IMPL_DIR+'/singleton']
gpu_lib_dirs = gpu_lib_dirs + [GPU_IMPL_DIR+'/dense']
gpu_lib_dirs = gpu_lib_dirs + [GPU_IMPL_DIR+'/orthogonal']
gpu_lib_dirs = gpu_lib_dirs + [GPU_IMPL_DIR+'/util']
gpu_lib_dirs = gpu_lib_dirs + [GPU_IMPL_DIR+'/readSVM']
gpu_lib_dirs = gpu_lib_dirs + [GPU_IMPL_DIR+'/']

#Grab all the required pyx files to compile
def scandir(directory, files=[]):
    for f in os.listdir(directory):
        path = os.path.join(directory, f)
        if os.path.isfile(path) and path.endswith(".pyx"):
            f_to_add = path.replace(os.path.sep, ".")[:-4]
            #if gpu is in the path, it is a wrapper for a GPU kernel. Compile it differently.
            if 'gpu' in f_to_add:
                if use_cuda:
                    files.append(f_to_add)
            else:
                files.append(f_to_add)
        elif os.path.isdir(path):
            scandir(path, files)
    return files

#For each pyx file setup an Extension object with compilation details
def makeExtension(extName):
    extPath = extName.replace(".", os.path.sep) + ".pyx"
    if 'gpu' in extPath:
        return Extension(
                extName,
                [extPath],
                include_dirs = gpu_inc_dirs,
                language="c++",
                library_dirs = gpu_lib_dirs,
                runtime_library_dirs = gpu_lib_dirs,
                #extra_objects=args.gpu_obj,
                #libraries=["cusparse", "cusolver", "cublas", "cudart"] +["denknngpu", "merge", "gemm", "reorder", "sort", "orthogonal", "readSVM", "transpose", "util"]+["spknngpu", "merge", "gemm", "reorder", "transpose", "orthogonal", "reorder", "sort", "readSVM", "util"],
                #libraries=["cuda", "cudart", "cublas", "cusparse", "cusolver", "util"] +["gemm", "util", "sort", "merge", "spknngpu", "transpose", "orthogonal", "reorder", "denknngpu"],

                libraries=["cusparse", "cusolver", "cublas", "cudart", "denknngpu","merge", "util", "readSVM", "transpose", "gemm", "reorder", "orthogonal", "spknngpu"],

                #libraries=["cusparse", "cusolver", "cublas", "cudart", "spknngpu","merge", "util", "readSVM", "transpose", "gemm", "reorder", "orthogonal", "denknngpu"],

                #libraries=["cusparse", "cusolver", "cublas", "cudart", "spknngpu","cusparse", "cusolver", "cublas", "cudart", "denknngpu", "merge", "util", "readSVM", "transpose", "gemm", "reorder", "orthogonal"],


                extra_compile_args=["-g", "-std=c++11", "-O3", "-fPIC", "-DPROD"],
                extra_link_args=["-ldl", "-lpthread", "-qopenmp"]#+["-Wl,--start-group", "-lcusparse", "-lcusolver", "-lcublas", "-lcudart", "-lspknngpu", "-ldenknngpu", "-lmerge", "-lutil", "-lreadSVM", "-ltranspose", "-lgemm", "-lreorder", "-lorthogonal", "-Wl,--end-group"]
                )
    return Extension(
            extName,
            [extPath],
            include_dirs = cpu_inc_dirs,
            language="c++",
            library_dirs = cpu_lib_dirs,
            runtime_library_dirs = cpu_lib_dirs,
            extra_compile_args=["-std=c++11", "-O3", "-fPIC", "-qopenmp", "-Wno-sign-compare","-xCORE-AVX2","-axCORE-AVX512", "-DEIGEN_USE_MKL_ALL", "-mkl"],
            extra_link_args=["-ldl", "-lpthread", "-qopenmp", "-lm", "-lgsknn", "-lspknncpu", "-lreadSVM", "-lexact", "-mkl"]
            )


extNames = scandir("pyrknn")
print("Found the following cython extensions (to be built): ")

extensions = [makeExtension(name) for name in reversed(extNames) if name is not None]

print("Extension Names:")
for e in extNames:
    print(e)


package_list=["pyrknn", "pyrknn.kernels", "pyrknn.kernels.cpu", "pyrknn.kdforest", "pyrknn.kdforest.reference"]
if use_cuda:
    package_list += ["pyrknn.kernels.gpu"]

setup(
        name="pyrknn",
        packages=package_list,
        ext_modules=extensions,
        package_data={
                '':['*.pxd', '.pyx']
            },
        zip_safe=False,
        include_package_data=True,
        cmdclass = {'build_ext': build_ext}
        )
