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

#TODO(p3)[Will] Make these not TACC specific. Set these in source script from base makefile

#ensure that we're using Intel compilers
os.environ["CC"] = "icc"
os.environ["CXX"] = "icpc"

use_cuda = False
try:
    use_cuda = bool(os.environ["PRKNN_USE_CUDA"])
except:
    print("Enviornment varibale PRKNN_USE_CUDA is not set. Assuming CUDA is not being used.")

CUDA_LIB = ""
try:
    CUDA_LIB = os.environ["CUDA_LIB"]
except:
    raise Exception("Enviornment variable CUDA_LIB is not set. Please set this to your CUDA library path.")

#TODO: Make this optional
GSKNN_DIR = ""
try:
    GSKNN_DIR = os.environ["GSKNN_DIR"]
except:
    raise Exception("Enviornment variable GSKNN_DIR is not set. Please set this to link against the GSKNN Library")


GPU_IMPL_DIR = "prknn/kernels/gpu/impl/"

#Parse command line arguments to get learn required object files
parser = argparse.ArgumentParser(description='Cython setup script to link against compiled object files')
parser.add_argument('--gpu_obj', '--cuda_obj', nargs="+", help='Pass a list of file names to link as cuda objects to the cython script', required=False, dest="gpu_obj", default=[])
parser.add_argument('--cpu_obj', nargs="+", help='Pass a list of file names to link as cuda objects to the cython script', required=False, dest="cpu_obj", default=[])
parser.add_argument("--setup", nargs="+", help="Pass setuptools arguments into setup.py (Required)", required=True, dest="setup")
args, unknown = parser.parse_known_args()

sys.argv = ['setup.py'] + args.setup + unknown

print(args.gpu_obj)

#TODO(p3)[Will] One of these includes/libraries doesn't need CUDA. Which way is it again?

#setup the shared include directories
inc_dirs = []
inc_dirs = inc_dirs + [np.get_include()]

#setup the cpu include directories
cpu_inc_dirs = inc_dirs
cpu_inc_dirs = cpu_inc_dirs + [GSKNN_DIR+'build/include']

#setup the gpu include directories
gpu_inc_dirs = inc_dirs + [CUDA_LIB]
gpu_inc_dirs = gpu_inc_dirs + [CUDA_LIB+'/stubs/']
gpu_inc_dirs = gpu_inc_dirs + [GPU_IMPL_DIR+'chao/sort']
gpu_inc_dirs = gpu_inc_dirs + [GPU_IMPL_DIR+'chao/merge']
gpu_inc_dirs = gpu_inc_dirs + [GPU_IMPL_DIR+'chao/util']
gpu_inc_dirs = gpu_inc_dirs + [GPU_IMPL_DIR+'chao/']

#setup the shared library directories
lib_dirs = []

#setup the cpu library directories
cpu_lib_dirs = lib_dirs
cpu_lib_dirs = cpu_lib_dirs + [GSKNN_DIR+'build/lib']

#setup the gpu library directories
gpu_lib_dirs = lib_dirs + [CUDA_LIB]
gpu_lib_dirs = gpu_lib_dirs + [CUDA_LIB+'/stubs/']
gpu_lib_dirs = gpu_lib_dirs + [GPU_IMPL_DIR+'chao/sort']
gpu_lib_dirs = gpu_lib_dirs + [GPU_IMPL_DIR+'chao/merge']
gpu_lib_dirs = gpu_lib_dirs + [GPU_IMPL_DIR+'chao/util']
gpu_lib_dirs = gpu_lib_dirs + [GPU_IMPL_DIR+'chao/']


#Grab all the required pyx files to compile
def scandir(directory, files=[]):
    for f in os.listdir(directory):
        path = os.path.join(directory, f)
        if os.path.isfile(path) and path.endswith(".pyx"):
            f_to_add = path.replace(os.path.sep, ".")[:-4]
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
                extra_objects=args.gpu_obj,
                libraries=["cuda", "cudart", "cublas", "util", "sortgpu", "mergegpu", "knngpu"], 
                extra_compile_args=["-std=c++11", "-O3", "-fPIC", "-DPROD"],
                extra_link_args=["-ldl", "-lpthread", "-qopenmp"]
                )
    return Extension(
            extName,
            [extPath],
            include_dirs = cpu_inc_dirs,
            language="c++",
            library_dirs = cpu_lib_dirs,
            runtime_library_dirs = cpu_lib_dirs,
            extra_objects=args.cpu_obj+[GSKNN_DIR+"/build/lib/libgsknn_shared.so", GSKNN_DIR+"/build/lib/libgsknn_ref_stl_shared.so"],
            extra_compile_args=["-std=c++11", "-O3", "-fPIC", "-qopenmp","-qopenmp-report 2", "-Wno-sign-compare"],
            extra_link_args=["-ldl", "-lpthread", "-qopenmp", "-lm", "-lgsknn"]
            )


extNames = scandir("prknn")
print("Found the following cython extensions (to be built): ")

extensions = [makeExtension(name) for name in reversed(extNames) if name is not None]

for e in extensions:
    print(e)

print("Linking against", args.gpu_obj)

setup(
        name="prknn",
        packages=["prknn", "prknn.kernels", "prknn.kernels.gpu", "prknn.kernels.cpu", "prknn.kdforest", "prknn.kdforest.reference"],
        ext_modules=extensions,
        package_data={
                '':['*.pxd', '.pyx']
            },
        zip_safe=False,
        include_package_data=True,
        cmdclass = {'build_ext': build_ext}
        )





