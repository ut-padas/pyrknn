from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy.distutils.misc_util
import sys, os
import numpy as np

#Check if Cython is installed
try:
    from Cython.Distutils import build_ext
except:
    print("You don't seem to have Cython installed")
    sys.exit(1)

os.environ["CC"] = "icc"
os.environ["CXX"] = "icpc"

CUDA_LIB = os.environ["TACC_CUDA_LIB"]

#include directories
inc_dirs = []
#inc_dirs = numpy.distutils.misc_util.get_numpy_include_dirs()
inc_dirs = inc_dirs + [np.get_include()]
inc_dirs = inc_dirs + [CUDA_LIB]
inc_dirs = inc_dirs + [CUDA_LIB+"/stubs/"] 
#library directories
lib_dirs = []
lib_dirs = lib_dirs + [CUDA_LIB + "/samples/common/inc"]
lib_dirs = lib_dirs + [CUDA_LIB]
lib_dirs = lib_dirs + [CUDA_LIB+"/stubs/"]

object_list=["dev/simple_cuda.o"]

def scandir(dir, files=[]):
    for file in os.listdir(dir):
        path = os.path.join(dir, file)
        if os.path.isfile(path) and path.endswith(".pyx"):
            files.append(path.replace(os.path.sep, ".")[:-4])
        elif os.path.isdir(path):
            scandir(path, files)
    return files

def makeExtension(extName):
    extPath = extName.replace(".", os.path.sep)+".pyx"
    return Extension(
        extName,
        [extPath],
        include_dirs = inc_dirs,
        language='c++',
        library_dirs = lib_dirs,
        runtime_library_dirs = lib_dirs,
        extra_objects=object_list,
        extra_compile_args=["-std=c++11","-O3", "-fPIC"],
        extra_link_args=["-Wl,--no-as-needed", "-Wl,--verbose", "-ldl", "-lpthread","-lcuda", "-lcudart"]
    )


extNames = scandir("cuda_wrapper")
print(extNames)
extensions = [makeExtension(name) for name in extNames]
print(extensions)

setup(
    name="cuda_wrapper",
    packages=["cuda_wrapper"],
    ext_modules=extensions,
    package_data={
        '':['*.pxd', '.pyx']
    },
    zip_safe=False,
    include_package_data=True,
    cmdclass = {'build_ext': build_ext}
    )
