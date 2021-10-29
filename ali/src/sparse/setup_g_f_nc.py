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

os.environ["CC"] = "cc"
os.environ["CXX"] = "c++"
#os.environ["CC"] = "icc"
#os.environ["CXX"] = "icpc"

CUDA_LIB = os.environ["TACC_CUDA_LIB"]
CUDA_INC = os.environ["TACC_CUDA_INC"]

#include directories
inc_dirs = []
#inc_dirs = inc_dirs + ['/work2/07544/ghafouri/frontera/gits/pyrknn/GeMM/include']
#inc_dirs = numpy.distutils.misc_util.get_numpy_include_dirs()
inc_dirs = inc_dirs + [np.get_include()]
inc_dirs = inc_dirs + [CUDA_LIB]
inc_dirs = inc_dirs + [CUDA_LIB+"/stubs/"] 
#library directories
lib_dirs = []
lib_dirs = lib_dirs + [CUDA_LIB]
lib_dirs = lib_dirs + [CUDA_LIB+"/stubs/"]
lib_dirs = lib_dirs + [CUDA_INC]

object_list=["guided_full_nodatacopy/Norms.o", "guided_full_nodatacopy/merge.o", \
            "guided_full_nodatacopy/GuidedBinSearch.o", "guided_full_nodatacopy/queryknn_guided.o"]

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
        extra_link_args=["-Wl,--no-as-needed", "-Wl,--verbose", "-ldl", "-lpthread","-lcuda", "-lcudart", "-lcublas"]
    )

extNames = scandir("guided_full_nodatacopy")
print(extNames)
extensions = [makeExtension(name) for name in extNames]
print(extensions)

setup(
    name="queryknn",
    packages=["queryknn"],
    ext_modules=extensions,
    package_data={
        '':['*.pxd']
    },
    zip_safe=False,
    include_package_data=True,
    cmdclass = {'build_ext': build_ext}
    )
