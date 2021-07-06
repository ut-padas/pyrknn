cmake_minimum_required(VERSION 3.9.1)

set(CMAKE_CXX_STANDARD 14)

#--Setup Project
project(pyrknn LANGUAGES C CXX)

#-- Look for Required Python/Cython 
find_package(PythonExtensions REQUIRED)
find_package(Cython REQUIRED)

#-- Look for MKL
#set(BLA_VENDOR intel10_64lp) 
find_package(BLAS REQUIRED)

if (BLAS_FOUND)
    link_libraries(${BLAS_LIBRARIES})
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${BLAS_LINKER_FLAGS} -mkl_rt")
endif()


find_package(MPI REQUIRED)

include_directories(SYSTEM ${MPI_C_INCLUDE_PATH})

#--
find_package(OpenMP)
if(OpenMP_FOUND)
    set ( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set ( CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

#-- Install GSKNN
#find_package(GSKNN)
add_subdirectory(extern/gsknn)

#-- Install CPU Dense & Sparse
add_subdirectory(src/pyrknn/kernels/gpu/impl/readSVM)
add_subdirectory(src/pyrknn/kernels/cpu/impl/exact)
add_subdirectory(src/pyrknn/kernels/cpu/impl/sparse)
add_subdirectory(src/pyrknn/kernels/cpu)
