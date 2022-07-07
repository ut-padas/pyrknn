cmake_minimum_required(VERSION 3.13.2)

set(CMAKE_CXX_STANDARD 17)

#--Setup Project
project(pyrknn LANGUAGES C CXX)

#-- Define Macros

macro(subdirlist result curdir)
    file(GLOB children RELATIVE ${curdir} ${curdir}/*)
    set(dirlist "")
    foreach(child ${children})
        if(IS_DIRECTORY ${curdir}/${child})
            list(APPEND dirlist ${child})
        endif()
    endforeach()
    set(${result} ${dirlist})
endmacro()

macro(descend_headers return_list)
    file(GLOB_RECURSE new_list *.h)
    set(dir_list "")
    foreach(file_path ${new_list})
        get_filename_component(dir_Path ${file_path} PATH)
        set(dir_list ${dir_list} ${dir_path})
    endforeach()
    list(REMOVE_DUPLICATES dir_list)
    set(${return_list} ${dir_list})
endmacro()

macro(descend_shared return_list)
    file(GLOB_RECURSE new_list *.so)
    set(dir_list "")
    foreach(file_path ${new_list})
        get_filename_component(dir_Path ${file_path} PATH)
        set(dir_list ${dir_list} ${dir_path})
    endforeach()
    list(REMOVE_DUPLICATES dir_list)
    set(${return_list} ${dir_list})
endmacro()

macro(descend_static return_list)
    file(GLOB_RECURSE new_list *.a)
    set(dir_list "")
    foreach(file_path ${new_list})
        get_filename_component(dir_Path ${file_path} PATH)
        set(dir_list ${dir_list} ${dir_path})
    endforeach()
    list(REMOVE_DUPLICATES dir_list)
    set(${return_list} ${dir_list})
endmacro()

#-- Look for Required Python/Cython
find_package(PythonExtensions REQUIRED)
find_package(Cython REQUIRED)
#find_package(NumPy REQUIRED)


set(Python3_ROOT_DIR "${CONDA_PREFIX}")
set(Python3_ROOT "${CONDA_PREFIX}")
set(Python3_FIND_STRATEGY "LOCATION")
find_package(Python3 REQUIRED COMPONENTS Interpreter NumPy)
#include_directories(${NumPy_INCLUDE_DIRS})
include_directories(${_Python3_NumPy_INCLUDE_DIR})

#-- Look for MKL
#set(BLA_VENDOR intel10_64lp)
#TODO: Ensure MKL is found (not openblas due to sparse library requirement)
find_package(BLAS REQUIRED)

if (BLAS_FOUND)
    link_libraries(${BLAS_LIBRARIES})
    #This is still missing something.... currently need to Preload with mkl Python modules.
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${BLAS_LINKER_FLAGS}")
endif()

#-- Look for MPI
# Note: This is currenntly required
find_package(MPI REQUIRED)

#--
find_package(OpenMP)
if(OpenMP_FOUND)
    set ( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set ( CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    include_directories(${OpenMP_CXX_INCLUDE_DIRS})
endif()

#-- Install GSKNN

#find_package(GSKNN)
if(USE_GSKNN)
    set(CYTHON_FLAGS "-E USE_GSKNN=True") 
    add_subdirectory(extern/gsknn)
else()
    set(CYTHON_FLAGS "-E USE_GSKNN=False")
endif()


#-- Install CPU Dense & Sparse
add_subdirectory(src/pyrknn/kernels/gpu/impl/readSVM)
add_subdirectory(src/pyrknn/kernels/cpu)

#--Install GPU Dense & Sparse
if(PYRKNN_USE_CUDA)
    enable_language(CUDA)
    add_subdirectory(src/pyrknn/kernels/gpu)
endif()







