

find_package(CUDAToolkit REQUIRED)

add_compile_options(-DFRONTERA -DPROD)

add_subdirectory(impl/util)
add_subdirectory(impl/gemm)
add_subdirectory(impl/orthogonal)
add_subdirectory(impl/sort)
add_subdirectory(impl/merge)
add_subdirectory(impl/transpose)
add_subdirectory(impl/reorder)
add_subdirectory(impl/dense)
add_subdirectory(impl/sparse)
add_subdirectory(impl/sfiknn)

#-- Build Dense

include_directories(${CUDAToolkit_LIBRARY_DIR})
include_directories(${CUDAToolkit_LIBRARY_ROOT})
include_directories(${CUDAToolkit_INCLUDE_DIRS})
include_directories(${CUDA_HOME}/samples/common/inc)

add_cython_target(core_gpu core_gpu.pyx CXX PY3)
add_library(core_gpu MODULE ${core_gpu})
#link_libraries(${MPI_C_LIBRARIES})


python_extension_module(core_gpu)
install(TARGETS core_gpu LIBRARY DESTINATION ${PYTHON_RELATIVE_SITE_PACKAGES_DIR}/pyrknn/kernels/gpu/)
target_include_directories(core_gpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(core_gpu ${PYTHON_LIBRARIES})
target_include_directories(core_gpu PUBLIC include)
target_include_directories(core_gpu PUBLIC impl)
target_include_directories(core_gpu PUBLIC impl/util)
target_include_directories(core_gpu PUBLIC impl/sort)
target_include_directories(core_gpu PUBLIC impl/gemm)
target_include_directories(core_gpu PUBLIC impl/reorder)
target_include_directories(core_gpu PUBLIC impl/transpose)
target_include_directories(core_gpu PUBLIC impl/dense)
target_include_directories(core_gpu PUBLIC impl/orthogonal)
target_include_directories(core_gpu PUBLIC impl/merge)
target_include_directories(core_gpu PUBLIC impl/readSVM)

target_include_directories(core_gpu PUBLIC ${PYTHON_INCLUDE_DIRS})
target_include_directories(core_gpu PUBLIC ${MPI_C_LIBRARIES})


message("Toolkit Dir" ${CUDAToolkit_LIBRARY_ROOT})

target_link_libraries(core_gpu ${MPI_C_LIBRARIES})

#target_link_libraries(core_gpu CUDA::cusparse CUDA::cusolver CUDA::cublas CUDA::cudart)
target_link_libraries(core_gpu denknngpu merge util readSVM transpose gemm reorder orthogonal)

#- Build Sparse

add_cython_target(core_sparse core_sparse.pyx CXX PY3)
add_library(core_sparse MODULE ${core_sparse})
#link_libraries(${MPI_C_LIBRARIES})


python_extension_module(core_sparse)
install(TARGETS core_sparse LIBRARY DESTINATION ${PYTHON_RELATIVE_SITE_PACKAGES_DIR}/pyrknn/kernels/gpu/)
install(TARGETS core_sparse LIBRARY DESTINATION pyrknn)


target_link_libraries(core_sparse ${PYTHON_LIBRARIES})
target_include_directories(core_sparse PUBLIC include)
target_include_directories(core_sparse PUBLIC impl)
target_include_directories(core_sparse PUBLIC impl/util)
target_include_directories(core_sparse PUBLIC impl/sort)
target_include_directories(core_sparse PUBLIC impl/gemm)
target_include_directories(core_sparse PUBLIC impl/reorder)
target_include_directories(core_sparse PUBLIC impl/transpose)
target_include_directories(core_sparse PUBLIC impl/dense)
target_include_directories(core_sparse PUBLIC impl/orthogonal)
target_include_directories(core_sparse PUBLIC impl/merge)
target_include_directories(core_sparse PUBLIC impl/readSVM)
target_include_directories(core_sparse PUBLIC impl/sfiknn)

target_include_directories(core_sparse PUBLIC ${PYTHON_INCLUDE_DIRS})
target_include_directories(core_sparse PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(core_sparse ${MPI_C_LIBRARIES})
#target_link_libraries(core_sparse CUDA::cusparse CUDA::cusolver CUDA::cublas CUDA::cudart)
target_link_libraries(core_sparse spknngpu merge util readSVM transpose gemm reorder orthogonal sfiknn)

