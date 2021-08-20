
enable_language(CUDA)

#Merge requires util, sort, modern_gpu, extended_lambda

file(GLOB SPARSE_SRC "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")
file(GLOB SPARSE_HDR ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

add_library(spknngpu SHARED ${SPARSE_SRC})

set_target_properties(spknngpu PROPERTIES PUBLIC_HEADER "${SPARSE_HDR}")
set_target_properties(spknngpu PROPERTIES POSITION_INDEPENDENT_CODE ON)

#set_target_properties(spknngpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#set_target_properties(spknngpu PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

set(GPU_ARCH $ENV{CUDA_ARCH})
if(GPU_ARCH)
    set_target_properties(spknngpu PROPERTIES CUDA_ARCHITECTURES ${GPU_ARCH})
else()
    set_target_properties(spknngpu PROPERTIES CUDA_ARCHITECTURES OFF)
endif()

target_compile_options(spknngpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda -DPROD>)

target_include_directories(spknngpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(spknngpu PUBLIC ../../../../../../extern/eigen/)
target_include_directories(spknngpu PUBLIC ../../../../../../extern/eigen/Eigen/)
target_include_directories(spknngpu PUBLIC ../../../../../../extern/mgpu/src)

target_link_libraries(spknngpu CUDA::cusparse CUDA::cublas CUDA::cusolver CUDA::cudart)

target_include_directories(spknngpu PUBLIC ../util)
target_include_directories(spknngpu PUBLIC ../sort)
target_include_directories(spknngpu PUBLIC ../transpose)
target_include_directories(spknngpu PUBLIC ../orthogonal)
target_include_directories(spknngpu PUBLIC ../gemm)
target_include_directories(spknngpu PUBLIC ../reorder)
target_include_directories(spknngpu PUBLIC ../merge)
target_include_directories(spknngpu PUBLIC ../singleton)
target_include_directories(spknngpu PUBLIC ../readSVM)

target_link_libraries(spknngpu sort merge util readSVM transpose gemm reorder orthogonal)

install(
        TARGETS spknngpu
        LIBRARY
            DESTINATION lib
        PUBLIC_HEADER
            DESTINATION include
    )


