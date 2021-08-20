
enable_language(CUDA)

#Merge requires util, sort, modern_gpu, extended_lambda

file(GLOB DENSE_SRC "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")
file(GLOB DENSE_HDR ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

add_library(denknngpu SHARED ${DENSE_SRC})

set_target_properties(denknngpu PROPERTIES PUBLIC_HEADER "${DENSE_HDR}")
set_target_properties(denknngpu PROPERTIES POSITION_INDEPENDENT_CODE ON)

#set_target_properties(denknngpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#set_target_properties(denknngpu PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

set(GPU_ARCH $ENV{CUDA_ARCH})
if(GPU_ARCH)
    set_target_properties(denknngpu PROPERTIES CUDA_ARCHITECTURES ${GPU_ARCH})
else()
    set_target_properties(denknngpu PROPERTIES CUDA_ARCHITECTURES OFF)
endif()

target_compile_options(denknngpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda -DPROD>)
target_compile_options(denknngpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
target_include_directories(denknngpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(denknngpu PUBLIC ../../../../../../extern/eigen/)
target_include_directories(denknngpu PUBLIC ../../../../../../extern/eigen/Eigen/)
target_include_directories(denknngpu PUBLIC ../../../../../../extern/mgpu/src)

target_link_libraries(denknngpu CUDA::cusparse CUDA::cublas CUDA::cusolver CUDA::cudart)

target_include_directories(denknngpu PUBLIC ../util)
target_include_directories(denknngpu PUBLIC ../sort)
target_include_directories(denknngpu PUBLIC ../transpose)
target_include_directories(denknngpu PUBLIC ../orthogonal)
target_include_directories(denknngpu PUBLIC ../gemm)
target_include_directories(denknngpu PUBLIC ../reorder)
target_include_directories(denknngpu PUBLIC ../merge)
target_include_directories(denknngpu PUBLIC ../singleton)
target_include_directories(denknngpu PUBLIC ../readSVM)

target_link_libraries(denknngpu sort merge util reorder orthogonal gemm)

install(
        TARGETS denknngpu
        LIBRARY
            DESTINATION lib
        PUBLIC_HEADER
            DESTINATION include
    )


