
enable_language(CUDA)

file(GLOB GEMM_SRC "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")
file(GLOB GEMM_HDR ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

add_library(gemm SHARED ${GEMM_SRC})

set_target_properties(gemm PROPERTIES PUBLIC_HEADER "${GEMM_HDR}")
set_target_properties(gemm PROPERTIES POSITION_INDEPENDENT_CODE ON)
#set_target_properties(gemm PROPERTIES CUDA_SEPERABLE_COMPILATION ON)
#set_target_properties(gemm PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)


set(GPU_ARCH $ENV{CUDA_ARCH})
if(GPU_ARCH)
    set_target_properties(gemm PROPERTIES CUDA_ARCHITECTURES ${GPU_ARCH})
else()
    set_target_properties(gemm PROPERTIES CUDA_ARCHITECTURES OFF)
endif()


target_include_directories(gemm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(gemm PUBLIC ../../../../../../extern/eigen/)
target_include_directories(gemm PUBLIC ../../../../../../extern/eigen/Eigen/)
target_link_libraries(gemm CUDA::cusparse CUDA::cublas)

target_include_directories(gemm PUBLIC ../util)
target_link_libraries(gemm util)

install(
        TARGETS gemm
        LIBRARY
            DESTINATION lib
        PUBLIC_HEADER
            DESTINATION include
    )


