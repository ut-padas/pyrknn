
enable_language(CUDA)

#Reorder requires util and gemm

file(GLOB REORDER_SRC "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")
file(GLOB REORDER_HDR ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

add_library(reorder SHARED ${REORDER_SRC})

set_target_properties(reorder PROPERTIES PUBLIC_HEADER "${REORDER_HDR}")
set_target_properties(reorder PROPERTIES POSITION_INDEPENDENT_CODE ON)

#set_target_properties(reorder PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#set_target_properties(reorder PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

set(GPU_ARCH $ENV{CUDA_ARCH})
if(GPU_ARCH)
    set_target_properties(reorder PROPERTIES CUDA_ARCHITECTURES ${GPU_ARCH})
else()
    set_target_properties(reorder PROPERTIES CUDA_ARCHITECTURES OFF)
endif()

target_include_directories(reorder PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(reorder PUBLIC ../../../../../../extern/eigen/)
target_include_directories(reorder PUBLIC ../../../../../../extern/eigen/Eigen/)
target_link_libraries(reorder CUDA::cusparse CUDA::cublas CUDA::cusolver)

target_include_directories(reorder PUBLIC ../util)
target_include_directories(reorder PUBLIC ../gemm)
target_link_libraries(reorder util gemm)

install(
        TARGETS reorder
        LIBRARY
            DESTINATION lib
        PUBLIC_HEADER
            DESTINATION include
    )


