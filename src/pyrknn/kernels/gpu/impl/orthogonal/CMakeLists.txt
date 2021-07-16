
enable_language(CUDA)

file(GLOB ORTH_SRC "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")
file(GLOB ORTH_HDR ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

add_library(orthogonal SHARED ${ORTH_SRC})

set_target_properties(orthogonal PROPERTIES PUBLIC_HEADER "${ORTH_HDR}")
set_target_properties(orthogonal PROPERTIES POSITION_INDEPENDENT_CODE ON)

#set_target_properties(orthogonal PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#set_target_properties(orthogonal PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

set(GPU_ARCH $ENV{CUDA_ARCH})
if(GPU_ARCH)
    set_target_properties(orthogonal PROPERTIES CUDA_ARCHITECTURES ${GPU_ARCH})
else()
    set_target_properties(orthogonal PROPERTIES CUDA_ARCHITECTURES OFF)
endif()

target_include_directories(orthogonal PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(orthogonal PUBLIC ../../../../../../extern/eigen/)
target_include_directories(orthogonal PUBLIC ../../../../../../extern/eigen/Eigen/)
target_link_libraries(orthogonal CUDA::cusparse CUDA::cublas CUDA::cusolver)

target_include_directories(orthogonal PUBLIC ../util)
target_link_libraries(orthogonal util)

install(
        TARGETS orthogonal
        LIBRARY
            DESTINATION lib
        PUBLIC_HEADER
            DESTINATION include
    )


