
enable_language(CUDA)

#Merge requires util, sort, modern_gpu, extended_lambda

file(GLOB MERGE_SRC "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")
file(GLOB MERGE_HDR ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

add_library(merge SHARED ${MERGE_SRC})

set_target_properties(merge PROPERTIES PUBLIC_HEADER "${MERGE_HDR}")
set_target_properties(merge PROPERTIES POSITION_INDEPENDENT_CODE ON)

#set_target_properties(merge PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#set_target_properties(merge PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

set(GPU_ARCH $ENV{CUDA_ARCH})
if(GPU_ARCH)
    set_target_properties(merge PROPERTIES CUDA_ARCHITECTURES ${GPU_ARCH})
else()
    set_target_properties(merge PROPERTIES CUDA_ARCHITECTURES OFF)
endif()

target_compile_options(merge PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda -DPROD>)

target_include_directories(merge PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(merge PUBLIC ../../../../../../extern/eigen/)
target_include_directories(merge PUBLIC ../../../../../../extern/eigen/Eigen/)
target_include_directories(merge PUBLIC ../../../../../../extern/mgpu/src)
target_link_libraries(merge CUDA::cusparse CUDA::cublas CUDA::cusolver)

target_include_directories(merge PUBLIC ../util)
target_include_directories(merge PUBLIC ../sort)
target_link_libraries(merge util sort)

install(
        TARGETS merge
        LIBRARY
            DESTINATION lib
        PUBLIC_HEADER
            DESTINATION include
    )


