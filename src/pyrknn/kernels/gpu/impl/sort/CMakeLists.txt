
enable_language(CUDA)

file(GLOB SORT_SRC "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")
file(GLOB SORT_HDR ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

add_library(sort SHARED ${SORT_SRC})

set_target_properties(sort PROPERTIES PUBLIC_HEADER "${SORT_HDR}")
set_target_properties(sort PROPERTIES POSITION_INDEPENDENT_CODE ON)
#set_target_properties(sort PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#set_target_properties(sort PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

set(GPU_ARCH $ENV{CUDA_ARCH})
if(GPU_ARCH)
    set_target_properties(sort PROPERTIES CUDA_ARCHITECTURES ${GPU_ARCH})
else()
    set_target_properties(sort PROPERTIES CUDA_ARCHITECTURES OFF)
endif()

target_compile_options(sort PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)

target_include_directories(sort PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(sort PUBLIC ../../../../../../extern/eigen/)
target_include_directories(sort PUBLIC ../../../../../../extern/eigen/Eigen/)
target_include_directories(sort PUBLIC ../../../../../../extern/mgpu/src)

target_include_directories(sort PUBLIC ../util)
target_link_libraries(sort util)

install(
        TARGETS sort
        LIBRARY
            DESTINATION lib
        PUBLIC_HEADER
            DESTINATION include
    )


