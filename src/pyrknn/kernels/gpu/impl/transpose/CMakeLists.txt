
enable_language(CUDA)

file(GLOB TRANS_SRC "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")
file(GLOB TRANS_HDR ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

add_library(transpose SHARED ${TRANS_SRC})

set_target_properties(transpose PROPERTIES PUBLIC_HEADER "${TRANS_HDR}")
set_target_properties(transpose PROPERTIES POSITION_INDEPENDENT_CODE ON)

#set_target_properties(transpose PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#set_target_properties(transpose PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

set(GPU_ARCH $ENV{CUDA_ARCH})
if(GPU_ARCH)
    set_target_properties(transpose PROPERTIES CUDA_ARCHITECTURES ${GPU_ARCH})
else()
    set_target_properties(transpose PROPERTIES CUDA_ARCHITECTURES OFF)
endif()

target_include_directories(transpose PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(transpose PUBLIC ../../../../../../extern/eigen/)
target_include_directories(transpose PUBLIC ../../../../../../extern/eigen/Eigen/)

target_include_directories(transpose PUBLIC ../util)
target_link_libraries(transpose util)

install(
        TARGETS transpose
        LIBRARY
            DESTINATION lib
        PUBLIC_HEADER
            DESTINATION include
    )


