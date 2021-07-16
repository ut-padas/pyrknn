
enable_language(CUDA)

file(GLOB UTIL_SRC "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")
file(GLOB UTIL_HDR ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

add_library(util SHARED ${UTIL_SRC})

set_target_properties(util PROPERTIES PUBLIC_HEADER "${util_HDR}")
set_target_properties(util PROPERTIES POSITION_INDEPENDENT_CODE ON)
#set_target_properties(util PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#set_target_properties(util PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

set(GPU_ARCH $ENV{CUDA_ARCH})
if(GPU_ARCH)
    set_target_properties(util PROPERTIES CUDA_ARCHITECTURES ${GPU_ARCH})
else()
    set_target_properties(util PROPERTIES CUDA_ARCHITECTURES OFF)
endif()


target_include_directories(util PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(util PUBLIC ../../../../../../extern/eigen/)
target_include_directories(util PUBLIC ../../../../../../extern/eigen/Eigen/)

install(
        TARGETS util
        LIBRARY
            DESTINATION lib
        PUBLIC_HEADER
            DESTINATION include
    )


