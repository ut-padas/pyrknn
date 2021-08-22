
enable_language(CUDA)
set(CUDA_DIR $ENV{CUDA_HOME})

file(GLOB SFIKNN_SRC "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")
file(GLOB SFIKNN_HDR "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp")

add_library(sfiknn SHARED ${SFIKNN_SRC})

set_target_properties(sfiknn PROPERTIES PUBLIC_HEADER "${SFIKNN_HDR}")

set(GPU_ARCH $ENV{CUDA_ARCH})
if(GPU_ARCH)
    set_target_properties(sfiknn PROPERTIES CUDA_ARCHITECTURES ${GPU_ARCH})
else()
    set_target_properties(sfiknn PROPERTIES CUDA_ARCHITECTURES OFF)
endif()

target_include_directories(sfiknn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(sfiknn util)
target_include_directories(sfiknn PUBLIC ${CUDA_HOME}/../samples/common/inc)

install(
        TARGETS sfiknn
        LIBRARY
            DESTINATION lib
        PUBLIC_HEADER
            DESTINATION include
    )


