
enable_language(CUDA)

file(GLOB FISKNN_SRC "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")
file(GLOB FISKNN_HDR "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp")

add_library(fisknn SHARED ${FISKNN_SRC})

set_target_properties(fisknn PROPERTIES PUBLIC_HEADER "${FISKNN_HDR}")

set(GPU_ARCH $ENV{CUDA_ARCH})
if(GPU_ARCH)
    set_target_properties(fisknn PROPERTIES CUDA_ARCHITECTURES ${GPU_ARCH})
else()
    set_target_properties(fisknn PROPERTIES CUDA_ARCHITECTURES OFF)
endif()

target_include_directories(fisknn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(fisknn util)

install(
        TARGETS fisknn
        LIBRARY
            DESTINATION lib
        PUBLIC_HEADER
            DESTINATION include
    )


