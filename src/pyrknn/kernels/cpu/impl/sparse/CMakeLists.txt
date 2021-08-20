
find_package(BLAS REQUIRED)
find_package(MPI REQUIRED)


#include_directories(SYSTEM ${MPI_C_INCLUDE_PATH})

find_package(OpenMP)
if(OpenMP_FOUND)
    set ( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set ( CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()



if (BLAS_FOUND)
    link_libraries(${BLAS_LIBRARIES})
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${BLAS_LINKER_FLAGS}")
endif()

file(GLOB SPARSE_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB SPARSE_HDR ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

add_library(sparse SHARED ${SPARSE_SRC})

target_include_directories(sparse PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(sparse PROPERTIES PUBLIC_HEADER "${SPARSE_HDR}")
set_target_properties(sparse PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(sparse PUBLIC ${SPARSE_HDR})

target_include_directories(sparse PUBLIC ../../../../../../extern/eigen/)
target_include_directories(sparse PUBLIC ../../../../../../extern/eigen/Eigen/)
target_include_directories(sparse PUBLIC ../../../gpu/impl/readSVM)
#target_include_directories(sparse PUBLIC ../exact)

target_link_libraries(sparse readSVM)
target_link_libraries(sparse exact)

install(
        TARGETS sparse
        LIBRARY
            DESTINATION lib
        PUBLIC_HEADER
            DESTINATION include
    )



