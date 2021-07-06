
find_package(BLAS REQUIRED)
find_package(MPI REQUIRED)


include_directories(SYSTEM ${MPI_C_INCLUDE_PATH})

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

file(GLOB EXACT_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB EXACT_HDR ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

add_library(exact SHARED ${EXACT_SRC})

target_include_directories(exact PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(exact PROPERTIES PUBLIC_HEADER "${EXACT_HDR}")
set_target_properties(exact PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(exact PUBLIC ${EXACT_HDR})
find_package(Eigen3)
target_include_directories(exact PUBLIC ../../../../../../extern/eigen/)
target_include_directories(exact PUBLIC ../../../../../../extern/eigen/Eigen/)

install(
        TARGETS exact
        LIBRARY
            DESTINATION lib
        PUBLIC_HEADER
            DESTINATION include
    )



