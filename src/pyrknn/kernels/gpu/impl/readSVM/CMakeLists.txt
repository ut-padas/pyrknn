
find_package(BLAS REQUIRED)
find_package(MPI REQUIRED)


include_directories(SYSTEM ${MPI_C_INCLUDE_PATH})



if (BLAS_FOUND)
    link_libraries(${BLAS_LIBRARIES})
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${BLAS_LINKER_FLAGS}")
endif()

file(GLOB SVM_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB SVM_HDR ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

add_library(readSVM SHARED ${SVM_SRC})

target_include_directories(readSVM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(readSVM PROPERTIES PUBLIC_HEADER "${SVM_HDR}")
set_target_properties(readSVM PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(readSVM PUBLIC ${SVM_HDR})
target_include_directories(readSVM PUBLIC ../../../../../../extern/eigen/)
target_include_directories(readSVM PUBLIC ../../../../../../extern/eigen/Eigen/)

install(
        TARGETS readSVM
        LIBRARY
            DESTINATION lib
        PUBLIC_HEADER
            DESTINATION include
    )



