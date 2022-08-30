

add_compile_options(-DFRONTERA)

add_subdirectory(impl/exact)

if(BUILD_SPARSE)
    add_subdirectory(impl/sparse)
endif()

find_package(NumPy REQUIRED)

set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c++17")
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

add_cython_target(core core.pyx CXX PY3)
add_library(core MODULE ${core})
#link_libraries(${MPI_C_LIBRARIES})


python_extension_module(core)
install(TARGETS core LIBRARY DESTINATION ${PYTHON_RELATIVE_SITE_PACKAGES_DIR}/pyrknn/kernels/cpu/)
#install(TARGETS core LIBRARY DESTINATION pyrknn)

if(USE_GSKNN)
    target_link_libraries(core gsknn)
endif()

target_link_libraries(core ${PYTHON_LIBRARIES})
target_link_libraries(core ${MPI_C_LIBRARIES})
target_include_directories(core PUBLIC ${NumPy_INCLUDE_DIRS})
target_include_directories(core PUBLIC include)
target_include_directories(core PUBLIC impl)
target_include_directories(core PUBLIC impl/exact)
if(BUILD_SPARSE)
    target_include_directories(core PUBLIC impl/sparse)
endif()
target_include_directories(core PUBLIC ${PYTHON_INCLUDE_DIRS})
target_include_directories(core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(core PUBLIC ${MPI_C_LIBRARIES})

target_link_libraries(core ${MPI_C_LIBRARIES})

target_link_libraries(core exact)

if(BUILD_SPARSE)
    target_link_libraries(core sparse)
endif()

#endif()
