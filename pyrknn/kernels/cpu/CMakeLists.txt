


add_cython_target(core core.pyx CXX PY3)
add_library(core MODULE ${core})
#link_libraries(${MPI_C_LIBRARIES})


python_extension_module(core)
install(TARGETS core LIBRARY DESTINATION pyrknn)
target_link_libraries(core gsknn)
target_link_libraries(core ${PYTHON_LIBRARIES})
target_include_directories(core PUBLIC include)
target_include_directories(core PUBLIC impl)
target_include_directories(core PUBLIC impl/exact)
target_include_directories(core PUBLIC impl/sparse)
target_include_directories(core PUBLIC ${PYTHON_INCLUDE_DIRS})
target_include_directories(core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(core ${MPI_C_LIBRARIES})
target_link_libraries(core exact)
target_link_libraries(core sparse)

