# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.19

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /scratch1/06081/wlruys/miniconda3/bin/cmake

# The command to remove a file.
RM = /scratch1/06081/wlruys/miniconda3/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /work/06081/wlruys/frontera/test_pyrknn/pyrknn/src/pyrknn/kernels/cpu/impl/sparse

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /work/06081/wlruys/frontera/test_pyrknn/pyrknn/src/pyrknn/kernels/cpu/impl/sparse/build

# Include any dependencies generated for this target.
include CMakeFiles/sparse.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/sparse.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/sparse.dir/flags.make

CMakeFiles/sparse.dir/backend_mkl.o: CMakeFiles/sparse.dir/flags.make
CMakeFiles/sparse.dir/backend_mkl.o: ../backend_mkl.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/work/06081/wlruys/frontera/test_pyrknn/pyrknn/src/pyrknn/kernels/cpu/impl/sparse/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/sparse.dir/backend_mkl.o"
	/opt/intel/compilers_and_libraries_2018.5.274/linux/bin/intel64/icpc $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sparse.dir/backend_mkl.o -c /work/06081/wlruys/frontera/test_pyrknn/pyrknn/src/pyrknn/kernels/cpu/impl/sparse/backend_mkl.cpp

CMakeFiles/sparse.dir/backend_mkl.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sparse.dir/backend_mkl.i"
	/opt/intel/compilers_and_libraries_2018.5.274/linux/bin/intel64/icpc $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /work/06081/wlruys/frontera/test_pyrknn/pyrknn/src/pyrknn/kernels/cpu/impl/sparse/backend_mkl.cpp > CMakeFiles/sparse.dir/backend_mkl.i

CMakeFiles/sparse.dir/backend_mkl.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sparse.dir/backend_mkl.s"
	/opt/intel/compilers_and_libraries_2018.5.274/linux/bin/intel64/icpc $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /work/06081/wlruys/frontera/test_pyrknn/pyrknn/src/pyrknn/kernels/cpu/impl/sparse/backend_mkl.cpp -o CMakeFiles/sparse.dir/backend_mkl.s

CMakeFiles/sparse.dir/driver.o: CMakeFiles/sparse.dir/flags.make
CMakeFiles/sparse.dir/driver.o: ../driver.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/work/06081/wlruys/frontera/test_pyrknn/pyrknn/src/pyrknn/kernels/cpu/impl/sparse/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/sparse.dir/driver.o"
	/opt/intel/compilers_and_libraries_2018.5.274/linux/bin/intel64/icpc $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sparse.dir/driver.o -c /work/06081/wlruys/frontera/test_pyrknn/pyrknn/src/pyrknn/kernels/cpu/impl/sparse/driver.cpp

CMakeFiles/sparse.dir/driver.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sparse.dir/driver.i"
	/opt/intel/compilers_and_libraries_2018.5.274/linux/bin/intel64/icpc $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /work/06081/wlruys/frontera/test_pyrknn/pyrknn/src/pyrknn/kernels/cpu/impl/sparse/driver.cpp > CMakeFiles/sparse.dir/driver.i

CMakeFiles/sparse.dir/driver.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sparse.dir/driver.s"
	/opt/intel/compilers_and_libraries_2018.5.274/linux/bin/intel64/icpc $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /work/06081/wlruys/frontera/test_pyrknn/pyrknn/src/pyrknn/kernels/cpu/impl/sparse/driver.cpp -o CMakeFiles/sparse.dir/driver.s

CMakeFiles/sparse.dir/spknn.o: CMakeFiles/sparse.dir/flags.make
CMakeFiles/sparse.dir/spknn.o: ../spknn.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/work/06081/wlruys/frontera/test_pyrknn/pyrknn/src/pyrknn/kernels/cpu/impl/sparse/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/sparse.dir/spknn.o"
	/opt/intel/compilers_and_libraries_2018.5.274/linux/bin/intel64/icpc $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sparse.dir/spknn.o -c /work/06081/wlruys/frontera/test_pyrknn/pyrknn/src/pyrknn/kernels/cpu/impl/sparse/spknn.cpp

CMakeFiles/sparse.dir/spknn.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sparse.dir/spknn.i"
	/opt/intel/compilers_and_libraries_2018.5.274/linux/bin/intel64/icpc $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /work/06081/wlruys/frontera/test_pyrknn/pyrknn/src/pyrknn/kernels/cpu/impl/sparse/spknn.cpp > CMakeFiles/sparse.dir/spknn.i

CMakeFiles/sparse.dir/spknn.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sparse.dir/spknn.s"
	/opt/intel/compilers_and_libraries_2018.5.274/linux/bin/intel64/icpc $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /work/06081/wlruys/frontera/test_pyrknn/pyrknn/src/pyrknn/kernels/cpu/impl/sparse/spknn.cpp -o CMakeFiles/sparse.dir/spknn.s

CMakeFiles/sparse.dir/timer.o: CMakeFiles/sparse.dir/flags.make
CMakeFiles/sparse.dir/timer.o: ../timer.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/work/06081/wlruys/frontera/test_pyrknn/pyrknn/src/pyrknn/kernels/cpu/impl/sparse/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/sparse.dir/timer.o"
	/opt/intel/compilers_and_libraries_2018.5.274/linux/bin/intel64/icpc $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sparse.dir/timer.o -c /work/06081/wlruys/frontera/test_pyrknn/pyrknn/src/pyrknn/kernels/cpu/impl/sparse/timer.cpp

CMakeFiles/sparse.dir/timer.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sparse.dir/timer.i"
	/opt/intel/compilers_and_libraries_2018.5.274/linux/bin/intel64/icpc $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /work/06081/wlruys/frontera/test_pyrknn/pyrknn/src/pyrknn/kernels/cpu/impl/sparse/timer.cpp > CMakeFiles/sparse.dir/timer.i

CMakeFiles/sparse.dir/timer.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sparse.dir/timer.s"
	/opt/intel/compilers_and_libraries_2018.5.274/linux/bin/intel64/icpc $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /work/06081/wlruys/frontera/test_pyrknn/pyrknn/src/pyrknn/kernels/cpu/impl/sparse/timer.cpp -o CMakeFiles/sparse.dir/timer.s

CMakeFiles/sparse.dir/util.o: CMakeFiles/sparse.dir/flags.make
CMakeFiles/sparse.dir/util.o: ../util.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/work/06081/wlruys/frontera/test_pyrknn/pyrknn/src/pyrknn/kernels/cpu/impl/sparse/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/sparse.dir/util.o"
	/opt/intel/compilers_and_libraries_2018.5.274/linux/bin/intel64/icpc $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sparse.dir/util.o -c /work/06081/wlruys/frontera/test_pyrknn/pyrknn/src/pyrknn/kernels/cpu/impl/sparse/util.cpp

CMakeFiles/sparse.dir/util.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sparse.dir/util.i"
	/opt/intel/compilers_and_libraries_2018.5.274/linux/bin/intel64/icpc $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /work/06081/wlruys/frontera/test_pyrknn/pyrknn/src/pyrknn/kernels/cpu/impl/sparse/util.cpp > CMakeFiles/sparse.dir/util.i

CMakeFiles/sparse.dir/util.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sparse.dir/util.s"
	/opt/intel/compilers_and_libraries_2018.5.274/linux/bin/intel64/icpc $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /work/06081/wlruys/frontera/test_pyrknn/pyrknn/src/pyrknn/kernels/cpu/impl/sparse/util.cpp -o CMakeFiles/sparse.dir/util.s

CMakeFiles/sparse.dir/util_eigen.o: CMakeFiles/sparse.dir/flags.make
CMakeFiles/sparse.dir/util_eigen.o: ../util_eigen.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/work/06081/wlruys/frontera/test_pyrknn/pyrknn/src/pyrknn/kernels/cpu/impl/sparse/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/sparse.dir/util_eigen.o"
	/opt/intel/compilers_and_libraries_2018.5.274/linux/bin/intel64/icpc $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sparse.dir/util_eigen.o -c /work/06081/wlruys/frontera/test_pyrknn/pyrknn/src/pyrknn/kernels/cpu/impl/sparse/util_eigen.cpp

CMakeFiles/sparse.dir/util_eigen.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sparse.dir/util_eigen.i"
	/opt/intel/compilers_and_libraries_2018.5.274/linux/bin/intel64/icpc $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /work/06081/wlruys/frontera/test_pyrknn/pyrknn/src/pyrknn/kernels/cpu/impl/sparse/util_eigen.cpp > CMakeFiles/sparse.dir/util_eigen.i

CMakeFiles/sparse.dir/util_eigen.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sparse.dir/util_eigen.s"
	/opt/intel/compilers_and_libraries_2018.5.274/linux/bin/intel64/icpc $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /work/06081/wlruys/frontera/test_pyrknn/pyrknn/src/pyrknn/kernels/cpu/impl/sparse/util_eigen.cpp -o CMakeFiles/sparse.dir/util_eigen.s

# Object files for target sparse
sparse_OBJECTS = \
"CMakeFiles/sparse.dir/backend_mkl.o" \
"CMakeFiles/sparse.dir/driver.o" \
"CMakeFiles/sparse.dir/spknn.o" \
"CMakeFiles/sparse.dir/timer.o" \
"CMakeFiles/sparse.dir/util.o" \
"CMakeFiles/sparse.dir/util_eigen.o"

# External object files for target sparse
sparse_EXTERNAL_OBJECTS =

libsparse.so: CMakeFiles/sparse.dir/backend_mkl.o
libsparse.so: CMakeFiles/sparse.dir/driver.o
libsparse.so: CMakeFiles/sparse.dir/spknn.o
libsparse.so: CMakeFiles/sparse.dir/timer.o
libsparse.so: CMakeFiles/sparse.dir/util.o
libsparse.so: CMakeFiles/sparse.dir/util_eigen.o
libsparse.so: CMakeFiles/sparse.dir/build.make
libsparse.so: /scratch1/06081/wlruys/miniconda3/lib/libmkl_intel_lp64.so
libsparse.so: /scratch1/06081/wlruys/miniconda3/lib/libmkl_intel_thread.so
libsparse.so: /scratch1/06081/wlruys/miniconda3/lib/libmkl_core.so
libsparse.so: /scratch1/06081/wlruys/miniconda3/lib/libiomp5.so
libsparse.so: CMakeFiles/sparse.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/work/06081/wlruys/frontera/test_pyrknn/pyrknn/src/pyrknn/kernels/cpu/impl/sparse/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Linking CXX shared library libsparse.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/sparse.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/sparse.dir/build: libsparse.so

.PHONY : CMakeFiles/sparse.dir/build

CMakeFiles/sparse.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/sparse.dir/cmake_clean.cmake
.PHONY : CMakeFiles/sparse.dir/clean

CMakeFiles/sparse.dir/depend:
	cd /work/06081/wlruys/frontera/test_pyrknn/pyrknn/src/pyrknn/kernels/cpu/impl/sparse/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /work/06081/wlruys/frontera/test_pyrknn/pyrknn/src/pyrknn/kernels/cpu/impl/sparse /work/06081/wlruys/frontera/test_pyrknn/pyrknn/src/pyrknn/kernels/cpu/impl/sparse /work/06081/wlruys/frontera/test_pyrknn/pyrknn/src/pyrknn/kernels/cpu/impl/sparse/build /work/06081/wlruys/frontera/test_pyrknn/pyrknn/src/pyrknn/kernels/cpu/impl/sparse/build /work/06081/wlruys/frontera/test_pyrknn/pyrknn/src/pyrknn/kernels/cpu/impl/sparse/build/CMakeFiles/sparse.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/sparse.dir/depend

