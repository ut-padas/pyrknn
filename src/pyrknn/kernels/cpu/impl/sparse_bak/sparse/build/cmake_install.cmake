# Install script for directory: /work/06081/wlruys/frontera/test_pyrknn/pyrknn/src/pyrknn/kernels/cpu/impl/sparse

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "0")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/opt/apps/gcc/6.3.0/bin/objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libsparse.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libsparse.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libsparse.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/work/06081/wlruys/frontera/test_pyrknn/pyrknn/src/pyrknn/kernels/cpu/impl/sparse/build/libsparse.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libsparse.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libsparse.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libsparse.so"
         OLD_RPATH "/scratch1/06081/wlruys/miniconda3/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/opt/apps/gcc/6.3.0/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libsparse.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES
    "/work/06081/wlruys/frontera/test_pyrknn/pyrknn/src/pyrknn/kernels/cpu/impl/sparse/NLA.hpp"
    "/work/06081/wlruys/frontera/test_pyrknn/pyrknn/src/pyrknn/kernels/cpu/impl/sparse/matrix.hpp"
    "/work/06081/wlruys/frontera/test_pyrknn/pyrknn/src/pyrknn/kernels/cpu/impl/sparse/merge.hpp"
    "/work/06081/wlruys/frontera/test_pyrknn/pyrknn/src/pyrknn/kernels/cpu/impl/sparse/spknn.hpp"
    "/work/06081/wlruys/frontera/test_pyrknn/pyrknn/src/pyrknn/kernels/cpu/impl/sparse/timer.hpp"
    "/work/06081/wlruys/frontera/test_pyrknn/pyrknn/src/pyrknn/kernels/cpu/impl/sparse/util.hpp"
    "/work/06081/wlruys/frontera/test_pyrknn/pyrknn/src/pyrknn/kernels/cpu/impl/sparse/util_eigen.hpp"
    )
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/work/06081/wlruys/frontera/test_pyrknn/pyrknn/src/pyrknn/kernels/cpu/impl/sparse/build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
