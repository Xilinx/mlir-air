# Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# cmake -DCMAKE_TOOLCHAIN_FILE=toolchain_crosscomp_aarch.cmake ..
#  -DCMAKE_SYSROOT="absolute path to the sysroot folder"

#set (CMAKE_SYSROOT /path/to/sysroot)

# give the system information
SET (CMAKE_SYSTEM_NAME Linux)
SET (CMAKE_SYSTEM_PROCESSOR aarch64)

# specify the cross compiler
set(CLANG_VER 10)
set(CMAKE_C_COMPILER clang-${CLANG_VER})
set(CMAKE_CXX_COMPILER clang++-${CLANG_VER})
set(CMAKE_ASM_COMPILER clang-${CLANG_VER})
set(CMAKE_STRIP llvm-strip-${CLANG_VER})
set(CLANG_LLD lld-${CLANG_VER} CACHE STRING "" FORCE)

set(CMAKE_SHARED_LINKER_FLAGS "-fuse-ld=${CLANG_LLD} " CACHE STRING "" FORCE)
set(CMAKE_EXE_LINKER_FLAGS "-fuse-ld=${CLANG_LLD} " CACHE STRING "" FORCE)
set(CMAKE_C_FLAGS "--sysroot=${CMAKE_SYSROOT} --target=aarch64-linux-gnu --gcc-toolchain=${CMAKE_SYSROOT}/usr -fuse-ld=${CLANG_LLD} -Wno-unused-command-line-argument" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS}" CACHE STRING "" FORCE)
set(CMAKE_ASM_FLAGS "${CMAKE_C_FLAGS}" CACHE STRING "" FORCE)

# set up cross compilation paths
set(CMAKE_FIND_ROOT_PATH  ${CMAKE_SYSROOT})
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
# Ensure that we build relocatable binaries
set(CMAKE_INSTALL_RPATH $ORIGIN/../lib/)
set(CMAKE_LIBRARY_PATH ${CMAKE_SYSROOT}/usr/lib)
set(CMAKE_INCLUDE_PATH ${CMAKE_SYSROOT}/usr/)
# adjust the default behavior of the find commands:
# search headers and libraries in the target environment
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
# search programs in the host environment
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)

set(CMAKE_C_IMPLICIT_LINK_LIBRARIES gcc_s CACHE STRING "" FORCE)
set(CMAKE_CXX_IMPLICIT_LINK_LIBRARIES gcc_s CACHE STRING "" FORCE)
set(LINKER_SUPPORTS_COLOR_DIAGNOSTICS OFF CACHE BOOL "" FORCE)

# Python env is detected incorrectly when cross-compiling. Setup manually
set(PYTHON_MODULE_EXTENSION ".cpython-38-aarch64-linux-gnu.so")
set(Python3_ROOT_DIR ${CMAKE_SYSROOT}/bin)
set(Python_ROOT ${CMAKE_SYSROOT}/usr/local/lib/python3.8/dist-packages)
set(Python3_NumPy_INCLUDE_DIR ${Python_ROOT}/numpy/ CACHE STRING "" FORCE)
set(pybind11_DIR ${Python_ROOT}/pybind11/share/cmake/pybind11 CACHE STRING "" FORCE)
