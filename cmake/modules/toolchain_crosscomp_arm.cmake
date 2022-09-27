# Copyright (C) 2018-2022, Xilinx Inc.
# Copyright (C) 2022, Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

#     Author: Kristof Denolf <kristof@xilinx.com>
#     Date:   2018/9/23

# cmake -DCMAKE_TOOLCHAIN_FILE=toolchain_crosscomp_arm.cmake ..
#  -DARM_SYSROOTSysroot="absolute path to the sysroot folder"

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_LIST_DIR})

#set(ARM_SYSROOT /group/xrlabs2/platforms/vck190_air_prod_2021.2_sysroot CACHE STRING "" FORCE)

# give the system information
SET (CMAKE_SYSTEM_NAME Linux)
SET (CMAKE_SYSTEM_PROCESSOR aarch64)

# specify the cross compiler
set(CLANG_VER 8)
set(CMAKE_C_COMPILER clang-${CLANG_VER})
set(CMAKE_CXX_COMPILER clang++-${CLANG_VER})
set(CMAKE_ASM_COMPILER clang-${CLANG_VER})
set(CMAKE_STRIP llvm-strip)
set(CLANG_LLD lld CACHE STRING "" FORCE)

set(CMAKE_SHARED_LINKER_FLAGS "-Wl,-z,notext -fuse-ld=lld -Wl,-rpath-link=${ARM_SYSROOT}/usr/lib/gcc/aarch64-linux-gnu/7" CACHE STRING "" FORCE)
set(CMAKE_EXE_LINKER_FLAGS "-Wl,-z,notext -fuse-ld=lld -Wl,-rpath-link=${ARM_SYSROOT}/usr/lib/gcc/aarch64-linux-gnu/7" CACHE STRING "" FORCE)
set(CMAKE_C_FLAGS "-Wl,-z,notext --sysroot=${ARM_SYSROOT} --target=aarch64-linux-gnu -fuse-ld=lld -Wno-unused-command-line-argument" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS}" CACHE STRING "" FORCE)
set(CMAKE_ASM_FLAGS "${CMAKE_C_FLAGS}" CACHE STRING "" FORCE)

# set up cross compilation paths
set(CMAKE_SYSROOT ${ARM_SYSROOT})
set(CMAKE_FIND_ROOT_PATH  ${ARM_SYSROOT})
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
# Ensure that we build relocatable binaries
set(CMAKE_INSTALL_RPATH $ORIGIN/../lib/)
set(CMAKE_LIBRARY_PATH ${ARM_SYSROOT}/usr/lib)
set(CMAKE_INCLUDE_PATH ${ARM_SYSROOT}/usr/)
# adjust the default behavior of the find commands:
# search headers and libraries in the target environment
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
# search programs in the host environment
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)

# set ACDC specifics for a cross compilation
set(CMAKE_BUILD_TYPE MinSizeRel CACHE STRING "build type" FORCE)
set(LLVM_TARGETS_TO_BUILD "ARM;AArch64;" CACHE STRING "Semicolon-separated list of LLVM targets" FORCE)
set(MLIR_BINDINGS_PYTHON_ENABLED ON CACHE BOOL "" FORCE)
set(CMAKE_C_IMPLICIT_LINK_LIBRARIES gcc_s CACHE STRING "" FORCE) 
set(CMAKE_CXX_IMPLICIT_LINK_LIBRARIES gcc_s CACHE STRING "" FORCE)
set(LLVM_ENABLE_PIC True CACHE BOOL "" FORCE)
set(MLIR_BUILD_UTILS ON CACHE BOOL "" FORCE)
set(MLIR_INCLUDE_TESTS ON CACHE BOOL "" FORCE)
set(MLIR_INCLUDE_INTEGRATION_TESTS OFF CACHE BOOL "" FORCE)
set(LINKER_SUPPORTS_COLOR_DIAGNOSTICS OFF CACHE BOOL "" FORCE)
set(LLVM_ENABLE_TERMINFO OFF CACHE BOOL "" FORCE)
set(LLVM_DEFAULT_TARGET_TRIPLE aarch64-linux-gnu CACHE STRING "" FORCE)
set(LLVM_TARGET_ARCH AArch64 CACHE STRING "" FORCE)
set(LLVM_ENABLE_ASSERTIONS ON CACHE BOOL "" FORCE)

# # Python
# We have to explicitly set this extension.  Normally it would be determined by FindPython3, but
# it's inference mechanism doesn't work when cross-compiling
set(PYTHON_MODULE_EXTENSION ".cpython-38-aarch64-linux-gnu.so")
set(Python3_ROOT_DIR ${ARM_SYSROOT}/bin)

set(Python_ROOT ${ARM_SYSROOT}/usr/local/lib/python3.8/dist-packages)
set(Python3_NumPy_INCLUDE_DIR ${Python_ROOT}/numpy/ CACHE STRING "" FORCE)
set(pybind11_DIR ${Python_ROOT}/pybind11/share/cmake/pybind11 CACHE STRING "" FORCE)
set(Torch_DIR ${Python_ROOT}/torch/share/cmake/Torch CACHE STRING "" FORCE)
