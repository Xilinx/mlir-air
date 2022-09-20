###############################################################################
#  Copyright (c) 2019, Xilinx, Inc.
#  All rights reserved.
# 
#  Redistribution and use in source and binary forms, with or without 
#  modification, are permitted provided that the following conditions are met:
#
#  1.  Redistributions of source code must retain the above copyright notice, 
#     this list of conditions and the following disclaimer.
#
#  2.  Redistributions in binary form must reproduce the above copyright 
#      notice, this list of conditions and the following disclaimer in the 
#      documentation and/or other materials provided with the distribution.
#
#  3.  Neither the name of the copyright holder nor the names of its 
#      contributors may be used to endorse or promote products derived from 
#      this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, 
#  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR 
#  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR 
#  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, 
#  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
#  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
#  OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
#  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR 
#  OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF 
#  ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
###############################################################################

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
