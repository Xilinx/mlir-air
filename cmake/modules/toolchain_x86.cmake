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

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_CURRENT_LIST_DIR}/cmakeModulesXilinx)

#message("toolchain_x86: ACDCSysroot = ${ACDCSysroot}")
#message("toolchain_x86: VitisSysroot = ${VitisSysroot}")
#set(ACDCSysroot /group/xrlabs/platforms/acdc_linux_sysroot CACHE STRING "" FORCE)
#set(VitisSysroot /group/xrlabs/platforms/acdc_linux_sysroot CACHE STRING "" FORCE)

# specify the cross compiler
set(CLANG_VER 8)
set(CMAKE_C_COMPILER clang-${CLANG_VER})
set(CMAKE_CXX_COMPILER clang++-${CLANG_VER})
set(CMAKE_ASM_COMPILER clang-${CLANG_VER})
set(CMAKE_STRIP llvm-strip)
set(CLANG_LLD lld CACHE STRING "" FORCE)

set(CMAKE_BUILD_TYPE Debug CACHE STRING "build type" FORCE)
set(LLVM_ENABLE_ASSERTIONS ON CACHE BOOL "" FORCE)
set(GCC_INSTALL_PREFIX /tools/batonroot/rodin/devkits/lnx64/gcc-8.3.0/ CACHE STRING "" FORCE)
