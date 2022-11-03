# Copyright (C) 2018-2022, Xilinx Inc. All rights reserved.
# Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

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
