# Copyright (C) 2022, Xilinx Inc. All rights reserved.
# Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

add_subdirectory(lib)
add_subdirectory(include)
add_subdirectory(test)

if(NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
  install(
    DIRECTORY include/air include/air-c
    DESTINATION include
    COMPONENT air-headers
    FILES_MATCHING
    PATTERN "*.def"
    PATTERN "*.h"
    PATTERN "*.inc"
    PATTERN "*.td"
    PATTERN "LICENSE.TXT")

  install(
    DIRECTORY ${PROJECT_BINARY_DIR}/include/air
    DESTINATION include
    COMPONENT air-headers
    FILES_MATCHING
    PATTERN "*.def"
    PATTERN "*.h"
    PATTERN "*.gen"
    PATTERN "*.inc"
    PATTERN "*.td"
    PATTERN "CMakeFiles" EXCLUDE
    PATTERN "config.h" EXCLUDE)

  if(NOT LLVM_ENABLE_IDE)
    add_llvm_install_targets(install-air-headers DEPENDS air-headers COMPONENT
                             air-headers)
  endif()
endif()
