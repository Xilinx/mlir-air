# Copyright (C) 2022, Xilinx Inc. All rights reserved. Copyright (C) 2022,
# Advanced Micro Devices, Inc. All rights reserved. SPDX-License-Identifier: MIT

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

add_subdirectory(aircc)
add_subdirectory(air-opt)
add_subdirectory(air-translate)
add_subdirectory(air-runner)
