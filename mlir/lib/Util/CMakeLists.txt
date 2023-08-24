# Copyright (C) 2022, Xilinx Inc. All rights reserved. Copyright (C) 2022,
# Advanced Micro Devices, Inc. All rights reserved. SPDX-License-Identifier: MIT

add_mlir_library(
  AIRUtil
  Util.cpp
  Outliner.cpp
  CostModel.cpp
  Runner.cpp
  Dependency.cpp
  LINK_LIBS
  PUBLIC
  MLIRIR
  MLIRTransforms)
