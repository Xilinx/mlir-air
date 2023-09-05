# Copyright (C) 2022, Xilinx Inc. All rights reserved.
# Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

add_mlir_library(
  AIRTargets
  AIRTargets.cpp
  AIRHerdToJSON.cpp

  LINK_LIBS PUBLIC
  AIRRtDialect
  AIRDialect
  MLIRIR
  MLIRSupport
  MLIRTransforms)
