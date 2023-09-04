# Copyright (C) 2022, Xilinx Inc. All rights reserved.
# Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

add_mlir_dialect_library(
  AIRRtDialect
  AIRRtDialect.cpp
  AIRRtOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/mlir/include/air/Dialect/AIRRt

  DEPENDS
  MLIRAIRRtOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR)
