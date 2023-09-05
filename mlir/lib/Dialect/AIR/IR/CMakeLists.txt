# Copyright (C) 2022, Xilinx Inc. All rights reserved.
# Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

add_mlir_dialect_library(
  AIRDialect
  AIRDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/mlir/include/air/Dialect/AIR

  DEPENDS
  MLIRAIRIncGen
  MLIRAIROpInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRIR)
