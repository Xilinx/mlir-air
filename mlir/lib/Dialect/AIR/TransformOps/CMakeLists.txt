# Copyright (C) 2022, Xilinx Inc. All rights reserved.
# Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

set(_depends
  AIRTransformOpsIncGen
)

if(BUILD_WITH_AIE)
  list(APPEND _depends AIRToAIETransformOpsIncGen)
  add_definitions(-DBUILD_WITH_AIE)
endif()

add_mlir_dialect_library(AIRTransformOps
  AIRTransformOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/AIR/

  DEPENDS
  ${_depends}

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPDLDialect
  MLIRTransformDialect
)
