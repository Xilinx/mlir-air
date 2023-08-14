# Copyright (C) 2022, Xilinx Inc. All rights reserved.
# Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
set(LLVM_OPTIONAL_SOURCES AIRtoAIETransformOps.cpp)

set(_sources
  AIRTransformOps.cpp
)
if(BUILD_WITH_AIE)
  list(APPEND _sources AIRtoAIETransformOps.cpp)
endif()

add_mlir_dialect_library(AIRTransformOps
  ${_sources}

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/AIR/

  DEPENDS
  AIRTransformOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPDLDialect
  MLIRTransformDialect
)
