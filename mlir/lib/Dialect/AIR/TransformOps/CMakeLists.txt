# Copyright (C) 2022, Xilinx Inc. All rights reserved.
# Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

add_mlir_dialect_library(AIRTransformOps
  AIRTransformOps.cpp
  AIRLinearizeVectors.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/AIR/

  DEPENDS
  AIRTransformOpsIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRTransformDialect
  MLIRTransformUtils
  MLIRUBDialect
  MLIRVectorDialect
  MLIRVectorTransforms
  MLIRVectorUtils
)
if(AIR_ENABLE_AIE)
  target_link_libraries(AIRTransformOps PUBLIC MLIRAIEVecDialect)
endif()
