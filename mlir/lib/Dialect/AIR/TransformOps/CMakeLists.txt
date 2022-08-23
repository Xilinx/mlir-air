# (c) Copyright 2022 Xilinx Inc.

add_mlir_dialect_library(AIRTransformOps
  AIRTransformOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/AIR/

  DEPENDS
  AIRTransformOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPDLDialect
  MLIRTransformDialect
)
