# (c) Copyright 2022 Xilinx Inc. All Rights Reserved.

add_mlir_library(AIRUtil
  Util.cpp
  Outliner.cpp
  CostModel.cpp
  Runner.cpp
  Dependency.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRTransforms
)