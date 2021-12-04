add_mlir_library(AIRUtil
  Util.cpp
  Outliner.cpp
  CostModel.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRStandard
)