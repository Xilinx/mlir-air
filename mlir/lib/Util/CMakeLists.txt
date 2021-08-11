add_mlir_library(AIRUtil
  Util.cpp
  Outliner.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRStandard
)