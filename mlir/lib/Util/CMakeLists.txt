add_mlir_library(AIRUtil
  Util.cpp
  Outliner.cpp
  CostModel.cpp
  Runner.cpp

  LINK_LIBS PUBLIC
  MLIRAffine
  MLIRLinalg
  MLIRIR
  MLIRTransforms
)