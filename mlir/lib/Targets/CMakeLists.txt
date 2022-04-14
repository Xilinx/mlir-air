add_air_library(AIRTargets
  AIRTargets.cpp

  LINK_LIBS

  PUBLIC
  AIRRtDialect
  AIRDialect
  MLIRAffine
  MLIRIR
  MLIRLLVMIR
  MLIRMemRef
  MLIRSCF
  MLIRSupport
  MLIRTensor
  MLIRTransforms

  MLIRVector
  )
