add_air_library(AIRTargets
  AIRTargets.cpp

  LINK_LIBS

  PUBLIC
  AIRRtDialect
  AIRDialect
  MLIRIR
  MLIRSupport
  MLIRTransforms
  )
