add_subdirectory(CAPI)
add_subdirectory(Conversion)
add_subdirectory(Dialect)
add_subdirectory(Targets)
add_subdirectory(Transform)
add_subdirectory(Util)

add_air_library(AIRInitAll
  InitAll.cpp

  LINK_LIBS
  AIRConversionPasses
  AIRTransformPasses
  AIRDialect
  AIRRtDialect
  AIRUtil
  MLIR
  )