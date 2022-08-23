# (c) Copyright 2021 Xilinx Inc.

add_subdirectory(CAPI)
add_subdirectory(Conversion)
add_subdirectory(Dialect)
add_subdirectory(Targets)
add_subdirectory(Transform)
add_subdirectory(Util)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_air_library(AIRInitAll
  InitAll.cpp

  DEPENDS
  AIRDialect
  AIRRtDialect

  LINK_LIBS
  AIRConversionPasses
  AIRTransformPasses
  AIRTransformOps
  AIRDialect
  AIRRtDialect
  AIRUtil
  MLIRSupport
  ${conversion_libs}
  ${dialect_libs}
  )