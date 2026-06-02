# Copyright (C) 2022, Xilinx Inc. All rights reserved.
# Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

add_subdirectory(CAPI)
add_subdirectory(Conversion)
add_subdirectory(Dialect)
add_subdirectory(Interfaces)
add_subdirectory(Targets)
add_subdirectory(Transform)
add_subdirectory(Util)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(_air_initall_link_libs
  AIRConversionPasses
  AIRTransformPasses
  AIRTransformOps
  AIRDialect
  AIRRtDialect
  AIRUtil
  AIRInterface
  MLIRSupport
  ${conversion_libs}
  ${dialect_libs})

if(AIR_ENABLE_AIE)
  # AIETransforms exposes registerAIEPasses() — wired into
  # registerAllPasses() so air-opt and aircc can invoke aie-place-tiles
  # on the LogicalTileOps emitted by AIR's lowering.
  list(APPEND _air_initall_link_libs AIETransforms)
endif()

add_mlir_library(
  AIRInitAll
  InitAll.cpp

  DEPENDS
  AIRDialect
  AIRRtDialect
  AIRTransformOpsIncGen
  AIRConversionPasses
  AIRTransformPasses
  AIRInterface

  LINK_LIBS
  ${_air_initall_link_libs})
