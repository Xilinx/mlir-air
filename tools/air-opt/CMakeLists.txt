# Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
# Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

llvm_map_components_to_libnames(llvm_libs support core irreader)

set(LIBS
  AIRDialect
  AIRRtDialect
  AIRUtil
  AIRConversionPasses
  AIRTransformPasses
  AIRInitAll
  MLIRAffineAnalysis
  MLIRAnalysis
  MLIRDialect
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRSupport
  MLIRIR
  MLIRTestTransformDialect
  MLIRTestTransforms
)

if(BUILD_WITH_AIE)
  list(APPEND LIBS AIE)
  add_definitions(-DBUILD_WITH_AIE)
endif()

# Now build our tools
add_llvm_tool(air-opt air-opt.cpp)
llvm_update_compile_flags(air-opt)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

target_link_libraries(air-opt PRIVATE ${LIBS})
