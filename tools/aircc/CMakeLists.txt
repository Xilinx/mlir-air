# Copyright (C) 2022, Xilinx Inc. All rights reserved.
# Copyright (C) 2022-2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# --- C++ aircc binary ---

llvm_map_components_to_libnames(llvm_libs support core irreader)

add_llvm_tool(aircc aircc.cpp)
llvm_update_compile_flags(aircc)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}
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
  MLIRRegisterAllDialects
  MLIRRegisterAllExtensions
  MLIRRegisterAllPasses
  MLIRTransforms
  MLIRTransformUtils
  MLIRSupport
  MLIRIR
  ${llvm_libs}
)

if(AIR_ENABLE_AIE)
  list(APPEND LIBS AIE)
endif()

target_link_libraries(aircc PRIVATE ${LIBS})
