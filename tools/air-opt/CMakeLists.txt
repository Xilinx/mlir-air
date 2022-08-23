# (c) Copyright 2021 Xilinx Inc.

llvm_map_components_to_libnames(llvm_libs support core irreader)

# Now build our tools
add_llvm_tool(air-opt air-opt.cpp)
llvm_update_compile_flags(air-opt)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LIBS
  AIRDialect
  AIRRtDialect
  AIRUtil
  AIRConversionPasses
  AIRTransformPasses
  AIRInitAll
  AIE
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

target_link_libraries(air-opt PRIVATE ${LIBS})
