# Copyright (C) 2022, Xilinx Inc. All rights reserved.
# Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

llvm_map_components_to_libnames(llvm_libs support core irreader)

# Now build our tools
add_llvm_tool(air-runner air-runner.cpp)
llvm_update_compile_flags(air-runner)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LIBS
AIRDialect
AIRRtDialect
AIRUtil
AIRConversionPasses
AIRTransformPasses
AIRInitAll
)

target_link_libraries(air-runner PRIVATE ${LIBS})
