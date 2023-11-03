# Copyright (C) 2022, Xilinx Inc. All rights reserved.
# Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

set(LLVM_LINK_COMPONENTS Core Support)

# Now build our tools
add_mlir_tool(air-runner air-runner.cpp)
llvm_update_compile_flags(air-runner)

set(LIBS
    AIRDialect
    AIRRtDialect
    AIRUtil
    AIRConversionPasses
    AIRTransformPasses
    AIRInitAll
    MLIRSupport)

target_link_libraries(air-runner PRIVATE ${LIBS})

export_executable_symbols(mlir-cpu-runner)
