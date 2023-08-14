# Copyright (C) 2022, Xilinx Inc. All rights reserved.
# Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name AIRConversion)
add_public_tablegen_target(AIRConversionPassIncGen)

add_mlir_doc(Passes AIRConversionPasses ./ -gen-pass-doc)

if(BUILD_WITH_AIE)
  set(LLVM_TARGET_DEFINITIONS AIRToAIEPasses.td)
  add_definitions(-DBUILD_WITH_AIE)
  mlir_tablegen(AIRToAIEPass.h.inc -gen-pass-decls -name AIRToAIEConversion)
  add_public_tablegen_target(AIRToAIEConversionPassIncGen)
  add_mlir_doc(Passes AIRToAIEConversionPasses ./ -gen-pass-doc)
endif()
