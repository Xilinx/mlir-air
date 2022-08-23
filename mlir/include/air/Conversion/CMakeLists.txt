# (c) Copyright 2022 Xilinx Inc. All Rights Reserved.
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name AIRConversion)
add_public_tablegen_target(AIRConversionPassIncGen)

add_mlir_doc(Passes AIRConversionPasses ./ -gen-pass-doc)