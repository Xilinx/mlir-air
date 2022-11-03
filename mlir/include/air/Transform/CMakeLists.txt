# Copyright (C) 2022, Xilinx Inc. All rights reserved.
# Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name AIRTransform)
add_public_tablegen_target(AIRTransformPassIncGen)

add_mlir_doc(Passes AIRTransformPasses ./ -gen-pass-doc)