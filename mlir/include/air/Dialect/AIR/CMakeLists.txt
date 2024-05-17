# Copyright (C) 2022, Xilinx Inc. All rights reserved.
# Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

add_mlir_dialect(AIR air)
add_mlir_doc(AIR AIRDialect ./ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS AIROpBase.td)
mlir_tablegen(AIROpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(AIROpInterfaces.cpp.inc -gen-op-interface-defs)
mlir_tablegen(AIREnums.h.inc -gen-enum-decls)
mlir_tablegen(AIREnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRAIROpInterfacesIncGen)
add_dependencies(air-headers MLIRAIROpInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS AIRTransformOps.td)
mlir_tablegen(AIRTransformOps.h.inc -gen-op-decls)
mlir_tablegen(AIRTransformOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(AIRTransformOpsIncGen)
add_dependencies(air-headers AIRTransformOpsIncGen)

add_mlir_doc(AIRTransformOps AIRTransformOps ./ -gen-op-doc)
