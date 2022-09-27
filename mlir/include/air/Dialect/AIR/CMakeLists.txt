
# Copyright (C) 2022, Xilinx Inc.
# Copyright (C) 2022, Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

set(dialect AIR)
set(dialect_namespace air)
set(LLVM_TARGET_DEFINITIONS ${dialect}.td)


mlir_tablegen(${dialect}.h.inc -gen-op-decls)
mlir_tablegen(${dialect}.cpp.inc -gen-op-defs)
mlir_tablegen(${dialect}Types.h.inc -gen-typedef-decls)
mlir_tablegen(${dialect}Types.cpp.inc -gen-typedef-defs)
mlir_tablegen(${dialect}Dialect.h.inc -gen-dialect-decls -dialect=${dialect_namespace})
mlir_tablegen(${dialect}Dialect.cpp.inc -gen-dialect-defs -dialect=${dialect_namespace})
mlir_tablegen(${dialect}Enums.h.inc -gen-enum-decls)
mlir_tablegen(${dialect}Enums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRAIRIncGen)

set(LLVM_TARGET_DEFINITIONS AIROpBase.td)
mlir_tablegen(AIROpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(AIROpInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRAIROpInterfacesIncGen)
add_dependencies(mlir-generic-headers MLIRAIROpInterfacesIncGen)

add_mlir_doc(AIR AIRDialect ./ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS AIRTransformOps.td)
mlir_tablegen(AIRTransformOps.h.inc -gen-op-decls)
mlir_tablegen(AIRTransformOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(AIRTransformOpsIncGen)

#add_mlir_doc(AIRTransformOps AIRTransformOps Dialects/ -gen-op-doc)
