//===- air.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2020-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

module {

func.func @foo(%arg0: i32) {
  %cst1 = arith.constant 1 : index
  air.herd tile(%tx, %ty) in (%size_x = %cst1, %size_y = %cst1) attributes {sym_name="cowfactory", link_with="beefmaker_kernel.o"} {
    %src0 = memref.alloc() {sym_name="beef"}: memref<1024xi32, 2>
    func.call @beefmaker_kernel(%src0) : (memref<1024xi32, 2>) -> ()
    air.herd_terminator
  }
  return
} 

func.func private @beefmaker_kernel(%A: memref<1024xi32, 2>) -> ()

}
