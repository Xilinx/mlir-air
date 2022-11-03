//===- air.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

module {

func.func @graph(%arg0 : memref<32xi32, 1>) -> () {
  %herd_cols = arith.constant 1 : index
  %herd_rows = arith.constant 1 : index
  air.herd tile(%tx, %ty) in (%size_x = %herd_cols, %size_y = %herd_rows) args(%ext0 = %arg0) : memref<32xi32, 1> attributes { sym_name="herd_0"} {
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %buf0 = memref.alloc() : memref<32xi32, 2>
    air.dma_memcpy_nd (%ext0[%c0][%c32][%c0], %buf0[%c0][%c32][%c0]) {id = 1 : i32} : (memref<32xi32, 1>, memref<32xi32, 2>)
    memref.dealloc %buf0 : memref<32xi32, 2>
    air.herd_terminator
  }
  return
}

}


