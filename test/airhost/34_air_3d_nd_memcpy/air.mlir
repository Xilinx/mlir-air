//===- air.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2020-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

module {

func.func @graph(%arg0 : memref<16x4x2xi32>) -> () {
  %herd_cols = arith.constant 1 : index
  %herd_rows = arith.constant 1 : index
  air.herd tile(%tx, %ty) in (%size_x = %herd_cols, %size_y = %herd_rows) args(%ext0 = %arg0) : memref<16x4x2xi32> attributes { sym_name="herd_0"} {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c16 = arith.constant 16 : index
    %c4 = arith.constant 4 : index
    %c2 = arith.constant 2 : index
    %buf0 = memref.alloc() : memref<4x2x2xi32, 2>
    air.dma_memcpy_nd (%buf0[%c0, %c0, %c0][%c2, %c2, %c4][%c64, %c16, %c0], %ext0[%c0, %c0, %c0][%c2, %c2, %c4][%c64, %c16, %c0]) {id = 1 : i32} : (memref<4x2x2xi32, 2>, memref<16x4x2xi32>)
    memref.dealloc %buf0 : memref<4x2x2xi32, 2>
    air.herd_terminator
  }
  return
}

}


