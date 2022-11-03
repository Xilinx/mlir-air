//===- air.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

module {

func.func @graph(%arg0 : memref<16x16xi32>, %arg1 : memref<16x16xi32>, %arg2 : memref<4x8xi32, 1>) -> () {
  %herd_cols = arith.constant 1 : index
  %herd_rows = arith.constant 1 : index
  air.herd tile(%tx, %ty) in (%size_x = %herd_cols, %size_y = %herd_rows) args(%ext0 = %arg0, %ext1 = %arg1, %sp0 = %arg2) : memref<16x16xi32>, memref<16x16xi32>, memref<4x8xi32, 1> attributes { sym_name="herd_0"} {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    air.dma_memcpy_nd (%sp0[%c0,%c0][%c8,%c4][%c4,%c0], %ext0[%c0,%c0][%c8,%c4][%c16,%c0]) {id = 1 : i32} : (memref<4x8xi32, 1>, memref<16x16xi32>)
    air.dma_memcpy_nd (%ext1[%c0,%c0][%c8,%c4][%c16,%c0], %sp0[%c0,%c0][%c8,%c4][%c4,%c0]) {id = 2 : i32} : (memref<16x16xi32>, memref<4x8xi32, 1>)
    air.herd_terminator
  }
  return
}

}


