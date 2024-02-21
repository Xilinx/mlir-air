//===- air.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2020-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

module {

func.func @graph(%arg0 : memref<256xi32>) -> () {
  %herd_cols = arith.constant 1 : index
  %herd_rows = arith.constant 1 : index
  air.herd tile(%tx, %ty) in (%size_x = %herd_cols, %size_y = %herd_rows) args(%ext0 = %arg0) : memref<256xi32>attributes { sym_name="segment_0"} {
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %buf0 = memref.alloc() : memref<256xi32, 2>
    air.dma_memcpy_nd (%buf0[%c0][%c256][%c0], %ext0[%c0][%c256][%c0]) {id = 1 : i32} : (memref<256xi32, 2>, memref<256xi32>)
    memref.dealloc %buf0 : memref<256xi32, 2>
    air.herd_terminator
  }
  return
}

}


