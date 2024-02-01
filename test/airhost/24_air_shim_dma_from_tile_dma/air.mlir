//===- air.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2020-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

module {

func.func @graph(%arg1 : memref<256xi32>) -> () {
  %herd_cols = arith.constant 1 : index
  %herd_rows = arith.constant 1 : index
  air.herd tile(%tx, %ty) in (%size_x = %herd_cols, %size_y = %herd_rows) args(%ext1 = %arg1) : memref<256xi32>, memref<256xi32> attributes { sym_name="herd_0"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %buf0 = memref.alloc() : memref<256xi32, 2>
    scf.for %arg2 = %c0 to %c256 step %c1 {
      %1 = memref.load %buf0[%arg2] : memref<256xi32, 2>
      %2 = arith.constant 16 : i32
      %3 = arith.index_cast %arg2 : index to i32
      %5 = arith.addi %2, %3 : i32
      memref.store %5, %buf0[%arg2] : memref<256xi32, 2>
    }
    air.dma_memcpy_nd (%ext1[%c0][%c256][%c1], %buf0[][][]) {id = 1 : i32} : (memref<256xi32>, memref<256xi32, 2>)
    memref.dealloc %buf0 : memref<256xi32, 2>
    air.herd_terminator
  }
  return
}

}
