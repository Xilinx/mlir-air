//===- air.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2020-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

func.func @graph(%arg0: memref<256xi32>) {
  %c1 = arith.constant 1 : index
  air.herd @herd_0  tile (%arg1, %arg2) in (%arg3=%c1, %arg4=%c1) args(%arg5=%arg0) : memref<256xi32> {
    %c0 = arith.constant 0 : index
    %c1_0 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<256xi32, 2>
    scf.for %arg6 = %c0 to %c256 step %c1_0 {
      %0 = arith.index_cast %arg6 : index to i32
      %c16_i32 = arith.constant 16 : i32
      %1 = arith.addi %0, %c16_i32 : i32
      memref.store %1, %alloc[%arg6] : memref<256xi32, 2>
    }
    air.dma_memcpy_nd (%arg5[%c0] [%c256] [%c0], %alloc[%c0] [%c256] [%c0]) {id = 1 : i32} : (memref<256xi32>, memref<256xi32, 2>)
    memref.dealloc %alloc : memref<256xi32, 2>
    air.herd_terminator
  }
  return
}
