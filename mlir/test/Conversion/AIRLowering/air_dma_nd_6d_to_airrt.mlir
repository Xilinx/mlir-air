//===- air_dma_nd_6d_to_airrt.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022-2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Verify that air-to-std correctly truncates >4D offset/size/stride lists
// to 4D for airrt.dma_memcpy_nd. The BD optimization pass and block-layout
// lowering can produce 6D patterns that must be truncated to fit the 4D
// hardware BD format.

// RUN: air-opt %s -air-to-std -cse | FileCheck %s

// CHECK-LABEL: func.func @dma_6d
// The 6D DMA is truncated to 4D: leading 2 (trivial) dimensions are dropped.
// The type signature confirms exactly 4 elements in each bracket group.
// CHECK: airrt.dma_memcpy_nd({{.*}}) : (i32, i64, i64, memref<64x64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64])
module {
  func.func @dma_6d(%arg0: memref<64x64xi32>) {
    %c2 = arith.constant 2 : index
    air.herd tile (%tx, %ty) in (%sx=%c2, %sy=%c2) args(%ext=%arg0) : memref<64x64xi32> attributes {sym_name = "herd_0"} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c32 = arith.constant 32 : index
      %c64 = arith.constant 64 : index
      %c2048 = arith.constant 2048 : index
      %buf = memref.alloc() : memref<32x64xi32, 2>
      // 6D offsets/sizes/strides: leading 2 dims are trivial (offset=0, size=1).
      air.dma_memcpy_nd (%buf[] [] [], %ext[%c0, %c0, %c0, %c0, %c0, %c0] [%c1, %c1, %c4, %c32, %c64, %c1] [%c0, %c0, %c2048, %c64, %c1, %c0]) {id = 1 : i32} : (memref<32x64xi32, 2>, memref<64x64xi32>)
      memref.dealloc %buf : memref<32x64xi32, 2>
    }
    return
  }
}
