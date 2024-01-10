//===- air_lower_herd_parallel.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-lower-herd-parallel %s | FileCheck %s

// CHECK-LABEL: func.func @f0
// CHECK: %[[VAL_11:.*]] = arith.constant 0 : index
// CHECK: %[[VAL_12:.*]] = arith.constant 128 : index
// CHECK: %[[VAL_13:.*]] = arith.constant 32 : index
// CHECK: scf.for %[[VAL_14:.*]] = %[[VAL_11]] to %[[VAL_12]] step %[[VAL_13]]
func.func @f0(%arg0: memref<128xf32>, %arg1: memref<128xf32>) -> memref<128xf32> {
  %c1 = arith.constant 1 : index
  %alloc = memref.alloc() : memref<128xf32>
  air.herd @herd_0  tile (%arg2, %arg3) in (%arg4=%c1, %arg5=%c1) args(%arg6=%arg0, %arg7=%alloc) : memref<128xf32>, memref<128xf32> {
    %c1_0 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c32 = arith.constant 32 : index
    scf.parallel (%arg8) = (%c0) to (%c128) step (%c32) {
      %alloc_1 = memref.alloc() : memref<32xf32, 2>
      air.dma_memcpy_nd (%alloc_1[] [] [], %arg6[%arg8] [%c32] [%c1_0]) : (memref<32xf32, 2>, memref<128xf32>)
      air.dma_memcpy_nd (%arg7[%arg8] [%c32] [%c1_0], %alloc_1[] [] []) : (memref<128xf32>, memref<32xf32, 2>)
      memref.dealloc %alloc_1 : memref<32xf32, 2>
    }
    air.herd_terminator
  }
  return %alloc : memref<128xf32>
}
