//===- air_L1L2_memcpy_async.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-std | FileCheck %s
// CHECK: %[[M0:.*]] = airrt.alloc : memref<1024xi32, 1>
// CHECK: %[[E0:.*]] = airrt.dma_memcpy_nd(%c1_i32, {{.*}}, {{.*}}, %0[{{.*}}], [{{.*}}], [{{.*}}]) : (i32, i64, i64, memref<1024xi32, 1>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
// CHECK: airrt.wait_all %[[E0]]
// CHECK: %[[E1:.*]] = airrt.dma_memcpy_nd(%c2_i32, {{.*}}, {{.*}}, %0[{{.*}}], [{{.*}}], [{{.*}}]) : (i32, i64, i64, memref<1024xi32, 1>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
// CHECK: airrt.wait_all %[[E0]], %[[E1]]
// CHECK: airrt.dealloc %[[M0]] : memref<1024xi32, 1>
module  {
  func.func @foo(%arg0: memref<1024xi32>, %arg1: memref<1024xi32>) {
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<1024xi32, 1>
    air.herd @h0 tile (%arg2, %arg3) in (%arg4=%c1, %arg5=%c1) args(%arg6=%0, %arg7=%arg1) : memref<1024xi32, 1>,memref<1024xi32> {
      %c0 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      %1 = memref.alloc() : memref<16xi32, 2>
      %e0 = air.dma_memcpy_nd async (%1[][][], %arg6[%c0][%c16][%c16]) {id = 1 : i32} : (memref<16xi32, 2>, memref<1024xi32, 1>)
      %e1 = air.dma_memcpy_nd async [%e0] (%arg6[%c16][%c0][%c16], %1[][][]) {id = 2 : i32} : (memref<1024xi32, 1>, memref<16xi32, 2>)
      air.wait_all [%e0, %e1]
    }
    memref.dealloc %0 : memref<1024xi32, 1>
    return
  }
}