//===- airrt_memcpy.mlir ---------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s | FileCheck %s
// CHECK: module
module  {
  func.func @foo(%arg0: memref<256x256xi32>, %arg1: memref<128x128xi32, 1>) {
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 1 {
        %c16 = arith.constant 16 : index
        %c0 = arith.constant 0 : i64
        %c1 = arith.constant 1 : i64
        %c1_i32 = arith.constant 1 : i32
        %c2_i32 = arith.constant 2 : i32
        %c128_i64 = arith.constant 128 : i64
        %c256_i64 = arith.constant 256 : i64
        %1 = arith.index_cast %arg3 : index to i64
        %2 = arith.index_cast %arg2 : index to i64
        %3 = arith.index_cast %c16 : index to i64
        // CHECK: airrt.dma_memcpy_nd(%c1_i32, %0, %1, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %2, %2], [%c0_i64, %c0_i64, %c256_i64]) : (i32, i64, i64, memref<256x256xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64])
        airrt.dma_memcpy_nd(%c1_i32, %1, %2, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %3, %3], [%c0, %c0, %c256_i64]) : (i32, i64, i64, memref<256x256xi32>, [i64,i64,i64,i64], [i64,i64,i64,i64], [i64,i64,i64])
        // CHECK: airrt.dma_memcpy_nd(%c2_i32, %0, %1, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %2, %2], [%c0_i64, %c0_i64, %c128_i64]) {attr = "attr"} : (i32, i64, i64, memref<128x128xi32, 1>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64])
        airrt.dma_memcpy_nd(%c2_i32, %1, %2, %arg1[%c0, %c0, %c0, %c0], [%c1, %c1, %3, %3], [%c0, %c0, %c128_i64]) {attr = "attr"} : (i32, i64, i64, memref<128x128xi32, 1>, [i64,i64,i64,i64], [i64,i64,i64,i64], [i64,i64,i64])
      } {air.herd_launch = "inner"}
    } {air.herd_launch = "outer"}
    return
  }
}