//===- unroll_inner_stride.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -airrt-to-npu %s | FileCheck %s

// A transfer with more dims than one BD holds is unrolled over its outermost
// dim. Each offset is in units of its stride, the innermost one included (here
// 2), so each BD starts at i * 262144 + 1 * 2 elements: 2, 262146 and 524290.
// A transfer that fits in one BD starts at the same sum: 3 * 64 + 5 * 2 = 202.

// CHECK-LABEL: aie.runtime_sequence @unroll_inner_stride
// CHECK: aie.dma_bd(%{{.*}} : memref<3x4x4096x16xi32> offset = 2 len = 16384 sizes = [2, 4, 512, 8] strides = [65536, 8192, 16, 2])
// CHECK: aie.dma_bd(%{{.*}} : memref<3x4x4096x16xi32> offset = 262146 len = 16384 sizes = [2, 4, 512, 8] strides = [65536, 8192, 16, 2])
// CHECK: aie.dma_bd(%{{.*}} : memref<3x4x4096x16xi32> offset = 524290 len = 16384 sizes = [2, 4, 512, 8] strides = [65536, 8192, 16, 2])

// CHECK-LABEL: aie.runtime_sequence @fits_inner_stride
// CHECK: aie.dma_bd(%{{.*}} : memref<1024xi32> offset = 202 len = 32 sizes = [4, 8] strides = [64, 2])

module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @airMemcpyId2(%tile_0_0, MM2S, 0)
    func.func @unroll_inner_stride(%arg0: memref<3x4x4096x16xi32>) {
      %z = arith.constant 0 : i64
      %one = arith.constant 1 : i64
      %id = arith.constant 2 : i32
      %e = airrt.dma_memcpy_nd(%id, %z, %z, %arg0[%z, %z, %z, %one], [3, 2, 2048, 8], [262144, 65536, 16, 2]) {metadata = @airMemcpyId2} : (i32, i64, i64, memref<3x4x4096x16xi32>) : !airrt.event
      return
    }
    func.func @fits_inner_stride(%arg0: memref<1024xi32>) {
      %z = arith.constant 0 : i64
      %three = arith.constant 3 : i64
      %five = arith.constant 5 : i64
      %id = arith.constant 2 : i32
      %e = airrt.dma_memcpy_nd(%id, %z, %z, %arg0[%z, %z, %three, %five], [1, 1, 4, 8], [0, 0, 64, 2]) {metadata = @airMemcpyId2} : (i32, i64, i64, memref<1024xi32>) : !airrt.event
      return
    }
  }
}
