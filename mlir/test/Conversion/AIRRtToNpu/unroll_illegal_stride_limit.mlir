//===- unroll_illegal_stride_limit.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -airrt-to-npu %s | FileCheck %s

// Folding an over-limit stride into base offsets costs one descriptor per entry
// of the folded dim, so a large wrap is NOT folded -- 64 descriptors is worse
// than the problem. The op is then left exactly as it was: this rewrite is an
// opportunity, not a gate, and must never fail a build that compiled before it
// existed. Whether the resulting BD is legal is downstream's call.

// CHECK-LABEL: aie.device(npu1)
// CHECK: aie.dma_bd(%arg0 : memref<268435456xbf16> offset = 0 len = 16384 sizes = [64, 256] strides = [4194304, 1])

module {
  aie.device(npu1) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @airMemcpyId4(%shim_noc_tile_0_0, S2MM, 0)
  } {sym_name = "forward_0"}
  airrt.module_metadata {
    airrt.segment_metadata attributes {sym_name = "forward_0"} {
      airrt.herd_metadata {size_x = 1 : i64, size_y = 1 : i64, loc_x = 0 : i64, loc_y = 0 : i64, sym_name = "herd_0"}
    }
  }
  func.func @forward(%arg0: memref<268435456xbf16>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c256_i64 = arith.constant 256 : i64
    %c4194304_i64 = arith.constant 4194304 : i64
    %c4_i32 = arith.constant 4 : i32
    %p = airrt.segment_load "forward_0" : i64
    %0 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c64_i64, %c256_i64], [%c0_i64, %c0_i64, %c4194304_i64, %c1_i64]) {metadata = @airMemcpyId4} : (i32, i64, i64, memref<268435456xbf16>) : !airrt.event
    return
  }
}
