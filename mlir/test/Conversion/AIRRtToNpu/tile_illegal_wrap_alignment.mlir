//===- tile_illegal_wrap_alignment.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -airrt-to-npu -split-input-file %s | FileCheck %s

// When tileIllegalWrapDim splits a wrap that exceeds the shim 10-bit limit
// (1023), the new innermost size in bytes must remain a multiple of the shim
// address granularity (4 B). For a bf16 transfer of length 131136 the only
// factors <= 1023 are 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 192, and
// 683. findLargestFactor would pick 683 (odd), giving an inner segment of
// 683 elem * 2 B = 1366 B, which aie.dma_bd verification rejects. The
// alignment-aware tiler instead picks 192 (largest even factor <= 1023),
// yielding an inner segment of 192 * 2 B = 384 B (4-B aligned).

// CHECK-LABEL: aie.device(npu1)
// CHECK: aie.dma_bd(%arg0 : memref<131136xbf16>, 0, 131136, [<size = 683, stride = 192>, <size = 192, stride = 1>])

module {
  aie.device(npu1) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @airMemcpyId4(%shim_noc_tile_0_0, MM2S, 0)
  } {sym_name = "forward_0"}
  airrt.module_metadata {
    airrt.segment_metadata attributes {sym_name = "forward_0"} {
      airrt.herd_metadata {size_x = 1 : i64, size_y = 1 : i64, loc_x = 0 : i64, loc_y = 0 : i64, sym_name = "herd_0"}
    }
  }
  func.func @forward(%arg0: memref<131136xbf16>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c131136_i64 = arith.constant 131136 : i64
    %c4_i32 = arith.constant 4 : i32
    %p = airrt.segment_load "forward_0" : i64
    %0 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c131136_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @airMemcpyId4} : (i32, i64, i64, memref<131136xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    return
  }
}

// -----

// f32 element type: addressGranularity / elemBits = 32 / 32 = 1 (no extra
// alignment constraint). The largest factor of 131136 <= 1023 is 683 — used
// directly because there is no even-factor requirement when each element
// already covers the full granularity.

// CHECK-LABEL: aie.device(npu1)
// CHECK: aie.dma_bd(%arg0 : memref<131136xf32>, 0, 131136, [<size = 192, stride = 683>, <size = 683, stride = 1>])

module {
  aie.device(npu1) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @airMemcpyId4(%shim_noc_tile_0_0, MM2S, 0)
  } {sym_name = "forward_0"}
  airrt.module_metadata {
    airrt.segment_metadata attributes {sym_name = "forward_0"} {
      airrt.herd_metadata {size_x = 1 : i64, size_y = 1 : i64, loc_x = 0 : i64, loc_y = 0 : i64, sym_name = "herd_0"}
    }
  }
  func.func @forward(%arg0: memref<131136xf32>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c131136_i64 = arith.constant 131136 : i64
    %c4_i32 = arith.constant 4 : i32
    %p = airrt.segment_load "forward_0" : i64
    %0 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c131136_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @airMemcpyId4} : (i32, i64, i64, memref<131136xf32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    return
  }
}
