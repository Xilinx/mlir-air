//===- tile_illegal_wrap_alignment.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -airrt-to-npu -split-input-file -verify-diagnostics %s | FileCheck %s

// All test cases below use a *non-contiguous* outer dim (stride != inner
// volume) so the shim BD cannot collapse to a single linear-mode transfer.
// The oversized inner wrap (>= 1024) therefore exercises the alignment-aware
// splitter in tileIllegalWrapDim. (Plain contiguous and outer-broadcast
// patterns are covered separately in linear_shim_bd.mlir, where they are
// expected to bypass tiling entirely.)

// When tileIllegalWrapDim splits a wrap that exceeds the shim 10-bit limit
// (1023), the new innermost size in bytes must remain a multiple of the shim
// address granularity (4 B). For a bf16 transfer of length 131136 the only
// factors <= 1023 are 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 192, and
// 683. findLargestFactor would pick 683 (odd), giving an inner segment of
// 683 elem * 2 B = 1366 B, which aie.dma_bd verification rejects. The
// alignment-aware tiler instead picks 192 (largest even factor <= 1023),
// yielding an inner segment of 192 * 2 B = 384 B (4-B aligned).

// CHECK-LABEL: aie.device(npu1)
// CHECK: aie.dma_bd(%arg0 : memref<262273xbf16>, 0, 131136, [<size = 2, stride = 131137>, <size = 1, stride = 0>, <size = 683, stride = 192>, <size = 192, stride = 1>])

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
  func.func @forward(%arg0: memref<262273xbf16>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %c131136_i64 = arith.constant 131136 : i64
    %c131137_i64 = arith.constant 131137 : i64
    %c4_i32 = arith.constant 4 : i32
    %p = airrt.segment_load "forward_0" : i64
    %0 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c2_i64, %c1_i64, %c1_i64, %c131136_i64], [%c131137_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @airMemcpyId4} : (i32, i64, i64, memref<262273xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    return
  }
}

// -----

// f32 element type: addressGranularity / elemBits = 32 / 32 = 1 (no extra
// alignment constraint). The largest factor of 131136 <= 1023 is 683 — used
// directly because there is no even-factor requirement when each element
// already covers the full granularity.

// CHECK-LABEL: aie.device(npu1)
// CHECK: aie.dma_bd(%arg0 : memref<262273xf32>, 0, 131136, [<size = 2, stride = 131137>, <size = 1, stride = 0>, <size = 192, stride = 683>, <size = 683, stride = 1>])

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
  func.func @forward(%arg0: memref<262273xf32>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %c131136_i64 = arith.constant 131136 : i64
    %c131137_i64 = arith.constant 131137 : i64
    %c4_i32 = arith.constant 4 : i32
    %p = airrt.segment_load "forward_0" : i64
    %0 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c2_i64, %c1_i64, %c1_i64, %c131136_i64], [%c131137_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @airMemcpyId4} : (i32, i64, i64, memref<262273xf32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    return
  }
}

// -----

// bf16 length 2049 = 3 * 683. The only factors <= 1023 are 1, 3, and 683 —
// all odd. No even factor exists, so findLargestAlignedFactor cannot satisfy
// the 2-element alignment that bf16 requires for a 4-byte shim BD. The pass
// should emit a diagnostic and fail rather than silently producing IR that
// the downstream aie.dma_bd verifier will reject.

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
  func.func @forward(%arg0: memref<4099xbf16>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %c2049_i64 = arith.constant 2049 : i64
    %c2050_i64 = arith.constant 2050 : i64
    %c4_i32 = arith.constant 4 : i32
    %p = airrt.segment_load "forward_0" : i64
    // expected-error @+1 {{cannot tile dim 3 of size 2049 into shim-legal chunks}}
    %0 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c2_i64, %c1_i64, %c1_i64, %c2049_i64], [%c2050_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @airMemcpyId4} : (i32, i64, i64, memref<4099xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    return
  }
}

// -----

// i8 element type: addressGranularity / elemBits = 32 / 8 = 4. Length 1028 =
// 4 * 257. Divisors <= 1023 are {1, 2, 4, 257, 514}. findLargestFactor would
// pick 514 (= 2 * 257), giving 514 elem * 1 B = 514 B (not 4-aligned). With
// alignment=4 the only multiple-of-4 factor in range is 4 itself, so the
// inner wrap drops to 4 (* 1 B = 4 B aligned) and the outer wrap becomes 257.

// CHECK-LABEL: aie.device(npu1)
// CHECK: aie.dma_bd(%arg0 : memref<2057xi8>, 0, 1028, [<size = 2, stride = 1029>, <size = 1, stride = 0>, <size = 257, stride = 4>, <size = 4, stride = 1>])

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
  func.func @forward(%arg0: memref<2057xi8>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %c1028_i64 = arith.constant 1028 : i64
    %c1029_i64 = arith.constant 1029 : i64
    %c4_i32 = arith.constant 4 : i32
    %p = airrt.segment_load "forward_0" : i64
    %0 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c2_i64, %c1_i64, %c1_i64, %c1028_i64], [%c1029_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @airMemcpyId4} : (i32, i64, i64, memref<2057xi8>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    return
  }
}

// -----

// NPU2 (AIE2P) has the same 32-bit address granularity as AIE2, so the bf16
// 131136 case picks the same 192-element inner wrap. Guards against future
// device divergence going unnoticed.

// CHECK-LABEL: aie.device(npu2)
// CHECK: aie.dma_bd(%arg0 : memref<262273xbf16>, 0, 131136, [<size = 2, stride = 131137>, <size = 1, stride = 0>, <size = 683, stride = 192>, <size = 192, stride = 1>])

module {
  aie.device(npu2) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @airMemcpyId4(%shim_noc_tile_0_0, MM2S, 0)
  } {sym_name = "forward_0"}
  airrt.module_metadata {
    airrt.segment_metadata attributes {sym_name = "forward_0"} {
      airrt.herd_metadata {size_x = 1 : i64, size_y = 1 : i64, loc_x = 0 : i64, loc_y = 0 : i64, sym_name = "herd_0"}
    }
  }
  func.func @forward(%arg0: memref<262273xbf16>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %c131136_i64 = arith.constant 131136 : i64
    %c131137_i64 = arith.constant 131137 : i64
    %c4_i32 = arith.constant 4 : i32
    %p = airrt.segment_load "forward_0" : i64
    %0 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c2_i64, %c1_i64, %c1_i64, %c131136_i64], [%c131137_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @airMemcpyId4} : (i32, i64, i64, memref<262273xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    return
  }
}
