//===- unroll_illegal_stride.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -airrt-to-npu -split-input-file -verify-diagnostics %s | FileCheck %s

// A shim BD encodes each dim's stride in a 20-bit field (max 1048576). A dim
// whose stride exceeds it cannot be tiled away -- splitting the dim leaves the
// stride unchanged -- but if its wrap is small the transfer is equivalent to
// `wrap` BDs that fold the dim into their base offset.
//
// This is the region-major KV-cache append shape: the per-group dim strides by
// ATTN_MAXL*REGION_W, which crosses the field once the context is long enough,
// while the dim itself stays at one entry per KV group. Here stride 4194304
// (= 16384 * 256) with wrap 2 becomes two contiguous 256-element BDs whose
// offsets differ by exactly that stride (16383 and 16383 + 4194304).

// CHECK-LABEL: aie.device(npu1)
// CHECK: aie.dma_bd(%arg0 : memref<268435456xbf16> offset = 16383 len = 256 sizes = [256] strides = [1])
// CHECK: aie.dma_bd(%arg0 : memref<268435456xbf16> offset = 4210687 len = 256 sizes = [256] strides = [1])

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
    %c2_i64 = arith.constant 2 : i64
    %c256_i64 = arith.constant 256 : i64
    %c16383_i64 = arith.constant 16383 : i64
    %c4194304_i64 = arith.constant 4194304 : i64
    %c4_i32 = arith.constant 4 : i32
    %p = airrt.segment_load "forward_0" : i64
    %0 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c16383_i64], [%c1_i64, %c1_i64, %c2_i64, %c256_i64], [%c0_i64, %c0_i64, %c4194304_i64, %c1_i64]) {metadata = @airMemcpyId4} : (i32, i64, i64, memref<268435456xbf16>) : !airrt.event
    return
  }
}

// -----

// A legal stride is left alone: the pass must be inert below the limit, so a
// build that does not need it is bit-identical to one from before the pass
// existed. Stride 1048576 is exactly at the bound and stays one strided BD.

// CHECK-LABEL: aie.device(npu1)
// CHECK: aie.dma_bd(%arg0 : memref<268435456xbf16> offset = 0 len = 512 sizes = [2, 256] strides = [1048576, 1])

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
    %c2_i64 = arith.constant 2 : i64
    %c256_i64 = arith.constant 256 : i64
    %c1048576_i64 = arith.constant 1048576 : i64
    %c4_i32 = arith.constant 4 : i32
    %p = airrt.segment_load "forward_0" : i64
    %0 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c2_i64, %c256_i64], [%c0_i64, %c0_i64, %c1048576_i64, %c1_i64]) {metadata = @airMemcpyId4} : (i32, i64, i64, memref<268435456xbf16>) : !airrt.event
    return
  }
}
