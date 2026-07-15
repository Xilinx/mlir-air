//===- linear_shim_bd.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -airrt-to-npu -split-input-file -verify-diagnostics %s | FileCheck %s

// Linear shim BDs (contiguous row-major body, optionally preceded by outer
// size==1 dummies or outer stride==0 repeat dims) lower to a single
// buffer_length transfer and bypass the per-dim 10-bit wrap-size limit.

// Plain contiguous bf16, inner 131136 (>> 1023): single BD, no tiling.

// CHECK-LABEL: aie.device(npu1)
// CHECK: aie.dma_bd(%arg0 : memref<131136xbf16> offset = 0 len = 131136 sizes = [131136] strides = [1])

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

// bf16 length 2049 has no even factor <= 1023. Previously errored in
// tileIllegalWrapDim; now passes through as a single linear BD.

// CHECK-LABEL: aie.device(npu1)
// CHECK: aie.dma_bd(%arg0 : memref<2049xbf16> offset = 0 len = 2049 sizes = [2049] strides = [1])

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
  func.func @forward(%arg0: memref<2049xbf16>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2049_i64 = arith.constant 2049 : i64
    %c4_i32 = arith.constant 4 : i32
    %p = airrt.segment_load "forward_0" : i64
    %0 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c2049_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @airMemcpyId4} : (i32, i64, i64, memref<2049xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    return
  }
}

// -----

// Outer repeat: size==8 stride==0 outer, dummy mids, inner 2048 stride==1.
// Folds into repeat_count=7 + one linear BD of 2048. Must NOT be tiled.

// CHECK-LABEL: aie.device(npu1)
// CHECK: aie.dma_bd(%arg0 : memref<2048xbf16> offset = 0 len = 2048 sizes = [2048] strides = [1])
// CHECK: repeat_count = 7 : i32

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
  func.func @forward(%arg0: memref<2048xbf16>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c8_i64 = arith.constant 8 : i64
    %c2048_i64 = arith.constant 2048 : i64
    %c4_i32 = arith.constant 4 : i32
    %p = airrt.segment_load "forward_0" : i64
    %0 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c8_i64, %c1_i64, %c1_i64, %c2048_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @airMemcpyId4} : (i32, i64, i64, memref<2048xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    return
  }
}

// -----

// NPU2: same linearization on AIE2P. Guards against device divergence.

// CHECK-LABEL: aie.device(npu2)
// CHECK: aie.dma_bd(%arg0 : memref<131136xbf16> offset = 0 len = 131136 sizes = [131136] strides = [1])

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
