//===- runtime_valued_size.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -airrt-to-npu -canonicalize -cse --split-input-file %s | FileCheck %s

// A loop-invariant runtime transfer size -- a context length known only at
// dispatch, not an IV-dependent shape -- reaches the shim descriptor instead of
// being folded away. It used to be silently replaced by 1, emitting a
// wrong-sized transfer with no diagnostic.
//
// The runtime case takes the aiex.npu.dma_memcpy_nd path rather than the
// DMA-task path: both accept mixed static/dynamic sizes, but only the memcpy_nd
// lowering carries the size's magnitude into buffer_length. The dma_task
// encoder folds a runtime size into an (n > 1) predicate and otherwise keeps
// the static extent, so the transfer would silently stay its compile-time size.

// CHECK-LABEL: aie.runtime_sequence @func0
// CHECK-SAME:    %[[ARG0:.*]]: memref<64xi32>, %[[N:.*]]: i64
// CHECK:         aiex.npu.dma_memcpy_nd(%[[ARG0]]
// CHECK-SAME:      %[[N]]
// CHECK-SAME:      metadata = @airMemcpyId2
module {
  aie.device(npu1_1col) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @airMemcpyId2(%shim_noc_tile_0_0, MM2S, 0)
  } {sym_name = "segment0"}
  func.func @func0(%arg0: memref<64xi32>, %n: i64) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i32 = arith.constant 2 : i32
    %c64_i64 = arith.constant 64 : i64
    %p = airrt.segment_load "segment0" : i64
    airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%n, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @airMemcpyId2} : (i32, i64, i64, memref<64xi32>)
    return
  }
}

// -----

// The shape the fused-decode KV readback emits: a runtime block count over a
// contiguous row-major nest. All four dimensions are preserved and the count
// rides in the mixed sizes list, which is what makes buffer_length scale with
// it (verified register-equivalent to the same transfer built statically).

// CHECK-LABEL: aie.runtime_sequence @func1
// CHECK-SAME:    %[[BUF:.*]]: memref<1048576xbf16>, %[[CB:.*]]: i64
// CHECK:         aiex.npu.dma_memcpy_nd(%[[BUF]]
// CHECK-SAME:      %[[CB]]
module {
  aie.device(npu2) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @airMemcpyId2(%shim_noc_tile_0_0, MM2S, 0)
  } {sym_name = "segment1"}
  func.func @func1(%arg0: memref<1048576xbf16>, %cb: i64) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i32 = arith.constant 2 : i32
    %c16_i64 = arith.constant 16 : i64
    %c256_i64 = arith.constant 256 : i64
    %c4096_i64 = arith.constant 4096 : i64
    %p = airrt.segment_load "segment1" : i64
    airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %cb, %c16_i64, %c256_i64], [%c0_i64, %c4096_i64, %c256_i64, %c1_i64]) {metadata = @airMemcpyId2} : (i32, i64, i64, memref<1048576xbf16>)
    return
  }
}
