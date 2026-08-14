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
// A pure repeat -- a runtime outermost size over a zero stride -- leaves the
// descriptor constant and drives the task's repeat_count, which mlir-aie takes
// as (n - 1).

// CHECK-LABEL: aie.runtime_sequence @func0
// CHECK-SAME:    %{{.*}}: memref<64xi32>, %[[N:[a-zA-Z0-9_]+]]: i64
// CHECK:         %[[T:.*]] = arith.trunci %[[N]]
// CHECK:         %[[R:.*]] = arith.subi %[[T]]
// CHECK:         aiex.dma_configure_task_for @airMemcpyId2 repeat %[[R]]
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
// CHECK-SAME:    %{{.*}}: memref<1048576xbf16>, %[[CB:[a-zA-Z0-9_]+]]: i64
// CHECK:         %[[M:.*]] = arith.muli %[[CB]]
// CHECK:         %[[L:.*]] = arith.trunci %[[M]]
// CHECK:         aie.dma_bd(%{{.*}} offset = 0 len = %[[L]])
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

// -----

// A runtime OFFSET with static sizes: the KV append writes this token at slot
// L-1, an address that moves every dispatch. constify() used to fold it to 0 --
// every token overwriting position 0. A runtime offset and a runtime size
// together are fine as well; the BD takes both as operands.

// CHECK-LABEL: aie.runtime_sequence @func2
// CHECK-SAME:    %{{.*}}: memref<1048576xbf16>, %[[OFF:[a-zA-Z0-9_]+]]: i64, %[[CB:[a-zA-Z0-9_]+]]: i64
// CHECK:         %[[O:.*]] = arith.trunci %[[OFF]]
// A moving address alone must not flatten the descriptor: the KV append writes
// NGRP chunks at a region stride, and one linear run would scatter all but the
// first. Only a runtime LENGTH (below) collapses the shape, and only because
// that case is checked contiguous.
// CHECK:         aie.dma_bd(%{{.*}} offset = %[[O]] len = 4096 sizes = [16, 256] strides = [256, 1])
// CHECK:         %[[M:.*]] = arith.muli %[[CB]]
// CHECK:         %[[L:.*]] = arith.trunci %[[M]]
// CHECK:         aie.dma_bd(%{{.*}} offset = %[[O]] len = %[[L]])
module {
  aie.device(npu2) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @airMemcpyId2(%shim_noc_tile_0_0, MM2S, 0)
  } {sym_name = "segment2"}
  func.func @func2(%arg0: memref<1048576xbf16>, %off: i64, %cb: i64) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i32 = arith.constant 2 : i32
    %c16_i64 = arith.constant 16 : i64
    %c256_i64 = arith.constant 256 : i64
    %c4096_i64 = arith.constant 4096 : i64
    %p = airrt.segment_load "segment2" : i64
    airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %off], [%c1_i64, %c1_i64, %c16_i64, %c256_i64], [%c0_i64, %c0_i64, %c256_i64, %c1_i64]) {metadata = @airMemcpyId2} : (i32, i64, i64, memref<1048576xbf16>)
    airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %off], [%c1_i64, %cb, %c16_i64, %c256_i64], [%c0_i64, %c4096_i64, %c256_i64, %c1_i64]) {metadata = @airMemcpyId2} : (i32, i64, i64, memref<1048576xbf16>)
    return
  }
}
