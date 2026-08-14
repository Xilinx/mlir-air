//===- runtime_valued_size.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -airrt-to-npu -canonicalize -cse %s | FileCheck %s

// A loop-invariant runtime transfer size (a context length known only at
// dispatch time, not an IV-dependent shape) reaches the shim BD instead of
// being folded away. With a zero outer stride the outermost dimension is a pure
// repeat, so it lowers to the task's runtime repeat count. It used to be
// silently replaced by 1, which emitted a wrong-sized transfer with no
// diagnostic.

// CHECK-LABEL: aie.runtime_sequence @func0
// CHECK-SAME:    %[[ARG0:.*]]: memref<64xi32>, %[[N:.*]]: i64
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : i32
// CHECK:         %[[T:.*]] = arith.trunci %[[N]] : i64 to i32
// CHECK:         %[[R:.*]] = arith.subi %[[T]], %[[C1]] : i32
// CHECK:         aiex.dma_configure_task_for @airMemcpyId2 repeat %[[R]] : i32
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
    airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%n, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c0_i64]) {metadata = @airMemcpyId2} : (i32, i64, i64, memref<64xi32>)
    return
  }
}
