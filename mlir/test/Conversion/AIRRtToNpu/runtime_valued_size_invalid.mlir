//===- runtime_valued_size_invalid.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: not air-opt -airrt-to-npu %s --split-input-file 2>&1 | FileCheck %s

// Runtime-valued access-pattern dimensions the shim BD lowering cannot carry
// yet. Each used to be silently replaced by a default (size 1, stride 0),
// emitting a wrong-sized transfer with no diagnostic.

// CHECK: op runtime-valued DMA size in dimension 2 requires a contiguous transfer
// CHECK: op runtime-valued outermost DMA size requires a zero outer stride


// A runtime size in an inner dimension has no repeat to fold into; it would
// need the mixed-operand aie.dma_bd form. Refused rather than defaulted.

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
    airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %n, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c0_i64]) {metadata = @airMemcpyId2} : (i32, i64, i64, memref<64xi32>)
    return
  }
}

// -----

// A runtime outermost size with a NON-zero outer stride is an iteration wrap,
// which the hardware reads from the descriptor dimensions rather than from the
// repeat count, so it cannot ride on repeat_count_val.

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
    airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%n, %c1_i64, %c1_i64, %c64_i64], [%c64_i64, %c0_i64, %c0_i64, %c0_i64]) {metadata = @airMemcpyId2} : (i32, i64, i64, memref<64xi32>)
    return
  }
}
