//===- shim_read_back_caps.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -airrt-to-npu %s | FileCheck %s

// A channel with six inputs, then a read-back on a second channel whose second
// input does not read back. Keeping the second channel's pacing peers in the
// order the channel-index weave issued them would put the sixth input behind
// the second channel's second input, which follows the read-back wait, which
// follows all six. That order cannot exist, so the peer constraint to a later
// input yields. The read-back wait still follows every input before it.

module {
  aie.device(npu2) {
    %t0 = aie.tile(0, 0)
    aie.shim_dma_allocation @D(%t0, S2MM, 0)
    %t1 = aie.tile(1, 0)
    aie.shim_dma_allocation @A(%t1, MM2S, 0)
    %t2 = aie.tile(2, 0)
    aie.shim_dma_allocation @B(%t2, MM2S, 0)
    %t3 = aie.tile(3, 0)
    aie.shim_dma_allocation @C(%t3, MM2S, 0)
  } {sym_name = "seg"}
  airrt.module_metadata{}
  // CHECK-LABEL: aie.runtime_sequence @late_peer
  // CHECK-COUNT-6: dma_configure_task_for @A
  // CHECK: dma_await_task
  // CHECK: dma_await_task
  // CHECK: dma_configure_task_for @B
  func.func @late_peer(%arg0: memref<64xi32>, %arg1: memref<64xi32>) {
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %c64 = arith.constant 64 : i64
    %c2i = arith.constant 2 : i32
    %c3i = arith.constant 3 : i32
    %c4i = arith.constant 4 : i32
    %p = airrt.segment_load "seg" : i64
    %D0 = airrt.dma_memcpy_nd(%c3i, %c0, %c0, %arg1[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @D} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A0 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A1 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A2 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A3 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A4 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A5 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %D0
    %B0 = airrt.dma_memcpy_nd(%c4i, %c0, %c0, %arg1[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @B} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %B1 = airrt.dma_memcpy_nd(%c4i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @B} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %D0, %A0, %A1, %A2, %A3, %A4, %A5, %B0, %B1
    return
  }
}
