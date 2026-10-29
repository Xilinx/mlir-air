//===- shim_read_back_caps.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -split-input-file -airrt-to-npu %s | FileCheck %s

// A drain armed after a feed may have to be armed before the cap wait on that
// feed. The fifth drain on a channel cannot be: its own cap waits for the first
// drain, which is read only after the feeds. The first three drains go ahead of
// the feeds' cap waits; the fifth follows the wait on the first drain.
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
  // CHECK-LABEL: aie.runtime_sequence @conflicting_caps
  // CHECK: dma_configure_task_for @D
  // CHECK: dma_configure_task_for @A
  // CHECK: dma_configure_task_for @A
  // CHECK: dma_configure_task_for @A
  // CHECK: dma_configure_task_for @A
  // CHECK: dma_configure_task_for @D
  // CHECK: dma_configure_task_for @D
  // CHECK: dma_configure_task_for @D
  // CHECK: dma_configure_task_for @A
  // CHECK: dma_await_task
  // CHECK: dma_configure_task_for @A
  // CHECK: dma_await_task
  // CHECK: dma_await_task
  // CHECK: dma_configure_task_for @D
  func.func @conflicting_caps(%arg0: memref<64xi32>, %arg1: memref<64xi32>, %arg2: memref<64xi32>) {
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %c64 = arith.constant 64 : i64
    %c2i = arith.constant 2 : i32
    %c3i = arith.constant 3 : i32
    %c4i = arith.constant 4 : i32
    %c5i = arith.constant 5 : i32
    %p = airrt.segment_load "seg" : i64
    %D0 = airrt.dma_memcpy_nd(%c3i, %c0, %c0, %arg1[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @D} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A0 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A1 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A2 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A3 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A4 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A5 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %D1 = airrt.dma_memcpy_nd(%c3i, %c0, %c0, %arg1[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @D} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %D2 = airrt.dma_memcpy_nd(%c3i, %c0, %c0, %arg1[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @D} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %D3 = airrt.dma_memcpy_nd(%c3i, %c0, %c0, %arg1[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @D} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %D4 = airrt.dma_memcpy_nd(%c3i, %c0, %c0, %arg1[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @D} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %D4
    %B0 = airrt.dma_memcpy_nd(%c4i, %c0, %c0, %arg1[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @B} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %D0, %D1, %D2, %D3, %D4, %A0, %A1, %A2, %A3, %A4, %A5, %B0
    return
  }
}

// -----

// Three channels: a drain read back, then two channels whose inputs may be
// consumed together. Each cap wait on C comes after the B feeds that precede it
// in the input, so a C transfer that needs a B transfer is not waited on first.
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
  // CHECK-LABEL: aie.runtime_sequence @peer_feed_cap
  // CHECK: dma_await_task
  // CHECK: dma_await_task
  // CHECK: dma_await_task
  // CHECK: dma_configure_task_for @B
  // CHECK: dma_configure_task_for @B
  // CHECK: dma_configure_task_for @B
  // CHECK: dma_configure_task_for @B
  // CHECK: dma_configure_task_for @B
  // CHECK: dma_await_task
  // CHECK: dma_start_task
  // CHECK: dma_configure_task_for @C
  // CHECK: dma_await_task
  func.func @peer_feed_cap(%arg0: memref<64xi32>, %arg1: memref<64xi32>, %arg2: memref<64xi32>) {
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %c64 = arith.constant 64 : i64
    %c2i = arith.constant 2 : i32
    %c3i = arith.constant 3 : i32
    %c4i = arith.constant 4 : i32
    %c5i = arith.constant 5 : i32
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
    %B1 = airrt.dma_memcpy_nd(%c4i, %c0, %c0, %arg1[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @B} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %B2 = airrt.dma_memcpy_nd(%c4i, %c0, %c0, %arg1[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @B} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %B3 = airrt.dma_memcpy_nd(%c4i, %c0, %c0, %arg1[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @B} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %B4 = airrt.dma_memcpy_nd(%c4i, %c0, %c0, %arg1[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @B} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %B5 = airrt.dma_memcpy_nd(%c4i, %c0, %c0, %arg1[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @B} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %C0 = airrt.dma_memcpy_nd(%c5i, %c0, %c0, %arg2[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @C} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %C1 = airrt.dma_memcpy_nd(%c5i, %c0, %c0, %arg2[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @C} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %C2 = airrt.dma_memcpy_nd(%c5i, %c0, %c0, %arg2[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @C} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %C3 = airrt.dma_memcpy_nd(%c5i, %c0, %c0, %arg2[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @C} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %C4 = airrt.dma_memcpy_nd(%c5i, %c0, %c0, %arg2[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @C} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %C5 = airrt.dma_memcpy_nd(%c5i, %c0, %c0, %arg2[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @C} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %D0, %A0, %A1, %A2, %A3, %A4, %A5, %B0, %B1, %B2, %B3, %B4, %B5, %C0, %C1, %C2, %C3, %C4, %C5
    return
  }
}
