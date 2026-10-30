//===- shim_read_back_ordering.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -split-input-file -airrt-to-npu %s | FileCheck %s

// A drain can consume inputs issued before it was armed. All three A feeds
// must start before the read-back wait, not just those after the drain.
// CHECK-LABEL: aie.runtime_sequence @earlier_producers
// CHECK: %[[A0:.*]] = aiex.dma_configure_task_for @A
// CHECK: aiex.dma_start_task(%[[A0]])
// CHECK: %[[A1:.*]] = aiex.dma_configure_task_for @A
// CHECK: aiex.dma_start_task(%[[A1]])
// CHECK: %[[D0:.*]] = aiex.dma_configure_task_for @D
// CHECK: aiex.dma_start_task(%[[D0]])
// CHECK: %[[A2:.*]] = aiex.dma_configure_task_for @A
// CHECK: aiex.dma_start_task(%[[A2]])
// CHECK-NEXT: aiex.dma_await_task(%[[D0]])
// CHECK-NEXT: %[[B0:.*]] = aiex.dma_configure_task_for @B
// CHECK: aiex.dma_start_task(%[[B0]])
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
  func.func @earlier_producers(%arg0: memref<64xi32>, %arg1: memref<64xi32>, %arg2: memref<64xi32>) {
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %c64 = arith.constant 64 : i64
    %c2i = arith.constant 2 : i32
    %c3i = arith.constant 3 : i32
    %c4i = arith.constant 4 : i32
    %c5i = arith.constant 5 : i32
    %p = airrt.segment_load "seg" : i64
    %A0 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A1 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A2 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %D0 = airrt.dma_memcpy_nd(%c3i, %c0, %c0, %arg1[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @D} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %D0
    %B0 = airrt.dma_memcpy_nd(%c4i, %c0, %c0, %arg1[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @B} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %A0, %A1, %A2, %D0, %B0
    return
  }
}

// -----

// Each read-back follows every preceding feed, including a preceding read-back.
// CHECK-LABEL: aie.runtime_sequence @chained_readbacks
// CHECK: %[[D0:.*]] = aiex.dma_configure_task_for @D
// CHECK: aiex.dma_start_task(%[[D0]])
// CHECK: %[[C0:.*]] = aiex.dma_configure_task_for @C
// CHECK: aiex.dma_start_task(%[[C0]])
// CHECK: %[[C1:.*]] = aiex.dma_configure_task_for @C
// CHECK: aiex.dma_start_task(%[[C1]])
// CHECK: %[[C2:.*]] = aiex.dma_configure_task_for @C
// CHECK: aiex.dma_start_task(%[[C2]])
// CHECK-NEXT: aiex.dma_await_task(%[[D0]])
// CHECK: %[[A0:.*]] = aiex.dma_configure_task_for @A
// CHECK: aiex.dma_start_task(%[[A0]])
// CHECK: %[[D1:.*]] = aiex.dma_configure_task_for @D
// CHECK: aiex.dma_start_task(%[[D1]])
// CHECK-NEXT: aiex.dma_await_task(%[[D1]])
// CHECK-NEXT: %[[B0:.*]] = aiex.dma_configure_task_for @B
// CHECK: aiex.dma_start_task(%[[B0]])
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
  func.func @chained_readbacks(%arg0: memref<64xi32>, %arg1: memref<64xi32>, %arg2: memref<64xi32>) {
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %c64 = arith.constant 64 : i64
    %c2i = arith.constant 2 : i32
    %c3i = arith.constant 3 : i32
    %c4i = arith.constant 4 : i32
    %c5i = arith.constant 5 : i32
    %p = airrt.segment_load "seg" : i64
    %D0 = airrt.dma_memcpy_nd(%c3i, %c0, %c0, %arg1[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @D} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %C0 = airrt.dma_memcpy_nd(%c5i, %c0, %c0, %arg2[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @C} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %C1 = airrt.dma_memcpy_nd(%c5i, %c0, %c0, %arg2[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @C} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %C2 = airrt.dma_memcpy_nd(%c5i, %c0, %c0, %arg2[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @C} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %D0
    %A0 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg1[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %D1 = airrt.dma_memcpy_nd(%c3i, %c0, %c0, %arg1[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @D} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %D1
    %B0 = airrt.dma_memcpy_nd(%c4i, %c0, %c0, %arg1[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @B} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %D0, %C0, %C1, %C2, %A0, %D1, %B0
    return
  }
}

// -----

// D consumes A0; B and C are peer inputs. C0 may need the delayed B stream.
// A and B fit the queue; C's cap can follow those inputs without a peer conflict.
// CHECK-LABEL: aie.runtime_sequence @peer_feed_cap
// CHECK: %[[D0:.*]] = aiex.dma_configure_task_for @D
// CHECK: aiex.dma_start_task(%[[D0]])
// CHECK: %[[A0:.*]] = aiex.dma_configure_task_for @A
// CHECK: aiex.dma_start_task(%[[A0]])
// CHECK: %[[C0:.*]] = aiex.dma_configure_task_for @C
// CHECK: aiex.dma_start_task(%[[C0]])
// CHECK: %[[A1:.*]] = aiex.dma_configure_task_for @A
// CHECK: aiex.dma_start_task(%[[A1]])
// CHECK: %[[C1:.*]] = aiex.dma_configure_task_for @C
// CHECK: aiex.dma_start_task(%[[C1]])
// CHECK: %[[A2:.*]] = aiex.dma_configure_task_for @A
// CHECK: aiex.dma_start_task(%[[A2]])
// CHECK: %[[C2:.*]] = aiex.dma_configure_task_for @C
// CHECK: aiex.dma_start_task(%[[C2]])
// CHECK: aiex.dma_await_task(%[[D0]])
// CHECK: %[[B0:.*]] = aiex.dma_configure_task_for @B
// CHECK: aiex.dma_start_task(%[[B0]])
// CHECK: %[[B1:.*]] = aiex.dma_configure_task_for @B
// CHECK: aiex.dma_start_task(%[[B1]])
// CHECK: %[[C3:.*]] = aiex.dma_configure_task_for @C
// CHECK: aiex.dma_start_task(%[[C3]])
// CHECK: %[[B2:.*]] = aiex.dma_configure_task_for @B
// CHECK: aiex.dma_start_task(%[[B2]])
// CHECK: %[[B3:.*]] = aiex.dma_configure_task_for @B
// CHECK: aiex.dma_start_task(%[[B3]])
// CHECK-NEXT: %[[C4:.*]] = aiex.dma_configure_task_for @C
// CHECK: aiex.dma_await_task(%[[C0]])
// CHECK: aiex.dma_start_task(%[[C4]])
// CHECK: %[[C5:.*]] = aiex.dma_configure_task_for @C
// CHECK: aiex.dma_start_task(%[[C5]])
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
    airrt.wait_all %D0
    %B0 = airrt.dma_memcpy_nd(%c4i, %c0, %c0, %arg1[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @B} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %B1 = airrt.dma_memcpy_nd(%c4i, %c0, %c0, %arg1[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @B} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %B2 = airrt.dma_memcpy_nd(%c4i, %c0, %c0, %arg1[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @B} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %B3 = airrt.dma_memcpy_nd(%c4i, %c0, %c0, %arg1[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @B} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %C0 = airrt.dma_memcpy_nd(%c5i, %c0, %c0, %arg2[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @C} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %C1 = airrt.dma_memcpy_nd(%c5i, %c0, %c0, %arg2[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @C} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %C2 = airrt.dma_memcpy_nd(%c5i, %c0, %c0, %arg2[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @C} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %C3 = airrt.dma_memcpy_nd(%c5i, %c0, %c0, %arg2[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @C} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %C4 = airrt.dma_memcpy_nd(%c5i, %c0, %c0, %arg2[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @C} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %C5 = airrt.dma_memcpy_nd(%c5i, %c0, %c0, %arg2[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @C} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %D0, %A0, %A1, %A2, %B0, %B1, %B2, %B3, %C0, %C1, %C2, %C3, %C4, %C5
    return
  }
}

// -----

// A0 can stream output into both D0 and D1. D1 must start before the new
// cap wait on A0, even though the following read-back feed is delayed.
// CHECK-LABEL: aie.runtime_sequence @middle_drain_cap
// CHECK: %[[D0:.*]] = aiex.dma_configure_task_for @D
// CHECK: aiex.dma_start_task(%[[D0]])
// CHECK: %[[A0:.*]] = aiex.dma_configure_task_for @A
// CHECK: aiex.dma_start_task(%[[A0]])
// CHECK: %[[A1:.*]] = aiex.dma_configure_task_for @A
// CHECK: aiex.dma_start_task(%[[A1]])
// CHECK: %[[A2:.*]] = aiex.dma_configure_task_for @A
// CHECK: aiex.dma_start_task(%[[A2]])
// CHECK: %[[A3:.*]] = aiex.dma_configure_task_for @A
// CHECK: aiex.dma_start_task(%[[A3]])
// CHECK-NEXT: %[[D1:.*]] = aiex.dma_configure_task_for @D
// CHECK: aiex.dma_start_task(%[[D1]])
// CHECK-NEXT: %[[A4:.*]] = aiex.dma_configure_task_for @A
// CHECK: aiex.dma_await_task(%[[A0]])
// CHECK: aiex.dma_start_task(%[[A4]])
// CHECK: %[[A5:.*]] = aiex.dma_configure_task_for @A
// CHECK: aiex.dma_start_task(%[[A5]])
// CHECK-NEXT: aiex.dma_await_task(%[[D0]])
// CHECK: %[[B0:.*]] = aiex.dma_configure_task_for @B
// CHECK: aiex.dma_start_task(%[[B0]])
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
  func.func @middle_drain_cap(%arg0: memref<64xi32>, %arg1: memref<64xi32>, %arg2: memref<64xi32>) {
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
    airrt.wait_all %D0
    %B0 = airrt.dma_memcpy_nd(%c4i, %c0, %c0, %arg1[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @B} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %D0, %A0, %A1, %A2, %A3, %A4, %A5, %D1, %B0
    return
  }
}

// -----

// Five leading drains exceed the queue cap. Input A7 (the third A transfer)
// must start before the cap wait on D0. D3/D4 fit after that ordinary feed
// and before the read-back wait in the same round.
// CHECK-LABEL: aie.runtime_sequence @leading_drain_cap
// CHECK: %[[D0:.*]] = aiex.dma_configure_task_for @D
// CHECK: aiex.dma_start_task(%[[D0]])
// CHECK: %[[A0:.*]] = aiex.dma_configure_task_for @A
// CHECK: aiex.dma_start_task(%[[A0]])
// CHECK: %[[D1:.*]] = aiex.dma_configure_task_for @D
// CHECK: aiex.dma_start_task(%[[D1]])
// CHECK: %[[A1:.*]] = aiex.dma_configure_task_for @A
// CHECK: aiex.dma_start_task(%[[A1]])
// CHECK: %[[D2:.*]] = aiex.dma_configure_task_for @D
// CHECK: aiex.dma_start_task(%[[D2]])
// CHECK: %[[A2:.*]] = aiex.dma_configure_task_for @A
// CHECK: aiex.dma_start_task(%[[A2]])
// CHECK-NEXT: %[[D3:.*]] = aiex.dma_configure_task_for @D
// CHECK: aiex.dma_start_task(%[[D3]])
// CHECK-NEXT: aiex.dma_await_task(%[[D0]])
// CHECK-NEXT: %[[D4:.*]] = aiex.dma_configure_task_for @D
// CHECK: aiex.dma_start_task(%[[D4]])
// CHECK-NEXT: aiex.dma_await_task(%[[D1]])
// CHECK-NEXT: aiex.dma_await_task(%[[D2]])
// CHECK-NEXT: aiex.dma_await_task(%[[D3]])
// CHECK-NEXT: aiex.dma_await_task(%[[D4]])
// CHECK-NEXT: %[[B0:.*]] = aiex.dma_configure_task_for @B
// CHECK: aiex.dma_start_task(%[[B0]])
// CHECK: %[[A3:.*]] = aiex.dma_configure_task_for @A
// CHECK: aiex.dma_start_task(%[[A3]])
// CHECK: %[[A4:.*]] = aiex.dma_configure_task_for @A
// CHECK: aiex.dma_start_task(%[[A4]])
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
  func.func @leading_drain_cap(%arg0: memref<64xi32>, %arg1: memref<64xi32>, %arg2: memref<64xi32>) {
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %c64 = arith.constant 64 : i64
    %c2i = arith.constant 2 : i32
    %c3i = arith.constant 3 : i32
    %c4i = arith.constant 4 : i32
    %c5i = arith.constant 5 : i32
    %p = airrt.segment_load "seg" : i64
    %D0 = airrt.dma_memcpy_nd(%c3i, %c0, %c0, %arg1[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @D} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %D1 = airrt.dma_memcpy_nd(%c3i, %c0, %c0, %arg1[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @D} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %D2 = airrt.dma_memcpy_nd(%c3i, %c0, %c0, %arg1[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @D} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %D3 = airrt.dma_memcpy_nd(%c3i, %c0, %c0, %arg1[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @D} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %D4 = airrt.dma_memcpy_nd(%c3i, %c0, %c0, %arg1[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @D} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A5 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A6 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A7 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %D4
    %B0 = airrt.dma_memcpy_nd(%c4i, %c0, %c0, %arg1[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @B} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A9 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A10 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %D0, %D1, %D2, %D3, %D4, %A5, %A6, %A7, %B0, %A9, %A10
    return
  }
}

// -----

// B is encountered before C, but C0 must precede B1. If an awaiting
// predecessor did not advance the round, channel iteration would emit B1 first.
// CHECK-LABEL: aie.runtime_sequence @awaiting_predecessor
// CHECK: %[[B0:.*]] = aiex.dma_configure_task_for @B
// CHECK: aiex.dma_start_task(%[[B0]])
// CHECK: %[[A0:.*]] = aiex.dma_configure_task_for @A
// CHECK: aiex.dma_start_task(%[[A0]])
// CHECK: %[[A1:.*]] = aiex.dma_configure_task_for @A
// CHECK: aiex.dma_start_task(%[[A1]])
// CHECK: %[[D0:.*]] = aiex.dma_configure_task_for @D
// CHECK: aiex.dma_start_task(%[[D0]])
// CHECK: %[[A2:.*]] = aiex.dma_configure_task_for @A
// CHECK: aiex.dma_start_task(%[[A2]])
// CHECK-NEXT: aiex.dma_await_task(%[[D0]])
// CHECK-NEXT: %[[C0:.*]] = aiex.dma_configure_task_for @C
// CHECK: aiex.dma_start_task(%[[C0]])
// CHECK: %[[D1:.*]] = aiex.dma_configure_task_for @D
// CHECK: aiex.dma_start_task(%[[D1]])
// CHECK-NEXT: aiex.dma_await_task(%[[D1]])
// CHECK-NEXT: %[[B1:.*]] = aiex.dma_configure_task_for @B
// CHECK: aiex.dma_start_task(%[[B1]])
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
  func.func @awaiting_predecessor(%arg0: memref<64xi32>, %arg1: memref<64xi32>, %arg2: memref<64xi32>) {
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %c64 = arith.constant 64 : i64
    %c2i = arith.constant 2 : i32
    %c3i = arith.constant 3 : i32
    %c4i = arith.constant 4 : i32
    %c5i = arith.constant 5 : i32
    %p = airrt.segment_load "seg" : i64
    %B0 = airrt.dma_memcpy_nd(%c4i, %c0, %c0, %arg1[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @B} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A0 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A1 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A2 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %D0 = airrt.dma_memcpy_nd(%c3i, %c0, %c0, %arg1[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @D} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %D0
    %C0 = airrt.dma_memcpy_nd(%c5i, %c0, %c0, %arg1[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @C} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %D1 = airrt.dma_memcpy_nd(%c3i, %c0, %c0, %arg1[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @D} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %D1
    %B1 = airrt.dma_memcpy_nd(%c4i, %c0, %c0, %arg1[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @B} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %B0, %A0, %A1, %A2, %D0, %C0, %D1, %B1
    return
  }
}

// -----

// A middle drain follows the assigned round of the following feed, not that
// feed's index within its channel. A trailing drain has no following feed.
// CHECK-LABEL: aie.runtime_sequence @middle_and_trailing
// CHECK: %[[D0:.*]] = aiex.dma_configure_task_for @D
// CHECK: aiex.dma_start_task(%[[D0]])
// CHECK: %[[A0:.*]] = aiex.dma_configure_task_for @A
// CHECK: aiex.dma_start_task(%[[A0]])
// CHECK: %[[A1:.*]] = aiex.dma_configure_task_for @A
// CHECK: aiex.dma_start_task(%[[A1]])
// CHECK-NEXT: %[[D1:.*]] = aiex.dma_configure_task_for @D
// CHECK: aiex.dma_start_task(%[[D1]])
// CHECK: %[[A2:.*]] = aiex.dma_configure_task_for @A
// CHECK: aiex.dma_start_task(%[[A2]])
// CHECK-NEXT: aiex.dma_await_task(%[[D0]])
// CHECK: %[[B0:.*]] = aiex.dma_configure_task_for @B
// CHECK: aiex.dma_start_task(%[[B0]])
// CHECK-NEXT: %[[D2:.*]] = aiex.dma_configure_task_for @D
// CHECK: aiex.dma_start_task(%[[D2]])
// CHECK: aiex.dma_await_task(%[[D1]])
// CHECK: aiex.dma_await_task(%[[D2]])
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
  func.func @middle_and_trailing(%arg0: memref<64xi32>, %arg1: memref<64xi32>, %arg2: memref<64xi32>) {
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
    %D1 = airrt.dma_memcpy_nd(%c3i, %c0, %c0, %arg1[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @D} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %D0
    %B0 = airrt.dma_memcpy_nd(%c4i, %c0, %c0, %arg1[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @B} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %D2 = airrt.dma_memcpy_nd(%c3i, %c0, %c0, %arg1[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @D} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %D0, %A0, %A1, %A2, %D1, %B0, %D2
    return
  }
}
