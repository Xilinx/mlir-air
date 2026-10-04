//===- shim_feed_burst_bound.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -airrt-to-npu %s | FileCheck %s

// A shim MM2S channel absorbs 6 tasks in flight (its DMA task queue plus an L2
// ping-pong); past that a push is dropped rather than deferred and the design
// deadlocks (Xilinx/mlir-air#1822). Runtime-loop tiling unrolls the feed puts,
// so a burst can be far deeper than that, and the conversion emits it
// channel-major: every task for A, then every task for B.
//
// Two channels x 6 feeds each, so both exceed the limit of 4. The burst comes
// back woven A/B/A/B -- if it stayed channel-major, the pacing awaits below
// could never retire, because B would not be fed until A had drained -- and
// each channel is capped at 4 in flight by awaiting task i-4 before starting
// task i. The awaited config gains issue_token (an MM2S task issues no
// completion token by default) and loses its fire-and-free, which an await
// would otherwise double.

// CHECK-LABEL: aie.runtime_sequence @burst
// The weave: the first four of each channel start back to back, unpaced.
// CHECK: %[[A0:.*]] = aiex.dma_configure_task_for @chanA
// CHECK: issue_token = true
// CHECK: aiex.dma_start_task(%[[A0]])
// CHECK: %[[B0:.*]] = aiex.dma_configure_task_for @chanB
// CHECK: issue_token = true
// CHECK: aiex.dma_start_task(%[[B0]])
// CHECK: %[[A1:.*]] = aiex.dma_configure_task_for @chanA
// CHECK: aiex.dma_start_task(%[[A1]])
// CHECK: %[[B1:.*]] = aiex.dma_configure_task_for @chanB
// CHECK: aiex.dma_start_task(%[[B1]])
// CHECK: %[[A2:.*]] = aiex.dma_configure_task_for @chanA
// CHECK: aiex.dma_start_task(%[[A2]])
// CHECK: %[[B2:.*]] = aiex.dma_configure_task_for @chanB
// CHECK: aiex.dma_start_task(%[[B2]])
// CHECK: %[[A3:.*]] = aiex.dma_configure_task_for @chanA
// CHECK: aiex.dma_start_task(%[[A3]])
// CHECK: %[[B3:.*]] = aiex.dma_configure_task_for @chanB
// CHECK: aiex.dma_start_task(%[[B3]])
// The fifth of each channel is the first that must wait: it may not start until
// the task four back has retired.
// CHECK: %[[A4:.*]] = aiex.dma_configure_task_for @chanA
// CHECK: aiex.dma_await_task(%[[A0]])
// CHECK: aiex.dma_start_task(%[[A4]])
// CHECK: %[[B4:.*]] = aiex.dma_configure_task_for @chanB
// CHECK: aiex.dma_await_task(%[[B0]])
// CHECK: aiex.dma_start_task(%[[B4]])
// CHECK: %[[A5:.*]] = aiex.dma_configure_task_for @chanA
// CHECK: aiex.dma_await_task(%[[A1]])
// CHECK: aiex.dma_start_task(%[[A5]])
// CHECK: %[[B5:.*]] = aiex.dma_configure_task_for @chanB
// CHECK: aiex.dma_await_task(%[[B1]])
// CHECK: aiex.dma_start_task(%[[B5]])
// An awaited task is freed by its await, so it must not also be freed here.
// CHECK-NOT: aiex.dma_free_task(%[[A0]])
// CHECK-NOT: aiex.dma_free_task(%[[B0]])
module {
  aie.device(npu1) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    aie.shim_dma_allocation @chanA(%shim_noc_tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @chanB(%shim_noc_tile_1_0, MM2S, 0)
  } {sym_name = "burst_seg"}
  airrt.module_metadata{}
  func.func @burst(%arg0: memref<64xi32>, %arg1: memref<64xi32>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %p = airrt.segment_load "burst_seg" : i64
    %0 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @chanA} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %1 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @chanA} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %2 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @chanA} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %3 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @chanA} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %4 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @chanA} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %5 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @chanA} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %6 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @chanB} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %7 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @chanB} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %8 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @chanB} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %9 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @chanB} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %10 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @chanB} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %11 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @chanB} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11
    return
  }
}

// -----

// A burst that already fits stays exactly as emitted: no weave, no awaits, no
// issue_token. Three feeds per channel, under the limit of 4.

// CHECK-LABEL: aie.runtime_sequence @short_burst
// CHECK-NOT: issue_token
// CHECK-NOT: aiex.dma_await_task
// CHECK: %[[S0:.*]] = aiex.dma_configure_task_for @shortA
// CHECK: aiex.dma_start_task(%[[S0]])
// CHECK: %[[S1:.*]] = aiex.dma_configure_task_for @shortA
// CHECK: aiex.dma_start_task(%[[S1]])
// CHECK: %[[S2:.*]] = aiex.dma_configure_task_for @shortA
// CHECK: aiex.dma_start_task(%[[S2]])
// CHECK: %[[T0:.*]] = aiex.dma_configure_task_for @shortB
// CHECK: aiex.dma_start_task(%[[T0]])
module {
  aie.device(npu1) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    aie.shim_dma_allocation @shortA(%shim_noc_tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @shortB(%shim_noc_tile_1_0, MM2S, 0)
  } {sym_name = "short_seg"}
  airrt.module_metadata{}
  func.func @short_burst(%arg0: memref<64xi32>, %arg1: memref<64xi32>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %p = airrt.segment_load "short_seg" : i64
    %0 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @shortA} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %1 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @shortA} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %2 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @shortA} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %3 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @shortB} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %4 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @shortB} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %5 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @shortB} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %0, %1, %2, %3, %4, %5
    return
  }
}

// -----

// An ordered drain (air.ordered_drain, from air-to-std's drain ordering for a
// launch that reads back its own drains) is armed, and an earlier one awaited,
// right before the feed that may read what the awaited drain wrote. Those ops
// must not end the burst: B, emitted after all of A, would then only be woven
// after the last drain, and its tasks pile up at the end. They travel with the
// next feed instead, so the drain await stays in front of A3 and B is still
// woven through the whole burst.

// CHECK-LABEL: aie.runtime_sequence @ordered_drain
// Within each round, B (no drain ops) goes ahead of A, so it is out before the
// control program blocks on A's drain await.
// CHECK: %[[OB0:.*]] = aiex.dma_configure_task_for @ordB
// CHECK: %[[D0:.*]] = aiex.dma_configure_task_for @ordOut
// CHECK: aiex.dma_start_task(%[[D0]])
// CHECK: %[[OA0:.*]] = aiex.dma_configure_task_for @ordA
// CHECK: %[[OB1:.*]] = aiex.dma_configure_task_for @ordB
// CHECK: %[[OA1:.*]] = aiex.dma_configure_task_for @ordA
// CHECK: %[[OB2:.*]] = aiex.dma_configure_task_for @ordB
// CHECK: %[[OA2:.*]] = aiex.dma_configure_task_for @ordA
// CHECK: %[[OB3:.*]] = aiex.dma_configure_task_for @ordB
// CHECK: %[[D1:.*]] = aiex.dma_configure_task_for @ordOut
// CHECK: aiex.dma_start_task(%[[D1]])
// CHECK: aiex.dma_await_task(%[[D0]])
// CHECK: %[[OA3:.*]] = aiex.dma_configure_task_for @ordA
// CHECK: %[[OB4:.*]] = aiex.dma_configure_task_for @ordB
// CHECK: %[[OA4:.*]] = aiex.dma_configure_task_for @ordA
// CHECK: %[[OB5:.*]] = aiex.dma_configure_task_for @ordB
// CHECK: %[[OA5:.*]] = aiex.dma_configure_task_for @ordA
// CHECK: aiex.dma_await_task(%[[D1]])
module {
  aie.device(npu1) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    aie.shim_dma_allocation @ordA(%shim_noc_tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @ordB(%shim_noc_tile_1_0, MM2S, 0)
    aie.shim_dma_allocation @ordOut(%shim_noc_tile_0_0, S2MM, 0)
  } {sym_name = "ord_seg"}
  airrt.module_metadata{}
  func.func @ordered_drain(%arg0: memref<64xi32>, %arg1: memref<64xi32>, %arg2: memref<64xi32>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %c4_i32 = arith.constant 4 : i32
    %p = airrt.segment_load "ord_seg" : i64
    %d0 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {air.ordered_drain, metadata = @ordOut} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %a0 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @ordA} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %a1 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @ordA} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %a2 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @ordA} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %d1 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {air.ordered_drain, metadata = @ordOut} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %d0
    %a3 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @ordA} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %a4 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @ordA} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %a5 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @ordA} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %b0 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @ordB} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %b1 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @ordB} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %b2 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @ordB} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %b3 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @ordB} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %b4 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @ordB} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %b5 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @ordB} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %d1, %a0, %a1, %a2, %a3, %a4, %a5, %b0, %b1, %b2, %b3, %b4, %b5
    return
  }
}
