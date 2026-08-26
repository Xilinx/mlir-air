//===- paced_shim_feed_sibling_channels.mlir -------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -airrt-to-npu %s | FileCheck %s

// Sibling paced channels in ONE segment must all be issued before any of them
// is drained.
//
// synthesizeDoubleBufferedAwaits paces per channel, and each segment's in-flight
// tail is drained so it cannot straddle the next segment. Anchoring that drain
// after the CHANNEL's own last start serializes siblings: a one-task channel is
// then awaited immediately after its own start, which turns a fire-and-forget
// feed into a synchronous one. When such a transfer can only complete if its
// consumer runs, and the consumer also needs a sibling channel that has not
// been issued yet, the design deadlocks.
//
// This is the LFM2 hybrid's KV read-back reduced to its shape: four one-task
// channels (K and V for two consumer pairs) that the consumers read in
// parallel. On device it survived while a whole region still fit in the memtile
// ring plus the consumers' own buffering (ATTN_MAXL 80, 5 blocks) and
// cold-deadlocked from 6 blocks up, at every ring depth.
//
// The drain therefore anchors after the last start in the SEGMENT. All four
// starts come first; the four awaits follow as a group.

// CHECK-LABEL: aie.runtime_sequence @kv_readback
// CHECK: %[[K0:.*]] = aiex.dma_configure_task_for @kvK0
// CHECK: aiex.dma_start_task(%[[K0]])
// CHECK-NOT: aiex.dma_await_task
// CHECK: %[[V0:.*]] = aiex.dma_configure_task_for @kvV0
// CHECK: aiex.dma_start_task(%[[V0]])
// CHECK-NOT: aiex.dma_await_task
// CHECK: %[[K1:.*]] = aiex.dma_configure_task_for @kvK1
// CHECK: aiex.dma_start_task(%[[K1]])
// CHECK-NOT: aiex.dma_await_task
// CHECK: %[[V1:.*]] = aiex.dma_configure_task_for @kvV1
// CHECK: aiex.dma_start_task(%[[V1]])
// Only now the tail drains, one per channel, as a group.
// CHECK: aiex.dma_await_task
// CHECK: aiex.dma_await_task
// CHECK: aiex.dma_await_task
// CHECK: aiex.dma_await_task

module {
  aie.device(npu1) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    %shim_noc_tile_2_0 = aie.tile(2, 0)
    %shim_noc_tile_3_0 = aie.tile(3, 0)
    aie.shim_dma_allocation @kvK0(%shim_noc_tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @kvV0(%shim_noc_tile_1_0, MM2S, 0)
    aie.shim_dma_allocation @kvK1(%shim_noc_tile_2_0, MM2S, 0)
    aie.shim_dma_allocation @kvV1(%shim_noc_tile_3_0, MM2S, 0)
  } {sym_name = "kv"}
  airrt.module_metadata{}
  func.func @kv_readback(%arg0: memref<64xi32>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c2_i32 = arith.constant 2 : i32
    %p = airrt.segment_load "kv" : i64
    %0 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @kvK0, air.preserve_shim_dma_order} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %1 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @kvV0, air.preserve_shim_dma_order} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %2 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @kvK1, air.preserve_shim_dma_order} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %3 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @kvV1, air.preserve_shim_dma_order} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %0, %1, %2, %3
    return
  }
}
