//===- shim_feed_no_pace_lowering.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -airrt-to-npu %s | FileCheck %s

// End-to-end consequence of the air.shim_feed_no_pace opt-out: airrt-to-npu
// gates its double-buffered pacing on air.preserve_shim_dma_order. An opted-out
// feed reaches this pass WITHOUT that marker (air-opt-shim-dma-bds excluded it),
// so it lowers fire-and-free (issue_token unset, dma_free_task after start, no
// await), while a sibling that kept the marker on the same channel is paced
// (issue_token set, depth-2 completion-token awaits).

// CHECK-LABEL: aie.runtime_sequence @mixed_feeds
// Paced feed: issue_token set, bounded (depth=2) awaits.
// CHECK: %[[P0:.*]] = aiex.dma_configure_task_for @paced
// CHECK: issue_token = true
// CHECK: aiex.dma_start_task(%[[P0]])
// CHECK: %[[P1:.*]] = aiex.dma_configure_task_for @paced
// CHECK: aiex.dma_start_task(%[[P1]])
// CHECK: %[[P2:.*]] = aiex.dma_configure_task_for @paced
// CHECK: aiex.dma_await_task(%[[P0]])
// CHECK: aiex.dma_start_task(%[[P2]])
// CHECK: aiex.dma_await_task(%[[P1]])
// CHECK: aiex.dma_await_task(%[[P2]])
// Fire-and-free feed (opted out upstream, no preserve marker): no issue_token,
// no await, freed immediately after start.
// CHECK: %[[F0:.*]] = aiex.dma_configure_task_for @free
// CHECK-NOT: issue_token
// CHECK-NOT: aiex.dma_await_task
// CHECK: aiex.dma_start_task(%[[F0]])
// CHECK-NEXT: aiex.dma_free_task(%[[F0]])
module {
  aie.device(npu1) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    aie.shim_dma_allocation @paced(%shim_noc_tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @free(%shim_noc_tile_1_0, MM2S, 0)
  } {sym_name = "mixed"}
  airrt.module_metadata{}
  func.func @mixed_feeds(%arg0: memref<64xi32>, %arg1: memref<64xi32>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %p = airrt.segment_load "mixed" : i64
    %0 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @paced, air.preserve_shim_dma_order} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    %1 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @paced, air.preserve_shim_dma_order} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    %2 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @paced, air.preserve_shim_dma_order} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    %3 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @free} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    airrt.wait_all %0, %1, %2, %3
    return
  }
}
