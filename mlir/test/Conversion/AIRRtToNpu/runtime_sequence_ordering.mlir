//===- runtime_sequence_ordering.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -airrt-to-npu -split-input-file %s | FileCheck %s

// airrt-to-npu reorders the generated runtime sequence to enforce ordering
// constraints that the async dependence graph cannot express. This test covers
// four such transforms.

// -----

// (1) air.runtime_hoist: an input feed tagged `air.runtime_hoist` is emitted at
// the FRONT of the runtime sequence -- ahead of other (untagged) shim feeds --
// even when it appears later in program order. Used when a feed drives a
// producer that a later feed's consumer transitively waits on: issuing the
// later feed first blocks the control program before the hoisted feed is ever
// issued, deadlocking the sequence.

// CHECK-LABEL: aie.runtime_sequence @runtime_hoist
// The tagged feed (@kvIn) is hoisted ahead of the untagged feed (@weightIn),
// reversing program order.
// CHECK: aiex.dma_configure_task_for @kvIn
// CHECK: aiex.dma_start_task
// CHECK: aiex.dma_configure_task_for @weightIn
// CHECK: aiex.dma_start_task
module {
  aie.device(npu1) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    aie.shim_dma_allocation @weightIn(%shim_noc_tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @kvIn(%shim_noc_tile_1_0, MM2S, 0)
  } {sym_name = "forward_0"}
  airrt.module_metadata{
    airrt.segment_metadata attributes {sym_name = "forward_0"} {
      airrt.herd_metadata {size_x = 1 : i64, size_y = 1 : i64, loc_x = 0 : i64, loc_y = 0 : i64, sym_name = "herd_0"}
    }
  }
  func.func @runtime_hoist(%arg0: memref<1024xbf16>, %arg1: memref<512xbf16>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c512_i64 = arith.constant 512 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %c4_i32 = arith.constant 4 : i32
    %c5_i32 = arith.constant 5 : i32
    %p = airrt.segment_load "forward_0" : i64
    // Untagged feed FIRST in program order.
    %0 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c1024_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @weightIn} : (i32, i64, i64, memref<1024xbf16>) : !airrt.event
    // Tagged feed SECOND, but hoisted to the front.
    %1 = airrt.dma_memcpy_nd(%c5_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c512_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @kvIn, air.runtime_hoist} : (i32, i64, i64, memref<512xbf16>) : !airrt.event
    return
  }
}

// -----

// (2) RTP-write + set_lock hoist: herd RTP writes and the herd-release set_lock
// ops are moved to the front of the runtime sequence, ahead of all data
// movement, so a persistent core latches its RTP (and is released) before any
// DMA that triggers it -- otherwise the core reads a stale (zero) RTP or is
// never released and produces no output.

// CHECK-LABEL: aie.runtime_sequence @rtp_hoist
// The RTP write and set_lock are hoisted ahead of the input DMA even though the
// herd_load that emits them appears after it in program order.
// CHECK: arith.constant 5 : i32
// CHECK: aiex.npu.rtp_write(@__air_herd_rtp_0_2, 0, %{{.*}}) : i32
// CHECK: aiex.set_lock(%__air_herd_lock_0_2, 1)
// CHECK: aiex.dma_configure_task_for @weightIn
module {
  aie.device(npu1) @segment_0 {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    %__air_herd_lock_0_2 = aie.lock(%tile_0_2, 0) {init = 0 : i32, sym_name = "__air_herd_lock_0_2"}
    %__air_herd_rtp_0_2 = aie.buffer(%tile_0_2) {sym_name = "__air_herd_rtp_0_2"} : memref<1xi32>
    aie.shim_dma_allocation @weightIn(%tile_0_0, MM2S, 0)
  } {sym_name = "segment_0"}
  airrt.module_metadata {
    airrt.segment_metadata attributes {sym_name = "segment_0"} {
      airrt.herd_metadata {size_x = 1 : i64, size_y = 1 : i64, loc_x = 0 : i64, loc_y = 2 : i64, sym_name = "herd_0"}
    }
  }
  func.func @rtp_hoist(%arg0: memref<1024xbf16>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %c4_i32 = arith.constant 4 : i32
    %p = airrt.segment_load "segment_0" : i64
    // Input DMA emitted BEFORE the herd_load.
    %0 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c1024_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @weightIn} : (i32, i64, i64, memref<1024xbf16>) : !airrt.event
    %c5_i32 = arith.constant 5 : i32
    %h = airrt.herd_load "herd_0" (%c5_i32) {segment_name = "segment_0"} : (i32) -> i64
    return
  }
}

// -----

// (3) air.preserve_shim_dma_order double-buffered pacing: MM2S shim feeds marked
// `air.preserve_shim_dma_order` are lockstep-coupled by a downstream broadcast
// consumer; with no backpressure the runtime over-commits one channel's BDs and
// deadlocks. Each such feed issues a token and gets bounded (depth=2) awaits:
// before reusing task i's BD (start i), task i-2 is awaited, then the final 2
// tasks are drained after the last start.

// CHECK-LABEL: aie.runtime_sequence @paced_feed
// CHECK: %[[T0:.*]] = aiex.dma_configure_task_for @feed
// CHECK: %[[T1:.*]] = aiex.dma_configure_task_for @feed
// CHECK: %[[T2:.*]] = aiex.dma_configure_task_for @feed
// Before starting task 2 (reusing task 0's BD), task 0 is awaited:
// CHECK: aiex.dma_await_task(%[[T0]])
// CHECK: aiex.dma_start_task(%[[T2]])
// The final depth=2 tasks are drained after the last start:
// CHECK: aiex.dma_await_task(%[[T1]])
// CHECK: aiex.dma_await_task(%[[T2]])
module {
  aie.device(npu1) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @feed(%shim_noc_tile_0_0, MM2S, 0)
  } {sym_name = "paced"}
  airrt.module_metadata{}
  func.func @paced_feed(%arg0: memref<64xi32>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c2_i32 = arith.constant 2 : i32
    %p = airrt.segment_load "paced" : i64
    %0 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @feed, air.preserve_shim_dma_order} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %1 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @feed, air.preserve_shim_dma_order} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %2 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @feed, air.preserve_shim_dma_order} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %0, %1, %2
    return
  }
}

// -----

// (4) paced-MM2S per-iteration segmentation: a two-iteration unrolled launch
// (two air.launch_end markers, NPU2, non-ELF) with 2 preserve_shim_dma_order
// feeds per iteration. synthesizeDoubleBufferedAwaits must split the 4 feeds
// into two 2-feed segments and fully drain each segment's tail before the next
// iteration's first start -- so the in-flight window never straddles an
// iteration boundary.
//
// With the whole-list fallback (no per-iteration segmentation), the 4 feeds
// would be paced as one list: no mid-stream awaits (4 > depth=2 but T0 await
// before T2 start), final drain of T2 and T3 only -- leaving T0 and T1
// in-flight across the iteration boundary and accumulating a lock imbalance.
// Correct behavior: iteration 0's T0+T1 are fully drained before T2 starts.

// CHECK-LABEL: aie.runtime_sequence @paced_multiiter
// Iteration 0: 2 paced feeds configured and started.
// CHECK: %[[T0:.*]] = aiex.dma_configure_task_for @feed
// CHECK: %[[T1:.*]] = aiex.dma_configure_task_for @feed
// Iteration 0 segment drain (fenceEnd=true, n=2<=depth=2): both tasks drained
// after T1's start, before iteration 1's T2.
// CHECK: aiex.dma_await_task(%[[T0]])
// CHECK: aiex.dma_await_task(%[[T1]])
// Iteration 1: 2 paced feeds independently segmented.
// CHECK: %[[T2:.*]] = aiex.dma_configure_task_for @feed
// CHECK: %[[T3:.*]] = aiex.dma_configure_task_for @feed
// Iteration 1 segment drain: both tasks drained after T3's start.
// CHECK: aiex.dma_await_task(%[[T2]])
// CHECK: aiex.dma_await_task(%[[T3]])
module {
  aie.device(npu2) @seg {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @feed(%shim_noc_tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @out(%shim_noc_tile_0_0, S2MM, 0)
  } {sym_name = "seg"}
  airrt.module_metadata {
    airrt.segment_metadata attributes {sym_name = "seg"} {
      airrt.herd_metadata {size_x = 1 : i64, size_y = 1 : i64,
                           loc_x = 0 : i64, loc_y = 2 : i64,
                           sym_name = "herd_0"}
    }
  }
  func.func @paced_multiiter(%arg0: memref<64xi32>, %arg1: memref<64xi32>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c2_i32 = arith.constant 2 : i32
    %c5_i32 = arith.constant 5 : i32
    %p = airrt.segment_load "seg" : i64
    // iteration 0: 2 paced MM2S feeds + 1 S2MM output
    %f0 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @feed, air.preserve_shim_dma_order} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %g0 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @feed, air.preserve_shim_dma_order} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %o0 = airrt.dma_memcpy_nd(%c5_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %o0 {"air.launch_end"}
    // iteration 1: 2 paced MM2S feeds + 1 S2MM output
    %f1 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @feed, air.preserve_shim_dma_order} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %g1 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @feed, air.preserve_shim_dma_order} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %o1 = airrt.dma_memcpy_nd(%c5_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %o1 {"air.launch_end"}
    return
  }
}
