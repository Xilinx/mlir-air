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
    %0 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c1024_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @weightIn} : (i32, i64, i64, memref<1024xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    // Tagged feed SECOND, but hoisted to the front.
    %1 = airrt.dma_memcpy_nd(%c5_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c512_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @kvIn, air.runtime_hoist} : (i32, i64, i64, memref<512xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
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
// CHECK: aiex.npu.rtp_write(@__air_herd_rtp_0_2, 0, 5)
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
    %0 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c1024_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @weightIn} : (i32, i64, i64, memref<1024xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    %c5_i32 = arith.constant 5 : i32
    %h = airrt.herd_load "herd_0" (%c5_i32) {segment_name = "segment_0"} : (i32) -> i64
    return
  }
}

// -----

// (3) air.await_appends / air.append_barrier: a same-DDR read-after-write that
// the dependence graph cannot see. A readback tagged `air.await_appends` must
// observe values written by device-side appends (S2MM drains into the same DDR
// buffer), each tagged `air.append_barrier`. Since an append's completion await
// is otherwise deferred, the append awaits are moved to just BEFORE the tagged
// readback's start, so the runtime blocks on append completion before reading.

// CHECK-LABEL: aie.runtime_sequence @await_appends
// The appends are configured/started first (program order)...
// CHECK: aiex.dma_configure_task_for @appendK
// CHECK: aiex.dma_configure_task_for @appendV
// ...and their completion awaits are hoisted before the readback's start task.
// CHECK: %[[RB:.*]] = aiex.dma_configure_task_for @readback
// CHECK: aiex.dma_await_task
// CHECK: aiex.dma_await_task
// CHECK: aiex.dma_start_task(%[[RB]])
module {
  aie.device(npu1_1col) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @readback(%shim_noc_tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @appendK(%shim_noc_tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @appendV(%shim_noc_tile_0_0, S2MM, 1)
  } {sym_name = "seg"}
  airrt.module_metadata{}
  func.func @await_appends(%arg0: memref<64xi32>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %c4_i32 = arith.constant 4 : i32
    %p = airrt.segment_load "seg" : i64
    // Appends (S2MM) tagged append_barrier, emitted first.
    %ak = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @appendK, air.append_barrier} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    %av = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @appendV, air.append_barrier} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    // Readback (MM2S) tagged await_appends, emitted after the appends.
    %r = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @readback, air.await_appends} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    airrt.wait_all %ak, %av
    return
  }
}

// -----

// (4) air.preserve_shim_dma_order double-buffered pacing: MM2S shim feeds marked
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
    %0 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @feed, air.preserve_shim_dma_order} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    %1 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @feed, air.preserve_shim_dma_order} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    %2 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @feed, air.preserve_shim_dma_order} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    airrt.wait_all %0, %1, %2
    return
  }
}

// -----

// (5) air.await_appends per-readback barrier: a runtime sequence may contain
// MORE THAN ONE independent readback (e.g. an unrolled loop with N
// append/readback pairs, each readback fed by several appends). Each append's
// completion await is ordered before the readback that CONSUMES it -- the first
// tagged readback that follows the append in program order -- NOT collapsed onto
// the first readback (which would move a later readback's append awaits ahead of
// an earlier readback, violating SSA dominance and the append->readback order).
//
// This mirrors the KV-cache shape: per iteration two appends (append0, append1)
// drain into a shared buffer, then one readback reads it back.

// CHECK-LABEL: aie.runtime_sequence @await_appends_multi
// Iteration 0: readback0 awaits ITS OWN two appends (by SSA identity), then starts.
// CHECK: %[[AK0:.*]] = aiex.dma_configure_task_for @appendK
// CHECK: %[[AV0:.*]] = aiex.dma_configure_task_for @appendV
// CHECK: %[[RB0:.*]] = aiex.dma_configure_task_for @readback
// CHECK: aiex.dma_await_task(%[[AK0]])
// CHECK: aiex.dma_await_task(%[[AV0]])
// CHECK: aiex.dma_start_task(%[[RB0]])
// Iteration 1: readback1 awaits ITS OWN two appends -- not iteration 0's, and
// iteration 1's appends are NOT awaited before readback0 above.
// CHECK: %[[AK1:.*]] = aiex.dma_configure_task_for @appendK
// CHECK: %[[AV1:.*]] = aiex.dma_configure_task_for @appendV
// CHECK: %[[RB1:.*]] = aiex.dma_configure_task_for @readback
// CHECK: aiex.dma_await_task(%[[AK1]])
// CHECK: aiex.dma_await_task(%[[AV1]])
// CHECK: aiex.dma_start_task(%[[RB1]])
module {
  aie.device(npu1_1col) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @readback(%shim_noc_tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @appendK(%shim_noc_tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @appendV(%shim_noc_tile_0_0, S2MM, 1)
  } {sym_name = "seg"}
  airrt.module_metadata{}
  func.func @await_appends_multi(%arg0: memref<64xi32>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %c4_i32 = arith.constant 4 : i32
    %p = airrt.segment_load "seg" : i64
    // iteration 0: two appends then a readback that consumes them
    %ak0 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @appendK, air.append_barrier} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    %av0 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @appendV, air.append_barrier} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    %r0 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @readback, air.await_appends} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    // iteration 1: two appends then a readback that consumes them
    %ak1 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @appendK, air.append_barrier} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    %av1 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @appendV, air.append_barrier} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    %r1 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @readback, air.await_appends} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    airrt.wait_all %ak0, %av0, %ak1, %av1
    return
  }
}

// -----

// (6) A trailing air.append_barrier append with NO tagged readback following it
// must be left deferred: its completion await is NOT hoisted before an earlier
// readback. Only an append the readback CONSUMES -- one that precedes it -- is
// awaited before that readback's start. This guards the "first following
// readback" match: an append with no following readback finds no target.

// CHECK-LABEL: aie.runtime_sequence @await_appends_trailing
// CHECK: %[[A0:.*]] = aiex.dma_configure_task_for @appendK
// CHECK: %[[RB:.*]] = aiex.dma_configure_task_for @readback
// The consumed append (before the readback) is awaited before the readback start.
// CHECK: aiex.dma_await_task(%[[A0]])
// CHECK: aiex.dma_start_task(%[[RB]])
// The trailing append is configured after the readback start; its await is NOT
// hoisted, so it stays after that start.
// CHECK: %[[A1:.*]] = aiex.dma_configure_task_for @appendV
// CHECK: aiex.dma_await_task(%[[A1]])
module {
  aie.device(npu1_1col) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @readback(%shim_noc_tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @appendK(%shim_noc_tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @appendV(%shim_noc_tile_0_0, S2MM, 1)
  } {sym_name = "seg"}
  airrt.module_metadata{}
  func.func @await_appends_trailing(%arg0: memref<64xi32>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %c4_i32 = arith.constant 4 : i32
    %p = airrt.segment_load "seg" : i64
    // append0 then the readback that consumes it
    %a0 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @appendK, air.append_barrier} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    %r0 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @readback, air.await_appends} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    // trailing append with no readback after it
    %a1 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @appendV, air.append_barrier} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    airrt.wait_all %a0, %a1
    return
  }
}
