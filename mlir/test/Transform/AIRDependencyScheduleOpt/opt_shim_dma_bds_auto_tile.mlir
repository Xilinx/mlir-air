//===- opt_shim_dma_bds_auto_tile.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-opt-shim-dma-bds="device=npu1 auto-derive-tile-sizes=true" | FileCheck %s

// Auto-derived shim DMA tile size from the BD-queue cost model:
//
//   N = min(trip_count, ⌊K / B⌋)
//
// K = XAIE_DMA_MAX_QUEUE_SIZE = 4 (per-channel hardware DMA start-queue
// depth, aie-rt/driver/src/dma/xaie_dma.c:45). B = max distinct BD
// patterns per shim channel in the loop body, counted via the
// chansMappedToEquivalentBDs predicate used by getRepeatCounts.
//
// Each func exercises one (B, trip count, channel layout) point of the
// model. The post-pass count of channel.put/get ops per source channel
// symbol equals trip × B (= number of unrolled iter-equivalents × source
// ops per iter), unless the loop fully absorbs into BD wrap-and-stride
// (then count collapses to the number of unique source ops on that
// channel after folding).
//
// The existing opt_shim_dma_bds.mlir suite covers:
//   - no flag / no tile sizes (today's default — no per-channel tiling)
//   - explicit `shim-dma-tile-sizes=N,M` (user override)
// Those paths are unchanged by this PR; auto-derive is opt-in via the
// pass option and aircc enables it by default.

module {

  // B = 1, trip = 8. tile = min(8, ⌊4/1⌋) = 4. Wrap-and-stride absorbs
  // the trip → 1 surviving put.
  //
  // CHECK-LABEL: func.func @b1_one_put_per_iter
  // CHECK-COUNT-1: air.channel.put async{{.*}}@ch_b1
  // CHECK-NOT: air.channel.put async{{.*}}@ch_b1
  air.channel @ch_b1 [1]
  func.func @b1_one_put_per_iter(%arg0: memref<512xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%i) in (%n=%c1) args(%a=%arg0) : memref<512xbf16> {
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c64 = arith.constant 64 : index
      %c1_0 = arith.constant 1 : index
      %tok0 = air.wait_all async
      %1 = scf.for %j = %c0 to %c8 step %c1_0
          iter_args(%tok = %tok0) -> (!air.async.token) {
        %2 = air.channel.put async [%tok] @ch_b1[]
            (%a[%j, %c0] [%c1_0, %c64] [%c64, %c1_0])
            {metadata = @airMemcpyId0} : (memref<512xbf16>)
        scf.yield %2 : !air.async.token
      }
    }
    return
  }

  // B = 2 (cascade-style: two distinct BD patterns on the same channel
  // per iter), trip = 8. tile = ⌊4/2⌋ = 2. After tiling+unroll: 8
  // unrolled iter-equivalents × 2 source ops = 16 ops on @ch_b2.
  //
  // CHECK-LABEL: func.func @b2_two_puts_same_channel
  // CHECK-COUNT-16: air.channel.put async{{.*}}@ch_b2
  // CHECK-NOT: air.channel.put async{{.*}}@ch_b2
  air.channel @ch_b2 [1]
  func.func @b2_two_puts_same_channel(%arg0: memref<512xbf16>, %arg1: memref<512xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%i) in (%n=%c1) args(%a=%arg0, %b=%arg1) : memref<512xbf16>, memref<512xbf16> {
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c64 = arith.constant 64 : index
      %c1_0 = arith.constant 1 : index
      %tok0 = air.wait_all async
      %1 = scf.for %j = %c0 to %c8 step %c1_0
          iter_args(%tok = %tok0) -> (!air.async.token) {
        %2 = air.channel.put async [%tok] @ch_b2[]
            (%a[%j, %c0] [%c1_0, %c64] [%c64, %c1_0])
            {metadata = @airMemcpyId1} : (memref<512xbf16>)
        %3 = air.channel.put async [%2] @ch_b2[]
            (%b[%j, %c0] [%c1_0, %c64] [%c64, %c1_0])
            {metadata = @airMemcpyId2} : (memref<512xbf16>)
        scf.yield %3 : !air.async.token
      }
    }
    return
  }

  // B = 3, trip = 8. tile = max(1, ⌊4/3⌋) = 1. 8 unrolled iters × 3
  // source ops = 24 ops on @ch_b3.
  //
  // CHECK-LABEL: func.func @b3_three_puts_same_channel
  // CHECK-COUNT-24: air.channel.put async{{.*}}@ch_b3
  // CHECK-NOT: air.channel.put async{{.*}}@ch_b3
  air.channel @ch_b3 [1]
  func.func @b3_three_puts_same_channel(%arg0: memref<512xbf16>, %arg1: memref<512xbf16>, %arg2: memref<512xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%i) in (%n=%c1) args(%a=%arg0, %b=%arg1, %c=%arg2) : memref<512xbf16>, memref<512xbf16>, memref<512xbf16> {
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c64 = arith.constant 64 : index
      %c1_0 = arith.constant 1 : index
      %tok0 = air.wait_all async
      %1 = scf.for %j = %c0 to %c8 step %c1_0
          iter_args(%tok = %tok0) -> (!air.async.token) {
        %2 = air.channel.put async [%tok] @ch_b3[]
            (%a[%j, %c0] [%c1_0, %c64] [%c64, %c1_0])
            {metadata = @airMemcpyId3} : (memref<512xbf16>)
        %3 = air.channel.put async [%2] @ch_b3[]
            (%b[%j, %c0] [%c1_0, %c64] [%c64, %c1_0])
            {metadata = @airMemcpyId4} : (memref<512xbf16>)
        %4 = air.channel.put async [%3] @ch_b3[]
            (%c[%j, %c0] [%c1_0, %c64] [%c64, %c1_0])
            {metadata = @airMemcpyId5} : (memref<512xbf16>)
        scf.yield %4 : !air.async.token
      }
    }
    return
  }

  // Trip clamp: B = 1, trip = 2. tile = min(2, 4) = 2. Loop still
  // fully absorbs into BD wrap-and-stride → 1 surviving put.
  //
  // CHECK-LABEL: func.func @trip_clamp_b1_trip2
  // CHECK-COUNT-1: air.channel.put async{{.*}}@ch_tc
  // CHECK-NOT: air.channel.put async{{.*}}@ch_tc
  air.channel @ch_tc [1]
  func.func @trip_clamp_b1_trip2(%arg0: memref<512xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%i) in (%n=%c1) args(%a=%arg0) : memref<512xbf16> {
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c1_0 = arith.constant 1 : index
      %tok0 = air.wait_all async
      %1 = scf.for %j = %c0 to %c2 step %c1_0
          iter_args(%tok = %tok0) -> (!air.async.token) {
        %2 = air.channel.put async [%tok] @ch_tc[]
            (%a[%j, %c0] [%c1_0, %c64] [%c64, %c1_0])
            {metadata = @airMemcpyId6} : (memref<512xbf16>)
        scf.yield %2 : !air.async.token
      }
    }
    return
  }

  // Put + Get mix on the same channel. ChannelInterface covers both, so
  // they are grouped together → B = 2 → tile = 2. 8 unrolled iter-
  // equivalents × (1 put + 1 get) = 8 puts + 8 gets = 16 total ops on
  // @ch_pg.
  //
  // CHECK-LABEL: func.func @put_get_mix_same_channel
  // CHECK-COUNT-16: air.channel.{{(put|get)}} async{{.*}}@ch_pg
  // CHECK-NOT: air.channel.{{(put|get)}} async{{.*}}@ch_pg
  air.channel @ch_pg [1]
  func.func @put_get_mix_same_channel(%arg0: memref<512xbf16>, %arg1: memref<512xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%i) in (%n=%c1) args(%a=%arg0, %b=%arg1) : memref<512xbf16>, memref<512xbf16> {
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c64 = arith.constant 64 : index
      %c1_0 = arith.constant 1 : index
      %tok0 = air.wait_all async
      %1 = scf.for %j = %c0 to %c8 step %c1_0
          iter_args(%tok = %tok0) -> (!air.async.token) {
        %2 = air.channel.put async [%tok] @ch_pg[]
            (%a[%j, %c0] [%c1_0, %c64] [%c64, %c1_0])
            {metadata = @airMemcpyId10} : (memref<512xbf16>)
        %3 = air.channel.get async [%2] @ch_pg[]
            (%b[%j, %c0] [%c1_0, %c64] [%c64, %c1_0])
            {metadata = @airMemcpyId11} : (memref<512xbf16>)
        scf.yield %3 : !air.async.token
      }
    }
    return
  }
}
