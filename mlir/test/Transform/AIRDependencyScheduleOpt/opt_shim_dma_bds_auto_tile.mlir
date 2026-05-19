//===- opt_shim_dma_bds_auto_tile.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Auto-derive returns tile=1 for every level of the perfectly-nested
// shim loop band. The user-override and default-off paths converge to
// the same final IR for these inputs after downstream wrap-and-stride
// folding; the three RUN lines guard against mode-specific regressions.

// RUN: air-opt %s -air-opt-shim-dma-bds="device=npu1 auto-derive-tile-sizes=true" | FileCheck %s
// RUN: air-opt %s -air-opt-shim-dma-bds="device=npu1 shim-dma-tile-sizes=1,1" | FileCheck %s
// RUN: air-opt %s -air-opt-shim-dma-bds="device=npu1" | FileCheck %s

module {

  // B=1, trip=8: tile=1 (B<=1 short-circuit); wrap-and-stride absorbs the loop.
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

  // B=2, trip=8: tile=2 → 16 puts after unroll.
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

  // B=3, trip=8: tile=1 → 24 puts after unroll.
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

  // B=1 short-circuit (trip=2): tile=1; loop absorbed.
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

  // Put+Get on same channel: ChannelInterface covers both, B=2, tile=2.
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

  // B=5 with auto-derive tile=1: 40 puts after downstream unroll.
  // CHECK-LABEL: func.func @b5_above_queue_depth
  // CHECK-COUNT-40: air.channel.put async{{.*}}@ch_b5
  // CHECK-NOT: air.channel.put async{{.*}}@ch_b5
  air.channel @ch_b5 [1]
  func.func @b5_above_queue_depth(%arg0: memref<512xbf16>, %arg1: memref<512xbf16>, %arg2: memref<512xbf16>, %arg3: memref<512xbf16>, %arg4: memref<512xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%i) in (%n=%c1) args(%a=%arg0, %b=%arg1, %c=%arg2, %d=%arg3, %e=%arg4) : memref<512xbf16>, memref<512xbf16>, memref<512xbf16>, memref<512xbf16>, memref<512xbf16> {
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c64 = arith.constant 64 : index
      %c1_0 = arith.constant 1 : index
      %tok0 = air.wait_all async
      %1 = scf.for %j = %c0 to %c8 step %c1_0
          iter_args(%tok = %tok0) -> (!air.async.token) {
        %2 = air.channel.put async [%tok] @ch_b5[]
            (%a[%j, %c0] [%c1_0, %c64] [%c64, %c1_0])
            {metadata = @airMemcpyId20} : (memref<512xbf16>)
        %3 = air.channel.put async [%2] @ch_b5[]
            (%b[%j, %c0] [%c1_0, %c64] [%c64, %c1_0])
            {metadata = @airMemcpyId21} : (memref<512xbf16>)
        %4 = air.channel.put async [%3] @ch_b5[]
            (%c[%j, %c0] [%c1_0, %c64] [%c64, %c1_0])
            {metadata = @airMemcpyId22} : (memref<512xbf16>)
        %5 = air.channel.put async [%4] @ch_b5[]
            (%d[%j, %c0] [%c1_0, %c64] [%c64, %c1_0])
            {metadata = @airMemcpyId23} : (memref<512xbf16>)
        %6 = air.channel.put async [%5] @ch_b5[]
            (%e[%j, %c0] [%c1_0, %c64] [%c64, %c1_0])
            {metadata = @airMemcpyId24} : (memref<512xbf16>)
        scf.yield %6 : !air.async.token
      }
    }
    return
  }

  // Two sibling shim for loops in one launch: first has B=1 (tile=1
  // short-circuit), second has B=2 (tile=2). Exercises per-loop tile
  // resolution.
  // CHECK-LABEL: func.func @two_shim_loops_per_launch
  // CHECK-COUNT-1: air.channel.put async{{.*}}@ch_ml_a
  // CHECK-NOT: air.channel.put async{{.*}}@ch_ml_a
  // CHECK-COUNT-16: air.channel.put async{{.*}}@ch_ml_b
  // CHECK-NOT: air.channel.put async{{.*}}@ch_ml_b
  air.channel @ch_ml_a [1]
  air.channel @ch_ml_b [1]
  func.func @two_shim_loops_per_launch(%arg0: memref<512xbf16>, %arg1: memref<512xbf16>, %arg2: memref<512xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%i) in (%n=%c1) args(%a=%arg0, %b=%arg1, %c=%arg2) : memref<512xbf16>, memref<512xbf16>, memref<512xbf16> {
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c64 = arith.constant 64 : index
      %c1_0 = arith.constant 1 : index
      %tok0 = air.wait_all async
      %1 = scf.for %j = %c0 to %c8 step %c1_0
          iter_args(%tok = %tok0) -> (!air.async.token) {
        %p = air.channel.put async [%tok] @ch_ml_a[]
            (%a[%j, %c0] [%c1_0, %c64] [%c64, %c1_0])
            {metadata = @airMemcpyId30} : (memref<512xbf16>)
        scf.yield %p : !air.async.token
      }
      %2 = scf.for %j = %c0 to %c8 step %c1_0
          iter_args(%tok = %tok0) -> (!air.async.token) {
        %p = air.channel.put async [%tok] @ch_ml_b[]
            (%b[%j, %c0] [%c1_0, %c64] [%c64, %c1_0])
            {metadata = @airMemcpyId31} : (memref<512xbf16>)
        %q = air.channel.put async [%p] @ch_ml_b[]
            (%c[%j, %c0] [%c1_0, %c64] [%c64, %c1_0])
            {metadata = @airMemcpyId32} : (memref<512xbf16>)
        scf.yield %q : !air.async.token
      }
    }
    return
  }
}
