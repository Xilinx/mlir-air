//===- opt_shim_dma_bds_no_pace_optout.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-opt-shim-dma-bds="device=npu2" -split-input-file | FileCheck %s

// air-opt-shim-dma-bds propagates a launch's air.preserve_shim_dma_order marker
// onto its air.channel.put/get ops (so airrt-to-npu paces them with bounded
// double-buffered backpressure), EXCEPT for ops that opt out via
// air.shim_feed_no_pace -- those lower fire-and-free instead. This covers both
// the default propagation and the per-op opt-out.

// (1) An untagged feed in a preserve launch INHERITS the preserve marker.
// CHECK-LABEL: func.func @propagates_preserve
// CHECK: air.channel.put
// CHECK-SAME: air.preserve_shim_dma_order
air.channel @feed_paced [1]
func.func @propagates_preserve(%arg0: memref<512xbf16>) {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%i) in (%n=%c1) args(%a=%arg0) : memref<512xbf16>
      attributes {air.preserve_shim_dma_order} {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c1_0 = arith.constant 1 : index
    %1 = air.channel.put async @feed_paced[]
        (%a[%c0] [%c64] [%c1_0]) {metadata = @m0} : (memref<512xbf16>)
  }
  return
}

// -----

// (2) A feed tagged air.shim_feed_no_pace is NOT given the preserve marker (it
// lowers fire-and-free), while still keeping its own opt-out attribute.
// CHECK-LABEL: func.func @respects_no_pace_optout
// CHECK: air.channel.put
// CHECK-SAME: air.shim_feed_no_pace
// CHECK-NOT: air.preserve_shim_dma_order
air.channel @feed_free [1]
func.func @respects_no_pace_optout(%arg0: memref<512xbf16>) {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%i) in (%n=%c1) args(%a=%arg0) : memref<512xbf16>
      attributes {air.preserve_shim_dma_order} {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c1_0 = arith.constant 1 : index
    %1 = air.channel.put async @feed_free[]
        (%a[%c0] [%c64] [%c1_0])
        {metadata = @m1, air.shim_feed_no_pace} : (memref<512xbf16>)
  }
  return
}

// -----

// (3) copyChannelSteeringAttrs must carry air.shim_feed_no_pace across the
// per-channel BD fold that rebuilds the op (here two scf.for loops collapse into
// one wrap/stride put). If the marker were dropped on rebuild it would be gone
// before the preserve-marker propagation reads it, silently re-pacing the feed.
// Non-preserve launch so folding actually runs.
// CHECK-LABEL: func.func @survives_bd_fold
// CHECK: air.channel.put
// CHECK-SAME: air.shim_feed_no_pace
air.channel @feed_fold [1]
func.func @survives_bd_fold(%arg0: memref<512x512xbf16>) {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%i) in (%n=%c1) args(%a=%arg0) : memref<512x512xbf16> {
    %c0 = arith.constant 0 : index
    %c1_0 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %c512 = arith.constant 512 : index
    %1 = air.wait_all async
    %2 = scf.for %arg6 = %c0 to %c512 step %c256 iter_args(%arg7 = %1) -> (!air.async.token) {
      %3 = scf.for %arg8 = %c0 to %c512 step %c256 iter_args(%arg9 = %arg7) -> (!air.async.token) {
        %4 = air.channel.put async [%arg9] @feed_fold[]
            (%a[%arg6, %arg8] [%c256, %c256] [%c512, %c1_0])
            {metadata = @m2, air.shim_feed_no_pace} : (memref<512x512xbf16>)
        scf.yield %4 : !air.async.token
      }
      scf.yield %3 : !air.async.token
    }
  }
  return
}
