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
