//===- opt_shim_dma_bds_pace_derivation.mlir -------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-opt-shim-dma-bds="device=npu2" -split-input-file | FileCheck %s

// air-opt-shim-dma-bds propagates a launch's air.preserve_shim_dma_order marker
// onto its air.channel.put/get ops (so airrt-to-npu paces them with bounded
// double-buffered backpressure). A host feed is exempted when it reaches no
// herd that also consumes a broadcast channel: with no lockstep sibling it has
// nothing to stay in step with, and lowers fire-and-free instead.

// (1) The feed lands in a herd that also reads a broadcast, so it is paced.
// CHECK-LABEL: func.func @paces_broadcast_coupled_feed
// CHECK: air.channel.put
// CHECK-SAME: air.preserve_shim_dma_order
air.channel @bcast [1, 1] {broadcast_shape = [1, 2]}
air.channel @coupled [1]
func.func @paces_broadcast_coupled_feed(%arg0: memref<512xbf16>) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %0 = air.launch async (%i) in (%n=%c1) args(%a=%arg0) : memref<512xbf16>
      attributes {air.preserve_shim_dma_order} {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c1_0 = arith.constant 1 : index
    %c2_0 = arith.constant 2 : index
    %1 = air.channel.put async @coupled[]
        (%a[%c0] [%c64] [%c1_0]) {metadata = @m0} : (memref<512xbf16>)
    %2 = air.segment async {
      %c1_1 = arith.constant 1 : index
      %c2_1 = arith.constant 2 : index
      %3 = air.herd @h async tile(%tx, %ty) in (%sx=%c1_1, %sy=%c2_1) {
        %b0 = memref.alloc() : memref<64xbf16, 2>
        %b1 = memref.alloc() : memref<64xbf16, 2>
        %4 = air.channel.get async @coupled[] (%b0[] [] []) : (memref<64xbf16, 2>)
        %5 = air.channel.get async @bcast[%tx, %ty] (%b1[] [] []) : (memref<64xbf16, 2>)
        memref.dealloc %b0 : memref<64xbf16, 2>
        memref.dealloc %b1 : memref<64xbf16, 2>
      }
    }
  }
  return
}

// -----

// (2) The feed lands in a herd with no broadcast consumer, so it is exempted.
// The marker is still propagated to the non-feed ops, so match the put's whole
// attribute dictionary: it holds metadata alone (an added marker would sort
// ahead of it).
// CHECK-LABEL: func.func @exempts_uncoupled_feed
// CHECK: air.channel.put
// CHECK-SAME: {metadata = @m1}
air.channel @lone [1]
func.func @exempts_uncoupled_feed(%arg0: memref<512xbf16>) {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%i) in (%n=%c1) args(%a=%arg0) : memref<512xbf16>
      attributes {air.preserve_shim_dma_order} {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c1_0 = arith.constant 1 : index
    %1 = air.channel.put async @lone[]
        (%a[%c0] [%c64] [%c1_0]) {metadata = @m1} : (memref<512xbf16>)
    %2 = air.segment async {
      %c1_1 = arith.constant 1 : index
      %c2_1 = arith.constant 2 : index
      %3 = air.herd @h async tile(%tx, %ty) in (%sx=%c1_1, %sy=%c2_1) {
        %b0 = memref.alloc() : memref<64xbf16, 2>
        %4 = air.channel.get async @lone[] (%b0[] [] []) : (memref<64xbf16, 2>)
        memref.dealloc %b0 : memref<64xbf16, 2>
      }
    }
  }
  return
}

// -----

// (3) The coupling is reached through an L2 relay -- the feed is consumed in the
// segment and re-emitted on another channel before it ever reaches the herd --
// so the reachability must be a fixpoint over the channel graph, not one hop.
// CHECK-LABEL: func.func @paces_through_l2_relay
// CHECK: air.channel.put
// CHECK-SAME: air.preserve_shim_dma_order
air.channel @bcast2 [1, 1] {broadcast_shape = [1, 2]}
air.channel @toL2 [1]
air.channel @l2ToL1 [1]
func.func @paces_through_l2_relay(%arg0: memref<512xbf16>) {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%i) in (%n=%c1) args(%a=%arg0) : memref<512xbf16>
      attributes {air.preserve_shim_dma_order} {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c1_0 = arith.constant 1 : index
    %1 = air.channel.put async @toL2[]
        (%a[%c0] [%c64] [%c1_0]) {metadata = @m2} : (memref<512xbf16>)
    %2 = air.segment async {
      %c1_1 = arith.constant 1 : index
      %c2_1 = arith.constant 2 : index
      %l2 = memref.alloc() : memref<64xbf16, 1>
      %3 = air.channel.get async @toL2[] (%l2[] [] []) : (memref<64xbf16, 1>)
      %4 = air.channel.put async [%3] @l2ToL1[] (%l2[] [] []) : (memref<64xbf16, 1>)
      %5 = air.herd @h async tile(%tx, %ty) in (%sx=%c1_1, %sy=%c2_1) {
        %b0 = memref.alloc() : memref<64xbf16, 2>
        %b1 = memref.alloc() : memref<64xbf16, 2>
        %6 = air.channel.get async @l2ToL1[] (%b0[] [] []) : (memref<64xbf16, 2>)
        %7 = air.channel.get async @bcast2[%tx, %ty] (%b1[] [] []) : (memref<64xbf16, 2>)
        memref.dealloc %b0 : memref<64xbf16, 2>
        memref.dealloc %b1 : memref<64xbf16, 2>
      }
      memref.dealloc %l2 : memref<64xbf16, 1>
    }
  }
  return
}
