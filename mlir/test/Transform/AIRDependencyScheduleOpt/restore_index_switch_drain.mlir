//===- restore_index_switch_drain.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-opt-shim-dma-bds="device=npu1" | FileCheck %s

// A launch-scope scf.index_switch selecting per-iteration host feeds is left
// SYNC (no async result) after air-dependency + canonicalize dropped its unused
// drain token, stranding the branch feeds' tokens. restoreIndexSwitchDrainToken
// (in air-opt-shim-dma-bds, via air::rebuildIndexSwitchWithTrailingAsyncToken)
// must rebuild the switch with a trailing !air.async.token result whose branches
// each yield a drain over their dangling feed tokens, so the launch-body drain
// (air.launch_end) consumes it -- otherwise the launch_end marker is never built
// and the fused multi-iteration launch is not detected downstream.

// CHECK-LABEL: func.func @index_switch_drain
// CHECK: air.launch
// CHECK: %[[SW:.*]] = scf.index_switch %{{.*}} -> !air.async.token
// CHECK: case 0 {
// CHECK: scf.yield %{{.*}} : !air.async.token
// CHECK: }
// CHECK: scf.yield %{{.*}} : !air.async.token
// CHECK: air.wait_all [%[[SW]]] {air.launch_end}
func.func @index_switch_drain(%arg0: memref<512x512xbf16>) {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%tx) in (%sx=%c1) args(%buf=%arg0) : memref<512x512xbf16> {
    %c0 = arith.constant 0 : index
    %c1_0 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c512 = arith.constant 512 : index
    %mode = arith.constant 0 : index
    scf.index_switch %mode
    case 0 {
      %p = air.channel.put async @channel_0[%c0, %c0] (%buf[%c0, %c0] [%c512, %c64] [%c512, %c1_0]) {id = 1 : i32} : (memref<512x512xbf16>)
      scf.yield
    }
    default {
      %q = air.channel.put async @channel_1[%c0, %c0] (%buf[%c0, %c0] [%c512, %c64] [%c512, %c1_0]) {id = 2 : i32} : (memref<512x512xbf16>)
      scf.yield
    }
  }
  return
}
