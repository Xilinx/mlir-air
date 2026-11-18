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

// Nested switches: the inner one is rebuilt first, and its token is what the
// outer case yields.

// CHECK-LABEL: func.func @nested_switch_drain
// CHECK: %[[OUT:.*]] = scf.index_switch %{{.*}} -> !air.async.token
// CHECK: case 0 {
// CHECK: %[[IN:.*]] = scf.index_switch %{{.*}} -> !air.async.token
// CHECK: air.channel.put async
// CHECK: air.channel.put async
// CHECK: scf.yield %[[IN]] : !air.async.token
// CHECK: air.wait_all [%[[OUT]]] {air.launch_end}
func.func @nested_switch_drain(%arg0: memref<512x512xbf16>, %m: index, %n: index) {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%tx) in (%sx=%c1) args(%buf=%arg0, %mode=%m, %sub=%n) : memref<512x512xbf16>, index, index {
    %c0 = arith.constant 0 : index
    %c1_0 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c512 = arith.constant 512 : index
    scf.index_switch %mode
    case 0 {
      scf.index_switch %sub
      case 0 {
        %p = air.channel.put async @channel_0[%c0, %c0] (%buf[%c0, %c0] [%c512, %c64] [%c512, %c1_0]) {id = 1 : i32} : (memref<512x512xbf16>)
        scf.yield
      }
      default {
        %r = air.channel.put async @channel_0[%c0, %c0] (%buf[%c0, %c64] [%c512, %c64] [%c512, %c1_0]) {id = 3 : i32} : (memref<512x512xbf16>)
        scf.yield
      }
      scf.yield
    }
    default {
      %q = air.channel.put async @channel_1[%c0, %c0] (%buf[%c0, %c0] [%c512, %c64] [%c512, %c1_0]) {id = 2 : i32} : (memref<512x512xbf16>)
      scf.yield
    }
  }
  return
}

// An inner switch that also yields a value keeps it next to its new token.

// CHECK-LABEL: func.func @nested_switch_with_value
// CHECK: scf.index_switch %{{.*}} -> !air.async.token
// CHECK: %[[R:.*]]:2 = scf.index_switch %{{.*}} -> index, !air.async.token
// CHECK: scf.yield %{{.*}}, %{{.*}} : index, !air.async.token
// CHECK: scf.yield %{{.*}}, %{{.*}} : index, !air.async.token
// CHECK: air.channel.put async {{.*}}[%{{.*}}, %[[R]]#0]
func.func @nested_switch_with_value(%arg0: memref<512x512xbf16>, %m: index, %n: index) {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%tx) in (%sx=%c1) args(%buf=%arg0, %mode=%m, %sub=%n) : memref<512x512xbf16>, index, index {
    %c0 = arith.constant 0 : index
    %c1_0 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c512 = arith.constant 512 : index
    scf.index_switch %mode
    case 0 {
      %col = scf.index_switch %sub -> index
      case 0 {
        %p = air.channel.put async @channel_0[%c0, %c0] (%buf[%c0, %c0] [%c512, %c64] [%c512, %c1_0]) {id = 1 : i32} : (memref<512x512xbf16>)
        scf.yield %c0 : index
      }
      default {
        scf.yield %c64 : index
      }
      %q = air.channel.put async @channel_1[%c0, %c0] (%buf[%c0, %col] [%c512, %c64] [%c512, %c1_0]) {id = 2 : i32} : (memref<512x512xbf16>)
      scf.yield
    }
    default {
      scf.yield
    }
  }
  return
}

// An outer switch that already has a drain token waits on the token its
// nested switch was just given.

// CHECK-LABEL: func.func @async_outer_switch
// CHECK: %[[OUT:.*]] = scf.index_switch %{{.*}} -> !air.async.token
// CHECK: case 0 {
// CHECK: %[[IN:.*]] = scf.index_switch %{{.*}} -> !air.async.token
// CHECK: air.channel.put async
// CHECK: scf.yield %[[IN]] : !air.async.token
// CHECK: default {
// CHECK: air.wait_all [%[[OUT]]] {air.launch_end}
func.func @async_outer_switch(%arg0: memref<512x512xbf16>, %m: index, %n: index) {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%tx) in (%sx=%c1) args(%buf=%arg0, %mode=%m, %sub=%n) : memref<512x512xbf16>, index, index {
    %c0 = arith.constant 0 : index
    %c1_0 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c512 = arith.constant 512 : index
    %t = scf.index_switch %mode -> !air.async.token
    case 0 {
      scf.index_switch %sub
      case 0 {
        %p = air.channel.put async @channel_0[%c0, %c0] (%buf[%c0, %c0] [%c512, %c64] [%c512, %c1_0]) {id = 1 : i32} : (memref<512x512xbf16>)
        scf.yield
      }
      default {
        scf.yield
      }
      %w = air.wait_all async
      scf.yield %w : !air.async.token
    }
    default {
      %w2 = air.wait_all async
      scf.yield %w2 : !air.async.token
    }
  }
  return
}
