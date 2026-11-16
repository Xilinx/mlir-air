//===- restore_if_drain.mlir ------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-opt-shim-dma-bds="device=npu1" | FileCheck %s

// A launch-scope scf.if holding host feeds and a drain is left SYNC after
// air-dependency + canonicalize dropped its unused token, which strands the
// branch transfers' tokens: the drain would never be awaited downstream.
// restoreIfDrainToken must rebuild it with a trailing !air.async.token result,
// each branch yielding a wait_all over its dangling tokens, so the launch-body
// drain (air.launch_end) consumes it. A missing else yields an empty wait_all.

// CHECK-LABEL: func.func @if_else_drain
// CHECK: air.launch
// CHECK: %[[IF:.*]] = scf.if %{{.*}} -> (!air.async.token) {
// CHECK:   %[[P0:.*]] = air.channel.put async {{ *}}@channel_0
// CHECK:   %[[G0:.*]] = air.channel.get async {{ *}}@channel_2
// CHECK:   %[[W0:.*]] = air.wait_all async [%[[P0]], %[[G0]]]
// CHECK:   scf.yield %[[W0]] : !air.async.token
// CHECK: } else {
// CHECK:   %[[P1:.*]] = air.channel.put async {{ *}}@channel_1
// CHECK:   %[[G1:.*]] = air.channel.get async {{ *}}@channel_2
// CHECK:   %[[W1:.*]] = air.wait_all async [%[[P1]], %[[G1]]]
// CHECK:   scf.yield %[[W1]] : !air.async.token
// CHECK: }
// CHECK: air.wait_all [%[[IF]]] {air.launch_end}
func.func @if_else_drain(%arg0: memref<512x512xbf16>, %arg1: memref<512x512xbf16>, %n: index) {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%tx) in (%sx=%c1) args(%buf=%arg0, %out=%arg1, %k=%n) : memref<512x512xbf16>, memref<512x512xbf16>, index {
    %c0 = arith.constant 0 : index
    %c1_0 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c512 = arith.constant 512 : index
    %cond = arith.cmpi slt, %c0, %k : index
    scf.if %cond {
      %p = air.channel.put async @channel_0[%c0, %c0] (%buf[%c0, %c0] [%c512, %c64] [%c512, %c1_0]) {id = 1 : i32} : (memref<512x512xbf16>)
      %g = air.channel.get async @channel_2[%c0, %c0] (%out[%c0, %c0] [%c512, %c64] [%c512, %c1_0]) {id = 2 : i32} : (memref<512x512xbf16>)
    } else {
      %q = air.channel.put async @channel_1[%c0, %c0] (%buf[%c0, %c0] [%c512, %c64] [%c512, %c1_0]) {id = 3 : i32} : (memref<512x512xbf16>)
      %h = air.channel.get async @channel_2[%c0, %c0] (%out[%c0, %c0] [%c512, %c64] [%c512, %c1_0]) {id = 4 : i32} : (memref<512x512xbf16>)
    }
  }
  return
}

// CHECK-LABEL: func.func @if_no_else_drain
// CHECK: air.launch
// CHECK: %[[IF:.*]] = scf.if %{{.*}} -> (!air.async.token) {
// CHECK:   %[[G:.*]] = air.channel.get async {{ *}}@channel_2
// CHECK:   scf.yield %[[G]] : !air.async.token
// CHECK: } else {
// CHECK:   %[[E:.*]] = air.wait_all async
// CHECK:   scf.yield %[[E]] : !air.async.token
// CHECK: }
// CHECK: air.wait_all [%[[IF]]] {air.launch_end}
func.func @if_no_else_drain(%arg1: memref<512x512xbf16>, %n: index) {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%tx) in (%sx=%c1) args(%out=%arg1, %k=%n) : memref<512x512xbf16>, index {
    %c0 = arith.constant 0 : index
    %c1_0 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c512 = arith.constant 512 : index
    %cond = arith.cmpi slt, %c0, %k : index
    scf.if %cond {
      %g = air.channel.get async @channel_2[%c0, %c0] (%out[%c0, %c0] [%c512, %c64] [%c512, %c1_0]) {id = 2 : i32} : (memref<512x512xbf16>)
    }
  }
  return
}

air.channel @channel_0 [1, 1]
air.channel @channel_1 [1, 1]
air.channel @channel_2 [1, 1]
