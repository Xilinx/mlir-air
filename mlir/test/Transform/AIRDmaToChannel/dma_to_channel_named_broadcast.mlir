//===- dma_to_channel_named_broadcast.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// A broadcast still works when the channel is named. The specialized set is
// forwarded to the put and the internal indices come from the affine.if that
// guards it, exactly as for an unnamed channel; only the SHAPE is taken from
// the declaration instead of being derived from the set.
//
// So this is silent while the two agree, and reports it when they do not: a
// declaration whose fan-out disagrees with the guard implementing it is a real
// bug, and silently preferring either one hides it.

// RUN: air-opt %s -air-dma-to-channel -split-input-file -verify-diagnostics | FileCheck %s

#set = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 1 >= 0)>
// CHECK: air.channel @bc [1, 1] {broadcast_shape = [1, 2]}
// CHECK-LABEL: func.func @named_broadcast_agrees
// CHECK: air.channel.put{{.*}}@bc[]
// CHECK-SAME: broadcast_set
air.channel @bc [1, 1] {broadcast_shape = [1, 2]}
func.func @named_broadcast_agrees(%arg0: memref<64x64xi32>) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  air.herd @herd_0 tile (%tx, %ty) in (%sx=%c1, %sy=%c2) args(%a=%arg0) : memref<64x64xi32> {
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %cst1 = arith.constant 1 : index
    %alloc = memref.alloc() : memref<32x32xi32, 2>
    affine.if #set()[%tx, %ty] {
      air.dma_memcpy_nd (%alloc[] [] [], %a[%c0, %c0] [%c32, %c32] [%c64, %cst1]) {id = 1 : i32, channel = @bc, broadcast_set = #set} : (memref<32x32xi32, 2>, memref<64x64xi32>)
    }
    memref.dealloc %alloc : memref<32x32xi32, 2>
  }
  return
}

// -----

// The declaration says the stream fans out to four destinations; the guard
// implementing it covers two. Report it.
#set = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 1 >= 0)>
air.channel @bc4 [1, 1] {broadcast_shape = [1, 4]}
func.func @named_broadcast_disagrees(%arg0: memref<64x64xi32>) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  air.herd @herd_0 tile (%tx, %ty) in (%sx=%c1, %sy=%c2) args(%a=%arg0) : memref<64x64xi32> {
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %cst1 = arith.constant 1 : index
    %alloc = memref.alloc() : memref<32x32xi32, 2>
    affine.if #set()[%tx, %ty] {
      // expected-warning @+1 {{implies a fan-out of [1, 2], but the channel it names, @bc4, declares broadcast_shape [1, 4]}}
      air.dma_memcpy_nd (%alloc[] [] [], %a[%c0, %c0] [%c32, %c32] [%c64, %cst1]) {id = 1 : i32, channel = @bc4, broadcast_set = #set} : (memref<32x32xi32, 2>, memref<64x64xi32>)
    }
    memref.dealloc %alloc : memref<32x32xi32, 2>
  }
  return
}
