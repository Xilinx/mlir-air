//===- channel_memory_effects.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -transform-interpreter -split-input-file | FileCheck %s

// air.channel.put reads its source and air.channel.get writes its
// destination; both also read and write the channel. Passes that go by
// memory effects see that.

// A buffer the core fills and sends is not dead: the write and the put stay.

// CHECK-LABEL: func.func @sent
// CHECK: %[[B:.*]] = memref.alloc()
// CHECK: vector.transfer_write %{{.*}}, %[[B]]
// CHECK: air.channel.put @c[] (%[[B]][] [] [])
air.channel @c [1, 1]
func.func @sent(%v: vector<16xf32>) {
  %c0 = arith.constant 0 : index
  %b = memref.alloc() : memref<16xf32, 2>
  vector.transfer_write %v, %b[%c0] {in_bounds = [true]} : vector<16xf32>, memref<16xf32, 2>
  air.channel.put @c[] (%b[] [] []) : (memref<16xf32, 2>)
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.memref.erase_dead_alloc_and_stores %f : (!transform.any_op) -> ()
    transform.yield
  }
}

// -----

// A get whose data nobody reads still takes it off the channel: it stays.

// CHECK-LABEL: func.func @received
// CHECK: %[[B:.*]] = memref.alloc()
// CHECK: air.channel.get @c[] (%[[B]][] [] [])
air.channel @c [1, 1]
func.func @received() {
  %b = memref.alloc() : memref<16xf32, 2>
  air.channel.get @c[] (%b[] [] []) : (memref<16xf32, 2>)
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.memref.erase_dead_alloc_and_stores %f : (!transform.any_op) -> ()
    transform.yield
  }
}

// -----

// A put of another buffer does not stop a vector written to a private buffer
// from being forwarded to its read; the private buffer goes, the sent one
// stays.

// CHECK-LABEL: func.func @beside
// CHECK-SAME: %[[V:.*]]: vector<16xf32>
// CHECK: %[[S:.*]] = memref.alloc()
// CHECK-NOT: memref.alloc
// CHECK: vector.transfer_write %[[V]], %[[S]]
// CHECK-NEXT: air.channel.put @c[] (%[[S]][] [] [])
// CHECK-NEXT: return %[[V]]
air.channel @c [1, 1]
func.func @beside(%v: vector<16xf32>) -> vector<16xf32> {
  %c0 = arith.constant 0 : index
  %p = arith.constant 0.0 : f32
  %s = memref.alloc() : memref<16xf32, 2>
  %b = memref.alloc() : memref<16xf32, 2>
  vector.transfer_write %v, %b[%c0] {in_bounds = [true]} : vector<16xf32>, memref<16xf32, 2>
  vector.transfer_write %v, %s[%c0] {in_bounds = [true]} : vector<16xf32>, memref<16xf32, 2>
  air.channel.put @c[] (%s[] [] []) : (memref<16xf32, 2>)
  %r = vector.transfer_read %b[%c0], %p {in_bounds = [true]} : memref<16xf32, 2>, vector<16xf32>
  return %r : vector<16xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.memref.erase_dead_alloc_and_stores %f : (!transform.any_op) -> ()
    transform.yield
  }
}

// -----

// The put between two writes reads the first: it is not dead.

// CHECK-LABEL: func.func @between
// CHECK: vector.transfer_write %{{.*}}, %[[B:[a-z0-9_]+]][
// CHECK-NEXT: air.channel.put @c[] (%[[B]][] [] [])
// CHECK-NEXT: vector.transfer_write
air.channel @c [1, 1]
func.func @between(%x: vector<16xf32>, %y: vector<16xf32>) {
  %c0 = arith.constant 0 : index
  %b = memref.alloc() : memref<16xf32, 2>
  vector.transfer_write %x, %b[%c0] {in_bounds = [true]} : vector<16xf32>, memref<16xf32, 2>
  air.channel.put @c[] (%b[] [] []) : (memref<16xf32, 2>)
  vector.transfer_write %y, %b[%c0] {in_bounds = [true]} : vector<16xf32>, memref<16xf32, 2>
  air.channel.put @c[] (%b[] [] []) : (memref<16xf32, 2>)
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.memref.erase_dead_alloc_and_stores %f : (!transform.any_op) -> ()
    transform.yield
  }
}
