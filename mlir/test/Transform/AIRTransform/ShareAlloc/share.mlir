//===- share.mlir ----------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -transform-interpreter -split-input-file -verify-diagnostics %s | FileCheck %s

// Two loops one after the other, each with a buffer: one buffer serves both,
// the one of another shape through a reshape of it.
// CHECK-LABEL: @two_loops
// CHECK: %[[A:.*]] = memref.alloc() : memref<4x8xf32, 2>
// CHECK-NOT: memref.alloc
// CHECK: %[[F:.*]] = memref.collapse_shape %[[A]] {{\[}}[0, 1]]
// CHECK: %[[B:.*]] = memref.expand_shape %[[F]] {{\[}}[0, 1, 2]] output_shape [2, 2, 8]
// CHECK: linalg.fill ins(%{{.*}}) outs(%[[A]]
// CHECK: linalg.fill ins(%{{.*}}) outs(%[[B]]
// CHECK: memref.dealloc %[[A]]
// CHECK-NOT: memref.dealloc
func.func @two_loops(%x: f32) {
  %a = memref.alloc() : memref<4x8xf32, 2>
  %b = memref.alloc() : memref<2x2x8xf32, 2>
  linalg.fill ins(%x : f32) outs(%a : memref<4x8xf32, 2>)
  linalg.fill ins(%x : f32) outs(%b : memref<2x2x8xf32, 2>)
  memref.dealloc %a : memref<4x8xf32, 2>
  memref.dealloc %b : memref<2x2x8xf32, 2>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %allocs = transform.structured.match ops{["memref.alloc"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %one = transform.air.share_alloc %allocs : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// Another element count cannot share the buffer.
func.func @count(%x: f32) {
  %a = memref.alloc() : memref<4x8xf32, 2>
  %b = memref.alloc() : memref<4x4xf32, 2>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %allocs = transform.structured.match ops{["memref.alloc"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{targets must have one element type, count and memory space}}
    %one = transform.air.share_alloc %allocs : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// The first target is kept, though later in the block: the buffer has its
// shape, and the earlier alloc becomes the reshape.
// CHECK-LABEL: @keep_first_target
// CHECK: %[[B:.*]] = memref.alloc() : memref<2x2x8xf32, 2>
// CHECK: %[[F:.*]] = memref.collapse_shape %[[B]] {{\[}}[0, 1, 2]]
// CHECK: %[[A:.*]] = memref.expand_shape %[[F]] {{\[}}[0, 1]] output_shape [4, 8]
// CHECK: linalg.fill ins(%{{.*}}) outs(%[[A]]
// CHECK: linalg.fill ins(%{{.*}}) outs(%[[B]]
func.func @keep_first_target(%x: f32) {
  %a = memref.alloc() : memref<4x8xf32, 2>
  %b = memref.alloc() : memref<2x2x8xf32, 2>
  linalg.fill ins(%x : f32) outs(%a : memref<4x8xf32, 2>)
  linalg.fill ins(%x : f32) outs(%b : memref<2x2x8xf32, 2>)
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %a = transform.structured.match ops{["memref.alloc"]} filter_result_type = memref<4x8xf32, 2> in %arg1 : (!transform.any_op) -> !transform.any_op
    %b = transform.structured.match ops{["memref.alloc"]} filter_result_type = memref<2x2x8xf32, 2> in %arg1 : (!transform.any_op) -> !transform.any_op
    %ba = transform.merge_handles %b, %a : !transform.any_op
    %one = transform.air.share_alloc %ba : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// The kept buffer takes the larger alignment and is freed after the last use
// of either buffer.
// CHECK-LABEL: @align_and_free
// CHECK: %[[A:.*]] = memref.alloc() alignment = 64 : memref<16xf32>
// CHECK: memref.store %{{.*}}, %[[A]]
// CHECK-NOT: memref.dealloc
// CHECK: memref.store %{{.*}}, %[[A]]
// CHECK-NEXT: memref.dealloc %[[A]]
// CHECK-NOT: memref.dealloc
func.func @align_and_free(%x: f32) {
  %c0 = arith.constant 0 : index
  %a = memref.alloc() alignment = 16 : memref<16xf32>
  memref.store %x, %a[%c0] : memref<16xf32>
  memref.dealloc %a : memref<16xf32>
  %b = memref.alloc() alignment = 64 : memref<16xf32>
  memref.store %x, %b[%c0] : memref<16xf32>
  memref.dealloc %b : memref<16xf32>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %allocs = transform.structured.match ops{["memref.alloc"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %one = transform.air.share_alloc %allocs : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// One buffer is freed and the other is returned: a shared free would end the
// second one's life too early.
func.func @mixed_free(%x: f32) -> memref<16xf32> {
  %c0 = arith.constant 0 : index
  %a = memref.alloc() : memref<16xf32>
  memref.store %x, %a[%c0] : memref<16xf32>
  memref.dealloc %a : memref<16xf32>
  %b = memref.alloc() : memref<16xf32>
  memref.store %x, %b[%c0] : memref<16xf32>
  return %b : memref<16xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %allocs = transform.structured.match ops{["memref.alloc"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-error@+1 {{either every target or none must be freed}}
    %one = transform.air.share_alloc %allocs : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// The first buffer is freed through a cast, which sharing would not move.
func.func @free_through_cast(%x: f32) {
  %c0 = arith.constant 0 : index
  %a = memref.alloc() : memref<16xf32>
  memref.store %x, %a[%c0] : memref<16xf32>
  %ac = memref.cast %a : memref<16xf32> to memref<?xf32>
  memref.dealloc %ac : memref<?xf32>
  %b = memref.alloc() : memref<16xf32>
  memref.store %x, %b[%c0] : memref<16xf32>
  memref.dealloc %b : memref<16xf32>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %allocs = transform.structured.match ops{["memref.alloc"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-error@+1 {{targets must not be freed through a view or cast}}
    %one = transform.air.share_alloc %allocs : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
