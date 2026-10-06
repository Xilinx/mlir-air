//===- preconditions.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -transform-interpreter -split-input-file %s | FileCheck %s

// An index that uses the IV twice advances by the sum of its coefficients.
// CHECK-LABEL: @repeated_iv
// CHECK: scf.for {{.*}} iter_args(%[[P:.*]] = %{{.*}}) -> (index)
// CHECK: vector.transfer_read %{{.*}}[%[[P]]]
// CHECK: %[[C2:.*]] = arith.constant 2 : index
// CHECK: %[[N:.*]] = arith.addi %[[P]], %[[C2]] : index
// CHECK: scf.yield %[[N]]
func.func @repeated_iv(%m: memref<64xf32, 2>, %out: memref<2xf32, 2>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %pad = arith.constant 0.0 : f32
  scf.for %i = %c0 to %c16 step %c1 {
    %idx = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%i, %i)
    %v = vector.transfer_read %m[%idx], %pad {in_bounds = [true]} : memref<64xf32, 2>, vector<2xf32>
    vector.transfer_write %v, %out[%c0] {in_bounds = [true]} : vector<2xf32>, memref<2xf32, 2>
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %l = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %h = transform.air.hoist_vector_transfer_pointers %l : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// Rows 16 apart in memory: the memref is not contiguous, so the loop is left
// alone.
// CHECK-LABEL: @non_contiguous_layout
// CHECK-NOT: memref.collapse_shape
// CHECK: vector.transfer_read %{{.*}}[%{{.*}}, %{{.*}}]{{.*}} : memref<8x8xi16, strided<[16, 1]>, 2>, vector<1x8xi16>
func.func @non_contiguous_layout(%m: memref<8x8xi16, strided<[16, 1]>, 2>, %out: memref<8xi16, 2>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %pad = arith.constant 0 : i16
  scf.for %i = %c0 to %c8 step %c1 {
    %v = vector.transfer_read %m[%i, %c0], %pad {in_bounds = [true, true]} : memref<8x8xi16, strided<[16, 1]>, 2>, vector<1x8xi16>
    %f = vector.shape_cast %v : vector<1x8xi16> to vector<8xi16>
    vector.transfer_write %f, %out[%c0] {in_bounds = [true]} : vector<8xi16>, memref<8xi16, 2>
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %l = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %h = transform.air.hoist_vector_transfer_pointers %l : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// A contiguous layout with an offset is collapsed to a rank-1 view that keeps
// the offset, and walked by the row stride.
// CHECK-LABEL: @contiguous_with_offset
// CHECK: %[[F:.*]] = memref.collapse_shape %{{.*}} {{\[}}[0, 1]] : memref<4x8xf32, strided<[8, 1], offset: 16>, 2> into memref<32xf32, strided<[1], offset: 16>, 2>
// CHECK: scf.for {{.*}} iter_args(%[[P:.*]] = %{{.*}}) -> (index)
// CHECK: vector.transfer_read %[[F]][%[[P]]]
// CHECK: %[[C8:.*]] = arith.constant 8 : index
// CHECK: arith.addi %[[P]], %[[C8]] : index
func.func @contiguous_with_offset(%m: memref<4x8xf32, strided<[8, 1], offset: 16>, 2>, %out: memref<8xf32, 2>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %pad = arith.constant 0.0 : f32
  scf.for %i = %c0 to %c4 step %c1 {
    %v = vector.transfer_read %m[%i, %c0], %pad {in_bounds = [true, true]} : memref<4x8xf32, strided<[8, 1], offset: 16>, 2>, vector<1x8xf32>
    %f = vector.shape_cast %v : vector<1x8xf32> to vector<8xf32>
    vector.transfer_write %f, %out[%c0] {in_bounds = [true]} : vector<8xf32>, memref<8xf32, 2>
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %l = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %h = transform.air.hoist_vector_transfer_pointers %l : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// A masked read keeps its mask: the loop is left alone.
// CHECK-LABEL: @masked_read
// CHECK-NOT: memref.collapse_shape
// CHECK: vector.transfer_read %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}}, %{{.*}} {in_bounds = [true, true]} : memref<4x8xf32, 2>, vector<1x8xf32>
func.func @masked_read(%m: memref<4x8xf32, 2>, %out: memref<8xf32, 2>, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %pad = arith.constant 0.0 : f32
  %mask = vector.create_mask %c1, %n : vector<1x8xi1>
  scf.for %i = %c0 to %c4 step %c1 {
    %v = vector.transfer_read %m[%i, %c0], %pad, %mask {in_bounds = [true, true]} : memref<4x8xf32, 2>, vector<1x8xf32>
    %f = vector.shape_cast %v : vector<1x8xf32> to vector<8xf32>
    vector.transfer_write %f, %out[%c0] {in_bounds = [true]} : vector<8xf32>, memref<8xf32, 2>
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %l = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %h = transform.air.hoist_vector_transfer_pointers %l : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
