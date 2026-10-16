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

// -----

// A flattened loop: the induction variable delinearized into (2, 4) indexes
// %m's leading dims, which together advance by one row per trip, so %m is
// walked by a pointer. %s takes the two indices in the other order (no
// constant advance) and is left as it is.
// CHECK-LABEL: @delinearized_iv
// CHECK: %[[F:.*]] = memref.collapse_shape %arg0
// CHECK: scf.for {{.*}} iter_args(%[[P:.*]] = %{{.*}}) -> (index)
// CHECK: vector.transfer_read %[[F]][%[[P]]]
// CHECK: vector.transfer_read %arg1[%{{.*}}, %{{.*}}, %{{.*}}]
// CHECK: %[[C8:.*]] = arith.constant 8 : index
// CHECK: arith.addi %[[P]], %[[C8]] : index
func.func @delinearized_iv(%m: memref<2x4x8xi16, 2>, %s: memref<4x2x8xi16, 2>, %out: memref<8xi16, 2>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %pad = arith.constant 0 : i16
  scf.for %i = %c0 to %c8 step %c1 {
    %a = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%i)
    %b = affine.apply affine_map<(d0) -> (d0 mod 4)>(%i)
    %v = vector.transfer_read %m[%a, %b, %c0], %pad {in_bounds = [true]} : memref<2x4x8xi16, 2>, vector<8xi16>
    %w = vector.transfer_read %s[%b, %a, %c0], %pad {in_bounds = [true]} : memref<4x2x8xi16, 2>, vector<8xi16>
    %x = arith.addi %v, %w : vector<8xi16>
    vector.transfer_write %x, %out[%c0] {in_bounds = [true]} : vector<8xi16>, memref<8xi16, 2>
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

// Four reads of one buffer a row apart, all moving one block per trip: one
// pointer, the others at constant offsets from it.
// CHECK-LABEL: @shared_pointer
// CHECK: scf.for {{.*}} iter_args(%[[P:.*]] = %{{.*}}) -> (index)
// CHECK: vector.transfer_read %{{.*}}[%[[P]]]
// CHECK: %[[C8:.*]] = arith.constant 8 : index
// CHECK: %[[P8:.*]] = arith.addi %[[P]], %[[C8]] : index
// CHECK: vector.transfer_read %{{.*}}[%[[P8]]]
func.func @shared_pointer(%m: memref<8x2x8xi16, 2>, %out: memref<2x8xi16, 2>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %pad = arith.constant 0 : i16
  scf.for %i = %c0 to %c8 step %c1 {
    %v = vector.transfer_read %m[%i, %c0, %c0], %pad {in_bounds = [true]} : memref<8x2x8xi16, 2>, vector<8xi16>
    %w = vector.transfer_read %m[%i, %c1, %c0], %pad {in_bounds = [true]} : memref<8x2x8xi16, 2>, vector<8xi16>
    vector.transfer_write %v, %out[%c0, %c0] {in_bounds = [true]} : vector<8xi16>, memref<2x8xi16, 2>
    vector.transfer_write %w, %out[%c1, %c0] {in_bounds = [true]} : vector<8xi16>, memref<2x8xi16, 2>
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

// The pointer's update for the next trip comes after the last access through
// it, so the accesses walk forward from it rather than reaching back from an
// update made early in the body.
// CHECK-LABEL: @update_after_last_access
// CHECK: scf.for {{.*}} iter_args(%[[P:.*]] = %{{.*}}) -> (index)
// CHECK: vector.transfer_write %{{.*}}, %{{.*}}[%[[P]]]
// CHECK: %[[P8:.*]] = arith.addi %[[P]], %{{.*}} : index
// CHECK: vector.transfer_write %{{.*}}, %{{.*}}[%[[P8]]]
// CHECK-NOT: vector.transfer_write
// CHECK: %[[C16:.*]] = arith.constant 16 : index
// CHECK: %[[N:.*]] = arith.addi %[[P]], %[[C16]] : index
// CHECK-NEXT: scf.yield %[[N]]
func.func @update_after_last_access(%in: memref<8xi16, 2>, %m: memref<8x2x8xi16, 2>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %pad = arith.constant 0 : i16
  scf.for %i = %c0 to %c8 step %c1 {
    %v = vector.transfer_read %in[%c0], %pad {in_bounds = [true]} : memref<8xi16, 2>, vector<8xi16>
    vector.transfer_write %v, %m[%i, %c0, %c0] {in_bounds = [true]} : vector<8xi16>, memref<8x2x8xi16, 2>
    vector.transfer_write %v, %m[%i, %c1, %c0] {in_bounds = [true]} : vector<8xi16>, memref<8x2x8xi16, 2>
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
