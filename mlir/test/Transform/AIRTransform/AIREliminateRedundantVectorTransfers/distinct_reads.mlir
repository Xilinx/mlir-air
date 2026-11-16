//===- distinct_reads.mlir -------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -transform-interpreter -split-input-file %s | FileCheck %s

// Reads at the same indices that differ in anything else must both stay.

// CHECK-LABEL: @different_permutation_map
// CHECK-COUNT-2: vector.transfer_read
func.func @different_permutation_map(%m: memref<8x8xi32>) -> (vector<4xi32>, vector<4xi32>) {
  %c0 = arith.constant 0 : index
  %p = arith.constant 0 : i32
  %0 = vector.transfer_read %m[%c0, %c0], %p {in_bounds = [true], permutation_map = affine_map<(d0, d1) -> (d1)>} : memref<8x8xi32>, vector<4xi32>
  %1 = vector.transfer_read %m[%c0, %c0], %p {in_bounds = [true], permutation_map = affine_map<(d0, d1) -> (d0)>} : memref<8x8xi32>, vector<4xi32>
  return %0, %1 : vector<4xi32>, vector<4xi32>
}

// CHECK-LABEL: @different_padding
// CHECK-COUNT-2: vector.transfer_read
func.func @different_padding(%m: memref<8xi32>, %i: index) -> (vector<4xi32>, vector<4xi32>) {
  %p0 = arith.constant 0 : i32
  %p1 = arith.constant 1 : i32
  %0 = vector.transfer_read %m[%i], %p0 : memref<8xi32>, vector<4xi32>
  %1 = vector.transfer_read %m[%i], %p1 : memref<8xi32>, vector<4xi32>
  return %0, %1 : vector<4xi32>, vector<4xi32>
}

// CHECK-LABEL: @different_mask
// CHECK-COUNT-2: vector.transfer_read
func.func @different_mask(%m: memref<8xi32>, %k0: vector<4xi1>, %k1: vector<4xi1>) -> (vector<4xi32>, vector<4xi32>) {
  %c0 = arith.constant 0 : index
  %p = arith.constant 0 : i32
  %0 = vector.transfer_read %m[%c0], %p, %k0 : memref<8xi32>, vector<4xi32>
  %1 = vector.transfer_read %m[%c0], %p, %k1 : memref<8xi32>, vector<4xi32>
  return %0, %1 : vector<4xi32>, vector<4xi32>
}

// CHECK-LABEL: @different_in_bounds
// CHECK-COUNT-2: vector.transfer_read
func.func @different_in_bounds(%m: memref<8xi32>, %i: index) -> (vector<4xi32>, vector<4xi32>) {
  %p = arith.constant 0 : i32
  %0 = vector.transfer_read %m[%i], %p {in_bounds = [true]} : memref<8xi32>, vector<4xi32>
  %1 = vector.transfer_read %m[%i], %p {in_bounds = [false]} : memref<8xi32>, vector<4xi32>
  return %0, %1 : vector<4xi32>, vector<4xi32>
}

// An op with unknown effects may write the memref.
func.func private @may_write(memref<8xi32>)

// CHECK-LABEL: @call_between_reads
// CHECK: vector.transfer_read
// CHECK: call @may_write
// CHECK: vector.transfer_read
func.func @call_between_reads(%m: memref<8xi32>) -> (vector<4xi32>, vector<4xi32>) {
  %c0 = arith.constant 0 : index
  %p = arith.constant 0 : i32
  %0 = vector.transfer_read %m[%c0], %p {in_bounds = [true]} : memref<8xi32>, vector<4xi32>
  func.call @may_write(%m) : (memref<8xi32>) -> ()
  %1 = vector.transfer_read %m[%c0], %p {in_bounds = [true]} : memref<8xi32>, vector<4xi32>
  return %0, %1 : vector<4xi32>, vector<4xi32>
}

// Padding from separate constants with the same value still matches.
// CHECK-LABEL: @equal_padding_constants
// CHECK: %[[R:.*]] = vector.transfer_read
// CHECK-NOT: vector.transfer_read
// CHECK: return %[[R]], %[[R]]
func.func @equal_padding_constants(%m: memref<8xi32>, %i: index) -> (vector<4xi32>, vector<4xi32>) {
  %p0 = arith.constant 0 : i32
  %p1 = arith.constant 0 : i32
  %0 = vector.transfer_read %m[%i], %p0 : memref<8xi32>, vector<4xi32>
  %1 = vector.transfer_read %m[%i], %p1 : memref<8xi32>, vector<4xi32>
  return %0, %1 : vector<4xi32>, vector<4xi32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %0 = transform.air.eliminate_redundant_vector_transfers %f : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// A write through another view of the buffer may change what the second read
// sees.
// CHECK-LABEL: @write_through_alias
// CHECK: vector.transfer_read
// CHECK: vector.transfer_write
// CHECK: vector.transfer_read
func.func @write_through_alias(%m: memref<2x4xi32>, %w: vector<4xi32>) -> (vector<4xi32>, vector<4xi32>) {
  %c0 = arith.constant 0 : index
  %p = arith.constant 0 : i32
  %flat = memref.collapse_shape %m [[0, 1]] : memref<2x4xi32> into memref<8xi32>
  %0 = vector.transfer_read %m[%c0, %c0], %p {in_bounds = [true]} : memref<2x4xi32>, vector<4xi32>
  vector.transfer_write %w, %flat[%c0] {in_bounds = [true]} : vector<4xi32>, memref<8xi32>
  %1 = vector.transfer_read %m[%c0, %c0], %p {in_bounds = [true]} : memref<2x4xi32>, vector<4xi32>
  return %0, %1 : vector<4xi32>, vector<4xi32>
}

// A write to a separate allocation does not block.
// CHECK-LABEL: @write_to_other_alloc
// CHECK: %[[R:.*]] = vector.transfer_read
// CHECK: vector.transfer_write
// CHECK-NOT: vector.transfer_read
// CHECK: return %[[R]], %[[R]]
func.func @write_to_other_alloc(%w: vector<4xi32>) -> (vector<4xi32>, vector<4xi32>) {
  %c0 = arith.constant 0 : index
  %p = arith.constant 0 : i32
  %a = memref.alloc() : memref<8xi32>
  %b = memref.alloc() : memref<8xi32>
  %0 = vector.transfer_read %a[%c0], %p {in_bounds = [true]} : memref<8xi32>, vector<4xi32>
  vector.transfer_write %w, %b[%c0] {in_bounds = [true]} : vector<4xi32>, memref<8xi32>
  %1 = vector.transfer_read %a[%c0], %p {in_bounds = [true]} : memref<8xi32>, vector<4xi32>
  return %0, %1 : vector<4xi32>, vector<4xi32>
}

// Indices from two loads are not known to be equal.
// CHECK-LABEL: @indices_from_loads
// CHECK-COUNT-2: vector.transfer_read
func.func @indices_from_loads(%m: memref<8xi32>, %idx: memref<1xindex>) -> (vector<4xi32>, vector<4xi32>) {
  %c0 = arith.constant 0 : index
  %p = arith.constant 0 : i32
  %i = memref.load %idx[%c0] : memref<1xindex>
  %0 = vector.transfer_read %m[%i], %p : memref<8xi32>, vector<4xi32>
  %j = memref.load %idx[%c0] : memref<1xindex>
  %1 = vector.transfer_read %m[%j], %p : memref<8xi32>, vector<4xi32>
  return %0, %1 : vector<4xi32>, vector<4xi32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %0 = transform.air.eliminate_redundant_vector_transfers %f : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// Different results of one multi-result op are different indices.
// CHECK-LABEL: @indices_from_different_results
// CHECK-COUNT-2: vector.transfer_read
func.func @indices_from_different_results(%m: memref<8x8xi32>, %i: index) -> (vector<4xi32>, vector<4xi32>) {
  %c0 = arith.constant 0 : index
  %p = arith.constant 0 : i32
  %d:2 = affine.delinearize_index %i into (8, 8) : index, index
  %0 = vector.transfer_read %m[%c0, %d#0], %p : memref<8x8xi32>, vector<4xi32>
  %1 = vector.transfer_read %m[%c0, %d#1], %p : memref<8x8xi32>, vector<4xi32>
  return %0, %1 : vector<4xi32>, vector<4xi32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %0 = transform.air.eliminate_redundant_vector_transfers %f : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
