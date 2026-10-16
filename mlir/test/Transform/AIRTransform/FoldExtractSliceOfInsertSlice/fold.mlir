//===- fold.mlir -----------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -transform-interpreter -split-input-file %s | FileCheck %s

// A slice inside the inserted box reads the source; one disjoint from it
// reads the destination.
// CHECK-LABEL: @inside_and_disjoint
// CHECK-SAME: (%[[S:.*]]: tensor<4x8xf32>, %[[D:.*]]: tensor<8x8xf32>)
// CHECK-DAG: tensor.extract_slice %[[S]][1, 0] [2, 8] [1, 1]
// CHECK-DAG: tensor.extract_slice %[[D]][6, 0] [2, 8] [1, 1]
func.func @inside_and_disjoint(%s: tensor<4x8xf32>, %d: tensor<8x8xf32>) -> (tensor<2x8xf32>, tensor<2x8xf32>) {
  %i = tensor.insert_slice %s into %d[0, 0] [4, 8] [1, 1] : tensor<4x8xf32> into tensor<8x8xf32>
  %a = tensor.extract_slice %i[1, 0] [2, 8] [1, 1] : tensor<8x8xf32> to tensor<2x8xf32>
  %b = tensor.extract_slice %i[6, 0] [2, 8] [1, 1] : tensor<8x8xf32> to tensor<2x8xf32>
  return %a, %b : tensor<2x8xf32>, tensor<2x8xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %r = transform.air.fold_extract_slice_of_insert_slice %f : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// The parts of a join are inserted rank-reducing (the join's new unit dim
// dropped): a slice of one part reads that part, its unit dim expanded back.
// CHECK-LABEL: @join_part
// CHECK-SAME: (%[[P0:.*]]: tensor<4x8xf32>, %[[P1:.*]]: tensor<4x8xf32>)
// CHECK: %[[X:.*]] = tensor.extract_slice %[[P1]][2, 0] [2, 8] [1, 1] : tensor<4x8xf32> to tensor<2x8xf32>
// CHECK: tensor.expand_shape %[[X]] {{\[}}[0], [1, 2]] output_shape [2, 8, 1]
func.func @join_part(%p0: tensor<4x8xf32>, %p1: tensor<4x8xf32>) -> tensor<2x8x1xf32> {
  %e = tensor.empty() : tensor<4x8x2xf32>
  %j0 = tensor.insert_slice %p0 into %e[0, 0, 0] [4, 8, 1] [1, 1, 1] : tensor<4x8xf32> into tensor<4x8x2xf32>
  %j1 = tensor.insert_slice %p1 into %j0[0, 0, 1] [4, 8, 1] [1, 1, 1] : tensor<4x8xf32> into tensor<4x8x2xf32>
  %a = tensor.extract_slice %j1[2, 0, 1] [2, 8, 1] [1, 1, 1] : tensor<4x8x2xf32> to tensor<2x8x1xf32>
  return %a : tensor<2x8x1xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %r = transform.air.fold_extract_slice_of_insert_slice %f : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// Two column parts of an accumulator, each written twice in turn: the first
// writes are overwritten, each part is written once.
// CHECK-LABEL: @overwritten_past_disjoint
// CHECK-SAME: (%[[Z:.*]]: tensor<4x8xf32>, %[[A:.*]]: tensor<4x6xf32>, %[[B:.*]]: tensor<4x2xf32>, %[[C:.*]]: tensor<4x6xf32>, %[[D:.*]]: tensor<4x2xf32>)
// CHECK: %[[I:.*]] = tensor.insert_slice %[[C]] into %[[Z]][0, 0] [4, 6]
// CHECK: %[[R:.*]] = tensor.insert_slice %[[D]] into %[[I]][0, 6] [4, 2]
// CHECK: return %[[R]]
func.func @overwritten_past_disjoint(%z: tensor<4x8xf32>, %a: tensor<4x6xf32>, %b: tensor<4x2xf32>, %c: tensor<4x6xf32>, %d: tensor<4x2xf32>) -> tensor<4x8xf32> {
  %0 = tensor.insert_slice %a into %z[0, 0] [4, 6] [1, 1] : tensor<4x6xf32> into tensor<4x8xf32>
  %1 = tensor.insert_slice %b into %0[0, 6] [4, 2] [1, 1] : tensor<4x2xf32> into tensor<4x8xf32>
  %2 = tensor.insert_slice %c into %1[0, 0] [4, 6] [1, 1] : tensor<4x6xf32> into tensor<4x8xf32>
  %3 = tensor.insert_slice %d into %2[0, 6] [4, 2] [1, 1] : tensor<4x2xf32> into tensor<4x8xf32>
  return %3 : tensor<4x8xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %r = transform.air.fold_extract_slice_of_insert_slice %f : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %r {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.yield
  }
}

// -----

// A row block of a tile through an expand that only added a unit dim.
// CHECK-LABEL: @slice_of_unit_expand
// CHECK-SAME: (%[[X:.*]]: tensor<4x32x128xbf16>, %[[I:.*]]: index)
// CHECK: %[[S:.*]] = tensor.extract_slice %[[X]][%[[I]], 0, 0] [1, 32, 128] [1, 1, 1] : tensor<4x32x128xbf16> to tensor<1x32x128xbf16>
// CHECK: return %[[S]]
func.func @slice_of_unit_expand(%x: tensor<4x32x128xbf16>, %i: index) -> tensor<1x32x128xbf16> {
  %e = tensor.expand_shape %x [[0], [1], [2, 3]] output_shape [4, 32, 128, 1] : tensor<4x32x128xbf16> into tensor<4x32x128x1xbf16>
  %s = tensor.extract_slice %e[%i, 0, 0, 0] [1, 32, 128, 1] [1, 1, 1, 1] : tensor<4x32x128x1xbf16> to tensor<1x32x128xbf16>
  return %s : tensor<1x32x128xbf16>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %r = transform.air.fold_extract_slice_of_insert_slice %f : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// The same box, read back with the inserted value's type, is upstream's
// extract_slice folder (which the pattern driver runs too); these patterns
// leave it to that folder.
// CHECK-LABEL: @same_box_left_to_upstream
// CHECK-NEXT: return %arg1
func.func @same_box_left_to_upstream(%d: tensor<8x16xf32>, %s: tensor<4x16xf32>) -> tensor<4x16xf32> {
  %i = tensor.insert_slice %s into %d[4, 0] [4, 16] [1, 1] : tensor<4x16xf32> into tensor<8x16xf32>
  %e = tensor.extract_slice %i[4, 0] [4, 16] [1, 1] : tensor<8x16xf32> to tensor<4x16xf32>
  return %e : tensor<4x16xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %r = transform.air.fold_extract_slice_of_insert_slice %f : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
