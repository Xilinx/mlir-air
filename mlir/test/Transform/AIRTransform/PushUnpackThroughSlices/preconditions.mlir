//===- preconditions.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -transform-interpreter -split-input-file -verify-diagnostics %s | FileCheck %s

// An elementwise op that reads linalg.index would count packed tiles instead
// of rows on the packed layout: it stays on the unpacked values.
// CHECK-LABEL: @uses_index
// CHECK: %[[G:.*]] = linalg.unpack
// CHECK: %[[V:.*]] = linalg.unpack
// CHECK: linalg.generic {{.*}} ins(%[[G]], %[[V]] : tensor<16x16xf32>, tensor<16x16xf32>)
// CHECK: linalg.index 0
#id = affine_map<(d0, d1) -> (d0, d1)>
func.func @uses_index(%acc: tensor<4x2x8x8xf32>) -> tensor<16x16xf32> {
  %e0 = tensor.empty() : tensor<16x32xf32>
  %u = linalg.unpack %acc outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %e0 : tensor<4x2x8x8xf32> -> tensor<16x32xf32>
  %x = tensor.expand_shape %u [[0], [1, 2]] output_shape [16, 16, 2] : tensor<16x32xf32> into tensor<16x16x2xf32>
  %g = tensor.extract_slice %x[0, 0, 0] [16, 16, 1] [1, 1, 1] : tensor<16x16x2xf32> to tensor<16x16xf32>
  %v = tensor.extract_slice %x[0, 0, 1] [16, 16, 1] [1, 1, 1] : tensor<16x16x2xf32> to tensor<16x16xf32>
  %o = tensor.empty() : tensor<16x16xf32>
  %h = linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel", "parallel"]}
      ins(%g, %v : tensor<16x16xf32>, tensor<16x16xf32>) outs(%o : tensor<16x16xf32>) {
  ^bb0(%a: f32, %b: f32, %c: f32):
    %i = linalg.index 0 : index
    %ii = arith.index_cast %i : index to i32
    %fi = arith.sitofp %ii : i32 to f32
    %m = arith.mulf %a, %b : f32
    %s = arith.addf %m, %fi : f32
    linalg.yield %s : f32
  } -> tensor<16x16xf32>
  return %h : tensor<16x16xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %u = transform.structured.match ops{["linalg.unpack"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %n = transform.air.push_unpack_through_slices %u : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// When one target does not match, nothing is rewritten, the matching one
// included (the error is suppressed so the IR can be checked).
// CHECK-LABEL: @one_bad_target
// CHECK: linalg.unpack %arg0
// CHECK: linalg.unpack %arg1
func.func @one_bad_target(%acc: tensor<4x2x8x8xf32>, %acc2: tensor<4x2x8x8xf32>) -> (tensor<16x16xf32>, tensor<16x16xf32>, tensor<16x32xf32>) {
  %e0 = tensor.empty() : tensor<16x32xf32>
  %u = linalg.unpack %acc outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %e0 : tensor<4x2x8x8xf32> -> tensor<16x32xf32>
  %x = tensor.expand_shape %u [[0], [1, 2]] output_shape [16, 16, 2] : tensor<16x32xf32> into tensor<16x16x2xf32>
  %g = tensor.extract_slice %x[0, 0, 0] [16, 16, 1] [1, 1, 1] : tensor<16x16x2xf32> to tensor<16x16xf32>
  %v = tensor.extract_slice %x[0, 0, 1] [16, 16, 1] [1, 1, 1] : tensor<16x16x2xf32> to tensor<16x16xf32>
  %o = tensor.empty() : tensor<16x16xf32>
  %e1 = tensor.empty() : tensor<16x32xf32>
  %w = linalg.unpack %acc2 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %e1 : tensor<4x2x8x8xf32> -> tensor<16x32xf32>
  return %g, %v, %w : tensor<16x16xf32>, tensor<16x16xf32>, tensor<16x32xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    transform.sequence %arg0 : !transform.any_op failures(suppress) {
    ^bb0(%root: !transform.any_op):
      %u = transform.structured.match ops{["linalg.unpack"]} in %root : (!transform.any_op) -> !transform.any_op
      %n = transform.air.push_unpack_through_slices %u : (!transform.any_op) -> !transform.any_op
    }
    transform.yield
  }
}

// -----

func.func @bad_target(%acc: tensor<4x2x8x8xf32>) -> tensor<16x32xf32> {
  %e1 = tensor.empty() : tensor<16x32xf32>
  %w = linalg.unpack %acc outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %e1 : tensor<4x2x8x8xf32> -> tensor<16x32xf32>
  return %w : tensor<16x32xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %u = transform.structured.match ops{["linalg.unpack"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    // expected-error @+1 {{the unpack's user is not an expand_shape}}
    %n = transform.air.push_unpack_through_slices %u : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
