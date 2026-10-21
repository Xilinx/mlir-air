//===- bubble.mlir ---------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -transform-interpreter -split-input-file %s | FileCheck %s

// A loop over the packets of a collapsed dim: the induction variable steps by
// the inner group's size, so the slice is one packet of the outer dim.
// CHECK-LABEL: @packet_loop
// CHECK: scf.for %[[I:.*]] = %c0 to %c32 step %c16
// CHECK: %[[P:.*]] = affine.apply #{{.*}}(%[[I]])
// CHECK: %[[S:.*]] = tensor.extract_slice %arg0[%[[P]], 0, 0] [1, 16, 4] [1, 1, 1]
// CHECK: tensor.collapse_shape %[[S]] {{\[}}[0, 1], [2]]
func.func @packet_loop(%t: tensor<2x16x4xf32>, %acc: tensor<16x4xf32>) -> tensor<16x4xf32> {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c = tensor.collapse_shape %t [[0, 1], [2]] : tensor<2x16x4xf32> into tensor<32x4xf32>
  %r = scf.for %i = %c0 to %c32 step %c16 iter_args(%a = %acc) -> tensor<16x4xf32> {
    %s = tensor.extract_slice %c[%i, 0] [16, 4] [1, 1] : tensor<32x4xf32> to tensor<16x4xf32>
    %n = arith.addf %a, %s : tensor<16x4xf32>
    scf.yield %n : tensor<16x4xf32>
  }
  return %r : tensor<16x4xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %r = transform.air.bubble_slice_through_collapse %f : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// A step that is not a multiple of the inner group: the slice may straddle
// two rows, so it stays below the collapse.
// CHECK-LABEL: @unaligned
// CHECK: tensor.collapse_shape %arg0
// CHECK: tensor.extract_slice %{{.*}}[%{{.*}}, 0] [16, 4]
func.func @unaligned(%t: tensor<2x16x4xf32>, %acc: tensor<16x4xf32>) -> tensor<16x4xf32> {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c = tensor.collapse_shape %t [[0, 1], [2]] : tensor<2x16x4xf32> into tensor<32x4xf32>
  %r = scf.for %i = %c0 to %c16 step %c8 iter_args(%a = %acc) -> tensor<16x4xf32> {
    %s = tensor.extract_slice %c[%i, 0] [16, 4] [1, 1] : tensor<32x4xf32> to tensor<16x4xf32>
    %n = arith.addf %a, %s : tensor<16x4xf32>
    scf.yield %n : tensor<16x4xf32>
  }
  return %r : tensor<16x4xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %r = transform.air.bubble_slice_through_collapse %f : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// Constant offsets go through the upstream bubble-up patterns the op also
// applies.
// CHECK-LABEL: @constant_offset
// CHECK: %[[S:.*]] = tensor.extract_slice %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}] [1, 8, 4, 16] [1, 1, 1, 1]
// CHECK: tensor.collapse_shape %[[S]]
func.func @constant_offset(%t: tensor<6x8x4x16xbf16>) -> tensor<32x16xbf16> {
  %c = tensor.collapse_shape %t [[0, 1, 2], [3]] : tensor<6x8x4x16xbf16> into tensor<192x16xbf16>
  %s = tensor.extract_slice %c[64, 0] [32, 16] [1, 1] : tensor<192x16xbf16> to tensor<32x16xbf16>
  return %s : tensor<32x16xbf16>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %r = transform.air.bubble_slice_through_collapse %f : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// An offset n * 6 with n unbounded may wrap, so it need not be a multiple of
// 6: left alone.
// CHECK-LABEL: @unbounded_offset
// CHECK: tensor.collapse_shape
// CHECK: tensor.extract_slice
func.func @unbounded_offset(%t: tensor<2x6xf32>, %n: index) -> tensor<6xf32> {
  %c6 = arith.constant 6 : index
  %o = arith.muli %n, %c6 : index
  %c = tensor.collapse_shape %t [[0, 1]] : tensor<2x6xf32> into tensor<12xf32>
  %s = tensor.extract_slice %c[%o] [6] [1] : tensor<12xf32> to tensor<6xf32>
  return %s : tensor<6xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %r = transform.air.bubble_slice_through_collapse %f : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// The slice is one row of the group's innermost dim, behind a unit dim: it
// lands inside the group, at one index of each dim before the innermost.
// CHECK-LABEL: @inner_dim
// CHECK: scf.for %[[I:.*]] = %c0 to %c64 step %c16
// CHECK: %[[O:.*]] = affine.apply #{{.*}}(%[[I]])
// CHECK: %[[P:.*]] = affine.apply #{{.*}}(%[[I]])
// CHECK: %[[S:.*]] = tensor.extract_slice %arg0[%[[O]], %[[P]], 0] [1, 1, 16] [1, 1, 1]
// CHECK: tensor.collapse_shape %[[S]] {{\[}}[0, 1, 2]]
func.func @inner_dim(%t: tensor<1x4x16xf32>, %acc: tensor<16xf32>) -> tensor<16xf32> {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c64 = arith.constant 64 : index
  %c = tensor.collapse_shape %t [[0, 1, 2]] : tensor<1x4x16xf32> into tensor<64xf32>
  %r = scf.for %i = %c0 to %c64 step %c16 iter_args(%a = %acc) -> tensor<16xf32> {
    %s = tensor.extract_slice %c[%i] [16] [1] : tensor<64xf32> to tensor<16xf32>
    %n = arith.addf %a, %s : tensor<16xf32>
    scf.yield %n : tensor<16xf32>
  }
  return %r : tensor<16xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %r = transform.air.bubble_slice_through_collapse %f : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
