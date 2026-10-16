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
