//===- outside_target.mlir -------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -transform-interpreter -split-input-file %s | FileCheck %s

// The rewrite stays inside the target. The slices inside it fold to %t; the
// insert_slice before the target loses its last use but is left in place.

// CHECK-LABEL: func.func @if_target
// CHECK-SAME: %[[T:[^:]*]]: tensor<16x1xf32>
// CHECK: tensor.insert_slice %[[T]]
// CHECK: scf.if
// CHECK-NEXT: scf.yield %[[T]]
// CHECK: else
// CHECK-NEXT: scf.yield %[[T]]
func.func @if_target(%c: i1, %a: tensor<64x1xf32>, %t: tensor<16x1xf32>) -> tensor<16x1xf32> {
  %ins = tensor.insert_slice %t into %a[0, 0] [16, 1] [1, 1] : tensor<16x1xf32> into tensor<64x1xf32>
  %r = scf.if %c -> tensor<16x1xf32> {
    %s = tensor.extract_slice %ins[0, 0] [16, 1] [1, 1] : tensor<64x1xf32> to tensor<16x1xf32>
    scf.yield %s : tensor<16x1xf32>
  } else {
    %s = tensor.extract_slice %ins[0, 0] [16, 1] [1, 1] : tensor<64x1xf32> to tensor<16x1xf32>
    scf.yield %s : tensor<16x1xf32>
  }
  return %r : tensor<16x1xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["scf.if"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %r = transform.air.fold_extract_slice_of_insert_slice %f : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: func.func @forall_target
// CHECK-SAME: %[[T:[^:]*]]: tensor<16x1xf32>
// CHECK: tensor.insert_slice %[[T]]
// CHECK: scf.forall
// CHECK: tensor.parallel_insert_slice %[[T]]
func.func @forall_target(%a: tensor<64x1xf32>, %t: tensor<16x1xf32>, %out: tensor<64x1xf32>) -> tensor<64x1xf32> {
  %ins = tensor.insert_slice %t into %a[0, 0] [16, 1] [1, 1] : tensor<16x1xf32> into tensor<64x1xf32>
  %r = scf.forall (%i) in (4) shared_outs(%o = %out) -> (tensor<64x1xf32>) {
    %off = affine.apply affine_map<(d0) -> (d0 * 16)>(%i)
    %s = tensor.extract_slice %ins[0, 0] [16, 1] [1, 1] : tensor<64x1xf32> to tensor<16x1xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %s into %o[%off, 0] [16, 1] [1, 1] : tensor<16x1xf32> into tensor<64x1xf32>
    }
  }
  return %r : tensor<64x1xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["scf.forall"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %r = transform.air.fold_extract_slice_of_insert_slice %f : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
