//===- preconditions.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -transform-interpreter -split-input-file -verify-diagnostics %s | FileCheck %s

// Two slices far apart would become one large copy: they stay.
// CHECK-LABEL: @far_apart
// CHECK-NOT: linalg.copy
func.func @far_apart(%t: tensor<4096xf32>) -> tensor<8xf32> {
  %a = tensor.extract_slice %t[0] [8] [1] : tensor<4096xf32> to tensor<8xf32>
  %b = tensor.extract_slice %t[4000] [8] [1] : tensor<4096xf32> to tensor<8xf32>
  %e = tensor.empty() : tensor<8xf32>
  %r = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%a, %b : tensor<8xf32>, tensor<8xf32>) outs(%e : tensor<8xf32>) {
  ^bb0(%x: f32, %y: f32, %z: f32):
    %s = arith.addf %x, %y : f32
    linalg.yield %s : f32
  } -> tensor<8xf32>
  return %r : tensor<8xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %g = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %c = transform.air.coalesce_slices %g : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// Row offsets that only simplify to constants, through a delinearized loop
// index: (4 * (j mod 4)) mod 4 is 0, and the other slice's is 1.
// CHECK-LABEL: @proven_constant
// CHECK: %[[R:.*]] = tensor.extract_slice %arg0[0, 0] [2, 8] [1, 1]
// CHECK: %[[C:.*]] = linalg.copy ins(%[[R]] : tensor<2x8xf32>)
// CHECK-DAG: tensor.extract_slice %[[C]][0, 0] [1, 8] [1, 1]
// CHECK-DAG: tensor.extract_slice %[[C]][1, 0] [1, 8] [1, 1]
func.func @proven_constant(%t: tensor<4x64xf32>, %j: index) -> tensor<1x8xf32> {
  %g:2 = affine.delinearize_index %j into (8, 4) : index, index
  %r0 = affine.apply affine_map<(d0) -> ((d0 * 4) mod 4)>(%g#1)
  %r1 = affine.apply affine_map<(d0) -> ((d0 * 4) mod 4 + 1)>(%g#1)
  %a = tensor.extract_slice %t[%r0, 0] [1, 8] [1, 1] : tensor<4x64xf32> to tensor<1x8xf32>
  %b = tensor.extract_slice %t[%r1, 0] [1, 8] [1, 1] : tensor<4x64xf32> to tensor<1x8xf32>
  %e = tensor.empty() : tensor<1x8xf32>
  %r = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%a, %b : tensor<1x8xf32>, tensor<1x8xf32>) outs(%e : tensor<1x8xf32>) {
  ^bb0(%x: f32, %y: f32, %z: f32):
    %s = arith.addf %x, %y : f32
    linalg.yield %s : f32
  } -> tensor<1x8xf32>
  return %r : tensor<1x8xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %g = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %c = transform.air.coalesce_slices %g : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// Targets must be destination-style ops.
func.func @not_dps(%t: tensor<8xf32>) -> tensor<8xf32> {
  return %t : tensor<8xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    // expected-error @+1 {{expects destination-style targets}}
    %c = transform.air.coalesce_slices %f : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
