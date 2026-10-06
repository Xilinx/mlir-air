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

// Column offsets that only simplify to constants through a delinearized loop
// index: the K step 4 * j split into (group, row block) has row block
// (4 * j) mod 4 = 0, so 8 times it is 0, next to the constants 32 and 40.
// CHECK-LABEL: @proven_constant
// CHECK: %[[R:.*]] = tensor.extract_slice %arg0[0, 0] [1, 48] [1, 1]
// CHECK: %[[C:.*]] = linalg.copy ins(%[[R]] : tensor<1x48xi32>)
// CHECK-DAG: tensor.extract_slice %[[C]][0, 0] [1, 32] [1, 1]
// CHECK-DAG: tensor.extract_slice %[[C]][0, 32] [1, 8] [1, 1]
// CHECK-DAG: tensor.extract_slice %[[C]][0, 40] [1, 8] [1, 1]
func.func @proven_constant(%t: tensor<4x48xi32>, %j: index) -> tensor<1x32xi32> {
  %k = affine.apply affine_map<(d0) -> (d0 * 4)>(%j)
  %g:2 = affine.delinearize_index %k into (48, 4) : index, index
  %c = affine.apply affine_map<(d0) -> (d0 * 8)>(%g#1)
  %a = tensor.extract_slice %t[0, %c] [1, 32] [1, 1] : tensor<4x48xi32> to tensor<1x32xi32>
  %s = tensor.extract_slice %t[0, 32] [1, 8] [1, 1] : tensor<4x48xi32> to tensor<1x8xi32>
  %m = tensor.extract_slice %t[0, 40] [1, 8] [1, 1] : tensor<4x48xi32> to tensor<1x8xi32>
  %e = tensor.empty() : tensor<1x32xi32>
  %r = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1 mod 8)>, affine_map<(d0, d1) -> (d0, d1 mod 8)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%a, %s, %m : tensor<1x32xi32>, tensor<1x8xi32>, tensor<1x8xi32>) outs(%e : tensor<1x32xi32>) {
  ^bb0(%x: i32, %y: i32, %z: i32, %o: i32):
    %v = arith.muli %x, %y : i32
    %w = arith.addi %v, %z : i32
    linalg.yield %w : i32
  } -> tensor<1x32xi32>
  return %r : tensor<1x32xi32>
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
