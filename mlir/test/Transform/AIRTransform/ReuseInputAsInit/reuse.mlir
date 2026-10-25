//===- reuse.mlir ----------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -transform-interpreter -split-input-file %s | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>

// Without `input`, the first input with the output's type and indexing map.
// CHECK-LABEL: @first_fitting_input
// CHECK-SAME: (%[[A:.*]]: tensor<64x1xbf16>, %[[B:.*]]: tensor<64x1xbf16>)
// CHECK: linalg.generic {{.*}} ins(%[[A]], %[[B]] : {{.*}}) outs(%[[A]] : tensor<64x1xbf16>)
// CHECK-NEXT: ^bb0(%{{.*}}: bf16, %[[Y:.*]]: bf16, %[[O:.*]]: bf16):
// CHECK-NEXT: arith.maximumf %[[O]], %[[Y]] : bf16
func.func @first_fitting_input(%a: tensor<64x1xbf16>, %b: tensor<64x1xbf16>) -> tensor<64x1xbf16> {
  %e = tensor.empty() : tensor<64x1xbf16>
  %r = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]}
      ins(%a, %b : tensor<64x1xbf16>, tensor<64x1xbf16>) outs(%e : tensor<64x1xbf16>) {
  ^bb0(%x: bf16, %y: bf16, %o: bf16):
    %m = arith.maximumf %x, %y : bf16
    linalg.yield %m : bf16
  } -> tensor<64x1xbf16>
  return %r : tensor<64x1xbf16>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %g = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %r = transform.air.reuse_input_as_init %g : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// `input` picks the second input.
// CHECK-LABEL: @chosen_input
// CHECK-SAME: (%[[A:.*]]: tensor<64x1xf32>, %[[B:.*]]: tensor<64x1xf32>)
// CHECK: linalg.generic {{.*}} outs(%[[B]] : tensor<64x1xf32>)
// CHECK-NEXT: ^bb0(%[[X:.*]]: f32, %{{.*}}: f32, %[[O:.*]]: f32):
// CHECK-NEXT: arith.addf %[[X]], %[[O]] : f32
func.func @chosen_input(%a: tensor<64x1xf32>, %b: tensor<64x1xf32>) -> tensor<64x1xf32> {
  %e = tensor.empty() : tensor<64x1xf32>
  %r = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]}
      ins(%a, %b : tensor<64x1xf32>, tensor<64x1xf32>) outs(%e : tensor<64x1xf32>) {
  ^bb0(%x: f32, %y: f32, %o: f32):
    %s = arith.addf %x, %y : f32
    linalg.yield %s : f32
  } -> tensor<64x1xf32>
  return %r : tensor<64x1xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %g = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %r = transform.air.reuse_input_as_init %g input = 1 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// The input is left unused; the upstream erase_unnecessary_inputs patterns
// drop it.
// CHECK-LABEL: @then_erase_input
// CHECK-SAME: (%[[A:.*]]: tensor<64x1xbf16>, %[[B:.*]]: tensor<64x1xbf16>)
// CHECK: linalg.generic {{.*}} ins(%[[B]] : tensor<64x1xbf16>) outs(%[[A]] : tensor<64x1xbf16>)
// CHECK-NEXT: ^bb0(%[[Y:.*]]: bf16, %[[O:.*]]: bf16):
// CHECK-NEXT: arith.maximumf %[[O]], %[[Y]] : bf16
func.func @then_erase_input(%a: tensor<64x1xbf16>, %b: tensor<64x1xbf16>) -> tensor<64x1xbf16> {
  %e = tensor.empty() : tensor<64x1xbf16>
  %r = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]}
      ins(%a, %b : tensor<64x1xbf16>, tensor<64x1xbf16>) outs(%e : tensor<64x1xbf16>) {
  ^bb0(%x: bf16, %y: bf16, %o: bf16):
    %m = arith.maximumf %x, %y : bf16
    linalg.yield %m : bf16
  } -> tensor<64x1xbf16>
  return %r : tensor<64x1xbf16>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %g = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %r = transform.air.reuse_input_as_init %g : (!transform.any_op) -> !transform.any_op
    %f = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f {
      transform.apply_patterns.linalg.erase_unnecessary_inputs
    } : !transform.any_op
    transform.yield
  }
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// A value updated in a loop, its rows split over a forall: with its previous
// value as the init, every tile reads and writes the loop's own buffer, with
// no allocation in the loop.
// CHECK-LABEL: @loop_update
// CHECK: memref.alloc
// CHECK: scf.for
// CHECK-NOT: memref.alloc
// CHECK: linalg.generic
// CHECK-NOT: memref.alloc
// CHECK: return
func.func @loop_update(%x: tensor<64x1xf32>, %n: index) -> tensor<64x1xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.0 : f32
  %e0 = tensor.empty() : tensor<64x1xf32>
  %init = linalg.fill ins(%cst : f32) outs(%e0 : tensor<64x1xf32>) -> tensor<64x1xf32>
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %init) -> tensor<64x1xf32> {
    %e = tensor.empty() : tensor<64x1xf32>
    %s = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]}
        ins(%acc, %x : tensor<64x1xf32>, tensor<64x1xf32>) outs(%e : tensor<64x1xf32>) {
    ^bb0(%a: f32, %b: f32, %o: f32):
      %v = arith.addf %a, %b : f32
      linalg.yield %v : f32
    } -> tensor<64x1xf32>
    scf.yield %s : tensor<64x1xf32>
  }
  return %r : tensor<64x1xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %g = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %r = transform.air.reuse_input_as_init %g : (!transform.any_op) -> !transform.any_op
    %t, %l = transform.structured.tile_using_forall %r tile_sizes [16, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %f = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %b = transform.bufferization.one_shot_bufferize %f : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
