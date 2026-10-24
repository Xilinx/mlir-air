//===- reuse.mlir ----------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -transform-interpreter -split-input-file -verify-diagnostics %s | FileCheck %s

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

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// The body reads the output: it is not an elementwise update of an input.
func.func @output_read(%a: tensor<64x1xf32>, %e: tensor<64x1xf32>) -> tensor<64x1xf32> {
  // expected-note @below {{when applied to this op}}
  %r = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]}
      ins(%a : tensor<64x1xf32>) outs(%e : tensor<64x1xf32>) {
  ^bb0(%x: f32, %o: f32):
    %s = arith.addf %x, %o : f32
    linalg.yield %s : f32
  } -> tensor<64x1xf32>
  return %r : tensor<64x1xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %g = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{the output is read}}
    %r = transform.air.reuse_input_as_init %g : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// A row sum: the output does not have the input's shape, and a reduction
// iterator is not an elementwise update.
func.func @reduction(%a: tensor<64x16xf32>, %e: tensor<64xf32>) -> tensor<64xf32> {
  // expected-note @below {{when applied to this op}}
  %r = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
                       iterator_types = ["parallel", "reduction"]}
      ins(%a : tensor<64x16xf32>) outs(%e : tensor<64xf32>) {
  ^bb0(%x: f32, %o: f32):
    linalg.yield %x : f32
  } -> tensor<64xf32>
  return %r : tensor<64xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %g = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{expected only parallel iterators}}
    %r = transform.air.reuse_input_as_init %g : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// The only input is broadcast along d1, so its indexing map differs.
func.func @no_fitting_input(%a: tensor<64x1xf32>, %e: tensor<64x16xf32>) -> tensor<64x16xf32> {
  // expected-note @below {{when applied to this op}}
  %r = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>],
                       iterator_types = ["parallel", "parallel"]}
      ins(%a : tensor<64x1xf32>) outs(%e : tensor<64x16xf32>) {
  ^bb0(%x: f32, %o: f32):
    linalg.yield %x : f32
  } -> tensor<64x16xf32>
  return %r : tensor<64x16xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %g = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{no input has the output's type and indexing map}}
    %r = transform.air.reuse_input_as_init %g : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

func.func @input_out_of_range(%a: tensor<64x1xf32>) -> tensor<64x1xf32> {
  %e = tensor.empty() : tensor<64x1xf32>
  // expected-note @below {{when applied to this op}}
  %r = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]}
      ins(%a : tensor<64x1xf32>) outs(%e : tensor<64x1xf32>) {
  ^bb0(%x: f32, %o: f32):
    linalg.yield %x : f32
  } -> tensor<64x1xf32>
  return %r : tensor<64x1xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %g = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{input 1 is out of range}}
    %r = transform.air.reuse_input_as_init %g input = 1 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// `input` names an input of another element type than the output: it cannot
// stand in as the output's init.
func.func @chosen_input_other_type(%a: tensor<64x1xbf16>, %b: tensor<64x1xf32>) -> tensor<64x1xf32> {
  %e = tensor.empty() : tensor<64x1xf32>
  // expected-note @below {{when applied to this op}}
  %r = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]}
      ins(%a, %b : tensor<64x1xbf16>, tensor<64x1xf32>) outs(%e : tensor<64x1xf32>) {
  ^bb0(%x: bf16, %y: f32, %o: f32):
    %xe = arith.extf %x : bf16 to f32
    %s = arith.addf %xe, %y : f32
    linalg.yield %s : f32
  } -> tensor<64x1xf32>
  return %r : tensor<64x1xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %g = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{input 0 differs from the output in type or indexing map}}
    %r = transform.air.reuse_input_as_init %g input = 0 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// Two outputs: which one the input would replace is ambiguous, so the op is
// rejected.
func.func @two_outputs(%a: tensor<64x1xf32>) -> (tensor<64x1xf32>, tensor<64x1xf32>) {
  %e0 = tensor.empty() : tensor<64x1xf32>
  %e1 = tensor.empty() : tensor<64x1xf32>
  // expected-note @below {{when applied to this op}}
  %r:2 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]}
      ins(%a : tensor<64x1xf32>) outs(%e0, %e1 : tensor<64x1xf32>, tensor<64x1xf32>) {
  ^bb0(%x: f32, %o0: f32, %o1: f32):
    %s = arith.addf %x, %x : f32
    linalg.yield %s, %x : f32, f32
  } -> (tensor<64x1xf32>, tensor<64x1xf32>)
  return %r#0, %r#1 : tensor<64x1xf32>, tensor<64x1xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %g = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{expected a linalg op on tensors with one output}}
    %r = transform.air.reuse_input_as_init %g : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
