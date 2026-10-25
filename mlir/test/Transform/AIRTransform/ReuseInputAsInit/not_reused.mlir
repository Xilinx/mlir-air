//===- not_reused.mlir -----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -transform-interpreter -split-input-file -verify-diagnostics %s

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
