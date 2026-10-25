//===- not_hoisted.mlir ----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -transform-interpreter -split-input-file -verify-diagnostics %s

// The expand_shape and the collapse_shape have the same types but group the
// dimensions differently, so the collapse does not undo the expand.

func.func @reshape_other_grouping(%init: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %init) -> tensor<16x16xf32> {
    %e = tensor.expand_shape %acc [[0, 1], [2]] output_shape [16, 1, 16] : tensor<16x16xf32> into tensor<16x1x16xf32>
    %x = math.exp %e : tensor<16x1x16xf32>
    %c = tensor.collapse_shape %x [[0], [1, 2]] : tensor<16x1x16xf32> into tensor<16x16xf32>
    scf.yield %c : tensor<16x16xf32>
  }
  return %r : tensor<16x16xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %loop = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{no iteration argument is packed and unpacked in the loop}}
    %new, %packs = transform.air.hoist_loop_carried_pack %loop : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// The pack's destination is an alloc_tensor that copies a value: it is not a
// fresh tensor, so it cannot be made again before the loop.

func.func @pack_dest_copies(%init: tensor<16x16xf32>, %src: tensor<2x2x8x8xf32>) -> tensor<16x16xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %init) -> tensor<16x16xf32> {
    %e = bufferization.alloc_tensor() copy(%src) : tensor<2x2x8x8xf32>
    %p = linalg.pack %acc inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %e : tensor<16x16xf32> -> tensor<2x2x8x8xf32>
    %x = math.exp %p : tensor<2x2x8x8xf32>
    %u = linalg.unpack %x inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %acc : tensor<2x2x8x8xf32> -> tensor<16x16xf32>
    scf.yield %u : tensor<16x16xf32>
  }
  return %r : tensor<16x16xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %loop = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{no iteration argument is packed and unpacked in the loop}}
    %new, %packs = transform.air.hoist_loop_carried_pack %loop : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// The unpack's destination is computed in the loop, so it cannot be made
// again after the loop.

func.func @unpack_dest_computed_in_loop(%init: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %zero = arith.constant 0.0 : f32
  %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %init) -> tensor<16x16xf32> {
    %e = tensor.empty() : tensor<2x2x8x8xf32>
    %p = linalg.pack %acc inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %e : tensor<16x16xf32> -> tensor<2x2x8x8xf32>
    %x = math.exp %p : tensor<2x2x8x8xf32>
    %d0 = tensor.empty() : tensor<16x16xf32>
    %d = linalg.fill ins(%zero : f32) outs(%d0 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %u = linalg.unpack %x inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %d : tensor<2x2x8x8xf32> -> tensor<16x16xf32>
    scf.yield %u : tensor<16x16xf32>
  }
  return %r : tensor<16x16xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %loop = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{no iteration argument is packed and unpacked in the loop}}
    %new, %packs = transform.air.hoist_loop_carried_pack %loop : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// A dynamically shaped value, its expansion's extent computed in the loop.
// Equal types do not prove equal shapes, and the extent is not available
// before the loop. It is not hoisted.

func.func @reshape_dynamic(%init: tensor<?x16xf32>, %rows: index) -> tensor<?x16xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %init) -> tensor<?x16xf32> {
    %h = arith.divui %rows, %c2 : index
    %e = tensor.expand_shape %acc [[0, 1], [2]] output_shape [%h, 2, 16] : tensor<?x16xf32> into tensor<?x2x16xf32>
    %x = math.exp %e : tensor<?x2x16xf32>
    %c = tensor.collapse_shape %x [[0, 1], [2]] : tensor<?x2x16xf32> into tensor<?x16xf32>
    scf.yield %c : tensor<?x16xf32>
  }
  return %r : tensor<?x16xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %loop = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{no iteration argument is packed and unpacked in the loop}}
    %new, %packs = transform.air.hoist_loop_carried_pack %loop : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// The carried value is what the unpack reads, not its destination: each trip
// transposes it again, so the trips do not cancel into one transpose after
// the loop.

func.func @carried_is_unpack_source(%init: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %r = scf.for %i = %c0 to %c2 step %c1 iter_args(%acc = %init) -> tensor<4x4xf32> {
    %e = tensor.empty() : tensor<4x4xf32>
    %p = linalg.pack %acc outer_dims_perm = [1, 0] inner_dims_pos = [] inner_tiles = [] into %e : tensor<4x4xf32> -> tensor<4x4xf32>
    %d = tensor.empty() : tensor<4x4xf32>
    %u = linalg.unpack %acc outer_dims_perm = [1, 0] inner_dims_pos = [] inner_tiles = [] into %d : tensor<4x4xf32> -> tensor<4x4xf32>
    scf.yield %u : tensor<4x4xf32>
  }
  return %r : tensor<4x4xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %loop = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{no iteration argument is packed and unpacked in the loop}}
    %new, %packs = transform.air.hoist_loop_carried_pack %loop : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
