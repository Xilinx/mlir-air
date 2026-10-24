//===- hoist_loop_carried_pack.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -transform-interpreter -split-input-file -verify-diagnostics %s | FileCheck %s

// A K loop whose accumulator is packed and unpacked every iteration carries
// the packed value instead: one pack before, one unpack after.

// CHECK-LABEL: func.func @accumulator
// CHECK-SAME: (%[[INIT:.*]]: tensor<16x16xf32>
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<2x2x8x8xf32>
// CHECK: %[[P0:.*]] = linalg.pack %[[INIT]] inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %[[EMPTY]]
// CHECK: %[[R:.*]]:2 = scf.for {{.*}} iter_args(%[[ACC:.*]] = %[[INIT]], %[[PACKED:.*]] = %[[P0]])
// CHECK-NOT: linalg.pack
// CHECK: %[[X:.*]] = math.exp %[[PACKED]]
// CHECK-NOT: linalg.unpack
// CHECK: scf.yield %[[ACC]], %[[X]]
// CHECK: %[[U:.*]] = linalg.unpack %[[R]]#1 inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %[[R]]#0
// CHECK: return %[[U]]
func.func @accumulator(%init: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %init) -> tensor<16x16xf32> {
    %e = tensor.empty() : tensor<2x2x8x8xf32>
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
    %new, %packs = transform.air.hoist_loop_carried_pack %loop : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// Other loop-carried values pass through untouched; the pack's destination
// may be defined outside the loop.

// CHECK-LABEL: func.func @with_offset
// CHECK: %[[P0:.*]] = linalg.pack %{{.*}} into %{{.*}}
// CHECK: %[[R:.*]]:3 = scf.for {{.*}} iter_args(%[[OFF:.*]] = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %[[P0]])
// CHECK: %[[NEXT:.*]] = arith.addi %[[OFF]]
// CHECK: scf.yield %[[NEXT]],
// CHECK: linalg.unpack %[[R]]#2
func.func @with_offset(%init: tensor<16x16xf32>, %dest: tensor<2x2x8x8xf32>, %off0: index) -> (tensor<16x16xf32>, index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %r:2 = scf.for %i = %c0 to %c4 step %c1 iter_args(%off = %off0, %acc = %init) -> (index, tensor<16x16xf32>) {
    %p = linalg.pack %acc inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %dest : tensor<16x16xf32> -> tensor<2x2x8x8xf32>
    %x = math.exp %p : tensor<2x2x8x8xf32>
    %u = linalg.unpack %x inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %acc : tensor<2x2x8x8xf32> -> tensor<16x16xf32>
    %next = arith.addi %off, %c1 : index
    scf.yield %next, %u : index, tensor<16x16xf32>
  }
  return %r#1, %r#0 : tensor<16x16xf32>, index
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %loop = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %new, %packs = transform.air.hoist_loop_carried_pack %loop : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// The accumulator is also read directly in the loop, so it is not only
// packed and unpacked: nothing to hoist.

func.func @other_use(%init: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %init) -> tensor<16x16xf32> {
    %e = tensor.empty() : tensor<2x2x8x8xf32>
    %p = linalg.pack %acc inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %e : tensor<16x16xf32> -> tensor<2x2x8x8xf32>
    %u = linalg.unpack %p inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %acc : tensor<2x2x8x8xf32> -> tensor<16x16xf32>
    %s = arith.addf %u, %acc : tensor<16x16xf32>
    scf.yield %s : tensor<16x16xf32>
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

// The pack and the unpack disagree on the layout.

func.func @layout_mismatch(%init: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %init) -> tensor<16x16xf32> {
    %e = tensor.empty() : tensor<2x2x8x8xf32>
    %p = linalg.pack %acc inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %e : tensor<16x16xf32> -> tensor<2x2x8x8xf32>
    %u = linalg.unpack %p outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %acc : tensor<2x2x8x8xf32> -> tensor<16x16xf32>
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

// The pack pads. Carrying the padded value would keep whatever the loop
// writes into the padding, which repacking every iteration resets.

func.func @padded(%init: tensor<15x15xf32>) -> tensor<15x15xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %zero = arith.constant 0.0 : f32
  %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %init) -> tensor<15x15xf32> {
    %e = tensor.empty() : tensor<2x2x8x8xf32>
    %p = linalg.pack %acc padding_value(%zero : f32) inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %e : tensor<15x15xf32> -> tensor<2x2x8x8xf32>
    %x = math.exp %p : tensor<2x2x8x8xf32>
    %u = linalg.unpack %x inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %acc : tensor<2x2x8x8xf32> -> tensor<15x15xf32>
    scf.yield %u : tensor<15x15xf32>
  }
  return %r : tensor<15x15xf32>
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

// An expand_shape and a collapse_shape with the same reassociation undo each
// other like a pack and an unpack: the loop carries the expanded value.

// CHECK-LABEL: func.func @reshape
// CHECK-SAME: (%[[INIT:.*]]: tensor<16x16xf32>
// CHECK: %[[E0:.*]] = tensor.expand_shape %[[INIT]] {{\[\[}}0, 1], [2]]
// CHECK: %[[R:.*]]:2 = scf.for {{.*}} iter_args(%[[ACC:.*]] = %[[INIT]], %[[E:.*]] = %[[E0]])
// CHECK-NOT: tensor.expand_shape
// CHECK: %[[X:.*]] = math.exp %[[E]]
// CHECK-NOT: tensor.collapse_shape
// CHECK: scf.yield %[[ACC]], %[[X]]
// CHECK: %[[C:.*]] = tensor.collapse_shape %[[R]]#1 {{\[\[}}0, 1], [2]]
// CHECK: return %[[C]]
func.func @reshape(%init: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %init) -> tensor<16x16xf32> {
    %e = tensor.expand_shape %acc [[0, 1], [2]] output_shape [2, 8, 16] : tensor<16x16xf32> into tensor<2x8x16xf32>
    %x = math.exp %e : tensor<2x8x16xf32>
    %c = tensor.collapse_shape %x [[0, 1], [2]] : tensor<2x8x16xf32> into tensor<16x16xf32>
    scf.yield %c : tensor<16x16xf32>
  }
  return %r : tensor<16x16xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %loop = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %new, %packs = transform.air.hoist_loop_carried_pack %loop : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}


// -----

// The pack writes into a bufferization.alloc_tensor and the unpack into a
// tensor.empty, both made in the loop. An unpack writes all of its
// destination, so it need not be the carried value: each destination is made
// again outside the loop for the copy there.

// CHECK-LABEL: func.func @fresh_dests
// CHECK: %[[A:.*]] = bufferization.alloc_tensor() : tensor<2x2x8x8xf32>
// CHECK: %[[P0:.*]] = linalg.pack %{{.*}} into %[[A]]
// CHECK: %[[R:.*]]:2 = scf.for {{.*}} iter_args(%{{.*}} = %{{.*}}, %[[PACKED:.*]] = %[[P0]])
// CHECK-NOT: linalg.pack
// CHECK: math.exp %[[PACKED]]
// CHECK-NOT: linalg.unpack
// CHECK: scf.yield
// CHECK: %[[D:.*]] = tensor.empty() : tensor<16x16xf32>
// CHECK: %[[U:.*]] = linalg.unpack %[[R]]#1 {{.*}} into %[[D]]
// CHECK: return %[[U]]
func.func @fresh_dests(%init: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %init) -> tensor<16x16xf32> {
    %e = bufferization.alloc_tensor() : tensor<2x2x8x8xf32>
    %p = linalg.pack %acc inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %e : tensor<16x16xf32> -> tensor<2x2x8x8xf32>
    %x = math.exp %p : tensor<2x2x8x8xf32>
    %d = tensor.empty() : tensor<16x16xf32>
    %u = linalg.unpack %x inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %d : tensor<2x2x8x8xf32> -> tensor<16x16xf32>
    scf.yield %u : tensor<16x16xf32>
  }
  return %r : tensor<16x16xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %loop = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %new, %packs = transform.air.hoist_loop_carried_pack %loop : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

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

// The unpack's destination is a tensor.empty sized by the induction variable:
// it cannot be made again after the loop.

func.func @unpack_dest_sized_in_loop(%init: tensor<?x16xf32>, %rows: index) -> tensor<?x16xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %tiles = arith.divui %rows, %c8 : index
  %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %init) -> tensor<?x16xf32> {
    %e = tensor.empty(%tiles) : tensor<?x2x8x8xf32>
    %p = linalg.pack %acc inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %e : tensor<?x16xf32> -> tensor<?x2x8x8xf32>
    %x = math.exp %p : tensor<?x2x8x8xf32>
    %n = arith.addi %rows, %i : index
    %d = tensor.empty(%n) : tensor<?x16xf32>
    %u = linalg.unpack %x inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %d : tensor<?x2x8x8xf32> -> tensor<?x16xf32>
    scf.yield %u : tensor<?x16xf32>
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
